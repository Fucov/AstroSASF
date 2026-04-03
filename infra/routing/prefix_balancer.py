"""
AstroSASF · Infra · Prefix-Aware Load Balancer (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
跨节点前缀感知负载均衡器。

核心功能：
1. **前缀哈希提取**：从 Prompt 中提取 SOP/System 前缀，计算 Prefix Hash
2. **全局路由表**：维护 "Prefix Hash -> 最近处理过该前缀的 IP:Port 列表"
3. **智能路由策略**：
   - Hash 命中 → 优先路由到历史实例（KV-Cache 复用）
   - Hash 未命中或实例不可用 → 降级为 Least-Connections

支持 SGLang RadixAttention 的 KV-Cache 复用感知。

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum as _Enum
from typing import Any

from infra.llm.instance_pool import (
    InstanceMetrics,
    LLMInstancePool,
    VRAM_HIGH_WATERMARK,
    InstanceStatus,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Prefix Extraction Patterns                                                  #
# --------------------------------------------------------------------------- #

SOP_PATTERN = re.compile(
    r'^\s*(?:SOP|sop|SOP:)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[-:\n]',
    re.IGNORECASE,
)

SYSTEM_PATTERN = re.compile(
    r'^\s*(?:SYSTEM|system|System)[:\s]',
    re.IGNORECASE,
)

CODE_BLOCK_PATTERN = re.compile(
    r'^\s*(?:```json|```markdown|```yaml|```)\s*\n?(.*?)(?:\n?```)',
    re.DOTALL,
)

TASK_PREFIX_PATTERN = re.compile(
    r'^\s*(?:请|帮我|我想|执行|开始|进行|完成)\s+(.{10,50}?)[\n。]',
    re.UNICODE,
)


# --------------------------------------------------------------------------- #
#  Routing Decision                                                            #
# --------------------------------------------------------------------------- #

class RoutingStrategy(_Enum):
    """路由策略枚举。"""
    PREFIX_AFFINITY = "prefix_affinity"   # 前缀亲和（KV-Cache 复用）
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    WEIGHTED = "weighted"                 # 权重加权
    FAILOVER = "failover"                # 故障转移


@dataclass(frozen=True)
class RoutingDecision:
    """路由决策结果。"""
    url: str
    strategy: RoutingStrategy
    reason: str
    prefix_hash: str
    vram_ratio: float
    active_connections: int
    is_cached: bool  # 是否命中 KV-Cache（历史路由成功）
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class RoutingContext:
    """路由上下文（用于决策分析）。"""
    prompt: str
    system_prompt: str | None
    prefix_hash: str
    priority: str  # "CRITICAL", "HIGH", "NORMAL", "LOW"
    tags: list[str]
    prefer_cached: bool = True


# --------------------------------------------------------------------------- #
#  PrefixAwareLoadBalancer                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class PrefixAwareLoadBalancer:
    """前缀感知负载均衡器（内核模块）。

    工作原理：
    1. **前缀提取**：从 prompt 中提取 SOP/System/任务前缀，计算稳定 Hash
    2. **全局路由表**：记录每个 Prefix Hash 最近处理过的实例
    3. **亲和路由**：Hash 命中且实例健康 → 直接路由（复用 KV-Cache）
    4. **降级路由**：Hash 未命中或实例不健康 → Least-Connections

    Example
    -------
    >>> balancer = PrefixAwareLoadBalancer(instance_pool=pool)
    >>> await balancer.start()
    >>> decision = await balancer.route(
    ...     prompt="SOP:bio_culture\n请帮我开始细胞培养实验",
    ...     system_prompt="你是太空实验柜助手",
    ...     priority="NORMAL",
    ... )
    >>> print(decision.url)  # "http://192.168.1.10:8000"
    >>> print(decision.strategy)  # RoutingStrategy.PREFIX_AFFINITY
    """

    instance_pool: LLMInstancePool
    _routing_table: dict[str, list[str]] = field(default_factory=dict)
    _instance_prefixes: dict[str, set[str]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _running: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.instance_pool.on_status_change(self._on_instance_status_change)

    @property
    def routing_table_size(self) -> int:
        """路由表大小。"""
        return len(self._routing_table)

    # ------------------------------------------------------------------------ #
    #  Prefix Extraction                                                        #
    # ------------------------------------------------------------------------ #

    def extract_prefix(self, prompt: str, system_prompt: str | None = None) -> str:
        """从 prompt 中提取稳定的前缀。

        优先级：
        1. SOP 标识符（SOP:bio_culture）
        2. System Prompt 片段
        3. 任务描述前缀
        4. 完整 prompt（兜底）
        """
        prompt = prompt.strip()

        # 1. 提取 SOP 标识符
        match = SOP_PATTERN.match(prompt)
        if match:
            sop_name = match.group(1)
            return f"SOP:{sop_name}"

        # 2. 检查 System Prompt 片段
        if system_prompt:
            system_prompt = system_prompt.strip()
            if len(system_prompt) > 200:
                system_prefix = system_prompt[:200]
            else:
                system_prefix = system_prompt
            return system_prefix

        # 3. 提取任务描述前缀
        match = TASK_PREFIX_PATTERN.match(prompt)
        if match:
            task_desc = match.group(1).strip()
            return task_desc[:80]

        # 4. 代码块内容提取
        match = CODE_BLOCK_PATTERN.match(prompt)
        if match:
            code_content = match.group(1).strip()[:200]
            return code_content

        # 5. 兜底：取 prompt 前 100 字符
        return prompt[:100]

    def compute_prefix_hash(self, prompt: str, system_prompt: str | None = None) -> str:
        """计算 prompt 前缀的稳定哈希（SHA256 前 8 字节 = 16 字符十六进制）。"""
        prefix = self.extract_prefix(prompt, system_prompt)
        hash_bytes = hashlib.sha256(prefix.encode("utf-8")).digest()
        return hash_bytes[:8].hex()

    # ------------------------------------------------------------------------ #
    #  Routing Logic                                                           #
    # ------------------------------------------------------------------------ #

    async def route(self, prompt: str, system_prompt: str | None = None,
                   priority: str = "NORMAL",
                   tags: list[str] | None = None,
                   prefer_cached: bool = True) -> RoutingDecision:
        """执行路由决策。

        决策流程：
        1. 计算 Prefix Hash
        2. 检查路由表是否有命中实例
        3. 验证命中实例是否健康、显存是否告警
        4. 若命中且健康 → 返回亲和路由
        5. 若未命中或不健康 → 降级为 Least-Connections
        """
        if not self._running:
            raise RuntimeError("LoadBalancer 未启动，请先调用 start()")

        prefix_hash = self.compute_prefix_hash(prompt, system_prompt)
        tags = tags or []

        async with self._lock:
            historical_urls = self._routing_table.get(prefix_hash, [])

            # CRITICAL 任务：跳过亲和路由，强制选择最健康实例
            if priority == "CRITICAL" and not prefer_cached:
                logger.debug(
                    "[LoadBalancer] CRITICAL 任务跳过亲和路由: hash=%s", prefix_hash
                )
                instance = self._select_least_loaded(tags=tags)
                if instance is None:
                    raise RuntimeError("无可用 LLM 实例")
                return RoutingDecision(
                    url=instance.url,
                    strategy=RoutingStrategy.LEAST_CONNECTIONS,
                    reason="CRITICAL 任务强制重新选择",
                    prefix_hash=prefix_hash,
                    vram_ratio=instance.vram_ratio,
                    active_connections=instance.active_connections,
                    is_cached=False,
                )

            # 策略 1：尝试亲和路由（SGLang RadixAttention KV-Cache 复用）
            if historical_urls and prefer_cached:
                for url in historical_urls:
                    instance = self.instance_pool.get_instance(url)
                    if instance and instance.is_available:
                        if priority != "CRITICAL" and instance.vram_ratio >= VRAM_HIGH_WATERMARK:
                            logger.debug(
                                "[LoadBalancer] 实例 %s 显存告警 (%.1f%%)，跳过亲和路由",
                                url, instance.vram_ratio * 100
                            )
                            continue
                        self._stats["prefix_hit"] += 1
                        logger.debug(
                            "[LoadBalancer] 前缀命中 (SGLang KV-Cache 复用): hash=%s -> %s (VRAM: %.1f%%, conn: %d)",
                            prefix_hash, url, instance.vram_ratio * 100, instance.active_connections
                        )
                        return RoutingDecision(
                            url=url,
                            strategy=RoutingStrategy.PREFIX_AFFINITY,
                            reason=f"KV-Cache 亲和路由 (历史处理过该前缀，支持 SGLang RadixAttention)",
                            prefix_hash=prefix_hash,
                            vram_ratio=instance.vram_ratio,
                            active_connections=instance.active_connections,
                            is_cached=True,
                        )

            # 策略 2：Least-Connections 降级
            self._stats["least_connections"] += 1
            instance = self._select_least_loaded(tags=tags)
            if instance is None:
                raise RuntimeError("无可用 LLM 实例")

            logger.debug(
                "[LoadBalancer] 降级路由: hash=%s -> %s (Least-Connections)",
                prefix_hash, instance.url
            )
            return RoutingDecision(
                url=instance.url,
                strategy=RoutingStrategy.LEAST_CONNECTIONS,
                reason="前缀未命中或实例不可用，降级为最少连接",
                prefix_hash=prefix_hash,
                vram_ratio=instance.vram_ratio,
                active_connections=instance.active_connections,
                is_cached=False,
            )

    def _select_least_loaded(self, tags: list[str] | None = None) -> InstanceMetrics | None:
        """选择负载最轻的实例。"""
        return self.instance_pool.get_least_loaded_instance(
            tags=tags,
            require_healthy=False,
        )

    # ------------------------------------------------------------------------ #
    #  Post-Route Updates                                                       #
    # ------------------------------------------------------------------------ #

    async def record_route_success(self, decision: RoutingDecision) -> None:
        """记录路由成功，更新路由表。"""
        if not decision.is_cached:
            async with self._lock:
                history = self._routing_table.get(decision.prefix_hash, [])
                if history and history[0] == decision.url:
                    return

                if decision.url in history:
                    history.remove(decision.url)
                history.insert(0, decision.url)
                self._routing_table[decision.prefix_hash] = history[:5]

                if decision.url not in self._instance_prefixes:
                    self._instance_prefixes[decision.url] = set()
                self._instance_prefixes[decision.url].add(decision.prefix_hash)

                logger.debug(
                    "[LoadBalancer] 路由表更新: hash=%s -> %s",
                    decision.prefix_hash, decision.url
                )

    async def record_route_failure(self, decision: RoutingDecision) -> None:
        """记录路由失败，降级实例热度。"""
        if not decision.is_cached:
            return

        async with self._lock:
            history = self._routing_table.get(decision.prefix_hash, [])
            if decision.url in history:
                history.remove(decision.url)
                self._routing_table[decision.prefix_hash] = history

                if decision.url in self._instance_prefixes:
                    self._instance_prefixes[decision.url].discard(decision.prefix_hash)

                self._stats["affinity_downgrade"] += 1
                logger.warning(
                    "[LoadBalancer] 亲和路由降级: hash=%s -> %s 失败，已移除",
                    decision.prefix_hash, decision.url
                )

    # ------------------------------------------------------------------------ #
    #  Instance Status Change Handler                                            #
    # ------------------------------------------------------------------------ #

    def _on_instance_status_change(self, url: str, old_status: InstanceStatus, new_status: InstanceStatus) -> None:
        """实例状态变更回调（同步）。当实例变为 UNHEALTHY 时，清除该实例的所有亲和路由记录。"""
        if new_status == InstanceStatus.UNHEALTHY:
            affected_hashes = []
            with self._lock:
                for prefix_hash, history in self._routing_table.items():
                    if url in history:
                        history.remove(url)
                        affected_hashes.append(prefix_hash)

                if url in self._instance_prefixes:
                    del self._instance_prefixes[url]

            if affected_hashes:
                logger.warning(
                    "[LoadBalancer] 实例 %s 不可用，清除了 %d 条亲和路由记录",
                    url, len(affected_hashes)
                )

    # ------------------------------------------------------------------------ #
    #  Lifecycle                                                                 #
    # ------------------------------------------------------------------------ #

    async def start(self) -> None:
        """启动负载均衡器。"""
        self._running = True
        logger.info(
            "[LoadBalancer] 前缀感知负载均衡器已启动，初始路由表: %d 条",
            len(self._routing_table)
        )

    async def stop(self) -> None:
        """停止负载均衡器。"""
        self._running = False
        logger.info("[LoadBalancer] 前缀感知负载均衡器已停止")

    # ------------------------------------------------------------------------ #
    #  Stats & Debug                                                            #
    # ------------------------------------------------------------------------ #

    def get_routing_table_snapshot(self) -> dict[str, Any]:
        """获取路由表快照（用于调试）。"""
        return {
            "routing_table_size": len(self._routing_table),
            "entries": {
                hash_val: {
                    "urls": urls,
                    "primary_url": urls[0] if urls else None,
                }
                for hash_val, urls in self._routing_table.items()
            },
            "stats": dict(self._stats),
        }

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。"""
        total = sum(self._stats.values())
        return {
            "total_routes": total,
            "prefix_hit_rate": f"{self._stats.get('prefix_hit', 0) / max(1, total) * 100:.1f}%",
            "affinity_downgrades": self._stats.get("affinity_downgrade", 0),
            "routing_table_size": len(self._routing_table),
            "instance_prefix_count": {url: len(prefixes) for url, prefixes in self._instance_prefixes.items()},
        }
