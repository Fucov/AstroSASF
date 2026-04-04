"""
AstroSASF · Infra · Prefix-Aware Load Balancer + Intent-Aware Model Router (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
跨节点前缀感知负载均衡器 + 异构算力意图路由。

V7.5 核心新增：异构计算调度
- **异构算力池**：实例按 compute_class 分为 heavy（7B+）和 light（1.5B）
- **意图感知路由**：基于 agent_id / tags 识别请求类型
    Planner Agent（复杂 DAG 规划）→ heavy（强制）
    Executor/QA Agent（简单 Tool Calling）→ light（优先）
- **Prefix Hash + 异构标签**：Hash 路由优先命中历史实例，结合 compute_class 过滤

Author: AstroSASF Team
Version: 7.5
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
    PREFIX_AFFINITY = "prefix_affinity"       # 前缀亲和（KV-Cache 复用）
    LEAST_CONNECTIONS = "least_connections"    # 最少连接
    WEIGHTED = "weighted"                       # 权重加权
    FAILOVER = "failover"                       # 故障转移
    INTENT_DIRECTED = "intent_directed"         # V7.5 意图路由（compute_class 过滤后最少连接）
    COMPUTE_DOWNGRADE = "compute_downgrade"     # V7.5 算力降级路由


@dataclass(frozen=True)
class RoutingDecision:
    """路由决策结果。

    V7.5 新增字段：
    - compute_class: 目标实例的算力分级（"heavy" | "light"）
    - is_downgraded: 是否经过算力降级
    - required_compute_class: 请求原本要求的算力级别（降级时与 compute_class 不同）
    """
    url: str
    strategy: RoutingStrategy
    reason: str
    prefix_hash: str
    vram_ratio: float
    active_connections: int
    is_cached: bool   # 是否命中 KV-Cache（历史路由成功）
    # V7.5 新增：算力分级
    compute_class: str = "heavy"
    is_downgraded: bool = False
    required_compute_class: str | None = None
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
    # V7.5 新增：意图感知
    agent_id: str | None = None
    required_compute_class: str = "heavy"


# --------------------------------------------------------------------------- #
#  Agent Intent Detection                                                      #
# --------------------------------------------------------------------------- #

class AgentIntent(_Enum):
    """V7.5 新增：Agent 意图枚举，决定算力分发策略。"""
    PLANNER = "planner"           # 复杂 DAG 规划 → heavy
    EXECUTOR = "executor"         # 简单 Tool Calling → light
    QA = "qa"                     # 问答 / 参数提取 → light
    FALLBACK = "fallback"         # 兜底默认 → heavy


# V7.5 新增：agent_id 前缀 → AgentIntent 映射规则
_AGENT_INTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^planner[_-]", re.IGNORECASE), "planner"),
    (re.compile(r"^executor[_-]", re.IGNORECASE), "executor"),
    (re.compile(r"^qa[_-]", re.IGNORECASE), "qa"),
    (re.compile(r"^worker[_-]", re.IGNORECASE), "executor"),
]

# V7.5 新增：tag → AgentIntent 映射规则（更灵活）
_TAG_INTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bplanner\b", re.IGNORECASE), "planner"),
    (re.compile(r"\bexecutor\b", re.IGNORECASE), "executor"),
    (re.compile(r"\b(exec|tool)\b", re.IGNORECASE), "executor"),
    (re.compile(r"\b(qa|question|answer)\b", re.IGNORECASE), "qa"),
]


def detect_agent_intent(agent_id: str | None, tags: list[str]) -> AgentIntent:
    """V7.5 新增：检测 Agent 意图类型。

    检测顺序：agent_id 前缀匹配 → tags 包含关键词匹配 → FALLBACK
    """
    if agent_id:
        for pattern, intent_name in _AGENT_INTENT_RULES:
            if pattern.match(agent_id):
                return AgentIntent(intent_name)

    all_tags = " ".join(tags).lower()
    for pattern, intent_name in _TAG_INTENT_RULES:
        if pattern.search(all_tags):
            return AgentIntent(intent_name)

    return AgentIntent.FALLBACK


def intent_to_compute_class(intent: AgentIntent) -> str:
    """V7.5 新增：将 Agent 意图转换为所需的算力级别。"""
    return {
        AgentIntent.PLANNER: "heavy",
        AgentIntent.EXECUTOR: "light",
        AgentIntent.QA: "light",
        AgentIntent.FALLBACK: "heavy",
    }.get(intent, "heavy")


# --------------------------------------------------------------------------- #
#  PrefixAwareLoadBalancer                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class PrefixAwareLoadBalancer:
    """前缀感知负载均衡器 + 异构算力意图路由（内核模块）。

    V7.5 核心工作流程：

    1. **意图检测**：从 agent_id / tags 识别 Agent 类型（Planner/Executor/QA）
    2. **算力映射**：Planner → heavy | Executor/QA → light
    3. **前缀哈希**：提取 Prompt 稳定前缀，计算 Prefix Hash（复用 KV-Cache）
    4. **亲和路由**：Hash 命中 + compute_class 匹配 + 实例健康 → 直连
    5. **降级路由**：Hash 命中但 compute_class 不匹配 → 降级为 compute_class 过滤后的最少连接
    6. **兜底路由**：Hash 未命中 → 按 compute_class 过滤后最少连接

    与 SGLang RadixAttention 配合：相同 Prefix Hash 的请求路由到历史处理实例，
    最大化 KV-Cache 命中率。
    """

    instance_pool: LLMInstancePool
    _routing_table: dict[str, list[str]] = field(default_factory=dict)
    _instance_prefixes: dict[str, set[str]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    # V7.5 新增：意图路由配置（默认从 config_loader 注入）
    _allow_downgrade: bool = field(default=True, init=False)
    _default_compute_class: str = field(default="heavy", init=False)

    def __post_init__(self) -> None:
        self.instance_pool.on_status_change(self._on_instance_status_change)

    @property
    def routing_table_size(self) -> int:
        return len(self._routing_table)

    def set_intent_config(self, default_compute_class: str, allow_downgrade: bool) -> None:
        """V7.5 新增：注入意图路由配置（在 GatewayProxy 初始化时调用）。"""
        self._default_compute_class = default_compute_class
        self._allow_downgrade = allow_downgrade

    def extract_prefix(self, prompt: str, system_prompt: str | None = None) -> str:
        """从 prompt 中提取稳定的前缀。"""
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

    async def route(
        self,
        prompt: str,
        system_prompt: str | None = None,
        priority: str = "NORMAL",
        tags: list[str] | None = None,
        prefer_cached: bool = True,
        agent_id: str | None = None,
        force_compute_class: str | None = None,
    ) -> RoutingDecision:
        """执行路由决策（V7.5 增强版：异构算力意图路由）。

        Parameters
        ----------
        agent_id : str | None
            V7.5 新增：Agent 标识符，用于意图检测（e.g. "planner_agent_01"）
        force_compute_class : str | None
            V7.5 新增：强制指定算力级别（覆盖意图检测结果）

        Returns
        -------
        RoutingDecision
            包含 url / strategy / compute_class / is_downgraded 等完整路由结果
        """
        if not self._running:
            raise RuntimeError("LoadBalancer 未启动，请先调用 start()")

        tags = tags or []
        prefix_hash = self.compute_prefix_hash(prompt, system_prompt)

        # ── Step 1: 意图检测 & 算力映射（V7.5 新增）───────────────────────── #
        if force_compute_class is not None:
            required_compute_class = force_compute_class
        else:
            intent = detect_agent_intent(agent_id, tags)
            required_compute_class = intent_to_compute_class(intent)
            logger.debug(
                "[LoadBalancer] 意图检测: agent_id=%s intent=%s → compute_class=%s",
                agent_id, intent.value, required_compute_class,
            )

        # ── Step 2: CRITICAL 任务跳过亲和路由 ──────────────────────────────── #
        if priority == "CRITICAL" and not prefer_cached:
            instance = self._select_least_loaded(
                tags=tags,
                compute_class=required_compute_class,
            )
            if instance is None:
                # 降级：CRITICAL 不降级，但放宽算力要求
                instance = self._select_least_loaded(tags=tags, compute_class=None)
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
                compute_class=self.instance_pool.configs[instance.url].compute_class,
            )

        async with self._lock:
            historical_urls = self._routing_table.get(prefix_hash, [])

            # ── Step 3: 亲和路由（带 compute_class 过滤）────────────────── #
            if historical_urls and prefer_cached:
                for url in historical_urls:
                    instance = self.instance_pool.get_instance(url)
                    if not instance or not instance.is_available:
                        continue

                    instance_cc = self.instance_pool.configs[url].compute_class
                    # 算力级别不匹配：记录降级，跳过
                    if instance_cc != required_compute_class:
                        continue

                    # VRAM 过载（对于非 CRITICAL 任务）
                    if priority != "CRITICAL" and instance.vram_ratio >= VRAM_HIGH_WATERMARK:
                        continue

                    self._stats["prefix_hit"] += 1
                    self._stats[f"compute_class_{instance_cc}_hit"] += 1
                    return RoutingDecision(
                        url=url,
                        strategy=RoutingStrategy.PREFIX_AFFINITY,
                        reason=f"KV-Cache 亲和路由 (前缀={prefix_hash}, compute_class={instance_cc})",
                        prefix_hash=prefix_hash,
                        vram_ratio=instance.vram_ratio,
                        active_connections=instance.active_connections,
                        is_cached=True,
                        compute_class=instance_cc,
                    )

                # 历史路由存在但算力级别不匹配：计入意图降级统计
                matched_historical = [
                    url for url in historical_urls
                    if self.instance_pool.get_instance(url) is not None
                ]
                if matched_historical and self._allow_downgrade:
                    self._stats["intent_mismatch_downgrade"] += 1
                    logger.debug(
                        "[LoadBalancer] 历史路由算力不匹配（%s 要求 %s），尝试降级路由",
                        matched_historical[0], required_compute_class,
                    )

            # ── Step 4: 意图降级路由（V7.5 新增）─────────────────────────── #
            if required_compute_class == "heavy" and self._allow_downgrade:
                # 高算力请求，但所有 heavy 实例都不可用或 VRAM 过载
                # 透明降级到 light 实例
                light_instance = self._select_least_loaded(tags=tags, compute_class="light")
                if light_instance is not None:
                    self._stats["compute_downgrade_heavy_to_light"] += 1
                    return RoutingDecision(
                        url=light_instance.url,
                        strategy=RoutingStrategy.COMPUTE_DOWNGRADE,
                        reason=(
                            f"高算力实例不可用，透明降级到 low 算力 "
                            f"(原要求={required_compute_class}, 实际={light_instance.url})"
                        ),
                        prefix_hash=prefix_hash,
                        vram_ratio=light_instance.vram_ratio,
                        active_connections=light_instance.active_connections,
                        is_cached=False,
                        compute_class="light",
                        is_downgraded=True,
                        required_compute_class=required_compute_class,
                    )

            # ── Step 5: Least-Connections 降级（按意图算力过滤）───────────── #
            instance = self._select_least_loaded(tags=tags, compute_class=required_compute_class)
            if instance is None and required_compute_class == "heavy" and self._allow_downgrade:
                # 兜底：尝试 light
                instance = self._select_least_loaded(tags=tags, compute_class="light")
                if instance is not None:
                    self._stats["fallback_to_light"] += 1

            if instance is None:
                raise RuntimeError("无可用 LLM 实例")

            selected_cc = self.instance_pool.configs[instance.url].compute_class
            is_downgraded = (
                required_compute_class == "heavy"
                and selected_cc == "light"
            )
            strategy = RoutingStrategy.COMPUTE_DOWNGRADE if is_downgraded else RoutingStrategy.LEAST_CONNECTIONS

            self._stats["least_connections"] += 1
            return RoutingDecision(
                url=instance.url,
                strategy=strategy,
                reason=(
                    f"前缀未命中，降级为最少连接 "
                    f"(compute_class={selected_cc}, {'透明降级' if is_downgraded else '正常'})"
                ),
                prefix_hash=prefix_hash,
                vram_ratio=instance.vram_ratio,
                active_connections=instance.active_connections,
                is_cached=False,
                compute_class=selected_cc,
                is_downgraded=is_downgraded,
                required_compute_class=required_compute_class if is_downgraded else None,
            )

    def _select_least_loaded(
        self,
        tags: list[str] | None = None,
        compute_class: str | None = None,
    ) -> InstanceMetrics | None:
        """V7.5 增强：按 compute_class 过滤后的最少连接选择。"""
        return self.instance_pool.get_least_loaded_instance(
            tags=tags,
            require_healthy=False,
            compute_class=compute_class,
        )

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

    def _on_instance_status_change(self, url: str, old_status: InstanceStatus, new_status: InstanceStatus) -> None:
        """实例状态变更回调（同步）。"""
        if new_status == InstanceStatus.UNHEALTHY:
            with self._lock:
                for prefix_hash, history in self._routing_table.items():
                    if url in history:
                        history.remove(url)
                if url in self._instance_prefixes:
                    del self._instance_prefixes[url]

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    def get_stats(self) -> dict[str, Any]:
        total = sum(
            v for k, v in self._stats.items()
            if k not in ("compute_downgrade_heavy_to_light", "intent_mismatch_downgrade",
                         "fallback_to_light", "compute_class_heavy_hit", "compute_class_light_hit")
        )
        return {
            "total_routes": total,
            "prefix_hit_rate": f"{self._stats.get('prefix_hit', 0) / max(1, total) * 100:.1f}%",
            "affinity_downgrades": self._stats.get("affinity_downgrade", 0),
            "compute_downgrade_count": self._stats.get("compute_downgrade_heavy_to_light", 0),
            "intent_mismatch_count": self._stats.get("intent_mismatch_downgrade", 0),
            "fallback_to_light_count": self._stats.get("fallback_to_light", 0),
            "heavy_routes": self._stats.get("compute_class_heavy_hit", 0),
            "light_routes": self._stats.get("compute_class_light_hit", 0),
            "routing_table_size": len(self._routing_table),
            "instance_prefix_count": {url: len(prefixes) for url, prefixes in self._instance_prefixes.items()},
        }
