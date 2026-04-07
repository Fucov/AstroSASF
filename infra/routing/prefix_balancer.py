"""
AstroSASF · Infra · Intent-Aware Heterogeneous Router + Time-Dimension Priority Switch (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
异构意图感知路由引擎 + 时间维度优先级切换 + 透明算力降级。

V8.0 核心新增（按需求规格逐条实现）：
1. **Intent-Aware 分流**：detect_agent_intent() 根据 agent_id 前缀自动映射至异构算力池。
   Planner → heavy (7B+) | Executor/QA → light (1.5B)
2. **透明降级机制 (Failover)**：当 heavy 实例 VRAM 触发 W_high (0.85) 阈值时，
   实现透明的模型名称重写（如 qwen-7b → qwen-1.5b）并路由至 light 池。
3. **前缀哈希亲和性**：确保 SHA-256 哈希路由与 compute_class 标签联动，
   实现跨节点的 KV-Cache 状态复用。
4. **时间维度优先级切换**：VRAM 水位动态调整路由策略，高负载时自动降级。

Author: AstroSASF Team
Version: 8.0
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
    VRAM_CRITICAL_WATERMARK,
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
#  Routing Decision & Context                                                  #
# --------------------------------------------------------------------------- #

class RoutingStrategy(_Enum):
    """V8.0 路由策略枚举。"""
    PREFIX_AFFINITY = "prefix_affinity"           # 前缀亲和（KV-Cache 复用）
    LEAST_CONNECTIONS = "least_connections"       # 最少连接
    WEIGHTED = "weighted"                         # 权重加权
    FAILOVER = "failover"                         # 故障转移
    INTENT_DIRECTED = "intent_directed"           # V8.0 意图路由（compute_class 过滤后最少连接）
    COMPUTE_DOWNGRADE = "compute_downgrade"       # V8.0 算力降级路由（VRAM 触发透明降级）
    TIME_PRIORITY_SWITCH = "time_priority_switch" # V8.0 时间维度优先级切换


# V8.0 新增：模型降级映射表（支持多级降级）
_MODEL_DOWNGRADE_MAP: dict[str, str] = {
    "qwen2.5:7b": "qwen2.5:1.5b",
    "qwen2.5:14b": "qwen2.5:7b",
    "qwen2.5:32b": "qwen2.5:14b",
    "qwen2.5:72b": "qwen2.5:32b",
    "llama3:8b": "llama3:1.5b",
    "llama3:70b": "llama3:8b",
    "deepseek:7b": "deepseek:1.5b",
    "deepseek:67b": "deepseek:7b",
    "mixtral:8x7b": "mixtral:8x22b",
    "mixtral:8x22b": "mixtral:8x7b",
}


@dataclass(frozen=True)
class RoutingDecision:
    """V8.0 路由决策结果（含完整异构降级元数据）。"""
    url: str
    strategy: RoutingStrategy
    reason: str
    prefix_hash: str
    vram_ratio: float
    active_connections: int
    is_cached: bool
    # 算力分级
    compute_class: str = "heavy"
    is_downgraded: bool = False
    required_compute_class: str | None = None
    # V8.0 新增：透明模型重写
    original_model_name: str | None = None
    effective_model_name: str | None = None
    # V8.0 新增：时间维度优先级
    time_priority: int = 0   # 时间维度优先级（0=正常，1=降级中，2=高负载）
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class RoutingContext:
    """路由上下文（用于决策分析）。"""
    prompt: str
    system_prompt: str | None
    prefix_hash: str
    priority: str
    tags: list[str]
    prefer_cached: bool = True
    agent_id: str | None = None
    required_compute_class: str = "heavy"


# --------------------------------------------------------------------------- #
#  Agent Intent Detection (V8.0 增强)                                         #
# --------------------------------------------------------------------------- #

class AgentIntent(_Enum):
    """V8.0 Agent 意图枚举。"""
    PLANNER = "planner"           # 复杂 DAG 规划 → heavy
    EXECUTOR = "executor"         # 简单 Tool Calling → light
    QA = "qa"                     # 问答 / 参数提取 → light
    FALLBACK = "fallback"         # 兜底默认 → heavy


# agent_id 前缀 → AgentIntent 映射规则
_AGENT_INTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^planner[_-]", re.IGNORECASE), "planner"),
    (re.compile(r"^executor[_-]", re.IGNORECASE), "executor"),
    (re.compile(r"^qa[_-]", re.IGNORECASE), "qa"),
    (re.compile(r"^worker[_-]", re.IGNORECASE), "executor"),
    (re.compile(r"^dag[_-]", re.IGNORECASE), "planner"),
    (re.compile(r"^sop[_-]", re.IGNORECASE), "planner"),
]

# tag → AgentIntent 映射规则
_TAG_INTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bplanner\b", re.IGNORECASE), "planner"),
    (re.compile(r"\bexecutor\b", re.IGNORECASE), "executor"),
    (re.compile(r"\b(exec|tool|tool_call)\b", re.IGNORECASE), "executor"),
    (re.compile(r"\b(qa|question|answer|query)\b", re.IGNORECASE), "qa"),
    (re.compile(r"\bdag\b", re.IGNORECASE), "planner"),
]


def detect_agent_intent(agent_id: str | None, tags: list[str]) -> AgentIntent:
    """V8.0 检测 Agent 意图类型（增强版）。

    检测顺序：agent_id 前缀匹配 → tags 包含关键词匹配 → FALLBACK
    """
    if agent_id:
        for pattern, intent_name in _AGENT_INTENT_RULES:
            if pattern.match(agent_id):
                logger.debug(
                    "[Intent] agent_id='%s' 匹配规则 '%s' → %s",
                    agent_id, pattern.pattern, intent_name,
                )
                return AgentIntent(intent_name)

    all_tags = " ".join(tags).lower()
    for pattern, intent_name in _TAG_INTENT_RULES:
        if pattern.search(all_tags):
            logger.debug(
                "[Intent] tags='%s' 匹配规则 '%s' → %s",
                all_tags, pattern.pattern, intent_name,
            )
            return AgentIntent(intent_name)

    return AgentIntent.FALLBACK


def intent_to_compute_class(intent: AgentIntent) -> str:
    """V8.0 将 Agent 意图转换为所需的算力级别。"""
    return {
        AgentIntent.PLANNER: "heavy",
        AgentIntent.EXECUTOR: "light",
        AgentIntent.QA: "light",
        AgentIntent.FALLBACK: "heavy",
    }.get(intent, "heavy")


# --------------------------------------------------------------------------- #
#  PrefixAwareLoadBalancer (V8.0)                                             #
# --------------------------------------------------------------------------- #

@dataclass
class PrefixAwareLoadBalancer:
    """V8.0 异构意图感知路由引擎。

    核心工作流程：
    1. **意图检测**：从 agent_id / tags 识别 Agent 类型（Planner/Executor/QA）
    2. **算力映射**：Planner → heavy | Executor/QA → light
    3. **前缀哈希**：提取 Prompt 稳定前缀，计算 SHA-256 Hash
    4. **亲和路由**：Hash 命中 + compute_class 匹配 + 实例健康 → 直连（KV-Cache 复用）
    5. **透明降级**：VRAM >= W_high (0.85) 时，自动重写模型名并路由至 light 池
    6. **时间维度优先级切换**：高负载时动态调整优先级策略

    前缀哈希亲和性联动 compute_class：
    - 路由表 key = (prefix_hash, compute_class) 元组
    - 确保相同前缀在不同算力池中的缓存独立维护
    - 跨节点 KV-Cache 复用通过 prefix_hash 一致性保证
    """

    instance_pool: LLMInstancePool
    # V8.0: 升级路由表结构为 (prefix_hash, compute_class) → [url]
    _routing_table: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    _instance_prefixes: dict[str, set[str]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _allow_downgrade: bool = field(default=True, init=False)
    _default_compute_class: str = field(default="heavy", init=False)

    # V8.0 新增：时间维度优先级状态
    _system_load_factor: float = field(default=1.0, init=False)   # 系统整体负载因子
    _vram_pressure_instances: dict[str, float] = field(default_factory=dict)  # 承压实例
    _downgrade_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.instance_pool.on_status_change(self._on_instance_status_change)

    @property
    def routing_table_size(self) -> int:
        return len(self._routing_table)

    def set_intent_config(self, default_compute_class: str, allow_downgrade: bool) -> None:
        """V8.0 注入意图路由配置。"""
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
        """计算 prompt 前缀的稳定哈希（SHA-256 前 8 字节）。"""
        prefix = self.extract_prefix(prompt, system_prompt)
        hash_bytes = hashlib.sha256(prefix.encode("utf-8")).digest()
        return hash_bytes[:8].hex()

    # --------------------------------------------------------------------------- #
    #  V8.0: 透明模型重写 (Model Name Rewriting)                                  #
    # --------------------------------------------------------------------------- #

    def _get_downgrade_model_name(self, original_model: str) -> str:
        """V8.0 新增：获取降级后的模型名称（透明降级机制）。"""
        return _MODEL_DOWNGRADE_MAP.get(original_model, original_model)

    def _get_composite_routing_key(self, prefix_hash: str, compute_class: str) -> str:
        """V8.0 新增：生成复合路由键（prefix_hash + compute_class 联动）。

        确保同一前缀在不同算力池中的 KV-Cache 独立维护。
        格式：prefix_hash#compute_class
        """
        return f"{prefix_hash}#{compute_class}"

    def _update_system_load_factor(self) -> None:
        """V8.0 新增：更新系统整体负载因子（时间维度优先级切换基础）。"""
        all_instances = [
            self.instance_pool.get_instance(url)
            for url in self.instance_pool.configs
        ]
        available = [inst for inst in all_instances if inst and inst.is_available]

        if not available:
            self._system_load_factor = 2.0
            return

        # 统计高负载实例比例
        high_load_count = sum(
            1 for inst in available
            if inst.vram_ratio >= VRAM_HIGH_WATERMARK
        )
        load_ratio = high_load_count / len(available)

        # 统计满连接实例比例
        overloaded_count = sum(
            1 for inst in available
            if inst.active_connections >= self.instance_pool.max_connections_per_instance
        )
        overload_ratio = overloaded_count / len(available)

        # 综合负载因子：正常=1.0，轻压=1.2，高压=1.5
        self._system_load_factor = 1.0 + load_ratio + overload_ratio

        logger.debug(
            "[LoadFactor] 系统负载因子=%.2f (高VRAM=%d/%d, 超连接=%d/%d)",
            self._system_load_factor, high_load_count, len(available),
            overloaded_count, len(available),
        )

    def _compute_time_priority(self, instance: InstanceMetrics, priority: str) -> int:
        """V8.0 新增：计算时间维度优先级。

        根据系统负载因子和实例 VRAM 水位，动态调整路由优先级。
        - 0: 正常（系统负载 < 1.2）
        - 1: 降级中（系统负载 1.2-1.5，或实例 VRAM >= W_high）
        - 2: 高负载（系统负载 > 1.5 或实例 VRAM >= W_critical）
        """
        if self._system_load_factor > 1.5 or instance.vram_ratio >= VRAM_CRITICAL_WATERMARK:
            return 2
        if self._system_load_factor > 1.2 or instance.vram_ratio >= VRAM_HIGH_WATERMARK:
            return 1
        return 0

    # --------------------------------------------------------------------------- #
    #  V8.0: 路由决策主流程                                                        #
    # --------------------------------------------------------------------------- #

    async def route(
        self,
        prompt: str,
        system_prompt: str | None = None,
        priority: str = "NORMAL",
        tags: list[str] | None = None,
        prefer_cached: bool = True,
        agent_id: str | None = None,
        force_compute_class: str | None = None,
        model_name: str | None = None,
    ) -> RoutingDecision:
        """V8.0 执行路由决策（异构意图路由 + 透明降级 + 时间维度优先级切换）。

        Parameters
        ----------
        agent_id : str | None
            Agent 标识符，用于意图检测（如 "planner_agent_01"）
        force_compute_class : str | None
            强制指定算力级别（覆盖意图检测结果）
        model_name : str | None
            V8.0 新增：指定模型名称（降级时自动重写）

        Returns
        -------
        RoutingDecision
            包含 url / strategy / compute_class / is_downgraded / effective_model_name
        """
        if not self._running:
            raise RuntimeError("LoadBalancer 未启动，请先调用 start()")

        tags = tags or []
        original_model = model_name or "qwen2.5:7b"

        # 更新系统负载因子（时间维度优先级切换）
        self._update_system_load_factor()

        # Step 1: 意图检测 & 算力映射
        if force_compute_class is not None:
            required_compute_class = force_compute_class
            detected_intent = AgentIntent.FALLBACK
        else:
            detected_intent = detect_agent_intent(agent_id, tags)
            required_compute_class = intent_to_compute_class(detected_intent)
            logger.debug(
                "[LoadBalancer] V8.0 意图检测: agent_id=%s intent=%s → compute_class=%s (系统负载=%.2f)",
                agent_id, detected_intent.value, required_compute_class, self._system_load_factor,
            )

        # Step 2: CRITICAL 任务跳过亲和路由
        if priority == "CRITICAL" and not prefer_cached:
            instance = self._select_least_loaded(
                tags=tags,
                compute_class=required_compute_class,
            )
            if instance is None:
                instance = self._select_least_loaded(tags=tags, compute_class=None)
                if instance is None:
                    raise RuntimeError("无可用 LLM 实例")

            tp = self._compute_time_priority(instance, priority)
            return RoutingDecision(
                url=instance.url,
                strategy=RoutingStrategy.LEAST_CONNECTIONS,
                reason="CRITICAL 任务强制重新选择",
                prefix_hash="",
                vram_ratio=instance.vram_ratio,
                active_connections=instance.active_connections,
                is_cached=False,
                compute_class=self.instance_pool.configs[instance.url].compute_class,
                original_model_name=original_model,
                effective_model_name=original_model,
                time_priority=tp,
            )

        prefix_hash = self.compute_prefix_hash(prompt, system_prompt)

        async with self._lock:
            # Step 3: 亲和路由（V8.0: 复合键 + compute_class 联动）
            routed_url = self._lookup_affinity_route(prefix_hash, required_compute_class)
            if routed_url and prefer_cached:
                instance = self.instance_pool.get_instance(routed_url)
                if instance and instance.is_available:
                    # 时间维度优先级切换：VRAM >= W_high 时触发降级
                    tp = self._compute_time_priority(instance, priority)
                    effective_model = original_model

                    if tp >= 1 and self._allow_downgrade:
                        effective_model = self._get_downgrade_model_name(original_model)
                        logger.info(
                            "[LoadBalancer] V8.0 时间维度降级: VRAM=%.2f, 时间优先级=%d, "
                            "模型重写: %s → %s",
                            instance.vram_ratio, tp, original_model, effective_model,
                        )

                    self._stats["prefix_hit"] += 1
                    self._stats[f"compute_class_{required_compute_class}_hit"] += 1
                    return RoutingDecision(
                        url=routed_url,
                        strategy=RoutingStrategy.PREFIX_AFFINITY,
                        reason=f"KV-Cache 亲和路由 (前缀={prefix_hash}, compute_class={required_compute_class})",
                        prefix_hash=prefix_hash,
                        vram_ratio=instance.vram_ratio,
                        active_connections=instance.active_connections,
                        is_cached=True,
                        compute_class=required_compute_class,
                        original_model_name=original_model,
                        effective_model_name=effective_model,
                        time_priority=tp,
                    )

            # Step 4: 意图降级路由（VRAM >= W_high 透明降级）
            if required_compute_class == "heavy" and self._allow_downgrade:
                # 检查 heavy 实例的 VRAM 水位
                heavy_instance = self._select_least_loaded(tags=tags, compute_class="heavy")
                if heavy_instance and heavy_instance.vram_ratio >= VRAM_HIGH_WATERMARK:
                    # V8.0: 透明降级 — 重写模型名 + 路由至 light 池
                    effective_model = self._get_downgrade_model_name(original_model)
                    light_instance = self._select_least_loaded(tags=tags, compute_class="light")

                    if light_instance is not None:
                        self._downgrade_count += 1
                        self._stats["compute_downgrade_heavy_to_light"] += 1
                        tp = self._compute_time_priority(light_instance, priority)
                        logger.info(
                            "[LoadBalancer] V8.0 透明降级触发: heavy VRAM=%.2f >= W_high(%.2f), "
                            "模型重写: %s → %s, 路由至 light 实例 %s",
                            heavy_instance.vram_ratio, VRAM_HIGH_WATERMARK,
                            original_model, effective_model, light_instance.url,
                        )
                        return RoutingDecision(
                            url=light_instance.url,
                            strategy=RoutingStrategy.COMPUTE_DOWNGRADE,
                            reason=(
                                f"heavy 实例 VRAM 过载 (%.2f >= %.2f)，透明降级 "
                                f"模型: {original_model} → {effective_model}"
                            ),
                            prefix_hash=prefix_hash,
                            vram_ratio=light_instance.vram_ratio,
                            active_connections=light_instance.active_connections,
                            is_cached=False,
                            compute_class="light",
                            is_downgraded=True,
                            required_compute_class=required_compute_class,
                            original_model_name=original_model,
                            effective_model_name=effective_model,
                            time_priority=tp,
                        )

            # Step 5: Least-Connections（按意图算力过滤）
            instance = self._select_least_loaded(tags=tags, compute_class=required_compute_class)
            if instance is None and required_compute_class == "heavy" and self._allow_downgrade:
                instance = self._select_least_loaded(tags=tags, compute_class="light")
                if instance is not None:
                    self._stats["fallback_to_light"] += 1

            if instance is None:
                raise RuntimeError("无可用 LLM 实例")

            selected_cc = self.instance_pool.configs[instance.url].compute_class
            is_downgraded = required_compute_class == "heavy" and selected_cc == "light"
            effective_model = (
                self._get_downgrade_model_name(original_model)
                if is_downgraded else original_model
            )
            tp = self._compute_time_priority(instance, priority)
            strategy = (
                RoutingStrategy.COMPUTE_DOWNGRADE
                if is_downgraded else RoutingStrategy.LEAST_CONNECTIONS
            )

            self._stats["least_connections"] += 1
            return RoutingDecision(
                url=instance.url,
                strategy=strategy,
                reason=(
                    f"前缀未命中，按 compute_class={selected_cc} 最少连接 "
                    f"({'透明降级' if is_downgraded else '正常'})"
                ),
                prefix_hash=prefix_hash,
                vram_ratio=instance.vram_ratio,
                active_connections=instance.active_connections,
                is_cached=False,
                compute_class=selected_cc,
                is_downgraded=is_downgraded,
                required_compute_class=required_compute_class if is_downgraded else None,
                original_model_name=original_model,
                effective_model_name=effective_model,
                time_priority=tp,
            )

    def _lookup_affinity_route(
        self,
        prefix_hash: str,
        compute_class: str,
    ) -> str | None:
        """V8.0 新增：按 (prefix_hash, compute_class) 复合键查找亲和路由。

        确保 KV-Cache 跨节点复用的同时，保持不同算力池的隔离。
        """
        # 方式1: 复合键查找（V8.0 新）
        composite_key = self._get_composite_routing_key(prefix_hash, compute_class)
        cc_table = self._routing_table.get(composite_key)
        if cc_table:
            urls = cc_table.get(compute_class, [])
            if urls:
                return urls[0]

        # 方式2: 旧式查找（兼容 V7.5）
        old_key = prefix_hash
        if old_key in self._routing_table:
            urls = self._routing_table[old_key]
            if isinstance(urls, list) and urls:
                return urls[0]
        return None

    def _select_least_loaded(
        self,
        tags: list[str] | None = None,
        compute_class: str | None = None,
    ) -> InstanceMetrics | None:
        """V8.0 按 compute_class 过滤后的最少连接选择。"""
        return self.instance_pool.get_least_loaded_instance(
            tags=tags,
            require_healthy=False,
            compute_class=compute_class,
        )

    async def record_route_success(self, decision: RoutingDecision) -> None:
        """V8.0 记录路由成功，更新复合键路由表。"""
        if decision.is_cached:
            return

        composite_key = self._get_composite_routing_key(
            decision.prefix_hash, decision.compute_class,
        )

        async with self._lock:
            if composite_key not in self._routing_table:
                self._routing_table[composite_key] = {}

            cc_table = self._routing_table[composite_key]
            history = cc_table.get(decision.compute_class, [])

            if history and history[0] == decision.url:
                return

            if decision.url in history:
                history.remove(decision.url)
            history.insert(0, decision.url)
            cc_table[decision.compute_class] = history[:5]
            self._routing_table[composite_key] = cc_table

            if decision.url not in self._instance_prefixes:
                self._instance_prefixes[decision.url] = set()
            self._instance_prefixes[decision.url].add(decision.prefix_hash)

    async def record_route_failure(self, decision: RoutingDecision) -> None:
        """V8.0 记录路由失败，降级实例热度。"""
        if not decision.is_cached:
            return

        async with self._lock:
            composite_key = self._get_composite_routing_key(
                decision.prefix_hash, decision.compute_class,
            )
            if composite_key in self._routing_table:
                cc_table = self._routing_table[composite_key]
                history = cc_table.get(decision.compute_class, [])
                if decision.url in history:
                    history.remove(decision.url)
                    cc_table[decision.compute_class] = history
                    self._routing_table[composite_key] = cc_table
                    if decision.url in self._instance_prefixes:
                        self._instance_prefixes[decision.url].discard(decision.prefix_hash)
                    self._stats["affinity_downgrade"] += 1

    def _on_instance_status_change(
        self,
        url: str,
        old_status: InstanceStatus,
        new_status: InstanceStatus,
    ) -> None:
        """实例状态变更回调（同步）。"""
        if new_status == InstanceStatus.UNHEALTHY:
            with self._lock:
                for composite_key, cc_table in list(self._routing_table.items()):
                    for cc, history in list(cc_table.items()):
                        if url in history:
                            history.remove(url)
                            cc_table[cc] = history
                    self._routing_table[composite_key] = cc_table
                if url in self._instance_prefixes:
                    del self._instance_prefixes[url]

    async def start(self) -> None:
        self._running = True
        logger.info(
            "[LoadBalancer] V8.0 异构意图路由引擎启动 | "
            "降级映射: %s | VRAM_High_Watermark: %.2f",
            _MODEL_DOWNGRADE_MAP, VRAM_HIGH_WATERMARK,
        )

    async def stop(self) -> None:
        self._running = False

    def get_stats(self) -> dict[str, Any]:
        total = sum(
            v for k, v in self._stats.items()
            if k not in (
                "compute_downgrade_heavy_to_light", "intent_mismatch_downgrade",
                "fallback_to_light", "compute_class_heavy_hit", "compute_class_light_hit",
            )
        )
        return {
            "version": "8.0",
            "total_routes": total,
            "prefix_hit_rate": f"{self._stats.get('prefix_hit', 0) / max(1, total) * 100:.1f}%",
            "affinity_downgrades": self._stats.get("affinity_downgrade", 0),
            "compute_downgrade_count": self._stats.get("compute_downgrade_heavy_to_light", 0),
            "intent_mismatch_count": self._stats.get("intent_mismatch_downgrade", 0),
            "fallback_to_light_count": self._stats.get("fallback_to_light", 0),
            "heavy_routes": self._stats.get("compute_class_heavy_hit", 0),
            "light_routes": self._stats.get("compute_class_light_hit", 0),
            "total_downgrade_count": self._downgrade_count,
            "system_load_factor": round(self._system_load_factor, 3),
            "routing_table_size": len(self._routing_table),
            "instance_prefix_count": {url: len(prefixes) for url, prefixes in self._instance_prefixes.items()},
            "model_downgrade_map": _MODEL_DOWNGRADE_MAP,
        }


__all__ = [
    "PrefixAwareLoadBalancer",
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingContext",
    "AgentIntent",
    "detect_agent_intent",
    "intent_to_compute_class",
]
