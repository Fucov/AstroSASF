"""
路由层 — 前缀感知负载均衡 + VRAM 熔断 + 异构意图路由 + 时间维度优先级切换（V8.0）
"""

from infra.routing.prefix_balancer import (
    PrefixAwareLoadBalancer,
    RoutingStrategy,
    RoutingDecision,
    RoutingContext,       # V8.0 新增
    AgentIntent,          # V7.5 新增
    detect_agent_intent,  # V7.5 新增
    intent_to_compute_class,  # V7.5 新增
)
from infra.routing.vram_breaker import (
    VRAMWatermarkBreaker,
    CircuitState,
    AdmissionResult,
)

__all__ = [
    "PrefixAwareLoadBalancer",
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingContext",            # V8.0 新增
    "AgentIntent",               # V7.5
    "detect_agent_intent",        # V7.5
    "intent_to_compute_class",   # V7.5
    "VRAMWatermarkBreaker",
    "CircuitState",
    "AdmissionResult",
]
