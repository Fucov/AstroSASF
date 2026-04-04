"""
路由层 — 前缀感知负载均衡 + VRAM 熔断 + 异构算力路由（V7.5）
"""

from infra.routing.prefix_balancer import (
    PrefixAwareLoadBalancer,
    RoutingStrategy,
    RoutingDecision,
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
    "AgentIntent",           # V7.5
    "detect_agent_intent",   # V7.5
    "intent_to_compute_class",  # V7.5
    "VRAMWatermarkBreaker",
    "CircuitState",
    "AdmissionResult",
]
