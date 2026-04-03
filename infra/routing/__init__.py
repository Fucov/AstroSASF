"""
路由层 — 前缀感知负载均衡 + VRAM 熔断
"""

from infra.routing.prefix_balancer import (
    PrefixAwareLoadBalancer,
    RoutingStrategy,
    RoutingDecision,
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
    "VRAMWatermarkBreaker",
    "CircuitState",
    "AdmissionResult",
]
