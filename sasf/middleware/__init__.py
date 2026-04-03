# AstroSASF · Middleware Layer
# ─────────────────────────────────────────────────────────────────────────────
# 导出核心中间件组件

from sasf.middleware.mcp_registry import (
    MCPToolContext,
    MCPToolRegistry,
    mcp_tool,
)
from sasf.middleware.a2a_protocol import (
    A2AIntent,
    A2AMessage,
    A2ARouter,
)
from sasf.middleware.codec import SpaceMCPCodec
from sasf.middleware.gateway import SpaceMCPGateway
from sasf.middleware.virtual_bus import VirtualSpaceWire

# ── V7.2 Distributed Gateway ──────────────────────────────────────────────── #
from sasf.middleware.llm_instance_pool import (
    LLMInstanceConfig,
    LLMInstancePool,
    InstanceMetrics,
    InstanceStatus,
    VRAM_HIGH_WATERMARK,
    VRAM_CRITICAL_WATERMARK,
    VRAM_LOW_WATERMARK,
)
from sasf.middleware.prefix_aware_load_balancer import (
    PrefixAwareLoadBalancer,
    RoutingDecision,
    RoutingContext,
    RoutingStrategy,
)
from sasf.middleware.vram_watermark_breaker import (
    VRAMWatermarkBreaker,
    AdmissionDecision,
    AdmissionResult,
    CircuitState,
    BreakerConfig,
)
from sasf.middleware.gateway_proxy import (
    GatewayProxy,
    GatewayRequest,
    GatewayResponse,
    GatewayError,
)

__all__ = [
    # Core (V5)
    "MCPToolContext",
    "MCPToolRegistry",
    "mcp_tool",
    "A2AIntent",
    "A2AMessage",
    "A2ARouter",
    "SpaceMCPCodec",
    "SpaceMCPGateway",
    "VirtualSpaceWire",
    # Distributed Gateway (V7.2)
    "LLMInstanceConfig",
    "LLMInstancePool",
    "InstanceMetrics",
    "InstanceStatus",
    "VRAM_HIGH_WATERMARK",
    "VRAM_CRITICAL_WATERMARK",
    "VRAM_LOW_WATERMARK",
    "PrefixAwareLoadBalancer",
    "RoutingDecision",
    "RoutingContext",
    "RoutingStrategy",
    "VRAMWatermarkBreaker",
    "AdmissionDecision",
    "AdmissionResult",
    "CircuitState",
    "BreakerConfig",
    "GatewayProxy",
    "GatewayRequest",
    "GatewayResponse",
    "GatewayError",
]
