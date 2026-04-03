"""
AstroSASF V7.2 · Distributed LLM Gateway · FastAPI Integration
================================================================
分布式 LLM 网关的 FastAPI 集成层。

将 LLMInstancePool、PrefixAwareLoadBalancer、VRAMWatermarkBreaker、GatewayProxy
接入 AstroSASF 现有的 FastAPI 服务（server.py）。

路由流程：
  HTTP Request → FastAPI → GatewayProxy → PrefixAwareLoadBalancer
                                          ↓
                                    VRAMWatermarkBreaker（准入检查）
                                          ↓
                                    LLMInstancePool（实例池）
                                          ↓
                                    SGLang/vLLM 节点

V7.2 新增：
- /api/v1/llm/chat 路由到 GatewayProxy（非流式）
- /api/v1/llm/stream 路由到 GatewayProxy（流式）
- /api/v1/admin/gateway/stats 网关统计
- /api/v1/admin/gateway/instances 管理实例

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from sasf.core.config_loader import load_config, SASFConfig
from sasf.middleware.llm_instance_pool import (
    LLMInstanceConfig,
    LLMInstancePool,
    VRAM_HIGH_WATERMARK,
    VRAM_CRITICAL_WATERMARK,
)
from sasf.middleware.prefix_aware_load_balancer import PrefixAwareLoadBalancer
from sasf.middleware.vram_watermark_breaker import VRAMWatermarkBreaker
from sasf.middleware.gateway_proxy import (
    GatewayProxy,
    GatewayRequest,
    GatewayResponse,
    GatewayError,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Pydantic Models                                                             #
# --------------------------------------------------------------------------- #

class LLMChatRequest(BaseModel):
    """LLM Chat 请求（V7.2 分布式网关版）。"""
    messages: list[dict[str, str]] = Field(..., description="消息列表")
    model: str = Field(default="qwen2.5-7b", description="模型名称")
    agent_id: str = Field(default="http_client", description="调用者 ID")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    stream: bool = Field(default=False, description="是否流式输出")
    priority: str = Field(
        default="NORMAL",
        description="任务优先级 (CRITICAL|HIGH|NORMAL|LOW)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="标签筛选（如 ['v100', 'node-1']）"
    )


class LLMChatResponse(BaseModel):
    """LLM Chat 响应。"""
    content: str
    model: str
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    request_id: str
    routed_to: str
    routing_strategy: str
    prefix_hash: str
    latency_ms: float
    streamed: bool = False


class InstanceRegisterRequest(BaseModel):
    """注册 LLM 实例请求。"""
    url: str = Field(..., description="实例 URL，如 http://192.168.1.10:8000")
    weight: int = Field(default=1, ge=1, description="路由权重")
    model_name: str = Field(default="qwen2.5-7b", description="模型名称")
    tags: list[str] = Field(default_factory=list, description="标签")


class InstanceInfo(BaseModel):
    """实例信息。"""
    url: str
    weight: int
    model_name: str
    tags: list[str]
    status: str
    vram_ratio: float
    vram_used_gb: float
    active_connections: int
    health_score: float
    avg_latency_ms: float


class GatewayStatsResponse(BaseModel):
    """网关统计响应。"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: str
    prefix_cached_rate: str
    instance_count: int
    healthy_instance_count: int
    routing_table_size: int
    breaker_states: dict[str, str]


# --------------------------------------------------------------------------- #
#  Gateway State                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class GatewayState:
    """网关全局状态。"""
    pool: LLMInstancePool = field(default=None)
    load_balancer: PrefixAwareLoadBalancer = field(default=None)
    breaker: VRAMWatermarkBreaker = field(default=None)
    proxy: GatewayProxy = field(default=None)
    startup_time: float = field(default_factory=time.time)
    _initialized: bool = field(default=False, init=False)


gateway_state = GatewayState()


# --------------------------------------------------------------------------- #
#  Gateway Initialization                                                      #
# --------------------------------------------------------------------------- #

async def init_distributed_gateway(
    instances: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """初始化分布式 LLM 网关。

    Parameters
    ----------
    instances : list[dict[str, Any]] | None
        初始实例列表，格式：
        [{"url": "http://192.168.1.10:8000", "weight": 1, "model_name": "qwen2.5-7b", "tags": ["v100"]}]
    config : dict[str, Any] | None
        网关配置，可选字段：
        - high_watermark: float (默认 0.85)
        - critical_watermark: float (默认 0.92)
    """
    config = config or {}
    instances = instances or []

    # 1. 创建实例池
    gateway_state.pool = LLMInstancePool()

    # 2. 注册初始实例
    for inst in instances:
        gateway_state.pool.register_instance(LLMInstanceConfig(
            url=inst["url"],
            weight=inst.get("weight", 1),
            model_name=inst.get("model_name", "qwen2.5-7b"),
            tags=inst.get("tags", []),
        ))
        logger.info(
            "[DistributedGateway] 注册实例: %s (weight=%d, model=%s, tags=%s)",
            inst["url"], inst.get("weight", 1),
            inst.get("model_name", "qwen2.5-7b"), inst.get("tags", [])
        )

    # 3. 创建子组件
    gateway_state.load_balancer = PrefixAwareLoadBalancer(
        instance_pool=gateway_state.pool,
    )
    gateway_state.breaker = VRAMWatermarkBreaker(
        instance_pool=gateway_state.pool,
    )
    gateway_state.proxy = GatewayProxy(
        instance_pool=gateway_state.pool,
        load_balancer=gateway_state.load_balancer,
        watermark_breaker=gateway_state.breaker,
    )

    # 4. 启动
    await gateway_state.proxy.start()
    gateway_state._initialized = True

    logger.info(
        "[DistributedGateway] 初始化完成: %d 个实例, high_wm=%.0f%%, critical_wm=%.0f%%",
        len(instances),
        (config.get("high_watermark", VRAM_HIGH_WATERMARK) * 100),
        (config.get("critical_watermark", VRAM_CRITICAL_WATERMARK) * 100),
    )


async def shutdown_distributed_gateway() -> None:
    """关闭分布式 LLM 网关。"""
    if gateway_state.proxy:
        await gateway_state.proxy.stop()
    gateway_state._initialized = False
    logger.info("[DistributedGateway] 已关闭")


# --------------------------------------------------------------------------- #
#  Lifespan Hook                                                               #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def distributed_gateway_lifespan(app: FastAPI):
    """分布式网关的生命周期管理器（独立启动，可被 server.py lifespan 覆盖）。"""
    # 默认不自动初始化，等待手动调用 init_distributed_gateway 或通过 admin API
    yield


# --------------------------------------------------------------------------- #
#  FastAPI Endpoints (挂载到 server.py 的 app 上)                                #
# --------------------------------------------------------------------------- #

def register_distributed_gateway_routes(app: FastAPI) -> None:
    """将分布式网关路由注册到 FastAPI 应用。"""

    @app.post(
        "/api/v1/llm/chat",
        response_model=LLMChatResponse,
        tags=["LLM (Distributed)"],
        summary="LLM Chat 推理（分布式网关）",
    )
    async def llm_chat_distributed(request: LLMChatRequest) -> dict[str, Any]:
        """分布式 LLM Chat 接口。

        支持：
        - 前缀感知路由（KV-Cache 复用）
        - VRAM 水线熔断
        - 优先级感知准入控制
        - 自动重路由与故障转移

        Parameters
        ----------
        request : LLMChatRequest
            推理请求

        Returns
        -------
        LLMChatResponse
            推理结果（含路由信息）
        """
        if not gateway_state._initialized:
            # 自动初始化（使用环境变量或配置）
            instances = _load_instances_from_env()
            await init_distributed_gateway(instances=instances)

        # 构建网关请求
        gw_request = GatewayRequest(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            priority=request.priority,
            tags=request.tags,
        )

        # 调用网关
        response = await gateway_state.proxy.chat(gw_request)

        if isinstance(response, GatewayError):
            raise HTTPException(
                status_code=503 if "CIRCUIT" in response.error_code else 500,
                detail={
                    "error": response.error,
                    "error_code": response.error_code,
                    "retry_after": response.retry_after,
                },
            )

        return {
            "content": response.content,
            "model": response.model,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "request_id": response.request_id,
            "routed_to": response.routed_to,
            "routing_strategy": response.routing_strategy,
            "prefix_hash": response.prefix_hash,
            "latency_ms": response.latency_ms,
            "streamed": response.streamed,
        }

    @app.post(
        "/api/v1/llm/stream",
        tags=["LLM (Distributed)"],
        summary="LLM Chat 推理（流式，分布式网关）",
    )
    async def llm_stream_distributed(request: LLMChatRequest):
        """分布式 LLM Chat 流式接口。

        直接透传 SGLang/vLLM 的 SSE 流式响应。

        Parameters
        ----------
        request : LLMChatRequest
            推理请求（stream=True）

        Yields
        ------
        Server-Sent Events
        """
        if not gateway_state._initialized:
            instances = _load_instances_from_env()
            await init_distributed_gateway(instances=instances)

        gw_request = GatewayRequest(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            priority=request.priority,
            tags=request.tags,
        )

        return StreamingResponse(
            gateway_state.proxy.stream_chat(gw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/api/v1/admin/gateway/stats",
        response_model=GatewayStatsResponse,
        tags=["Admin (Distributed Gateway)"],
        summary="获取网关统计信息",
    )
    async def get_gateway_stats() -> dict[str, Any]:
        """获取网关运行时统计信息。"""
        if not gateway_state._initialized:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": "0.0%",
                "prefix_cached_rate": "0.0%",
                "instance_count": 0,
                "healthy_instance_count": 0,
                "routing_table_size": 0,
                "breaker_states": {},
            }

        stats = gateway_state.proxy.get_stats()

        return {
            "total_requests": stats["gateway"]["total_requests"],
            "successful_requests": stats["gateway"]["successful_requests"],
            "failed_requests": stats["gateway"]["failed_requests"],
            "success_rate": stats["gateway"]["success_rate"],
            "prefix_cached_rate": stats["gateway"]["prefix_cached_rate"],
            "instance_count": stats["instance_pool"]["total_instances"],
            "healthy_instance_count": stats["instance_pool"]["healthy_instances"],
            "routing_table_size": stats["load_balancer"]["routing_table_size"],
            "breaker_states": stats["watermark_breaker"]["breaker_states"],
        }

    @app.get(
        "/api/v1/admin/gateway/instances",
        response_model=list[InstanceInfo],
        tags=["Admin (Distributed Gateway)"],
        summary="获取所有实例状态",
    )
    async def get_gateway_instances() -> list[dict[str, Any]]:
        """获取所有注册实例的详细状态。"""
        if not gateway_state._initialized:
            return []

        pool_stats = gateway_state.proxy.instance_pool.get_stats()
        instances = []

        for url, inst_stats in pool_stats.get("instances", {}).items():
            config = gateway_state.proxy.instance_pool.configs.get(url)
            instances.append({
                "url": url,
                "weight": config.weight if config else 1,
                "model_name": config.model_name if config else "",
                "tags": config.tags if config else [],
                "status": inst_stats["status"],
                "vram_ratio": float(inst_stats["vram_ratio"].replace("%", "")) / 100,
                "vram_used_gb": float(inst_stats["vram_used_gb"]),
                "active_connections": inst_stats["active_connections"],
                "health_score": float(inst_stats["health_score"]),
                "avg_latency_ms": float(inst_stats["avg_latency_ms"]),
            })

        return instances

    @app.post(
        "/api/v1/admin/gateway/instances",
        status_code=status.HTTP_201_CREATED,
        response_model=InstanceInfo,
        tags=["Admin (Distributed Gateway)"],
        summary="注册新实例",
    )
    async def register_instance(req: InstanceRegisterRequest) -> dict[str, Any]:
        """注册一个新的 LLM 实例。"""
        if not gateway_state._initialized:
            await init_distributed_gateway(instances=[])

        config = LLMInstanceConfig(
            url=req.url,
            weight=req.weight,
            model_name=req.model_name,
            tags=req.tags,
        )

        gateway_state.proxy.register_instance(config)
        logger.info(
            "[Admin] 注册实例: %s (weight=%d, model=%s, tags=%s)",
            req.url, req.weight, req.model_name, req.tags
        )

        # 返回注册后的状态
        inst = gateway_state.proxy.instance_pool.get_instance(req.url)
        return {
            "url": req.url,
            "weight": req.weight,
            "model_name": req.model_name,
            "tags": req.tags,
            "status": inst.status.name if inst else "UNKNOWN",
            "vram_ratio": inst.vram_ratio if inst else 0.0,
            "vram_used_gb": inst.vram_used_gb if inst else 0.0,
            "active_connections": inst.active_connections if inst else 0,
            "health_score": inst.health_score if inst else 0.0,
            "avg_latency_ms": inst.avg_latency_ms if inst else 0.0,
        }

    @app.delete(
        "/api/v1/admin/gateway/instances/{url:path}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Admin (Distributed Gateway)"],
        summary="注销实例",
    )
    async def unregister_instance(url: str) -> None:
        """注销一个 LLM 实例。"""
        if not gateway_state._initialized:
            raise HTTPException(status_code=404, detail="网关未初始化")

        success = gateway_state.proxy.instance_pool.unregister_instance(url)
        if not success:
            raise HTTPException(status_code=404, detail=f"实例不存在: {url}")

        logger.info("[Admin] 注销实例: %s", url)

    @app.get(
        "/api/v1/admin/gateway/routing-table",
        tags=["Admin (Distributed Gateway)"],
        summary="获取路由表快照",
    )
    async def get_routing_table() -> dict[str, Any]:
        """获取前缀感知路由表快照（用于调试）。"""
        if not gateway_state._initialized:
            return {"routing_table_size": 0, "entries": {}, "stats": {}}

        return gateway_state.proxy.load_balancer.get_routing_table_snapshot()


# --------------------------------------------------------------------------- #
#  Helper Functions                                                            #
# --------------------------------------------------------------------------- #

def _load_instances_from_env() -> list[dict[str, Any]]:
    """从环境变量加载初始实例配置。

    格式：ASTRO_LLM_INSTANCES='[{"url":"http://192.168.1.10:8000","weight":1,"model_name":"qwen2.5-7b","tags":["v100","node-1"]}]'
    """
    import json
    import os

    env_var = os.environ.get("ASTRO_LLM_INSTANCES", "")
    if not env_var:
        return []

    try:
        instances = json.loads(env_var)
        if isinstance(instances, list):
            return instances
        return []
    except json.JSONDecodeError:
        logger.warning("[DistributedGateway] ASTRO_LLM_INSTANCES 格式错误，跳过")
        return []


# --------------------------------------------------------------------------- #
#  Usage Example                                                                #
# --------------------------------------------------------------------------- #
"""
V7.2 分布式网关使用示例：

1. 在 config.yaml 中添加实例配置：
```yaml
distributed_llm:
  instances:
    - url: "http://192.168.1.10:8000"
      weight: 1
      model_name: "qwen2.5-7b"
      tags: ["v100", "node-1"]
    - url: "http://192.168.1.11:8000"
      weight: 1
      model_name: "qwen2.5-7b"
      tags: ["v100", "node-2"]
  watermarks:
    high: 0.85
    critical: 0.92
```

2. 在 server.py 中注册路由：
```python
from sasf.middleware.distributed_gateway_api import register_distributed_gateway_routes

register_distributed_gateway_routes(app)
```

3. 通过 Admin API 动态注册实例：
```bash
curl -X POST http://localhost:8000/api/v1/admin/gateway/instances \
  -H "Content-Type: application/json" \
  -d '{"url": "http://192.168.1.12:8000", "weight": 1, "model_name": "qwen2.5-7b", "tags": ["v100"]}'
```

4. 调用分布式 Chat 接口：
```bash
curl -X POST http://localhost:8000/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "你是太空实验柜助手"},
      {"role": "user", "content": "SOP:bio_culture\\n请帮我开始细胞培养实验"}
    ],
    "model": "qwen2.5-7b",
    "priority": "NORMAL"
  }'
```
"""
