"""
AstroSASF · Interface · FastAPI Server
=====================================
北向 API 网关 — 将 AstroSASF 暴露为 RESTful Web 服务。

V7.2 重构自 server.py，整合了：
- 分布式 LLM 网关路由（PrefixBalancer + VRAMBreaker）
- 实验室管理 API
- MCP Tool 执行 API

启动方式：
    uvicorn interface.server:app --reload --host 0.0.0.0 --port 8000

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path（支持 uv run interface/server.py 等多种运行方式）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
from contextlib import asynccontextmanager
from typing import Any
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from scheduler.models import TaskPriority
from interface.state import AppState, state  # 全局状态（单例）

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Pydantic Models                                                            #
# --------------------------------------------------------------------------- #

class ExecuteRequest(BaseModel):
    """工具调用请求。"""
    tool_name: str = Field(..., description="MCP Tool 名称")
    params: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    agent_id: str = Field(default="http_client", description="调用者 ID")
    priority: int = Field(default=2, description="优先级 (0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW)")


class ExecuteResponse(BaseModel):
    """工具调用响应。"""
    status: str
    tool_name: str
    execution_id: str
    agent_id: str
    detail: str | None = None
    result: dict[str, Any] | None = None
    fsm_states: dict[str, str] | None = None
    execution_time_ms: float = 0.0


class LLMChatRequest(BaseModel):
    """LLM 推理请求。"""
    messages: list[dict[str, str]] = Field(..., description="消息列表")
    agent_id: str = Field(default="http_client", description="调用者 ID")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class LLMChatResponse(BaseModel):
    """LLM 推理响应。"""
    content: str
    task_id: str
    elapsed_ms: float
    model: str
    error: str | None = None


class LabInfo(BaseModel):
    """实验舱简要信息。"""
    lab_id: str
    name: str
    description: str


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    version: str
    timestamp: float
    loaded_labs: list[str]


# --------------------------------------------------------------------------- #
#  Lifespan — 启动 / 关闭顺序                                                   #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("")
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  🌌 AstroSASF V7.2 北向 API 网关启动中...                      ║")
    logger.info("╚" + "═" * 70 + "╝")

    try:
        from infra.llm.config_loader import load_config
        from labs.lab_loader import create_loader

        config_path = str(_ROOT / "config.yaml")
        state.config = load_config(config_path)
        logger.info("[Server] 配置加载完成: %s", config_path)

        # ── 1. 加载实验舱（使用 demo 专用 labs 目录）── #
        demo_labs_dir = str(_ROOT / "demo" / "assets" / "labs")
        state.loader = create_loader(demo_labs_dir)
        loaded_labs = await state.loader.discover_and_load()
        logger.info("[Server] 实验舱加载完成: %s", list(loaded_labs.keys()))

        # ── 2. 初始化分布式 LLM 网关（PrefixBalancer + VRAMBreaker）── #
        from infra.llm.instance_pool import LLMInstancePool, LLMInstanceConfig
        from infra.routing.prefix_balancer import PrefixAwareLoadBalancer
        from infra.routing.vram_breaker import VRAMWatermarkBreaker
        from infra.gateway.proxy import GatewayProxy

        pool = LLMInstancePool()

        # 从 config.yaml 读取 LLM 地址，注册为默认实例
        llm_cfg = state.config.llm
        pool.register_instance(LLMInstanceConfig(
            url=llm_cfg.base_url,
            weight=1,
            model_name=llm_cfg.model_name,
            tags=["default"],
        ))
        logger.info(
            "[Server] LLM 实例注册: %s (%s)",
            llm_cfg.base_url, llm_cfg.model_name,
        )

        lb = PrefixAwareLoadBalancer(instance_pool=pool)
        breaker = VRAMWatermarkBreaker(instance_pool=pool)
        state.gateway_proxy = GatewayProxy(
            instance_pool=pool,
            load_balancer=lb,
            watermark_breaker=breaker,
        )
        await state.gateway_proxy.start()
        logger.info("[Server] LLM GatewayProxy 启动完成（PrefixBalancer + VRAMBreaker）")

        # ── 3. 初始化 API Facade ── #
        from interface.facade import APIFacade
        state.facade = APIFacade(
            lab_loader=state.loader,
            gateway_proxy=state.gateway_proxy,
            scheduler=None,
        )

        logger.info("")
        logger.info("╔" + "═" * 70 + "╗")
        logger.info("║  ✅ AstroSASF V7.2 服务已就绪                                   ║")
        logger.info(f"║  API 文档: http://localhost:8000/docs                         ║")
        logger.info(f"║  LLM:      {llm_cfg.provider} @ {llm_cfg.base_url} ({llm_cfg.model_name})     ║")
        logger.info("╚" + "═" * 70 + "╝")

        yield

    finally:
        logger.info("[Server] 正在关闭...")
        if state.gateway_proxy:
            await state.gateway_proxy.stop()
        if state.facade:
            await state.facade.shutdown()
        logger.info("[Server] 已关闭 ✓")


app = FastAPI(
    title="AstroSASF V7.2 — Northbound API",
    description="太空智能体调度内核的北向接口 — C/S 架构服务端",
    version="7.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  API Endpoints                                                               #
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """健康检查端点。"""
    return HealthResponse(
        status="healthy",
        version="7.2.0",
        timestamp=time.time(),
        loaded_labs=state.loader.list_lab_ids() if state.loader else [],
    )


@app.get("/api/v1/labs", response_model=list[LabInfo], tags=["Labs"])
async def list_labs() -> list[dict[str, str]]:
    """获取所有已注册的实验舱列表。"""
    if state.loader is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    labs = []
    for lab_id in state.loader.list_lab_ids():
        metadata = state.loader.get_lab_metadata(lab_id)
        if metadata:
            labs.append({
                "lab_id": metadata["lab_id"],
                "name": metadata["name"],
                "description": metadata["description"],
            })

    return labs


@app.get("/api/v1/labs/{lab_id}/meta", response_model=dict, tags=["Labs"])
async def get_lab_metadata(lab_id: str) -> dict[str, Any]:
    """获取指定实验舱的完整元数据。"""
    if state.loader is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    metadata = state.loader.get_lab_metadata(lab_id)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"实验舱 '{lab_id}' 不存在",
        )

    return metadata


@app.post("/api/v1/labs/{lab_id}/execute", response_model=ExecuteResponse, tags=["Execution"])
async def execute_tool(lab_id: str, request: ExecuteRequest) -> dict[str, Any]:
    """执行工具调用。"""
    if state.facade is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    priority_map = {
        0: TaskPriority.CRITICAL,
        1: TaskPriority.HIGH,
        2: TaskPriority.NORMAL,
        3: TaskPriority.LOW,
    }
    priority = priority_map.get(request.priority, TaskPriority.NORMAL)

    result = await state.facade.execute_tool_call(
        lab_id=lab_id,
        tool_name=request.tool_name,
        params=request.params,
        agent_id=request.agent_id,
        priority=priority,
    )

    logger.info(
        "[API] /api/v1/labs/%s/execute [%s] %s → %s (%.1fms)",
        lab_id, request.agent_id, request.tool_name,
        result.status, result.execution_time_ms,
    )

    return result.to_dict()


@app.post("/api/v1/llm/chat", response_model=LLMChatResponse, tags=["LLM"])
async def llm_chat(request: LLMChatRequest) -> dict[str, Any]:
    """调用底层 LLM 进行推理（经分布式网关 Proxy → PrefixBalancer → VRAMBreaker → LLM 实例）。"""
    if state.gateway_proxy is None:
        raise HTTPException(status_code=503, detail="LLM 网关未初始化，请先启动框架服务")

    from infra.gateway.proxy import GatewayRequest
    priority_map = {0: "CRITICAL", 1: "HIGH", 2: "NORMAL", 3: "LOW"}
    priority = priority_map.get(request.priority, "NORMAL")
    model_name = state.config.llm.model_name if state.config else "qwen2.5:7b"

    gw_request = GatewayRequest(
        messages=request.messages,
        model=model_name,
        temperature=request.temperature or 0.1,
        max_tokens=request.max_tokens or 2048,
        priority=priority,
    )

    try:
        response = await state.gateway_proxy.chat(gw_request)

        # 处理错误响应
        if isinstance(response, Exception):
            err = str(response)
            logger.warning("[API] LLM 网关异常: %s", err)
            return LLMChatResponse(
                content="[LLM ERROR] " + err,
                task_id=getattr(response, "request_id", "unknown") or "unknown",
                elapsed_ms=getattr(response, "latency_ms", 0) or 0,
                model=model_name,
                error=err,
            )

        # 检查是否是 GatewayError
        from infra.gateway.proxy import GatewayError as GWError
        if isinstance(response, GWError):
            logger.warning("[API] LLM 网关返回错误: %s [%s]", response.error, response.error_code)
            return LLMChatResponse(
                content="[LLM ERROR] " + response.error,
                task_id=response.request_id or "unknown",
                elapsed_ms=response.latency_ms if hasattr(response, "latency_ms") else 0,
                model=model_name,
                error=response.error,
            )

        return LLMChatResponse(
            content=response.content or "",
            task_id=response.request_id or "unknown",
            elapsed_ms=response.latency_ms or 0,
            model=response.model or model_name,
        )

    except Exception as exc:
        logger.exception("[API] LLM 调用失败: %s", exc)
        return LLMChatResponse(
            content="[LLM ERROR] " + str(exc),
            task_id="error",
            elapsed_ms=0,
            model=model_name,
            error=str(exc),
        )


# --------------------------------------------------------------------------- #
#  Error Handlers                                                             #
# --------------------------------------------------------------------------- #

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.exception("[Server] 未处理的异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


__all__ = ["app"]
