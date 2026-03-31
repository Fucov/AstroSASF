"""
AstroSASF V7.1 · Server · FastAPI Application
=============================================
北向 API 网关 — 将 AstroSASF 暴露为 RESTful Web 服务。

启动方式：
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

API 端点：
    GET  /api/v1/labs                    — 获取所有实验舱列表
    GET  /api/v1/labs/{lab_id}/meta     — 获取实验舱完整元数据
    POST /api/v1/labs/{lab_id}/execute  — 执行工具调用
    POST /api/v1/llm/chat               — LLM 推理接口
    GET  /health                        — 健康检查

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sasf.core.config_loader import load_config, SASFConfig
from sasf.core.lab_loader import LabLoader, create_loader, LoadedLab
from sasf.core.models import TaskPriority
from sasf.core.os_gateway import AstroOSFacade, ExecutionStatus

# --------------------------------------------------------------------------- #
#  Logging Configuration                                                       #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Pydantic Models (Request/Response)                                         #
# --------------------------------------------------------------------------- #

class ExecuteRequest(BaseModel):
    """工具调用请求。"""
    tool_name: str = Field(..., description="MCP Tool 名称")
    params: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    agent_id: str = Field(default="http_client", description="调用者 ID")
    priority: int = Field(default=2, description="优先级 (0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool_name": "read_sensor",
                "params": {"channel": 1},
                "agent_id": "agent_001",
                "priority": 2,
            }
        }
    }


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
    preempted_by: str | None = None


class LLMChatRequest(BaseModel):
    """LLM 推理请求。"""
    messages: list[dict[str, str]] = Field(..., description="消息列表")
    agent_id: str = Field(default="http_client", description="调用者 ID")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "system", "content": "你是一个太空实验助手。"},
                    {"role": "user", "content": "我需要读取传感器数据。"},
                ],
                "agent_id": "agent_001",
            }
        }
    }


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
#  Application State                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class AppState:
    """应用全局状态。"""
    facade: AstroOSFacade = field(default=None)
    loader: LabLoader = field(default=None)
    config: SASFConfig = field(default=None)
    startup_time: float = field(default_factory=time.time)


state = AppState()


# --------------------------------------------------------------------------- #
#  Lifecycle Events                                                            #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("")
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  🌌 AstroSASF V7.1 北向 API 网关启动中...                      ║")
    logger.info("╚" + "═" * 70 + "╝")

    try:
        # ── 加载配置 ── #
        state.config = load_config("config.yaml")
        logger.info("[Server] 配置加载完成")

        # ── 创建 Facade ── #
        state.facade = AstroOSFacade.from_config_obj(state.config)
        await state.facade.initialize()
        logger.info("[Server] AstroOSFacade 初始化完成")

        # ── 加载实验舱 ── #
        state.loader = create_loader()
        loaded_labs = await state.loader.discover_and_load(state.config)

        # ── 注册到 Facade ── #
        for lab_id, loaded in loaded_labs.items():
            await state.facade.register_lab(loaded.environment)
            logger.info("[Server] 实验舱 '%s' 已注册", lab_id)

        logger.info("")
        logger.info("╔" + "═" * 70 + "╗")
        logger.info("║  ✅ AstroSASF V7.1 服务已就绪                                   ║")
        logger.info(f"║  API 文档: http://localhost:8000/docs                         ║")
        logger.info("╚" + "═" * 70 + "╝")

        yield  # 应用运行中

    finally:
        # ── 关闭 ── #
        logger.info("[Server] 正在关闭...")
        if state.facade:
            await state.facade.shutdown()
        logger.info("[Server] 已关闭 ✓")


# --------------------------------------------------------------------------- #
#  FastAPI Application                                                        #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="AstroSASF V7.1 — Northbound API",
    description="太空智能体调度内核的北向接口 — C/S 架构服务端",
    version="7.1.0",
    lifespan=lifespan,
)

# ── CORS 中间件（允许外部多智能体系统跨域调用）─ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  API Endpoints                                                              #
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """健康检查端点。"""
    return HealthResponse(
        status="healthy",
        version="7.1.0",
        timestamp=time.time(),
        loaded_labs=state.loader.list_lab_ids() if state.loader else [],
    )


@app.get("/api/v1/labs", response_model=list[LabInfo], tags=["Labs"])
async def list_labs() -> list[dict[str, str]]:
    """获取所有已注册的实验舱列表。

    Returns
    -------
    list[LabInfo]
        实验舱简要信息列表
    """
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

    logger.info("[API] /api/v1/labs 返回 %d 个实验舱", len(labs))
    return labs


@app.get("/api/v1/labs/{lab_id}/meta", response_model=dict, tags=["Labs"])
async def get_lab_metadata(lab_id: str) -> dict[str, Any]:
    """获取指定实验舱的完整元数据。

    这是**核心接口**！外部智能体系统启动时调用此接口获取：
    - 所有 MCP Tools（含 JSON Schema）
    - 所有 Macros
    - FSM 状态和联锁规则
    - 可用 Skills

    Parameters
    ----------
    lab_id : str
        实验舱 ID

    Returns
    -------
    dict
        完整元数据
    """
    if state.loader is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    metadata = state.loader.get_lab_metadata(lab_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"实验舱 '{lab_id}' 不存在",
        )

    logger.info("[API] /api/v1/labs/%s/meta 返回元数据", lab_id)
    return metadata


@app.post(
    "/api/v1/labs/{lab_id}/execute",
    response_model=ExecuteResponse,
    tags=["Execution"],
)
async def execute_tool(lab_id: str, request: ExecuteRequest) -> dict[str, Any]:
    """执行工具调用。

    接收外部 Agent 的原子操作请求，经 FSM + Guard 校验后执行。

    Parameters
    ----------
    lab_id : str
        目标实验舱 ID
    request : ExecuteRequest
        执行请求

    Returns
    -------
    ExecuteResponse
        执行结果
    """
    if state.facade is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    # 验证实验舱存在
    if lab_id not in (state.loader.list_lab_ids() if state.loader else []):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"实验舱 '{lab_id}' 不存在",
        )

    # 转换优先级
    priority_map = {
        0: TaskPriority.CRITICAL,
        1: TaskPriority.HIGH,
        2: TaskPriority.NORMAL,
        3: TaskPriority.LOW,
    }
    priority = priority_map.get(request.priority, TaskPriority.NORMAL)

    # 执行工具调用
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
        result.status.value, result.execution_time_ms,
    )

    return result.to_dict()


@app.post("/api/v1/llm/chat", response_model=LLMChatResponse, tags=["LLM"])
async def llm_chat(request: LLMChatRequest) -> dict[str, Any]:
    """调用底层 LLM 进行推理。

    外部 Agent 可使用此接口进行自然语言理解。

    Parameters
    ----------
    request : LLMChatRequest
        推理请求

    Returns
    -------
    LLMChatResponse
        推理结果
    """
    if state.facade is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        response = await state.facade.chat_completion(
            messages=request.messages,
            agent_id=request.agent_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return LLMChatResponse(
            content=response.get("content", ""),
            task_id=response.get("task_id", ""),
            elapsed_ms=response.get("elapsed_ms", 0),
            model=response.get("model", state.config.llm.model_name),
        )

    except Exception as exc:
        logger.exception("[API] LLM 调用失败: %s", exc)
        return LLMChatResponse(
            content="",
            task_id="",
            elapsed_ms=0,
            model=state.config.llm.model_name,
            error=str(exc),
        )


# --------------------------------------------------------------------------- #
#  Error Handlers                                                             #
# --------------------------------------------------------------------------- #

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.exception("[Server] 未处理的异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# --------------------------------------------------------------------------- #
#  Entry Point                                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
