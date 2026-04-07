"""
AstroSASF · Interface · API Facade (V8.0)
=========================================
北向 API 统一门面，聚合 LLM Gateway、Lab Context 与 Scheduler。

V8.0 新增特性（按需求规格逐条实现）：
1. **统一 /llm/chat 接口**：对上层屏蔽降级与乱序细节
2. **响应头元数据**：
   - X-Astro-OoO: true — 表示本次请求经过了乱序调度优化
   - X-Astro-Downgraded: true — 表示本次请求经过了模型透明降级
   - X-Astro-Routing-Strategy — 路由策略枚举值
   - X-Astro-Compute-Class — 本次请求使用的算力分级
   - X-Astro-Model — 实际使用的模型名称（可能与请求不同）
   - X-Astro-Time-Priority — 时间维度优先级（0=正常，1=降级，2=高压）

设计原则：
- 对上层屏蔽所有底层调度细节（OoO 越级、模型降级、VRAM 熔断等）
- 响应头仅用于实验记录，不影响业务逻辑
- 所有元数据通过 RoutingDecision 传递，保持透明性

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from scheduler.models import TaskPriority
from scheduler.a2a_protocol import A2AMessage, A2AIntent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data Classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class ToolCallResult:
    """工具调用结果。"""
    status: str
    tool_name: str
    execution_id: str
    agent_id: str
    detail: str | None = None
    result: dict[str, Any] | None = None
    fsm_states: dict[str, str] | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        fsm_states = None
        if self.fsm_states:
            fsm_states = {
                k: str(v) for k, v in self.fsm_states.items()
            }
        return {
            "status": self.status,
            "tool_name": self.tool_name,
            "execution_id": self.execution_id,
            "agent_id": self.agent_id,
            "detail": self.detail,
            "result": self.result,
            "fsm_states": fsm_states,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class LLMResponseMetadata:
    """V8.0 LLM 响应元数据（携带实验记录信息）。"""
    is_ooo: bool = False              # 是否经过乱序调度优化
    is_downgraded: bool = False        # 是否经过模型透明降级
    routing_strategy: str = ""         # 路由策略枚举值
    compute_class: str = "heavy"       # 本次使用的算力分级
    original_model: str = ""           # 原始请求的模型名
    effective_model: str = ""          # 实际使用的模型名
    time_priority: int = 0             # 时间维度优先级（0=正常，1=降级，2=高压）
    prefix_hash: str = ""             # 前缀哈希（用于 KV-Cache 追踪）
    agent_intent: str = ""             # 检测到的 Agent 意图

    def to_response_headers(self) -> dict[str, str]:
        """V8.0 将元数据转换为 HTTP 响应头（用于实验记录）。"""
        headers = {}
        if self.is_ooo:
            headers["X-Astro-OoO"] = "true"
        if self.is_downgraded:
            headers["X-Astro-Downgraded"] = "true"
        if self.routing_strategy:
            headers["X-Astro-Routing-Strategy"] = self.routing_strategy
        if self.compute_class:
            headers["X-Astro-Compute-Class"] = self.compute_class
        if self.original_model:
            headers["X-Astro-Original-Model"] = self.original_model
        if self.effective_model:
            headers["X-Astro-Model"] = self.effective_model
        headers["X-Astro-Time-Priority"] = str(self.time_priority)
        if self.prefix_hash:
            headers["X-Astro-Prefix-Hash"] = self.prefix_hash
        if self.agent_intent:
            headers["X-Astro-Agent-Intent"] = self.agent_intent
        return headers


@dataclass
class LLMChatResult:
    """V8.0 LLM 聊天结果（含元数据）。"""
    content: str
    task_id: str
    elapsed_ms: float
    model: str
    metadata: LLMResponseMetadata = field(default_factory=LLMResponseMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "task_id": self.task_id,
            "elapsed_ms": self.elapsed_ms,
            "model": self.model,
            "metadata": {
                "is_ooo": self.metadata.is_ooo,
                "is_downgraded": self.metadata.is_downgraded,
                "routing_strategy": self.metadata.routing_strategy,
                "compute_class": self.metadata.compute_class,
                "original_model": self.metadata.original_model,
                "effective_model": self.metadata.effective_model,
                "time_priority": self.metadata.time_priority,
                "prefix_hash": self.metadata.prefix_hash,
                "agent_intent": self.metadata.agent_intent,
            },
        }


# --------------------------------------------------------------------------- #
#  APIFacade — V8.0 统一门面                                                  #
# --------------------------------------------------------------------------- #

class APIFacade:
    """V8.0 北向 API 统一门面。

    聚合以下组件：
    - ``gateway_proxy`` : LLM 推理代理（负载均衡 + 熔断 + 异构路由）
    - ``lab_loader``    : 实验舱加载器（Registry + Engine + Bus）
    - ``scheduler``     : DAG 任务调度器（V8.0 OoO 乱序调度）
    - ``metrics_collector`` : 指标收集器（V8.0 新增，可选）

    V8.0 关键变化：
    - LLM 响应包含完整的元数据头（X-Astro-*），用于实验记录
    - 透明化 OoO 和降级细节，对上层不暴露内部实现
    - 支持 A2A Router 注入
    """

    def __init__(
        self,
        lab_loader: Any,
        gateway_proxy: Any = None,
        scheduler: Any = None,
        metrics_collector: Any = None,
        a2a_router: Any = None,
    ) -> None:
        self._lab_loader = lab_loader
        self._gateway_proxy = gateway_proxy
        self._scheduler = scheduler
        self._metrics_collector = metrics_collector
        self._a2a_router = a2a_router
        self._running = True

        logger.info(
            "[APIFacade] V8.0 统一门面初始化 | "
            "Gateway: %s | Scheduler: %s | Metrics: %s | A2A: %s",
            bool(gateway_proxy), bool(scheduler), bool(metrics_collector), bool(a2a_router),
        )

    # --------------------------------------------------------------------------- #
    #  生命周期                                                                #
    # --------------------------------------------------------------------------- #

    async def shutdown(self) -> None:
        """V8.0 优雅关闭。"""
        logger.info("[APIFacade] 正在关闭...")
        self._running = False

        if self._scheduler:
            await self._scheduler.shutdown()

        if self._metrics_collector:
            self._metrics_collector.stop()

        logger.info("[APIFacade] 已关闭")

    # --------------------------------------------------------------------------- #
    #  工具调用                                                                #
    # --------------------------------------------------------------------------- #

    async def execute_tool_call(
        self,
        lab_id: str,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> ToolCallResult:
        """执行 MCP Tool 调用（含 Guard 校验与联锁检查）。"""
        execution_id = uuid.uuid4().hex[:12]
        t0 = time.monotonic()

        ctx = self._lab_loader.get_lab(lab_id)
        if ctx is None:
            return ToolCallResult(
                status="error",
                tool_name=tool_name,
                execution_id=execution_id,
                agent_id=agent_id,
                detail=f"实验舱 '{lab_id}' 未找到",
                execution_time_ms=0.0,
            )

        from labs.mcp_registry import MCPToolContext
        mcp_ctx = MCPToolContext(engine=ctx.engine, bus=ctx.bus, lab_id=lab_id)

        # 联锁预检查
        try:
            await ctx.engine.check_interlocks()
        except Exception as interlock_exc:
            return ToolCallResult(
                status="interlock_blocked",
                tool_name=tool_name,
                execution_id=execution_id,
                agent_id=agent_id,
                detail=str(interlock_exc),
                fsm_states=ctx.engine.current_states,
                execution_time_ms=(time.monotonic() - t0) * 1000,
            )

        # 工具执行
        try:
            result = await ctx.registry.invoke(tool_name, params, mcp_ctx)
            elapsed_ms = (time.monotonic() - t0) * 1000

            return ToolCallResult(
                status=result.get("status", "ok"),
                tool_name=tool_name,
                execution_id=execution_id,
                agent_id=agent_id,
                detail=result.get("detail"),
                result=result.get("result"),
                fsm_states=ctx.engine.current_states,
                execution_time_ms=elapsed_ms,
            )

        except Exception as tool_exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[APIFacade] [%s] 工具 '%s' 执行失败: %s",
                agent_id, tool_name, tool_exc,
            )
            return ToolCallResult(
                status="error",
                tool_name=tool_name,
                execution_id=execution_id,
                agent_id=agent_id,
                detail=str(tool_exc),
                fsm_states=ctx.engine.current_states,
                execution_time_ms=elapsed_ms,
            )

    # --------------------------------------------------------------------------- #
    #  V8.0: LLM 推理（含元数据响应头）                                            #
    # --------------------------------------------------------------------------- #

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "http_client",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMChatResult:
        """V8.0 LLM 对话补全（含透明元数据响应头）。

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式消息列表
        agent_id : str
            调用者 Agent ID（用于意图检测和路由）
        temperature : float, optional
            采样温度
        max_tokens : int, optional
            最大生成长度
        model : str, optional
            模型名称（V8.0：可能因降级而与请求不同）

        Returns
        -------
        LLMChatResult
            包含 content / task_id / elapsed_ms / model / metadata
            metadata 中包含 X-Astro-* 响应头所需的所有字段
        """
        task_id = uuid.uuid4().hex[:12]
        t0 = time.monotonic()

        # V8.0: 初始化元数据
        metadata = LLMResponseMetadata(
            original_model=model or "qwen2.5:7b",
            effective_model=model or "qwen2.5:7b",
        )

        if self._gateway_proxy is None:
            return LLMChatResult(
                content="[FACADE] LLM Gateway 未配置",
                task_id=task_id,
                elapsed_ms=0.0,
                model="mock",
                metadata=metadata,
            )

        from infra.gateway.proxy import GatewayRequest
        gw_request = GatewayRequest(
            messages=messages,
            model=model or "qwen2.5:7b",
            temperature=temperature or 0.1,
            max_tokens=max_tokens or 2048,
            priority="NORMAL",
            agent_id=agent_id,
        )

        try:
            response = await self._gateway_proxy.chat(gw_request)
            elapsed_ms = (time.monotonic() - t0) * 1000

            from infra.gateway.proxy import GatewayError as GWErr
            if isinstance(response, GWErr):
                return LLMChatResult(
                    content=f"[ERROR] LLM 网关错误: {response.error}",
                    task_id=response.request_id or task_id,
                    elapsed_ms=elapsed_ms,
                    model=response.model or metadata.original_model,
                    metadata=metadata,
                )

            # V8.0: 从 response 中提取路由元数据
            routing_meta = getattr(response, "_routing_metadata", None)
            if routing_meta is not None:
                metadata.is_downgraded = getattr(routing_meta, "is_downgraded", False)
                metadata.routing_strategy = (
                    getattr(routing_meta, "strategy", None).__class__.__name__
                    if hasattr(routing_meta, "strategy") else
                    str(getattr(routing_meta, "strategy", ""))
                )
                metadata.compute_class = getattr(routing_meta, "compute_class", "heavy")
                metadata.time_priority = getattr(routing_meta, "time_priority", 0)
                metadata.prefix_hash = getattr(routing_meta, "prefix_hash", "")
                metadata.effective_model = getattr(routing_meta, "effective_model_name", response.model or metadata.original_model)
                metadata.agent_intent = getattr(routing_meta, "agent_intent", "")

            # V8.0: 检查是否经过 OoO 调度优化
            if self._scheduler is not None:
                status = self._scheduler.status_summary
                if status.get("ooo_execution_count", 0) > 0:
                    metadata.is_ooo = True

            return LLMChatResult(
                content=response.content or "",
                task_id=response.request_id or task_id,
                elapsed_ms=elapsed_ms,
                model=response.model or metadata.effective_model,
                metadata=metadata,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[APIFacade] LLM 调用失败 [%s]: %s", agent_id, exc)
            return LLMChatResult(
                content=f"[ERROR] LLM 调用失败: {exc}",
                task_id=task_id,
                elapsed_ms=elapsed_ms,
                model=metadata.effective_model,
                metadata=metadata,
            )

    async def chat_completion_with_headers(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "http_client",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[LLMChatResult, dict[str, str]]:
        """V8.0 LLM 对话补全，返回 (结果, 响应头)。

        响应头包含：
        - X-Astro-OoO: true — 经过乱序调度优化
        - X-Astro-Downgraded: true — 经过模型透明降级
        - X-Astro-Routing-Strategy — 路由策略
        - X-Astro-Compute-Class — 算力分级
        - X-Astro-Model — 实际使用的模型
        - X-Astro-Time-Priority — 时间维度优先级
        """
        result = await self.chat_completion(
            messages=messages,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        headers = result.metadata.to_response_headers()
        return result, headers

    # --------------------------------------------------------------------------- #
    #  A2A 通信                                                                #
    # --------------------------------------------------------------------------- #

    def broadcast_a2a(
        self,
        from_agent: str,
        to_agent: str,
        intent: A2AIntent,
        payload: dict[str, Any],
    ) -> None:
        """通过 A2A 协议向指定代理发送消息。"""
        msg = A2AMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            intent=intent,
            payload=payload,
        )
        if self._a2a_router:
            try:
                self._a2a_router.route(msg)
            except Exception as exc:
                logger.warning("[APIFacade] A2A 路由失败: %s", exc)

    # --------------------------------------------------------------------------- #
    #  调度器 & 指标                                                            #
    # --------------------------------------------------------------------------- #

    async def submit_dag(self, dag_def: dict[str, Any], agent_id: str) -> str:
        """提交 DAG 到调度器（V8.0 支持指标收集）。"""
        if self._scheduler is None:
            return ""

        graph_id = await self._scheduler.submit_dag(dag_def)

        # V8.0: 记录 DAG 开始到指标收集器
        if self._metrics_collector is not None:
            node_count = len(dag_def.get("nodes", []))
            self._metrics_collector.on_dag_start(graph_id, node_count)

        return graph_id

    def get_scheduler_status(self) -> dict[str, Any]:
        """V8.0 获取调度器状态（含 OoO 指标）。"""
        if self._scheduler is None:
            return {"available": False}
        return self._scheduler.status_summary

    def get_metrics_report(self) -> dict[str, Any] | None:
        """V8.0 获取指标报告（如已配置指标收集器）。"""
        if self._metrics_collector is None:
            return None
        return self._metrics_collector.get_quick_stats()


__all__ = [
    "APIFacade",
    "ToolCallResult",
    "LLMResponseMetadata",
    "LLMChatResult",
]
