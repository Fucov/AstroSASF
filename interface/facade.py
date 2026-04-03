"""
AstroSASF · Interface · API Facade
===================================
北向 API 的统一门面，聚合 LLM Gateway、Lab Context 与 Scheduler。

V7.2: 从 server.py 中提取，解除直接循环依赖。

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from scheduler.models import TaskPriority
from scheduler.a2a_protocol import A2AMessage, A2AIntent

logger = logging.getLogger(__name__)


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
        # 确保 fsm_states 所有值都是字符串（FastAPI Pydantic 校验）
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


class APIFacade:
    """北向 API 统一门面 (V7.2)。

    聚合以下组件：
    - ``gateway_proxy`` : LLM 推理代理（负载均衡 + 熔断）
    - ``lab_loader``    : 实验舱加载器（Registry + Engine + Bus）
    - ``scheduler``     : DAG 任务调度器（可选，用于后台任务）

    提供 REST API 层所需的全部方法。
    """

    def __init__(
        self,
        lab_loader: Any,
        gateway_proxy: Any = None,
        scheduler: Any = None,
    ) -> None:
        self._lab_loader = lab_loader
        self._gateway_proxy = gateway_proxy
        self._scheduler = scheduler
        self._running = True
        self._a2a_router = Any  # A2A 路由，由 LabContext 提供

    # --------------------------------------------------------------------------- #
    #  生命周期                                                                    #
    # --------------------------------------------------------------------------- #

    async def shutdown(self) -> None:
        """优雅关闭。"""
        logger.info("[APIFacade] 正在关闭...")
        self._running = False

        if self._scheduler:
            await self._scheduler.shutdown()

        logger.info("[APIFacade] 已关闭 ✓")

    # --------------------------------------------------------------------------- #
    #  工具调用                                                                    #
    # --------------------------------------------------------------------------- #

    async def execute_tool_call(
        self,
        lab_id: str,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> ToolCallResult:
        """执行 MCP Tool 调用（含 Guard 校验与联锁检查）。

        Parameters
        ----------
        lab_id : str
            实验舱 ID
        tool_name : str
            MCP Tool 名称
        params : dict
            工具参数
        agent_id : str
            调用者代理 ID（用于 A2A 日志）
        priority : TaskPriority
            任务优先级（影响 VRAM 熔断器决策）

        Returns
        -------
        ToolCallResult
        """
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

        # ── 联锁预检查 ── #
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

        # ── 工具执行 ── #
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
    #  LLM 推理                                                                    #
    # --------------------------------------------------------------------------- #

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "http_client",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """调用 LLM Gateway 完成对话补全。

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式消息列表
        agent_id : str
            调用者 ID
        temperature : float, optional
            采样温度
        max_tokens : int, optional
            最大生成长度

        Returns
        -------
        dict
            包含 content、task_id、elapsed_ms、model 的字典
        """
        if self._gateway_proxy is None:
            return {
                "content": "[FACADE] LLM Gateway 未配置（demo 模式）",
                "task_id": uuid.uuid4().hex[:12],
                "elapsed_ms": 0.0,
                "model": "mock",
            }

        task_id = uuid.uuid4().hex[:12]
        t0 = time.monotonic()

        try:
            response = await self._gateway_proxy.chat(
                messages=messages,
                agent_id=agent_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            return {
                "content": response.get("content", ""),
                "task_id": task_id,
                "elapsed_ms": elapsed_ms,
                "model": response.get("model", "qwen2.5:7b"),
            }
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[APIFacade] LLM 调用失败 [%s]: %s", agent_id, exc)
            return {
                "content": f"[ERROR] LLM 调用失败: {exc}",
                "task_id": task_id,
                "elapsed_ms": elapsed_ms,
                "model": "qwen2.5:7b",
            }

    # --------------------------------------------------------------------------- #
    #  A2A 通信                                                                    #
    # --------------------------------------------------------------------------- #

    def broadcast_a2a(
        self,
        from_agent: str,
        to_agent: str,
        intent: A2AIntent,
        payload: dict[str, Any],
    ) -> None:
        """通过 A2A 协议向指定代理发送消息（用于调度层）。"""
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
    #  调度器                                                                      #
    # --------------------------------------------------------------------------- #

    async def submit_dag(self, dag_def: dict[str, Any], agent_id: str) -> str:
        """提交 DAG 到调度器。"""
        if self._scheduler is None:
            return ""
        return await self._scheduler.submit_dag(dag_def, agent_id)


__all__ = ["APIFacade", "ToolCallResult"]
