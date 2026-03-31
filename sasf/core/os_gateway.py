"""
AstroSASF · Core · AstroOSFacade (V7.1 — Northbound API Adapter)
================================================================
北向接口层 —— 封装 AstroSASF 内核，对外暴露统一的「太空操作系统」API。

外部多智能体框架（对方团队）通过此门面接入 AstroSASF：
1. **工具发现** — 获取实验舱的 MCP Tool Schema
2. **LLM 算力** — 调用底层 LLM 进行推理
3. **原子调度** — 提交 Tool Call，经 FSM + Guard 校验后执行
4. **事件订阅** — 硬件报警抢占时接收回调通知

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum

from sasf.core.config_loader import SASFConfig, LLMConfig, load_config, create_llm
from sasf.core.models import TaskPriority, DAGNode, HardwareInterruptTask
from sasf.core.orchestrator import DAGOrchestrator
from sasf.core.environment import LaboratoryEnvironment
from sasf.middleware.mcp_registry import MCPToolContext, MCPToolRegistry
from sasf.physics.interlock_engine import SecurityGuardrailException

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Error Codes & Enums                                                        #
# --------------------------------------------------------------------------- #

class ExecutionStatus(str, Enum):
    """Tool Call 执行状态码。"""
    SUCCESS = "SUCCESS"
    FSM_BLOCKED = "FSM_BLOCKED"
    GUARD_BLOCKED = "GUARD_BLOCKED"
    PREEMPTED = "PREEMPTED"
    NOT_FOUND = "NOT_FOUND"
    INVALID_PARAMS = "INVALID_PARAMS"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# --------------------------------------------------------------------------- #
#  Hardware Alert Callback Types                                              #
# --------------------------------------------------------------------------- #

HardwareAlertCallback = Callable[[HardwareInterruptTask], Awaitable[None]]


@dataclass
class ToolCallResult:
    """Tool Call 执行结果（统一返回格式）。"""
    status: ExecutionStatus
    tool_name: str
    agent_id: str
    execution_id: str
    detail: str | None = None
    result: dict[str, Any] | None = None
    fsm_states: dict[str, str] | None = None
    execution_time_ms: float = 0.0
    preempted_by: str | None = None  # 如果被抢占，记录抢占者

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "tool_name": self.tool_name,
            "agent_id": self.agent_id,
            "execution_id": self.execution_id,
            "detail": self.detail,
            "result": self.result,
            "fsm_states": self.fsm_states,
            "execution_time_ms": self.execution_time_ms,
            "preempted_by": self.preempted_by,
        }


# --------------------------------------------------------------------------- #
#  LLM Manager                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class LLMManager:
    """LLM 生命周期管理器 —— 统一管理底层 LLM 算力。

    对外暴露简洁的 chat_completion 接口，支持参数透传。
    所有 LLM 推理任务自动注册到 Orchestrator，供硬件中断时 Cancel。
    """

    config: LLMConfig
    _llm: Any = field(default=None, init=False, repr=False)
    _orchestrator: DAGOrchestrator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._llm = create_llm(self.config)
        logger.info(
            "[LLMManager] 初始化完成: provider=%s, model=%s",
            self.config.provider, self.config.model_name,
        )

    def bind_orchestrator(self, orchestrator: DAGOrchestrator) -> None:
        """绑定 Orchestrator，用于 LLM 任务 Cancel 追踪。"""
        self._orchestrator = orchestrator

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "unknown",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """发起 LLM 推理请求。

        Parameters
        ----------
        messages : list[dict[str, str]]
            标准消息列表，格式：[{"role": "user", "content": "..."}]
        agent_id : str
            调用者 Agent ID（用于追踪和日志）
        temperature : float, optional
            采样温度，覆盖默认值
        max_tokens : int, optional
            最大生成 token 数
        stop : list[str], optional
            停止词列表

        Returns
        -------
        dict[str, Any]
            标准 LLM 响应，包含 "content"（生成的文本）
        """
        task_id = f"llm_{agent_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()

        # 构建调用参数（合并默认配置和传入参数）
        # 注意：LangChain ChatModel 使用 "input" 作为消息列表的参数名
        # Ollama 不支持在 invoke 时传递 temperature，需使用 LLM 实例级别的配置
        invoke_params: dict[str, Any] = {"input": messages}
        # temperature 在 LLM 实例初始化时设置，此处不再传递
        if max_tokens is not None:
            invoke_params["max_tokens"] = max_tokens
        if stop is not None:
            invoke_params["stop"] = stop

        # 注册 LLM 任务到 Orchestrator（供硬件中断 Cancel）
        llm_task: asyncio.Task | None = None
        if self._orchestrator is not None:
            # 创建包装任务以自动注册/注销
            async def _wrapped_llm() -> dict[str, Any]:
                try:
                    return await self._llm.ainvoke(**invoke_params)
                finally:
                    if self._orchestrator is not None:
                        self._orchestrator.unregister_llm_task(task_id)

            llm_task = asyncio.create_task(_wrapped_llm(), name=task_id)
            self._orchestrator.register_llm_task(task_id, llm_task)

        try:
            logger.debug(
                "[LLMManager] [%s] Agent '%s' 请求 LLM 推理 (task_id=%s)",
                agent_id, task_id, task_id,
            )

            if llm_task is not None:
                response = await llm_task
            else:
                response = await self._llm.ainvoke(**invoke_params)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            # 提取文本内容
            content = ""
            if hasattr(response, "content"):
                content = response.content
            elif isinstance(response, dict):
                content = response.get("content", "")
            elif isinstance(response, str):
                content = response

            result = {
                "content": content,
                "task_id": task_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "model": self.config.model_name,
            }

            logger.info(
                "[LLMManager] [%s] LLM 推理完成 (task_id=%s, elapsed=%.1fms)",
                agent_id, task_id, elapsed_ms,
            )

            return result

        except asyncio.CancelledError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[LLMManager] [%s] LLM 推理被 Cancel (task_id=%s, elapsed=%.1fms)",
                agent_id, task_id, elapsed_ms,
            )
            raise  # 重新抛出 CancelledError，让上层处理

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.exception(
                "[LLMManager] [%s] LLM 推理异常 (task_id=%s): %s",
                agent_id, task_id, exc,
            )
            return {
                "error": str(exc),
                "task_id": task_id,
                "elapsed_ms": round(elapsed_ms, 2),
            }


# --------------------------------------------------------------------------- #
#  AstroOSFacade — 北向接口门面                                                #
# --------------------------------------------------------------------------- #

@dataclass
class AstroOSFacade:
    """AstroSASF 北向接口门面 (V7.1)。

    对外提供统一的「太空操作系统」API，封装内部复杂调度逻辑。

    Example
    -------
    >>> facade = AstroOSFacade.from_config()
    >>> await facade.initialize()
    >>>
    >>> # 1. 获取工具列表
    >>> tools = await facade.get_lab_tools("bio_lab_01")
    >>>
    >>> # 2. 调用 LLM 思考
    >>> response = await facade.chat_completion(
    ...     messages=[{"role": "user", "content": "分析传感器数据"}],
    ...     agent_id="agent_001",
    ... )
    >>>
    >>> # 3. 提交 Tool Call
    >>> result = await facade.execute_tool_call(
    ...     lab_id="bio_lab_01",
    ...     tool_name="read_sensor",
    ...     params={"channel": 1},
    ...     agent_id="agent_001",
    ... )
    >>>
    >>> # 4. 订阅硬件报警
    >>> async def on_alert(interrupt):
    ...     print(f"硬件报警: {interrupt.description}")
    >>> facade.register_hardware_alert_callback(on_alert)
    """

    config: SASFConfig
    _orchestrator: DAGOrchestrator = field(init=False, repr=False)
    _labs: dict[str, LaboratoryEnvironment] = field(default_factory=dict, init=False)
    _llm_manager: LLMManager = field(init=False, repr=False)
    _hardware_alert_callbacks: list[HardwareAlertCallback] = field(
        default_factory=list, init=False,
    )
    _initialized: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @classmethod
    def from_config(cls, config_path: str | None = None) -> AstroOSFacade:
        """从配置文件创建 Facade 实例。"""
        config = load_config(config_path)
        instance = cls(config=config)
        logger.info("[AstroOSFacade] 从配置文件创建实例")
        return instance

    @classmethod
    def from_config_obj(cls, config: SASFConfig) -> AstroOSFacade:
        """从配置对象创建 Facade 实例。"""
        return cls(config=config)

    # --------------------------------------------------------------------------- #
    #  生命周期管理                                                               #
    # --------------------------------------------------------------------------- #

    async def initialize(self) -> None:
        """初始化 Facade —— 启动 Orchestrator 和 LLM Manager。"""
        async with self._lock:
            if self._initialized:
                logger.warning("[AstroOSFacade] 已初始化，跳过")
                return

            logger.info("")
            logger.info("╔" + "═" * 60 + "╗")
            logger.info("║  🌌 AstroSASF 北向接口层初始化 (V7.1)                   ║")
            logger.info("╚" + "═" * 60 + "╝")

            # 1) 初始化 LLM Manager
            self._llm_manager = LLMManager(config=self.config.llm)

            # 2) 初始化 DAG Orchestrator
            self._orchestrator = DAGOrchestrator(
                config=self.config,
                max_workers=self.config.orchestrator.max_concurrent_labs,
            )

            # 绑定 LLM Manager 到 Orchestrator（用于硬件中断 Cancel）
            self._llm_manager.bind_orchestrator(self._orchestrator)

            # 3) 启动 Orchestrator Workers
            await self._orchestrator.start()

            self._initialized = True
            logger.info("[AstroOSFacade] 初始化完成 ✓")

    async def shutdown(self) -> dict[str, Any]:
        """优雅关闭 Facade。"""
        async with self._lock:
            if not self._initialized:
                return {"status": "already_shutdown"}

            logger.info("[AstroOSFacade] 正在关闭...")

            # 关闭 Orchestrator
            if hasattr(self, "_orchestrator") and self._orchestrator is not None:
                await self._orchestrator.shutdown()

            self._initialized = False
            logger.info("[AstroOSFacade] 已关闭 ✓")

            return {"status": "shutdown_complete"}

    # --------------------------------------------------------------------------- #
    #  任务 1: 工具发现接口 (Tool Discovery)                                       #
    # --------------------------------------------------------------------------- #

    async def get_lab_tools(self, lab_id: str) -> list[dict[str, Any]]:
        """获取指定实验舱的所有 MCP 工具（含 Schema）。

        Parameters
        ----------
        lab_id : str
            实验舱 ID

        Returns
        -------
        list[dict[str, Any]]
            工具列表，每项包含：
            - name: 工具名
            - description: 工具描述
            - json_schema: OpenAI Function Calling 格式的 Schema
            - is_macro: 是否为宏指令

        Raises
        ------
        ValueError
            如果实验舱不存在
        """
        lab = self._labs.get(lab_id)
        if lab is None:
            raise ValueError(f"实验舱 '{lab_id}' 未注册")

        tools = lab.registry.list_tools()
        logger.debug("[%s] 工具发现: 返回 %d 个工具", lab_id, len(tools))
        return tools

    async def list_lab_ids(self) -> list[str]:
        """获取所有已注册的实验舱 ID。"""
        return list(self._labs.keys())

    async def register_lab(self, lab: LaboratoryEnvironment) -> None:
        """注册实验舱到 Facade。

        Parameters
        ----------
        lab : LaboratoryEnvironment
            已初始化的实验舱环境
        """
        self._labs[lab.lab_id] = lab
        self._orchestrator.register_lab(lab)

        # 注册硬件报警回调
        await self._setup_lab_hardware_callbacks(lab)

        logger.info("[AstroOSFacade] 注册实验舱: %s", lab.lab_id)

    async def _setup_lab_hardware_callbacks(self, lab: LaboratoryEnvironment) -> None:
        """为实验舱设置硬件报警回调（内部方法）。"""
        bus = getattr(lab, "_bus", None)
        if bus is None:
            return

        def _alert_wrapper(interrupt: HardwareInterruptTask) -> None:
            """报警回调包装 —— 在事件循环外被调用，转为异步处理。"""
            asyncio.create_task(
                self._dispatch_hardware_alert(interrupt),
                name=f"hw-alert-{interrupt.interrupt_id}",
            )

        bus.register_alarm_callback(_alert_wrapper)

    async def _dispatch_hardware_alert(self, interrupt: HardwareInterruptTask) -> None:
        """分发硬件报警到所有注册的回调。"""
        logger.warning(
            "🚨 [AstroOSFacade] 硬件报警触发: %s → %s",
            interrupt.interrupt_id, interrupt.description,
        )

        for callback in self._hardware_alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(interrupt)
                else:
                    callback(interrupt)
            except Exception as exc:
                logger.exception(
                    "[AstroOSFacade] 硬件报警回调执行失败: %s → %s",
                    callback, exc,
                )

    # --------------------------------------------------------------------------- #
    #  任务 2: LLM 生命周期管理 (LLM Provider)                                     #
    # --------------------------------------------------------------------------- #

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "unknown",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """发起 LLM 推理请求。

        Parameters
        ----------
        messages : list[dict[str, str]]
            标准消息列表
        agent_id : str
            调用者 Agent ID
        temperature : float, optional
            采样温度
        max_tokens : int, optional
            最大 token 数

        Returns
        -------
        dict[str, Any]
            LLM 响应，包含 "content" 字段
        """
        if not self._initialized:
            await self.initialize()

        return await self._llm_manager.chat_completion(
            messages=messages,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def llm_stats(self) -> dict[str, Any]:
        """获取 LLM 统计信息。"""
        return {
            "provider": self.config.llm.provider,
            "model": self.config.llm.model_name,
            "base_url": self.config.llm.base_url,
            "llm_tasks_in_progress": (
                len(self._orchestrator.get_llm_task_ids())
                if hasattr(self, "_orchestrator") and self._orchestrator
                else 0
            ),
        }

    # --------------------------------------------------------------------------- #
    #  任务 3: 原子工具调度与执行 (Tool Execution Engine)                         #
    # --------------------------------------------------------------------------- #

    async def execute_tool_call(
        self,
        lab_id: str,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str = "unknown",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 60.0,
    ) -> ToolCallResult:
        """执行原子工具调用 —— 核心接口。

        执行流程：
        1. 参数校验
        2. FSM (InterlockEngine) 联锁校验
        3. Guard (ToolGuard) 前置校验
        4. 进入 ReadyQueue 等待 Worker 执行
        5. 返回执行结果或错误码

        Parameters
        ----------
        lab_id : str
            目标实验舱 ID
        tool_name : str
            MCP Tool 名称
        params : dict[str, Any]
            工具调用参数
        agent_id : str
            调用者 Agent ID
        priority : TaskPriority
            任务优先级（默认 NORMAL）
        timeout : float
            执行超时时间（秒）

        Returns
        -------
        ToolCallResult
            统一执行结果，包含状态码和详情
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = time.monotonic()

        # ── Step 1: 参数校验 ── #
        lab = self._labs.get(lab_id)
        if lab is None:
            return ToolCallResult(
                status=ExecutionStatus.NOT_FOUND,
                tool_name=tool_name,
                agent_id=agent_id,
                execution_id=execution_id,
                detail=f"实验舱 '{lab_id}' 不存在",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        registry = lab.registry
        if not registry.has_tool(tool_name):
            return ToolCallResult(
                status=ExecutionStatus.NOT_FOUND,
                tool_name=tool_name,
                agent_id=agent_id,
                execution_id=execution_id,
                detail=f"Tool '{tool_name}' 未注册",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── Step 2: FSM (InterlockEngine) 联锁校验 ── #
        engine = lab.engine
        bus = getattr(lab, "_bus", None)

        try:
            telemetry = await bus.snapshot() if bus else {}
            await engine.check_interlocks(tool_name=tool_name, telemetry=telemetry)
        except SecurityGuardrailException as exc:
            logger.warning(
                "[%s] [execute_tool_call] FSM 拦截: %s (execution_id=%s)",
                lab_id, exc, execution_id,
            )
            return ToolCallResult(
                status=ExecutionStatus.FSM_BLOCKED,
                tool_name=tool_name,
                agent_id=agent_id,
                execution_id=execution_id,
                detail=f"FSM 联锁拦截: {exc}",
                fsm_states=engine.current_states,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── Step 3: Guard 前置校验 ── #
        descriptor = registry.get_tool(tool_name)
        if descriptor and descriptor.guard:
            ctx = MCPToolContext(engine=engine, bus=bus, lab_id=lab_id)
            try:
                # Guard 校验在 invoke() 内部完成，这里手动触发以便返回特定错误码
                await _validate_guard(descriptor.guard, engine, bus, lab_id, tool_name)
            except SecurityGuardrailException as exc:
                logger.warning(
                    "[%s] [execute_tool_call] Guard 拦截: %s (execution_id=%s)",
                    lab_id, exc, execution_id,
                )
                return ToolCallResult(
                    status=ExecutionStatus.GUARD_BLOCKED,
                    tool_name=tool_name,
                    agent_id=agent_id,
                    execution_id=execution_id,
                    detail=f"Guard 拦截: {exc}",
                    fsm_states=engine.current_states,
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

        # ── Step 4: 创建 DAGNode 并加入 ReadyQueue ── #
        node_id = f"{agent_id}_{tool_name}_{uuid.uuid4().hex[:8]}"
        dag_node = DAGNode(
            node_id=node_id,
            skill_name=tool_name,
            params=params,
            dependencies=[],
            priority=priority,
            description=f"[{agent_id}] {tool_name}",
            lab_id=lab_id,
        )

        # 使用 asyncio.create_task 创建执行任务（不等待结果）
        execution_task = asyncio.create_task(
            self._execute_node_with_tracking(dag_node, execution_id),
            name=f"exec-{execution_id}",
        )

        # ── Step 5: 等待执行结果或超时 ── #
        try:
            result = await asyncio.wait_for(execution_task, timeout=timeout)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # 检查结果是否为 FSM/Guard 拦截
            result_detail = result.get("detail", "")
            if "联锁拦截" in result_detail or "FSM" in result_detail:
                return ToolCallResult(
                    status=ExecutionStatus.FSM_BLOCKED,
                    tool_name=tool_name,
                    agent_id=agent_id,
                    execution_id=execution_id,
                    detail=result_detail,
                    fsm_states=engine.current_states,
                    execution_time_ms=elapsed_ms,
                )
            elif "Guard" in result_detail:
                return ToolCallResult(
                    status=ExecutionStatus.GUARD_BLOCKED,
                    tool_name=tool_name,
                    agent_id=agent_id,
                    execution_id=execution_id,
                    detail=result_detail,
                    fsm_states=engine.current_states,
                    execution_time_ms=elapsed_ms,
                )

            return ToolCallResult(
                status=ExecutionStatus.SUCCESS if result.get("status") == "success"
                       else ExecutionStatus.INTERNAL_ERROR,
                tool_name=tool_name,
                agent_id=agent_id,
                execution_id=execution_id,
                result=result,
                fsm_states=engine.current_states,
                execution_time_ms=elapsed_ms,
            )

        except asyncio.TimeoutError:
            execution_task.cancel()
            logger.warning(
                "[%s] [execute_tool_call] 执行超时 (execution_id=%s, timeout=%.1fs)",
                lab_id, execution_id, timeout,
            )
            return ToolCallResult(
                status=ExecutionStatus.INTERNAL_ERROR,
                tool_name=tool_name,
                agent_id=agent_id,
                execution_id=execution_id,
                detail=f"执行超时 ({timeout}s)",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        except asyncio.CancelledError:
            logger.warning(
                "[%s] [execute_tool_call] 执行被取消 (execution_id=%s)",
                lab_id, execution_id,
            )
            return ToolCallResult(
                status=ExecutionStatus.PREEMPTED,
                tool_name=tool_name,
                agent_id=agent_id,
                execution_id=execution_id,
                detail="执行被硬件中断抢占",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

    async def _execute_node_with_tracking(
        self,
        node: DAGNode,
        execution_id: str,
    ) -> dict[str, Any]:
        """执行 DAGNode 并跟踪状态。"""
        lab = self._labs.get(node.lab_id)
        if lab is None:
            return {"status": "error", "detail": f"Lab not found: {node.lab_id}"}

        try:
            result = await lab.gateway.invoke_tool(
                tool_name=node.skill_name,
                params=node.params,
            )
            return result
        except SecurityGuardrailException as exc:
            return {"status": "error", "detail": str(exc)}
        except Exception as exc:
            logger.exception(
                "[_execute_node_with_tracking] 执行异常 (execution_id=%s)",
                execution_id,
            )
            return {"status": "error", "detail": f"Internal error: {exc}"}

    # --------------------------------------------------------------------------- #
    #  任务 4: 事件订阅 (Event Pub/Sub)                                           #
    # --------------------------------------------------------------------------- #

    def register_hardware_alert_callback(
        self,
        callback: HardwareAlertCallback,
    ) -> str:
        """注册硬件报警回调函数。

        当底层 TelemetryBus 触发逃生任务时，所有注册的回调都会被调用。

        Parameters
        ----------
        callback : HardwareAlertCallback
            异步回调函数，签名：
            ``async def on_alert(interrupt: HardwareInterruptTask) -> None``

        Returns
        -------
        str
            回调订阅 ID（可用于取消订阅）

        Example
        -------
        >>> async def on_fire_alert(interrupt):
        ...     print(f"火灾报警! 需要执行: {interrupt.action_skill}")
        ...
        >>> sub_id = facade.register_hardware_alert_callback(on_fire_alert)
        >>> # 取消订阅：
        >>> facade.unregister_hardware_alert_callback(sub_id)
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
        wrapped_callback = _wrap_alert_callback(callback, subscription_id)
        self._hardware_alert_callbacks.append(wrapped_callback)

        logger.info(
            "[AstroOSFacade] 注册硬件报警回调 (id=%s, total=%d)",
            subscription_id, len(self._hardware_alert_callbacks),
        )
        return subscription_id

    def unregister_hardware_alert_callback(self, subscription_id: str) -> bool:
        """取消注册硬件报警回调。"""
        original_count = len(self._hardware_alert_callbacks)
        self._hardware_alert_callbacks = [
            cb for cb in self._hardware_alert_callbacks
            if getattr(cb, "_subscription_id", None) != subscription_id
        ]
        removed = original_count - len(self._hardware_alert_callbacks)

        if removed > 0:
            logger.info(
                "[AstroOSFacade] 取消硬件报警回调 (id=%s)",
                subscription_id,
            )
        return removed > 0

    @property
    def hardware_alert_subscription_count(self) -> int:
        """获取当前订阅数量。"""
        return len(self._hardware_alert_callbacks)

    # --------------------------------------------------------------------------- #
    #  便捷方法 & 状态查询                                                         #
    # --------------------------------------------------------------------------- #

    @property
    def status_summary(self) -> dict[str, Any]:
        """获取 Facade 状态摘要。"""
        return {
            "initialized": self._initialized,
            "registered_labs": list(self._labs.keys()),
            "orchestrator": (
                self._orchestrator.status_summary
                if hasattr(self, "_orchestrator") and self._orchestrator
                else None
            ),
            "llm_stats": self.llm_stats if hasattr(self, "_llm_manager") else None,
            "hardware_alert_subscriptions": self.hardware_alert_subscription_count,
        }

    async def get_lab_status(self, lab_id: str) -> dict[str, Any] | None:
        """获取实验舱详细状态。"""
        lab = self._labs.get(lab_id)
        if lab is None:
            return None

        telemetry = await lab.get_telemetry()
        return {
            "lab_id": lab_id,
            "engine_states": lab.engine_states,
            "telemetry": telemetry,
            "available_tools_count": lab.registry.count,
            "macro_count": lab.registry.macro_count,
        }


# --------------------------------------------------------------------------- #
#  Internal Helper Functions                                                  #
# --------------------------------------------------------------------------- #

async def _validate_guard(
    guard: Any,
    engine: Any,
    bus: Any,
    lab_id: str,
    tool_name: str,
) -> None:
    """手动执行 Guard 前置校验（抛出 SecurityGuardrailException）。"""
    from sasf.physics.interlock_engine import safe_eval_bool

    # require_states 校验
    for subsystem, required_state in guard.require_states.items():
        try:
            current = engine.get_subsystem_state(subsystem)
            if current != required_state:
                raise SecurityGuardrailException(
                    f"[{lab_id}] Guard 拦截 '{tool_name}': "
                    f"子系统 '{subsystem}' 需要 '{required_state}' "
                    f"但当前为 '{current}'"
                )
        except KeyError:
            raise SecurityGuardrailException(
                f"[{lab_id}] Guard 拦截 '{tool_name}': 未知子系统 '{subsystem}'"
            )

    # forbid_states 校验
    for subsystem, forbidden_state in guard.forbid_states.items():
        try:
            current = engine.get_subsystem_state(subsystem)
            if current == forbidden_state:
                raise SecurityGuardrailException(
                    f"[{lab_id}] Guard 拦截 '{tool_name}': "
                    f"子系统 '{subsystem}' 处于禁止状态 '{forbidden_state}'"
                )
        except KeyError:
            raise SecurityGuardrailException(
                f"[{lab_id}] Guard 拦截 '{tool_name}': 未知子系统 '{subsystem}'"
            )

    # telemetry_rules 校验
    if guard.telemetry_rules and bus:
        # 直接创建协程并 await，避免嵌套的 asyncio.run()
        async def _get_telemetry() -> dict:
            return await bus.snapshot()

        try:
            telemetry = await _get_telemetry()
        except Exception:
            telemetry = {}
        for rule_expr in guard.telemetry_rules:
            try:
                if not safe_eval_bool(rule_expr, telemetry):
                    raise SecurityGuardrailException(
                        f"[{lab_id}] Guard 拦截 '{tool_name}': "
                        f"遥测条件不满足: {rule_expr}"
                    )
            except SecurityGuardrailException:
                raise
            except Exception as exc:
                raise SecurityGuardrailException(
                    f"[{lab_id}] Guard 校验异常 '{tool_name}': {exc}"
                ) from exc


@dataclass
class _WrappedAlertCallback:
    """包装的报警回调（带订阅 ID）。"""
    _original_callback: HardwareAlertCallback
    _subscription_id: str
    _is_async: bool = field(init=False)

    def __post_init__(self) -> None:
        self._is_async = asyncio.iscoroutinefunction(self._original_callback)

    async def __call__(self, interrupt: HardwareInterruptTask) -> None:
        try:
            if self._is_async:
                await self._original_callback(interrupt)
            else:
                self._original_callback(interrupt)
        except Exception as exc:
            logger.exception(
                "[_WrappedAlertCallback] 回调执行失败 (id=%s): %s",
                self._subscription_id, exc,
            )


def _wrap_alert_callback(
    callback: HardwareAlertCallback,
    subscription_id: str,
) -> _WrappedAlertCallback:
    """包装报警回调以便追踪。"""
    return _WrappedAlertCallback(
        _original_callback=callback,
        _subscription_id=subscription_id,
    )


__all__ = [
    "AstroOSFacade",
    "LLMManager",
    "ExecutionStatus",
    "ToolCallResult",
    "HardwareAlertCallback",
]
