"""
AstroSASF · Core · LaboratoryEnvironment (V4.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境 —— Middleware-First Architecture。

V4.1 修复:
- build_lab_graph 现在返回 (compiled_graph, memory_saver) 元组
- run() 使用 HITL 中断循环方式调用图，支持 Human-in-the-Loop
- 提供 collect_stats() 方法，即使崩溃也能提取已产生的统计数据
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from sasf.cognition.graph_builder import build_lab_graph
from sasf.cognition.state import LabGraphState
from sasf.core.config_loader import SASFConfig, create_llm
from sasf.middleware.a2a_protocol import A2ARouter
from sasf.middleware.codec import SpaceMCPCodec
from sasf.middleware.gateway import SpaceMCPGateway
from sasf.middleware.skill_registry import SkillRegistry
from sasf.middleware.virtual_bus import VirtualSpaceWire
from sasf.physics.shadow_fsm import ShadowFSM, register_default_skills
from sasf.physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)


@dataclass
class LaboratoryEnvironment:
    """单个实验柜的完整运行时环境 (V4.1 HITL + Crash-safe stats)。"""

    lab_id: str
    config: SASFConfig

    # ---- 内部组件 --------------------------------------------------------- #
    _bus: TelemetryBus = field(init=False, repr=False)
    _fsm: ShadowFSM = field(init=False, repr=False)
    _registry: SkillRegistry = field(init=False, repr=False)
    _a2a_router: A2ARouter = field(init=False, repr=False)
    _codec: SpaceMCPCodec = field(init=False, repr=False)
    _space_wire: VirtualSpaceWire = field(init=False, repr=False)
    _gateway: SpaceMCPGateway = field(init=False, repr=False)
    _graph: Any = field(init=False, repr=False)
    _memory: MemorySaver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mw_cfg = self.config.middleware

        # 1) Physics Layer
        self._bus = TelemetryBus(lab_id=self.lab_id)
        self._fsm = ShadowFSM(lab_id=self.lab_id)

        # 2) Middleware Layer
        self._registry = SkillRegistry(lab_id=self.lab_id)
        self._a2a_router = A2ARouter(lab_id=self.lab_id)
        self._codec = SpaceMCPCodec(lab_id=self.lab_id)
        self._space_wire = VirtualSpaceWire(
            lab_id=self.lab_id,
            bandwidth_kbps=mw_cfg.spacewire_bandwidth_kbps,
        )
        self._gateway = SpaceMCPGateway(
            lab_id=self.lab_id,
            fsm=self._fsm,
            bus=self._bus,
            codec=self._codec,
            space_wire=self._space_wire,
            registry=self._registry,
            a2a_router=self._a2a_router,
        )

        # 3) Physics → Middleware 注册
        register_default_skills(
            registry=self._registry,
            fsm=self._fsm,
            bus=self._bus,
        )

        # 4) Cognition Layer (返回 graph + memory)
        llm = create_llm(self.config.llm)
        self._graph, self._memory = build_lab_graph(
            gateway=self._gateway,
            llm=llm,
            lab_id=self.lab_id,
            a2a_router=self._a2a_router,
        )

    # ---- 状态查询 --------------------------------------------------------- #

    async def get_telemetry(self) -> dict[str, Any]:
        return await self._bus.snapshot()

    @property
    def fsm_state(self) -> str:
        return self._fsm.current_state.name

    @property
    def available_skills(self) -> list[dict[str, Any]]:
        return self._gateway.list_skills()

    @property
    def codec_stats(self) -> dict[str, Any]:
        return self._codec.stats

    @property
    def bus_stats(self) -> dict[str, Any]:
        return self._space_wire.stats

    @property
    def a2a_stats(self) -> dict[str, Any]:
        return self._a2a_router.stats

    def collect_stats(self) -> dict[str, Any]:
        """安全地收集当前所有统计信息 —— 即使环境崩溃也可调用。"""
        stats: dict[str, Any] = {"lab_id": self.lab_id}
        try:
            stats["fsm_state"] = self._fsm.current_state.name
        except Exception:
            stats["fsm_state"] = "UNKNOWN"
        try:
            stats["codec_stats"] = self._codec.stats
        except Exception:
            stats["codec_stats"] = {}
        try:
            stats["bus_stats"] = self._space_wire.stats
        except Exception:
            stats["bus_stats"] = {}
        try:
            stats["a2a_stats"] = self._a2a_router.stats
        except Exception:
            stats["a2a_stats"] = {}
        return stats

    # ---- 任务运行 (HITL 中断循环) ----------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        """启动 LangGraph 工作流 —— 带 HITL 中断的执行循环。"""
        logger.info("=" * 64)
        logger.info("[%s] 🚀 实验柜环境启动 (V4.1 HITL)", self.lab_id)
        logger.info(
            "[%s] 📋 已注册 Skills: %s",
            self.lab_id,
            [s["name"] for s in self.available_skills],
        )
        logger.info("=" * 64)

        all_results: list[dict[str, Any]] = []

        for task_desc in tasks:
            logger.info("")
            logger.info("[%s] 📥 提交任务: %s", self.lab_id, task_desc)

            initial_state: LabGraphState = {
                "original_task": task_desc,
                "plan": [],
                "current_step_index": 0,
                "current_step": None,
                "fsm_feedback": None,
                "execution_log": [],
                "error_count": 0,
                "final_result": None,
            }

            # 每个任务一个独立的 thread_id
            thread_config = {"configurable": {"thread_id": uuid.uuid4().hex}}

            # ── HITL 中断循环 ── #
            result = await self._hitl_loop(initial_state, thread_config)
            all_results.append(result)

            logger.info(
                "[%s] 📤 任务完成: %s",
                self.lab_id,
                result.get("status", "N/A"),
            )

        # 收集环境结果
        final_telemetry = await self._bus.snapshot()
        env_result = {
            "lab_id": self.lab_id,
            "fsm_state": self._fsm.current_state.name,
            "final_telemetry": final_telemetry,
            "codec_stats": self._codec.stats,
            "bus_stats": self._space_wire.stats,
            "a2a_stats": self._a2a_router.stats,
            "task_results": all_results,
        }

        logger.info("")
        logger.info("-" * 64)
        logger.info("[%s] ✅ 环境关闭  FSM=%s", self.lab_id, env_result["fsm_state"])
        logger.info("[%s] 最终遥测: %s", self.lab_id, final_telemetry)
        logger.info("[%s] A2A 统计: %s", self.lab_id, self._a2a_router.stats)
        logger.info("-" * 64)

        return env_result

    async def _hitl_loop(
        self,
        initial_state: LabGraphState,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """HITL (Human-in-the-loop) 执行循环。

        图在 execute_node 前中断。每次中断时打印待执行的 Skill，
        等待外部回调决定是否继续（由 demo 脚本通过 stdin 输入控制）。
        """
        # 首次运行（planner → operator → 中断于 execute_node 前）
        state = await self._graph.ainvoke(initial_state, config)
        snapshot = self._graph.get_state(config)

        while snapshot.next:
            # 此时图中断在 execute_node 前
            step = state.get("current_step")
            if step and isinstance(step, dict):
                logger.info("")
                logger.info(
                    "[%s] ⏸️  HITL 中断 — 即将执行:",
                    self.lab_id,
                )
                logger.info(
                    "[%s]    Skill : %s", self.lab_id, step.get("skill"),
                )
                logger.info(
                    "[%s]    Params: %s",
                    self.lab_id,
                    json.dumps(step.get("params", {}), ensure_ascii=False),
                )

            # 人类审批：从 stdin 读取
            user_input = await self._get_human_approval(step)

            if user_input == "abort":
                logger.warning("[%s] ❌ 用户中止执行", self.lab_id)
                return {
                    "status": "aborted_by_user",
                    "total_steps": len(state.get("plan", [])),
                    "execution_log": list(state.get("execution_log") or []),
                }

            if user_input == "approve":
                # 继续执行（不修改 state）
                state = await self._graph.ainvoke(None, config)
            else:
                # 用户提供了修正后的 JSON 参数
                try:
                    corrected = json.loads(user_input)
                    if isinstance(corrected, dict):
                        self._graph.update_state(
                            config,
                            {"current_step": corrected},
                        )
                        logger.info(
                            "[%s] ✏️  用户修正参数: %s", self.lab_id, corrected,
                        )
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "[%s] 无法解析用户输入，按原计划继续", self.lab_id,
                    )
                state = await self._graph.ainvoke(None, config)

            snapshot = self._graph.get_state(config)

        # 图运行结束
        final = state.get("final_result")
        if final and isinstance(final, dict):
            return final
        return {
            "status": "completed",
            "total_steps": len(state.get("plan", [])),
            "execution_log": list(state.get("execution_log") or []),
        }

    @staticmethod
    async def _get_human_approval(step: dict[str, Any] | None) -> str:
        """从 stdin 获取人类审批。

        输入约定:
        - ``y`` / 回车 → approve（继续执行）
        - ``n``       → abort（中止）
        - JSON 字符串  → 修正参数
        """
        import asyncio
        import sys

        if step:
            skill = step.get("skill", "unknown")
            params = json.dumps(step.get("params", {}), ensure_ascii=False)
            prompt = (
                f"\n{'─' * 60}\n"
                f"🛡️ HITL | 即将执行: {skill}({params})\n"
                f"  [y/回车] 批准  |  [n] 中止  |  [JSON] 修正参数\n"
                f"{'─' * 60}\n"
                f">>> "
            )
        else:
            prompt = "\n>>> 继续? [y/n]: "

        # 在事件循环中安全读取 stdin
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: input(prompt))

        raw = raw.strip()
        if raw == "" or raw.lower() in ("y", "yes"):
            return "approve"
        if raw.lower() in ("n", "no"):
            return "abort"
        return raw  # 尝试作为 JSON 解析
