"""
AstroSASF · Core · LaboratoryEnvironment (V4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境 —— Middleware-First Architecture。

V4 装配流程:
1. Physics Layer — 创建 FSM + TelemetryBus
2. Middleware Layer — 创建 SkillRegistry + A2ARouter + Codec + SpaceWire + Gateway
3. Physics → Middleware — 物理设备主动注册 Skills 到 SkillRegistry
4. Cognition Layer — config_loader 创建 LLM，构建 LangGraph
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
    """单个实验柜的完整运行时环境 (V4 Middleware-First)。"""

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

        # 3) Physics → Middleware: 物理设备主动注册 Skills
        register_default_skills(
            registry=self._registry,
            fsm=self._fsm,
            bus=self._bus,
        )

        # 4) Cognition Layer
        llm = create_llm(self.config.llm)
        self._graph = build_lab_graph(
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

    # ---- 任务运行 --------------------------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        logger.info("=" * 64)
        logger.info("[%s] 🚀 实验柜环境启动 (V4 Middleware-First)", self.lab_id)
        logger.info("[%s] 📋 已注册 Skills: %s", self.lab_id, [s["name"] for s in self.available_skills])
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

            final_state = await self._graph.ainvoke(initial_state)
            result = final_state.get("final_result", {})
            all_results.append(result)

            logger.info("[%s] 📤 任务完成: %s", self.lab_id, result.get("status", "N/A"))

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
