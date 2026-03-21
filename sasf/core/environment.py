"""
AstroSASF · Core · LaboratoryEnvironment (V5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境。

V5 变化：
- ``ShadowFSM`` → ``InterlockEngine``
- ``tool_registrar`` 签名改为 ``(registry, engine, bus) → None``
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sasf.cognition.graph_builder import build_lab_graph
from sasf.cognition.skill_loader import OpenAISkillCatalog
from sasf.cognition.state import LabGraphState
from sasf.core.config_loader import SASFConfig, create_llm
from sasf.middleware.a2a_protocol import A2ARouter
from sasf.middleware.codec import SpaceMCPCodec
from sasf.middleware.gateway import SpaceMCPGateway
from sasf.middleware.mcp_registry import MCPToolRegistry
from sasf.middleware.virtual_bus import VirtualSpaceWire
from sasf.physics.interlock_engine import InterlockEngine
from sasf.physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class LaboratoryEnvironment:
    """单个实验柜运行时环境 (V5)。

    Parameters
    ----------
    lab_id : str
    config : SASFConfig
    engine : InterlockEngine
        正交联锁引擎（外部构造）。
    skills_catalog_dir : str | Path
    tool_registrar : callable, optional
        业务 MCP Tool 注册函数，签名 ``(registry, engine, bus) → None``。
    macro_registrar : callable, optional
        Macro 绑定函数，签名 ``(registry) → None``。
    initial_telemetry : dict
    """

    lab_id: str
    config: SASFConfig
    engine: InterlockEngine
    skills_catalog_dir: str | Path = "skills_catalog"
    tool_registrar: Any = None
    macro_registrar: Any = None
    initial_telemetry: dict[str, Any] = field(default_factory=dict)

    # ---- 内部组件 --------------------------------------------------------- #
    _bus: TelemetryBus = field(init=False, repr=False)
    _registry: MCPToolRegistry = field(init=False, repr=False)
    _a2a_router: A2ARouter = field(init=False, repr=False)
    _codec: SpaceMCPCodec = field(init=False, repr=False)
    _space_wire: VirtualSpaceWire = field(init=False, repr=False)
    _gateway: SpaceMCPGateway = field(init=False, repr=False)
    _skill_catalog: OpenAISkillCatalog = field(init=False, repr=False)
    _graph: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mw_cfg = self.config.middleware

        # 1) Physics Layer
        self._bus = TelemetryBus(
            lab_id=self.lab_id,
            initial_state=self.initial_telemetry,
        )
        # 绑定遥测总线到联锁引擎（使 set_subsystem_state 自动获取遥测）
        self.engine.bind_telemetry_bus(self._bus)

        # 2) Middleware Layer
        self._registry = MCPToolRegistry(lab_id=self.lab_id)
        self._a2a_router = A2ARouter(lab_id=self.lab_id)

        # 3) 业务层注册 MCP Tools
        if self.tool_registrar is not None:
            self.tool_registrar(self._registry, self.engine, self._bus)

        # 4) 业务层绑定 Macros（需在 Tool 注册后）
        if self.macro_registrar is not None:
            self.macro_registrar(self._registry)

        # 5) 动态 Codec 协商（Macro 名自动包含）
        vocabulary = self._registry.all_vocabulary()
        self._codec = SpaceMCPCodec(
            lab_id=self.lab_id,
            vocabulary=vocabulary,
        )
        self._space_wire = VirtualSpaceWire(
            lab_id=self.lab_id,
            bandwidth_kbps=mw_cfg.spacewire_bandwidth_kbps,
        )
        self._gateway = SpaceMCPGateway(
            lab_id=self.lab_id,
            engine=self.engine,
            bus=self._bus,
            codec=self._codec,
            space_wire=self._space_wire,
            registry=self._registry,
            a2a_router=self._a2a_router,
        )

        # 6) Cognition Layer
        catalog_path = Path(self.skills_catalog_dir)
        if not catalog_path.is_absolute():
            catalog_path = _ROOT / catalog_path

        self._skill_catalog = OpenAISkillCatalog(
            catalog_dir=catalog_path,
            registry=self._registry,
        )

        llm = create_llm(self.config.llm)
        self._graph = build_lab_graph(
            gateway=self._gateway,
            llm=llm,
            lab_id=self.lab_id,
            a2a_router=self._a2a_router,
            skill_catalog=self._skill_catalog,
        )

    # ---- 暴露给应用层 ----------------------------------------------------- #

    @property
    def graph(self) -> Any:
        return self._graph

    @property
    def gateway(self) -> SpaceMCPGateway:
        return self._gateway

    @property
    def a2a_router(self) -> A2ARouter:
        return self._a2a_router

    @property
    def registry(self) -> MCPToolRegistry:
        return self._registry

    # ---- 状态查询 --------------------------------------------------------- #

    async def get_telemetry(self) -> dict[str, Any]:
        return await self._bus.snapshot()

    @property
    def engine_states(self) -> dict[str, str]:
        return self.engine.current_states

    @property
    def available_tools(self) -> list[dict[str, Any]]:
        return self._gateway.list_tools()

    @property
    def codec_stats(self) -> dict[str, Any]:
        return self._codec.stats

    @property
    def codec_dictionary(self) -> dict[str, int]:
        return self._codec.dictionary_table

    @property
    def bus_stats(self) -> dict[str, Any]:
        return self._space_wire.stats

    @property
    def a2a_stats(self) -> dict[str, Any]:
        return self._a2a_router.stats

    @property
    def loaded_skills(self) -> list[dict[str, str]]:
        return self._skill_catalog.list_skills()

    def collect_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {"lab_id": self.lab_id}
        try:
            stats["engine_states"] = self.engine.current_states
        except Exception:
            stats["engine_states"] = {}
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

    # ---- Headless 运行 --------------------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        """Headless 全自动运行。"""
        logger.info("=" * 64)
        logger.info("[%s] 🚀 实验柜启动 (V5 Headless)", self.lab_id)
        logger.info(
            "[%s] 🔧 已注册 Tools: %s (含 %d 个 Macro)",
            self.lab_id,
            [t["name"] for t in self.available_tools],
            self._registry.macro_count,
        )
        logger.info("=" * 64)

        compiled = self._graph.compile()

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

            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            state = await compiled.ainvoke(initial_state, config)

            final = state.get("final_result")
            if final and isinstance(final, dict):
                all_results.append(final)
            else:
                all_results.append({
                    "status": "completed",
                    "total_steps": len(state.get("plan", [])),
                    "execution_log": list(state.get("execution_log") or []),
                })

        final_telemetry = await self._bus.snapshot()
        return {
            "lab_id": self.lab_id,
            "engine_states": self.engine.current_states,
            "final_telemetry": final_telemetry,
            "codec_stats": self._codec.stats,
            "bus_stats": self._space_wire.stats,
            "a2a_stats": self._a2a_router.stats,
            "task_results": all_results,
        }
