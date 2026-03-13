"""
AstroSASF · Core · LaboratoryEnvironment (V4.3 — Headless)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境 —— 框架默认 Headless 全自动运行。

V4.3 变化：
- 删除 HITL（_hitl_loop / _get_human_approval / MemorySaver）
- 默认直接 ``graph.ainvoke()`` 全自动执行完所有步骤
- 暴露 ``build()`` 方法，供应用层获取未编译图自行注入 HITL
- FSM 由外部传入（通用规则引擎不含业务词汇）
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
from sasf.physics.shadow_fsm import ShadowFSM
from sasf.physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)

# 项目根目录
_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class LaboratoryEnvironment:
    """单个实验柜运行时环境 (V4.3 Headless)。

    Parameters
    ----------
    lab_id : str
    config : SASFConfig
    fsm : ShadowFSM
        由外部构造的通用 FSM 实例（框架不硬编码业务规则）。
    skills_catalog_dir : str | Path
        OpenAI Skills 知识目录。
    tool_registrar : callable, optional
        业务 MCP Tool 注册函数，签名 ``(registry, fsm, bus) → None``。
    """

    lab_id: str
    config: SASFConfig
    fsm: ShadowFSM
    skills_catalog_dir: str | Path = "skills_catalog"
    tool_registrar: Any = None
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

        # 2) Middleware Layer
        self._registry = MCPToolRegistry(lab_id=self.lab_id)
        self._a2a_router = A2ARouter(lab_id=self.lab_id)

        # 3) 业务层注册 MCP Tools
        if self.tool_registrar is not None:
            self.tool_registrar(self._registry, self.fsm, self._bus)

        # 4) 动态 Codec 协商
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
            fsm=self.fsm,
            bus=self._bus,
            codec=self._codec,
            space_wire=self._space_wire,
            registry=self._registry,
            a2a_router=self._a2a_router,
        )

        # 5) Cognition Layer
        catalog_path = Path(self.skills_catalog_dir)
        if not catalog_path.is_absolute():
            catalog_path = _ROOT / catalog_path

        self._skill_catalog = OpenAISkillCatalog(catalog_dir=catalog_path)

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
        """未编译的 StateGraph —— 供应用层自行注入 checkpointer / interrupt。"""
        return self._graph

    @property
    def gateway(self) -> SpaceMCPGateway:
        return self._gateway

    @property
    def a2a_router(self) -> A2ARouter:
        return self._a2a_router

    # ---- 状态查询 --------------------------------------------------------- #

    async def get_telemetry(self) -> dict[str, Any]:
        return await self._bus.snapshot()

    @property
    def fsm_state(self) -> str:
        return self.fsm.current_state

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
        """安全地收集所有统计信息。"""
        stats: dict[str, Any] = {"lab_id": self.lab_id}
        try:
            stats["fsm_state"] = self.fsm.current_state
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

    # ---- Headless 运行 --------------------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        """Headless 全自动运行 —— 无 HITL、无中断。"""
        logger.info("=" * 64)
        logger.info("[%s] 🚀 实验柜启动 (V4.3 Headless)", self.lab_id)
        logger.info(
            "[%s] 🔧 已注册 MCP Tools: %s",
            self.lab_id,
            [t["name"] for t in self.available_tools],
        )
        logger.info(
            "[%s] 📚 已加载 OpenAI Skills: %s",
            self.lab_id,
            [s["name"] for s in self.loaded_skills],
        )
        logger.info("=" * 64)

        # Headless 编译：无 checkpointer、无 interrupt
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

            logger.info(
                "[%s] 📤 任务完成: %s", self.lab_id,
                all_results[-1].get("status", "N/A"),
            )

        final_telemetry = await self._bus.snapshot()
        env_result = {
            "lab_id": self.lab_id,
            "fsm_state": self.fsm.current_state,
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
        logger.info("-" * 64)

        return env_result
