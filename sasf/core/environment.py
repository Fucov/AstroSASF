"""
AstroSASF · Core · LaboratoryEnvironment (V4.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境 —— MCP Tools + OpenAI Skills 解耦架构。

V4.2 装配流程:
1. Physics Layer — 创建 FSM + TelemetryBus
2. Middleware Layer — 创建 MCPToolRegistry + A2ARouter
3. Physics → Middleware — 物理设备通过 @mcp_tool 注册 MCP Tools
4. 动态 Codec 协商 — 从 Registry 获取词汇表 → 构建 SpaceMCPCodec
5. Cognition Layer — 加载 SKILL.md 知识 + config_loader 创建 LLM → 构建 LangGraph
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from sasf.cognition.graph_builder import build_lab_graph
from sasf.cognition.skill_loader import OpenAISkillCatalog
from sasf.cognition.state import LabGraphState
from sasf.core.config_loader import SASFConfig, create_llm
from sasf.middleware.a2a_protocol import A2ARouter
from sasf.middleware.codec import SpaceMCPCodec
from sasf.middleware.gateway import SpaceMCPGateway
from sasf.middleware.mcp_registry import MCPToolRegistry
from sasf.middleware.virtual_bus import VirtualSpaceWire
from sasf.physics.shadow_fsm import ShadowFSM, register_default_tools
from sasf.physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)


@dataclass
class LaboratoryEnvironment:
    """单个实验柜运行时环境 (V4.2 MCP Tools + OpenAI Skills)。"""

    lab_id: str
    config: SASFConfig
    skills_catalog_dir: str | Path = "skills_catalog"

    # ---- 内部组件 --------------------------------------------------------- #
    _bus: TelemetryBus = field(init=False, repr=False)
    _fsm: ShadowFSM = field(init=False, repr=False)
    _registry: MCPToolRegistry = field(init=False, repr=False)
    _a2a_router: A2ARouter = field(init=False, repr=False)
    _codec: SpaceMCPCodec = field(init=False, repr=False)
    _space_wire: VirtualSpaceWire = field(init=False, repr=False)
    _gateway: SpaceMCPGateway = field(init=False, repr=False)
    _skill_catalog: OpenAISkillCatalog = field(init=False, repr=False)
    _graph: Any = field(init=False, repr=False)
    _memory: MemorySaver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mw_cfg = self.config.middleware

        # 1) Physics Layer
        self._bus = TelemetryBus(lab_id=self.lab_id)
        self._fsm = ShadowFSM(lab_id=self.lab_id)

        # 2) Middleware Layer — MCPToolRegistry
        self._registry = MCPToolRegistry(lab_id=self.lab_id)
        self._a2a_router = A2ARouter(lab_id=self.lab_id)

        # 3) Physics → Middleware: 通过 @mcp_tool 注册 MCP Tools
        register_default_tools(
            registry=self._registry,
            fsm=self._fsm,
            bus=self._bus,
        )

        # 4) 动态 Codec 协商 — 从 Registry 自动获取词汇表
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
            fsm=self._fsm,
            bus=self._bus,
            codec=self._codec,
            space_wire=self._space_wire,
            registry=self._registry,
            a2a_router=self._a2a_router,
        )

        # 5) Cognition Layer — Load Skills + Build Graph
        self._skill_catalog = OpenAISkillCatalog(
            catalog_dir=self.skills_catalog_dir,
        )

        llm = create_llm(self.config.llm)
        self._graph, self._memory = build_lab_graph(
            gateway=self._gateway,
            llm=llm,
            lab_id=self.lab_id,
            a2a_router=self._a2a_router,
            skill_catalog=self._skill_catalog,
        )

    # ---- 状态查询 --------------------------------------------------------- #

    async def get_telemetry(self) -> dict[str, Any]:
        return await self._bus.snapshot()

    @property
    def fsm_state(self) -> str:
        return self._fsm.current_state.name

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
        """安全地收集所有统计信息 —— 即使环境崩溃也可调用。"""
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

    # ---- 任务运行 (HITL) ------------------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        logger.info("=" * 64)
        logger.info("[%s] 🚀 实验柜启动 (V4.2 MCP Tools + OpenAI Skills)", self.lab_id)
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

            thread_config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            result = await self._hitl_loop(initial_state, thread_config)
            all_results.append(result)

            logger.info(
                "[%s] 📤 任务完成: %s",
                self.lab_id,
                result.get("status", "N/A"),
            )

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
        state = await self._graph.ainvoke(initial_state, config)
        snapshot = self._graph.get_state(config)

        while snapshot.next:
            step = state.get("current_step")
            if step and isinstance(step, dict):
                logger.info("")
                logger.info(
                    "[%s] ⏸️  HITL 中断 — 即将执行:", self.lab_id,
                )
                logger.info(
                    "[%s]    Tool  : %s", self.lab_id, step.get("skill"),
                )
                logger.info(
                    "[%s]    Params: %s",
                    self.lab_id,
                    json.dumps(step.get("params", {}), ensure_ascii=False),
                )

            user_input = await self._get_human_approval(step)

            if user_input == "abort":
                logger.warning("[%s] ❌ 用户中止执行", self.lab_id)
                return {
                    "status": "aborted_by_user",
                    "total_steps": len(state.get("plan", [])),
                    "execution_log": list(state.get("execution_log") or []),
                }

            if user_input == "approve":
                state = await self._graph.ainvoke(None, config)
            else:
                try:
                    corrected = json.loads(user_input)
                    if isinstance(corrected, dict):
                        self._graph.update_state(
                            config, {"current_step": corrected},
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
        import asyncio

        if step:
            skill = step.get("skill", "unknown")
            params = json.dumps(step.get("params", {}), ensure_ascii=False)
            prompt = (
                f"\n{'─' * 60}\n"
                f"🛡️ HITL | 即将执行 MCP Tool: {skill}({params})\n"
                f"  [y/回车] 批准  |  [n] 中止  |  [JSON] 修正参数\n"
                f"{'─' * 60}\n"
                f">>> "
            )
        else:
            prompt = "\n>>> 继续? [y/n]: "

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: input(prompt))

        raw = raw.strip()
        if raw == "" or raw.lower() in ("y", "yes"):
            return "approve"
        if raw.lower() in ("n", "no"):
            return "abort"
        return raw
