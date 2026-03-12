"""
AstroSASF · Core · LaboratoryEnvironment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境 —— 多系统并发隔离的基本单元。

V3: 认知层升级为 LangGraph StateGraph + ChatOllama，
移除旧的 AgentNexus / PlannerAgent / OperatorAgent。
Middleware 和 Physics 层保持不变。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_ollama import ChatOllama

from cognition.graph_builder import build_lab_graph
from cognition.state import LabGraphState
from middleware.codec import SpaceMCPCodec
from middleware.gateway import SpaceMCPGateway
from middleware.virtual_bus import VirtualSpaceWire
from physics.shadow_fsm import ShadowFSM
from physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Default LLM Config                                                          #
# --------------------------------------------------------------------------- #

_DEFAULT_MODEL: str = "qwen2.5:7b"
_DEFAULT_BASE_URL: str = "http://localhost:11434"


# --------------------------------------------------------------------------- #
#  LaboratoryEnvironment                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class LaboratoryEnvironment:
    """单个实验柜的完整运行时环境。

    V3 架构变更：
    - 认知层：LangGraph StateGraph (planner→operator→execute 循环)
    - LLM：ChatOllama（本地 Ollama 服务）
    - Middleware / Physics：复用 V2 Space-MCP 全链路

    每个实例拥有独立的 LLM、编译后的 Graph、Gateway，实现多系统并发隔离。
    """

    lab_id: str
    model: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_BASE_URL

    # ---- 内部组件（延迟初始化） --------------------------------------------- #
    _bus: TelemetryBus = field(init=False, repr=False)
    _fsm: ShadowFSM = field(init=False, repr=False)
    _codec: SpaceMCPCodec = field(init=False, repr=False)
    _space_wire: VirtualSpaceWire = field(init=False, repr=False)
    _gateway: SpaceMCPGateway = field(init=False, repr=False)
    _llm: ChatOllama = field(init=False, repr=False)
    _graph: Any = field(init=False, repr=False)  # CompiledGraph

    def __post_init__(self) -> None:
        # 1) Physics Layer
        self._bus = TelemetryBus(lab_id=self.lab_id)
        self._fsm = ShadowFSM(lab_id=self.lab_id)

        # 2) Middleware Layer (V2 Space-MCP 三件套)
        self._codec = SpaceMCPCodec(lab_id=self.lab_id)
        self._space_wire = VirtualSpaceWire(lab_id=self.lab_id)
        self._gateway = SpaceMCPGateway(
            lab_id=self.lab_id,
            fsm=self._fsm,
            bus=self._bus,
            codec=self._codec,
            space_wire=self._space_wire,
        )

        # 3) Cognition Layer (V3 LangGraph)
        self._llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.1,  # 低温度 → 更确定性的输出
        )
        self._graph = build_lab_graph(
            gateway=self._gateway,
            llm=self._llm,
            lab_id=self.lab_id,
        )

    # ---- 状态查询 --------------------------------------------------------- #

    async def get_telemetry(self) -> dict[str, Any]:
        return await self._bus.snapshot()

    @property
    def fsm_state(self) -> str:
        return self._fsm.current_state.name

    @property
    def available_skills(self) -> list[dict[str, str]]:
        return self._gateway.list_skills()

    @property
    def codec_stats(self) -> dict[str, Any]:
        return self._codec.stats

    @property
    def bus_stats(self) -> dict[str, Any]:
        return self._space_wire.stats

    # ---- 任务运行 --------------------------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        """启动 LangGraph 工作流处理任务序列。"""
        logger.info("=" * 64)
        logger.info(
            "[%s] 🚀 实验柜环境启动 (V3 LangGraph + %s)", self.lab_id, self.model,
        )
        logger.info("=" * 64)

        all_results: list[dict[str, Any]] = []

        for task_desc in tasks:
            logger.info("")
            logger.info("[%s] 📥 提交任务: %s", self.lab_id, task_desc)

            # 构造初始状态
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

            # 调用编译后的 LangGraph
            final_state = await self._graph.ainvoke(initial_state)

            result = final_state.get("final_result", {})
            all_results.append(result)

            logger.info(
                "[%s] 📤 任务「%s」完成: %s",
                self.lab_id, task_desc, result.get("status", "N/A"),
            )

        # 收集环境结果
        final_telemetry = await self._bus.snapshot()
        env_result = {
            "lab_id": self.lab_id,
            "fsm_state": self._fsm.current_state.name,
            "final_telemetry": final_telemetry,
            "codec_stats": self._codec.stats,
            "bus_stats": self._space_wire.stats,
            "task_results": all_results,
        }

        logger.info("")
        logger.info("-" * 64)
        logger.info("[%s] ✅ 实验柜环境关闭  FSM=%s", self.lab_id, env_result["fsm_state"])
        logger.info("[%s] 最终遥测: %s", self.lab_id, final_telemetry)
        logger.info("-" * 64)

        return env_result
