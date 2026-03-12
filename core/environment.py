"""
AstroSASF · Core · LaboratoryEnvironment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单个实验舱的上下文环境 —— 多系统并发隔离的基本单元。

V3.1: 支持 Human-in-the-loop (HITL) 审核机制。
图在 execute_node 前暂停，等待科学家确认后才执行硬件操作。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from cognition.graph_builder import build_lab_graph
from cognition.state import LabGraphState
from middleware.codec import SpaceMCPCodec
from middleware.gateway import SpaceMCPGateway
from middleware.virtual_bus import VirtualSpaceWire
from physics.shadow_fsm import ShadowFSM
from physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Default Config                                                              #
# --------------------------------------------------------------------------- #

_DEFAULT_MODEL: str = "qwen2.5:7b"
_DEFAULT_BASE_URL: str = "http://localhost:11434"


# --------------------------------------------------------------------------- #
#  LaboratoryEnvironment                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class LaboratoryEnvironment:
    """单个实验柜的完整运行时环境。

    V3.1 新增 Human-in-the-loop：图在 ``execute_node`` 前中断，
    在控制台打印即将执行的 Skill，等待科学家输入确认。
    """

    lab_id: str
    model: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_BASE_URL

    # ---- 内部组件 --------------------------------------------------------- #
    _bus: TelemetryBus = field(init=False, repr=False)
    _fsm: ShadowFSM = field(init=False, repr=False)
    _codec: SpaceMCPCodec = field(init=False, repr=False)
    _space_wire: VirtualSpaceWire = field(init=False, repr=False)
    _gateway: SpaceMCPGateway = field(init=False, repr=False)
    _llm: ChatOllama = field(init=False, repr=False)
    _graph: Any = field(init=False, repr=False)
    _checkpointer: MemorySaver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # 1) Physics Layer
        self._bus = TelemetryBus(lab_id=self.lab_id)
        self._fsm = ShadowFSM(lab_id=self.lab_id)

        # 2) Middleware Layer
        self._codec = SpaceMCPCodec(lab_id=self.lab_id)
        self._space_wire = VirtualSpaceWire(lab_id=self.lab_id)
        self._gateway = SpaceMCPGateway(
            lab_id=self.lab_id,
            fsm=self._fsm,
            bus=self._bus,
            codec=self._codec,
            space_wire=self._space_wire,
        )

        # 3) Cognition Layer (LangGraph + HITL)
        self._llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.1,
        )
        self._graph, self._checkpointer = build_lab_graph(
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

    # ---- Human-in-the-loop 任务运行 ---------------------------------------- #

    async def run(self, tasks: list[str]) -> dict[str, Any]:
        """启动 LangGraph 工作流，在 execute_node 前暂停等待人工审核。"""
        logger.info("=" * 64)
        logger.info(
            "[%s] 🚀 实验柜环境启动 (V3.1 LangGraph + HITL)", self.lab_id,
        )
        logger.info("=" * 64)

        all_results: list[dict[str, Any]] = []

        for task_idx, task_desc in enumerate(tasks):
            logger.info("")
            logger.info("[%s] 📥 提交任务 %d: %s", self.lab_id, task_idx + 1, task_desc)

            # 每个任务使用独立的 thread_id
            thread_id = f"{self.lab_id}::task_{task_idx}"
            config = {"configurable": {"thread_id": thread_id}}

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

            # 首次调用 → 图运行到 interrupt_before execute_node 时暂停
            state = await self._graph.ainvoke(initial_state, config)

            # HITL 循环：检查是否中断在 execute_node 前
            while True:
                graph_state = self._graph.get_state(config)
                next_nodes = graph_state.next

                if not next_nodes:
                    # 图已结束
                    state = graph_state.values
                    break

                if "execute_node" in next_nodes:
                    # 图暂停在 execute_node 前 → 等待人工审核
                    current_state = graph_state.values
                    step = current_state.get("current_step")

                    if step and isinstance(step, dict):
                        skill_name = step.get("skill", "unknown")
                        params = step.get("params", {})

                        # 打印审核信息
                        print("")
                        print("┌" + "─" * 58 + "┐")
                        print(f"│  🛡️  Human-in-the-loop 审核 — {self.lab_id}")
                        print("├" + "─" * 58 + "┤")
                        print(f"│  即将执行 Skill: {skill_name}")
                        print(f"│  参数: {json.dumps(params, ensure_ascii=False)}")
                        print("├" + "─" * 58 + "┤")
                        print("│  [y] 确认执行")
                        print("│  [n] 终止任务")
                        print("│  [输入JSON] 覆盖参数，例如: {\"target\": 40.0}")
                        print("└" + "─" * 58 + "┘")

                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, input, ">>> 请输入: ",
                        )
                        user_input = user_input.strip()

                        if user_input.lower() == "n":
                            logger.info("[%s] ⏹️ 用户终止任务", self.lab_id)
                            state = current_state
                            state["final_result"] = {
                                "status": "aborted_by_user",
                                "total_steps": len(current_state.get("plan", [])),
                                "execution_log": current_state.get("execution_log", []),
                            }
                            break

                        if user_input.lower() != "y" and user_input.startswith("{"):
                            # 用户提供了修正参数
                            try:
                                override = json.loads(user_input)
                                updated_step = {**step, "params": override}
                                logger.info(
                                    "[%s] 🔧 用户覆盖参数: %s",
                                    self.lab_id, override,
                                )
                                # 更新图状态中的当前步骤
                                self._graph.update_state(
                                    config,
                                    {"current_step": updated_step},
                                )
                            except json.JSONDecodeError:
                                logger.warning(
                                    "[%s] 参数解析失败，使用原始参数继续",
                                    self.lab_id,
                                )

                    # 继续执行（传入 None → 从中断点恢复）
                    state = await self._graph.ainvoke(None, config)
                else:
                    # 在其他节点中断（不应发生），继续
                    state = await self._graph.ainvoke(None, config)

            result = state.get("final_result", {}) if isinstance(state, dict) else {}
            all_results.append(result)

            logger.info(
                "[%s] 📤 任务「%s」完成: %s",
                self.lab_id,
                task_desc,
                result.get("status", "N/A") if isinstance(result, dict) else "N/A",
            )

        # 环境结果
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
