"""
AstroSASF · Cognition · SpecializedAgents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
两大核心智能体实现：

- **PlannerAgent** — 接收外部任务 → 生成分步 JSON 计划 → 发布至 Nexus
- **OperatorAgent** — 监听计划 → 调用 MCPGateway Skills → 反馈执行结果

Planner 与 Operator 通过 AgentNexus 完全解耦，可在同一个
LaboratoryEnvironment 内异步并发运行。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from cognition.agent_nexus import AgentNexus, Message, Topic
from cognition.base_agent import BaseAgent
from middleware.gateway import SpaceMCPGateway

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  PlannerAgent                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class PlannerAgent(BaseAgent):
    """规划智能体 —— 将自然语言任务拆解为分步执行计划。

    V1 使用 Mock LLM 返回预设的计划结构。
    """

    _task_queue: asyncio.Queue[str] = field(
        default_factory=asyncio.Queue, init=False, repr=False,
    )
    _feedback_queue: asyncio.Queue[Message] = field(
        default=None, init=False, repr=False,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        # 订阅执行反馈
        self._feedback_queue = self.nexus.subscribe(Topic.EXECUTION_FEEDBACK)

    # ---- 外部任务提交接口 -------------------------------------------------- #

    async def submit_task(self, task_description: str) -> None:
        """外部（Environment）向 Planner 提交任务。"""
        await self._task_queue.put(task_description)
        logger.info("[%s] 收到新任务: %s", self.agent_id, task_description)

    # ---- 主循环 ----------------------------------------------------------- #

    async def run(self) -> None:
        self._running = True
        logger.info("[%s] Planner 启动", self.agent_id)

        while self._running:
            try:
                # 等待新任务（带超时，方便优雅退出）
                task_desc = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            # 生成执行计划（Mock LLM）
            plan = await self._generate_plan(task_desc)
            logger.info("[%s] 生成计划: %d 个步骤", self.agent_id, len(plan))

            # 发布计划到 Nexus
            await self.nexus.publish(Message(
                topic=Topic.PLANNING_RESULT,
                sender=self.agent_id,
                payload={"task": task_desc, "steps": plan},
            ))

            # 等待并处理执行反馈（非阻塞轮询）
            await self._process_feedback()

        logger.info("[%s] Planner 退出", self.agent_id)

    # ---- 内部方法 --------------------------------------------------------- #

    async def _generate_plan(
        self, task_description: str,
    ) -> list[dict[str, Any]]:
        """模拟 LLM 规划 —— 根据任务描述返回预设的分步计划。"""
        plan_templates: dict[str, list[dict[str, Any]]] = {
            "heat_and_move": [
                {"skill": "set_temperature", "params": {"target": 50.0}},
                {"skill": "move_robotic_arm", "params": {"target_angle": 45.0}},
            ],
            "overheat_test": [
                {"skill": "set_temperature", "params": {"target": 90.0}},
                {"skill": "set_temperature", "params": {"target": 95.0}},
            ],
            "vacuum_sequence": [
                {"skill": "toggle_vacuum_pump", "params": {"activate": True}},
                {"skill": "move_robotic_arm", "params": {"target_angle": 90.0}},
                {"skill": "toggle_vacuum_pump", "params": {"activate": False}},
            ],
        }

        # 匹配模板 —— 用关键词做简单路由
        for key, steps in plan_templates.items():
            if key in task_description.lower():
                preset = steps
                break
        else:
            # 默认计划
            preset = [
                {"skill": "set_temperature", "params": {"target": 36.5}},
                {"skill": "move_robotic_arm", "params": {"target_angle": 30.0}},
            ]

        return await self._call_llm(  # type: ignore[return-value]
            prompt=f"将任务拆解为操作步骤: {task_description}",
            preset_response=preset,
        )

    async def _process_feedback(self) -> None:
        """消费所有待处理的执行反馈（非阻塞）。"""
        while not self._feedback_queue.empty():
            msg = await self._feedback_queue.get()
            logger.info(
                "[%s] 收到执行反馈: %s",
                self.agent_id, msg.payload,
            )


# --------------------------------------------------------------------------- #
#  OperatorAgent                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class OperatorAgent(BaseAgent):
    """操作智能体 —— 监听执行计划并调用 MCP Skills。

    处理 FSM 安全拦截异常，尝试生成修正方案。
    """

    gateway: SpaceMCPGateway = field(default=None)  # type: ignore[assignment]
    _plan_queue: asyncio.Queue[Message] = field(
        default=None, init=False, repr=False,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        # 订阅规划结果
        self._plan_queue = self.nexus.subscribe(Topic.PLANNING_RESULT)

    # ---- 主循环 ----------------------------------------------------------- #

    async def run(self) -> None:
        self._running = True
        logger.info("[%s] Operator 启动", self.agent_id)

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self._plan_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            plan_payload = msg.payload
            task_name = plan_payload.get("task", "unknown")
            steps: list[dict[str, Any]] = plan_payload.get("steps", [])

            logger.info(
                "[%s] 接收到计划「%s」, 共 %d 步",
                self.agent_id, task_name, len(steps),
            )

            results: list[dict[str, Any]] = []
            for idx, step in enumerate(steps, 1):
                result = await self._execute_step(idx, step)
                results.append(result)

            # 将执行结果反馈给 Planner
            await self.nexus.publish(Message(
                topic=Topic.EXECUTION_FEEDBACK,
                sender=self.agent_id,
                payload={
                    "task": task_name,
                    "results": results,
                },
            ))

        logger.info("[%s] Operator 退出", self.agent_id)

    # ---- 单步执行 --------------------------------------------------------- #

    async def _execute_step(
        self, step_index: int, step: dict[str, Any],
    ) -> dict[str, Any]:
        """执行计划中的单个步骤。"""
        skill_name: str = step.get("skill", "")
        params: dict[str, Any] = step.get("params", {})

        logger.info(
            "[%s] 执行步骤 %d: skill=%s params=%s",
            self.agent_id, step_index, skill_name, params,
        )

        # 通过 MCP Gateway 调用 Skill（内含 FSM 校验）
        result = await self.gateway.invoke_skill(skill_name, params)

        if result.get("status") == "error":
            logger.warning(
                "[%s] 步骤 %d 失败: %s — 尝试修正",
                self.agent_id, step_index, result.get("detail"),
            )
            # 尝试自动修正：请求 Mock LLM 生成替代方案
            correction = await self._attempt_correction(skill_name, params, result)
            result["correction_attempted"] = correction

        return {
            "step": step_index,
            "skill": skill_name,
            **result,
        }

    async def _attempt_correction(
        self,
        skill_name: str,
        original_params: dict[str, Any],
        error_result: dict[str, Any],
    ) -> dict[str, Any]:
        """当 Skill 执行失败时，尝试生成修正方案。"""
        correction = await self._call_llm(
            prompt=(
                f"Skill '{skill_name}' 执行失败: {error_result.get('detail')}. "
                f"原始参数: {original_params}. 请给出修正建议。"
            ),
            preset_response={
                "action": "skip_and_log",
                "reason": "FSM 安全约束，建议降低目标值或等待环境条件改善",
            },
        )
        logger.info(
            "[%s] 修正建议: %s", self.agent_id, correction,
        )
        return correction  # type: ignore[return-value]
