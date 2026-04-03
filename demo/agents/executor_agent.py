"""
AstroSASF · Demo · Agents · Executor Agent
==========================================
执行 Agent：负责按计划顺序调用 MCP Tools，汇报执行结果。
Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from demo.agents.base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStepResult:
    """单步执行结果。"""
    step: int
    tool_name: str
    params: dict[str, Any]
    status: str
    detail: str | None = None
    execution_time_ms: float = 0.0
    fsm_states: dict[str, str] | None = None


@dataclass
class ExecutionReport:
    """完整执行报告。"""
    plan_id: str
    lab_id: str
    mission: str
    total_steps: int
    success_count: int
    failure_count: int
    results: list[ExecutionStepResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    final_fsm_states: dict[str, str] | None = None

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            f"  执行报告 [{self.plan_id}]",
            "=" * 60,
            f"  实验舱: {self.lab_id}",
            f"  任务: {self.mission}",
            f"  成功率: {self.success_count}/{self.total_steps}",
            f"  总耗时: {self.total_time_ms:.1f} ms",
            "-" * 60,
        ]
        for r in self.results:
            icon = "✅" if r.status == "ok" else "❌" if r.status == "error" else "⚠️"
            lines.append(
                f"  {icon} Step {r.step}: {r.tool_name} "
                f"({r.params}) → {r.status} ({r.execution_time_ms:.1f}ms)"
            )
            if r.detail and r.status != "ok":
                lines.append(f"       详情: {r.detail}")
        if self.final_fsm_states:
            lines.append("-" * 60)
            lines.append("  最终 FSM 状态:")
            for subsystem, state in self.final_fsm_states.items():
                lines.append(f"    {subsystem}: {state}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ExecutorAgent(BaseAgent):
    """执行 Agent (V7.2)。

    接收来自 Planner Agent 的执行计划，
    按顺序调用 MCP Tools，生成执行报告，
    并将结果汇报给 Q&A Agent。
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        super().__init__(agent_id="Executor-Agent", base_url=base_url)

    async def think(self, message: AgentMessage | None) -> AgentMessage | None:
        """接收执行计划，运行工具调用。"""
        if message is None:
            return None

        if message.metadata.get("type") != "execution_request":
            return None

        lab_id = message.metadata["lab_id"]
        plan_id = message.metadata["plan_id"]
        steps = message.metadata["steps"]
        priority = message.metadata.get("priority", 2)
        mission = message.metadata.get("original_task", "unknown")

        logger.info(
            "[Executor] 开始执行计划 [%s]，共 %d 步",
            plan_id, len(steps),
        )

        t0 = time.monotonic()
        results: list[ExecutionStepResult] = []
        success_count = 0
        failure_count = 0
        final_fsm_states: dict[str, str] | None = None

        for i, step_def in enumerate(steps):
            step_num = i + 1
            tool_name = step_def.get("tool", "unknown")
            params = step_def.get("params", {})

            step_t0 = time.monotonic()

            try:
                resp = await self.execute_tool(
                    lab_id=lab_id,
                    tool_name=tool_name,
                    params=params,
                    priority=priority,
                )
                step_elapsed = (time.monotonic() - step_t0) * 1000
                status = resp.get("status", "ok")
                detail = resp.get("detail")
                fsm_states = resp.get("fsm_states")

                if status in ("ok", "success"):
                    success_count += 1
                else:
                    failure_count += 1

                result = ExecutionStepResult(
                    step=step_num,
                    tool_name=tool_name,
                    params=params,
                    status=status,
                    detail=detail,
                    execution_time_ms=step_elapsed,
                    fsm_states=fsm_states,
                )
                results.append(result)
                final_fsm_states = fsm_states

                logger.info(
                    "[Executor] Step %d/%d: %s → %s (%.1fms)",
                    step_num, len(steps), tool_name, status, step_elapsed,
                )

            except Exception as exc:
                step_elapsed = (time.monotonic() - step_t0) * 1000
                failure_count += 1
                result = ExecutionStepResult(
                    step=step_num,
                    tool_name=tool_name,
                    params=params,
                    status="error",
                    detail=str(exc),
                    execution_time_ms=step_elapsed,
                )
                results.append(result)
                logger.warning(
                    "[Executor] Step %d/%d: %s → ERROR: %s",
                    step_num, len(steps), tool_name, exc,
                )

        total_time_ms = (time.monotonic() - t0) * 1000

        report = ExecutionReport(
            plan_id=plan_id,
            lab_id=lab_id,
            mission=mission,
            total_steps=len(steps),
            success_count=success_count,
            failure_count=failure_count,
            results=results,
            total_time_ms=total_time_ms,
            final_fsm_states=final_fsm_states,
        )

        logger.info(
            "[Executor] ✅ 执行完成 [%s]: %d/%d 成功 (%.1fms)",
            plan_id, success_count, len(steps), total_time_ms,
        )

        return AgentMessage(
            sender=self.agent_id,
            content=report.to_text(),
            metadata={
                "type": "execution_complete",
                "plan_id": plan_id,
                "lab_id": lab_id,
                "success_count": success_count,
                "failure_count": failure_count,
                "total_time_ms": total_time_ms,
                "report_obj": report,
            },
        )
