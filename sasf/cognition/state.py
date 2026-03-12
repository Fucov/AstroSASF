"""
AstroSASF · Cognition · State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态定义。
"""

from __future__ import annotations

from typing import Any, TypedDict


class SkillStep(TypedDict):
    """单个 Skill 调用步骤。"""
    skill: str
    params: dict[str, Any]


class ExecutionLogEntry(TypedDict, total=False):
    """执行日志条目。"""
    step_index: int
    skill: str
    params: dict[str, Any]
    result: dict[str, Any]
    status: str
    correction: str | None


class LabGraphState(TypedDict, total=False):
    """LangGraph 状态图的完整状态。"""
    original_task: str
    plan: list[SkillStep]
    current_step_index: int
    current_step: SkillStep | None
    fsm_feedback: dict[str, Any] | None
    execution_log: list[ExecutionLogEntry]
    error_count: int
    final_result: dict[str, Any] | None
