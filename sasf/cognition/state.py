"""
AstroSASF · Cognition · State (V6.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态定义。

V6.2 变化：
- 新增 ``selected_skill`` 字段（router_node 写入，planner_node 读取）
- 新增 ``error_msg`` 字段（LLM 拒绝/无法解析时存储自然语言说明）
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
    """LangGraph 状态图的完整状态 (V6.2)。"""
    original_task: str
    selected_skill: str | None       # V6.2: router_node 选中的 SOP 名称
    plan: list[SkillStep]
    current_step_index: int
    current_step: SkillStep | None
    fsm_feedback: dict[str, Any] | None
    execution_log: list[ExecutionLogEntry]
    error_count: int
    error_msg: str | None             # V6.2: LLM 拒绝/解析失败时的说明
    final_result: dict[str, Any] | None
