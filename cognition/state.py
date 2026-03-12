"""
AstroSASF · Cognition · State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态定义 —— 定义状态图中流转的全部上下文信息。

使用 ``TypedDict`` 作为 LangGraph StateGraph 的状态模式，
包含任务描述、执行计划、当前步骤索引、FSM 反馈和最终结果。
"""

from __future__ import annotations

from typing import Any, TypedDict


# --------------------------------------------------------------------------- #
#  Step Schema                                                                 #
# --------------------------------------------------------------------------- #

class SkillStep(TypedDict):
    """单个 Skill 调用步骤。"""
    skill: str
    params: dict[str, Any]


# --------------------------------------------------------------------------- #
#  Execution Log Entry                                                         #
# --------------------------------------------------------------------------- #

class ExecutionLogEntry(TypedDict, total=False):
    """执行日志条目。"""
    step_index: int
    skill: str
    params: dict[str, Any]
    result: dict[str, Any]
    status: str            # "success" | "error" | "corrected"
    correction: str | None


# --------------------------------------------------------------------------- #
#  LangGraph State                                                             #
# --------------------------------------------------------------------------- #

class LabGraphState(TypedDict, total=False):
    """LangGraph 状态图的完整状态。

    Attributes
    ----------
    original_task : str
        用户提交的原始任务描述。
    plan : list[SkillStep]
        Planner LLM 生成的分步执行计划。
    current_step_index : int
        当前正在执行的步骤索引（0-based）。
    current_step : SkillStep | None
        当前提取出的待执行步骤。
    fsm_feedback : dict | None
        最近一次 execute_node 返回的 FSM / Gateway 反馈。
    execution_log : list[ExecutionLogEntry]
        完整的执行历史日志。
    error_count : int
        当前步骤的连续错误次数（用于限制重试）。
    final_result : dict | None
        最终汇总结果。
    """
    original_task: str
    plan: list[SkillStep]
    current_step_index: int
    current_step: SkillStep | None
    fsm_feedback: dict[str, Any] | None
    execution_log: list[ExecutionLogEntry]
    error_count: int
    final_result: dict[str, Any] | None
