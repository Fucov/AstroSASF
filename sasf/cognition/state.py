"""
AstroSASF · Cognition · State (V7.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态定义。

V7.1 变化：
- 新增 ``planner_llm_calls`` 字段（理论智能体调用大模型的次数）
- 新增 ``planner_time_ms`` 字段（生成 DAG 的纯规划耗时）

V7.0 变化：
- 新增 ``dag_graph`` 字段（dag_planner_node 生成）
- 新增 ``dag_error`` 字段（DAG 生成/验证失败说明）
"""

from __future__ import annotations

from typing import Any, TypedDict

from sasf.core.models import DAGTaskGraph


class SkillStep(TypedDict):
    """单个 Skill 调用步骤。"""
    id: str
    skill: str
    params: dict[str, Any]
    depends_on: list[str]


class ExecutionLogEntry(TypedDict, total=False):
    """执行日志条目。"""
    step_index: int
    skill: str
    params: dict[str, Any]
    result: dict[str, Any]
    status: str
    correction: str | None


class LabGraphState(TypedDict, total=False):
    """LangGraph 状态图的完整状态 (V7.1)。

    V7.1 新增埋点字段：
    - planner_llm_calls: 理论智能体调用大模型的次数
    - planner_time_ms: 生成 DAG 的纯规划耗时（毫秒）
    """
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
    # V7.0: DAG 模式
    dag_graph: DAGTaskGraph | None
    dag_error: str | None
    # V7.1: LLM 算力与耗时埋点
    planner_llm_calls: int            # 理论智能体调用大模型的次数
    planner_time_ms: float            # 生成 DAG 的纯规划耗时（毫秒）
