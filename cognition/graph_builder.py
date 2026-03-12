"""
AstroSASF · Cognition · GraphBuilder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于 LangGraph 的状态图工作流构建器。

将 Planner / Operator / Executor 三个职责抽象为状态图节点，
通过条件边实现带循环的闭环执行流。

状态图拓扑
~~~~~~~~~~
::

    ┌─────────┐     ┌───────────────┐     ┌──────────────┐
    │ START   │────▶│ planner_node  │────▶│ operator_node│
    └─────────┘     └───────────────┘     └──────┬───────┘
                                                  │
                                    ┌─────────────┼─────────────┐
                                    │ has_step    │ done        │
                                    ▼             ▼             │
                              ┌──────────┐   ┌───────┐         │
                              │ execute  │   │  END  │         │
                              │ _node    │   └───────┘         │
                              └────┬─────┘                     │
                                   │                           │
                                   └───────────────────────────┘
                                   (回到 operator_node)

每个 LaboratoryEnvironment 实例调用 ``build_lab_graph`` 时，
通过闭包将该环境独有的 ``SpaceMCPGateway`` 和 ``LLM`` 绑定到
节点函数中，实现多系统并发隔离。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from cognition.state import LabGraphState, SkillStep
from middleware.gateway import SpaceMCPGateway

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

_MAX_RETRIES_PER_STEP: int = 2  # 单步最大重试次数

# 可用 Skills 的描述（注入 Planner prompt）
_SKILLS_DESCRIPTION = """
可用的 Skills（工具）清单：
1. set_temperature  — 设置舱内温度，参数: {"target": <float, ℃>}
2. move_robotic_arm — 移动机械臂，参数: {"target_angle": <float, °>}
3. toggle_vacuum_pump — 切换真空泵，参数: {"activate": <bool>}
""".strip()


# --------------------------------------------------------------------------- #
#  Prompt Templates                                                            #
# --------------------------------------------------------------------------- #

_PLANNER_PROMPT = """你是太空实验柜的规划智能体 (Planner)。
你的任务是将用户的自然语言指令拆解为一系列可执行的 Skill 调用步骤。

{skills_description}

请严格按以下 JSON 格式输出计划（不要添加任何其他文字解释）：
[
  {{"skill": "<skill_name>", "params": {{...}} }},
  ...
]

用户指令: {task}"""

_CORRECTION_PROMPT = """你是太空实验柜的操作智能体 (Operator)。
上一步 Skill 执行失败，FSM 安全护栏返回了错误。

失败的 Skill: {skill_name}
原始参数: {original_params}
错误信息: {error_detail}

请分析错误原因，给出修正后的参数。严格仅输出修正后的 JSON 对象（不要添加任何其他解释）：
{{"skill": "<skill_name>", "params": {{...}} }}

如果该步骤无法修正（例如物理约束无法绕过），请输出：
{{"skill": "skip", "params": {{}}, "reason": "<原因>"}}"""


# --------------------------------------------------------------------------- #
#  JSON 提取工具                                                                #
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> Any:
    """从 LLM 输出中提取 JSON —— 容忍 markdown 代码块包裹。"""
    # 尝试 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个 [ ... ] 或 { ... }
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:200]}")


# --------------------------------------------------------------------------- #
#  Graph Builder                                                               #
# --------------------------------------------------------------------------- #

def build_lab_graph(
    gateway: SpaceMCPGateway,
    llm: ChatOllama,
    lab_id: str,
) -> Any:
    """构建并编译实验柜的 LangGraph 状态图。

    Parameters
    ----------
    gateway : SpaceMCPGateway
        该实验柜独有的 Space-MCP 网关实例。
    llm : ChatOllama
        LLM 实例（ChatOllama）。
    lab_id : str
        实验柜标识（用于日志前缀）。

    Returns
    -------
    CompiledGraph
        编译后的状态图，支持 ``ainvoke(state)`` 调用。
    """

    # ================================================================== #
    #  Node: planner_node                                                 #
    # ================================================================== #

    async def planner_node(state: LabGraphState) -> dict[str, Any]:
        """调用 LLM 将原始任务拆解为分步 Skill 计划。"""
        task = state["original_task"]
        logger.info("[%s] 🧠 Planner: 正在规划任务「%s」...", lab_id, task)

        prompt = _PLANNER_PROMPT.format(
            skills_description=_SKILLS_DESCRIPTION,
            task=task,
        )

        response = await llm.ainvoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        logger.info("[%s] 🧠 Planner LLM 原始输出:\n%s", lab_id, raw_text)

        try:
            plan = _extract_json(raw_text)
            if not isinstance(plan, list):
                plan = [plan]
        except ValueError as exc:
            logger.warning("[%s] Planner JSON 解析失败: %s", lab_id, exc)
            plan = []

        logger.info("[%s] 🧠 Planner: 生成 %d 步计划", lab_id, len(plan))
        for i, step in enumerate(plan):
            logger.info("[%s]    步骤 %d: %s", lab_id, i + 1, step)

        return {
            "plan": plan,
            "current_step_index": 0,
            "current_step": None,
            "fsm_feedback": None,
            "execution_log": [],
            "error_count": 0,
        }

    # ================================================================== #
    #  Node: operator_node                                                #
    # ================================================================== #

    async def operator_node(state: LabGraphState) -> dict[str, Any]:
        """判断下一步动作：提取步骤 / 处理错误 / 标记完成。"""
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        feedback = state.get("fsm_feedback")
        error_count = state.get("error_count", 0)
        log = list(state.get("execution_log", []))

        # ── Case 1: 上一步执行失败，尝试 LLM 修正 ── #
        if feedback and feedback.get("status") == "error":
            if error_count >= _MAX_RETRIES_PER_STEP:
                logger.warning(
                    "[%s] ⚠️ Operator: 步骤 %d 连续失败 %d 次，跳过",
                    lab_id, index + 1, error_count,
                )
                log.append({
                    "step_index": index,
                    "skill": feedback.get("skill", "unknown"),
                    "params": {},
                    "result": feedback,
                    "status": "error",
                    "correction": None,
                })
                return {
                    "current_step_index": index + 1,
                    "current_step": None,
                    "fsm_feedback": None,
                    "execution_log": log,
                    "error_count": 0,
                }

            # 调用 LLM 尝试修正
            logger.info(
                "[%s] 🔄 Operator: 步骤 %d 失败，调用 LLM 修正 (第 %d 次)",
                lab_id, index + 1, error_count + 1,
            )

            current = plan[index] if index < len(plan) else {}
            correction_prompt = _CORRECTION_PROMPT.format(
                skill_name=current.get("skill", "unknown"),
                original_params=json.dumps(current.get("params", {}), ensure_ascii=False),
                error_detail=feedback.get("detail", "未知错误"),
            )

            response = await llm.ainvoke(correction_prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            logger.info("[%s] 🔄 Operator LLM 修正输出:\n%s", lab_id, raw_text)

            try:
                corrected = _extract_json(raw_text)
                if corrected.get("skill") == "skip":
                    logger.info("[%s] 🔄 LLM 建议跳过: %s", lab_id, corrected.get("reason"))
                    log.append({
                        "step_index": index,
                        "skill": current.get("skill", "unknown"),
                        "params": current.get("params", {}),
                        "result": feedback,
                        "status": "error",
                        "correction": corrected.get("reason"),
                    })
                    return {
                        "current_step_index": index + 1,
                        "current_step": None,
                        "fsm_feedback": None,
                        "execution_log": log,
                        "error_count": 0,
                    }
                else:
                    return {
                        "current_step": corrected,
                        "fsm_feedback": None,
                        "error_count": error_count + 1,
                    }
            except ValueError:
                logger.warning("[%s] 修正 JSON 解析失败，跳过此步", lab_id)
                log.append({
                    "step_index": index,
                    "skill": current.get("skill", "unknown"),
                    "params": current.get("params", {}),
                    "result": feedback,
                    "status": "error",
                    "correction": "LLM 修正 JSON 解析失败",
                })
                return {
                    "current_step_index": index + 1,
                    "current_step": None,
                    "fsm_feedback": None,
                    "execution_log": log,
                    "error_count": 0,
                }

        # ── Case 2: 上一步成功，记录并推进 ── #
        if feedback and feedback.get("status") == "success":
            log.append({
                "step_index": index,
                "skill": feedback.get("skill", "unknown"),
                "params": {},
                "result": feedback,
                "status": "success",
                "correction": None,
            })
            return {
                "current_step_index": index + 1,
                "current_step": None,
                "fsm_feedback": None,
                "execution_log": log,
                "error_count": 0,
            }

        # ── Case 3: 提取当前步骤 ── #
        if index < len(plan):
            step = plan[index]
            logger.info(
                "[%s] 📋 Operator: 提取步骤 %d/%d → %s",
                lab_id, index + 1, len(plan), step,
            )
            return {"current_step": step}

        # ── Case 4: 全部完成 ── #
        logger.info("[%s] ✅ Operator: 所有步骤执行完毕", lab_id)
        return {
            "current_step": None,
            "final_result": {
                "status": "completed",
                "total_steps": len(plan),
                "execution_log": log,
            },
        }

    # ================================================================== #
    #  Node: execute_node                                                 #
    # ================================================================== #

    async def execute_node(state: LabGraphState) -> dict[str, Any]:
        """通过 SpaceMCPGateway 调用当前步骤的 Skill。"""
        step = state.get("current_step")
        if not step:
            return {"fsm_feedback": {"status": "error", "detail": "无有效步骤"}}

        skill_name = step.get("skill", "")
        params = step.get("params", {})

        logger.info(
            "[%s] ⚙️ Execute: 调用 Skill '%s' params=%s",
            lab_id, skill_name, params,
        )

        result = await gateway.invoke_skill(skill_name, params)
        logger.info(
            "[%s] ⚙️ Execute: 结果 → %s", lab_id, result.get("status"),
        )

        return {"fsm_feedback": result}

    # ================================================================== #
    #  Conditional Edge: operator → execute or END                        #
    # ================================================================== #

    def should_continue(state: LabGraphState) -> str:
        """条件边：判断是继续执行还是结束。"""
        if state.get("final_result") is not None:
            return "done"
        if state.get("current_step") is not None:
            return "has_step"
        # 安全兜底：如果既没有步骤也没有结果，结束
        return "done"

    # ================================================================== #
    #  Build the StateGraph                                               #
    # ================================================================== #

    graph = StateGraph(LabGraphState)

    # 注册节点
    graph.add_node("planner_node", planner_node)
    graph.add_node("operator_node", operator_node)
    graph.add_node("execute_node", execute_node)

    # 设置入口
    graph.set_entry_point("planner_node")

    # planner → operator
    graph.add_edge("planner_node", "operator_node")

    # operator → conditional (execute or END)
    graph.add_conditional_edges(
        "operator_node",
        should_continue,
        {
            "has_step": "execute_node",
            "done": END,
        },
    )

    # execute → operator（循环回去处理结果）
    graph.add_edge("execute_node", "operator_node")

    return graph.compile()
