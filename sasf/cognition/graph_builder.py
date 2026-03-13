"""
AstroSASF · Cognition · GraphBuilder (V4.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态图工作流构建器。

V4.1 修复:
- 加回 MemorySaver + interrupt_before=["execute_node"] (HITL)
- 修复 operator_node 所有分支确保返回非 None dict
- should_continue 边增加防御性判断
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from sasf.cognition.state import LabGraphState
from sasf.middleware.a2a_protocol import A2AIntent, A2ARouter
from sasf.middleware.gateway import SpaceMCPGateway

logger = logging.getLogger(__name__)

_MAX_RETRIES_PER_STEP: int = 2


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
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:200]}")


# --------------------------------------------------------------------------- #
#  Graph Builder                                                               #
# --------------------------------------------------------------------------- #

def build_lab_graph(
    gateway: SpaceMCPGateway,
    llm: Any,
    lab_id: str,
    a2a_router: A2ARouter,
) -> tuple[Any, MemorySaver]:
    """构建并编译实验柜的 LangGraph 状态图。

    Returns
    -------
    tuple[CompiledGraph, MemorySaver]
        编译后的图 和 MemorySaver（HITL 检查点后端）。
    """

    # 从 Registry 动态获取 Skill 描述
    skills_desc = "\n".join(
        f"- {s['name']}: {s['description']}  参数: {s.get('param_schema', {})}"
        for s in gateway.list_skills()
    )

    # ================================================================== #
    #  Node: planner_node                                                 #
    # ================================================================== #

    async def planner_node(state: LabGraphState) -> dict[str, Any]:
        """调用 LLM 将原始任务拆解为分步 Skill 计划。"""
        task = state["original_task"]
        logger.info("[%s] 🧠 Planner: 正在规划「%s」...", lab_id, task)

        a2a_router.route(
            sender="System", receiver="Planner",
            intent=A2AIntent.TASK_REQUEST,
            payload={"task": task},
        )

        prompt = _PLANNER_PROMPT.format(skills_description=skills_desc, task=task)
        response = await llm.ainvoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        logger.info("[%s] 🧠 Planner LLM 输出:\n%s", lab_id, raw_text)

        try:
            plan = _extract_json(raw_text)
            if not isinstance(plan, list):
                plan = [plan]
        except ValueError as exc:
            logger.warning("[%s] Planner JSON 解析失败: %s", lab_id, exc)
            plan = []

        a2a_router.route(
            sender="Planner", receiver="Operator",
            intent=A2AIntent.PLAN_GENERATED,
            payload={"steps": len(plan), "plan": plan},
        )

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
            "final_result": None,
        }

    # ================================================================== #
    #  Node: operator_node                                                #
    # ================================================================== #

    async def operator_node(state: LabGraphState) -> dict[str, Any]:
        """判断下一步动作。**所有分支必须返回 dict，禁止返回 None**。"""
        plan = state.get("plan") or []
        index = state.get("current_step_index", 0)
        feedback = state.get("fsm_feedback")
        error_count = state.get("error_count", 0)
        log = list(state.get("execution_log") or [])

        # ── Case 1: 上一步执行失败，尝试 LLM 修正 ── #
        if isinstance(feedback, dict) and feedback.get("status") == "error":
            if error_count >= _MAX_RETRIES_PER_STEP:
                logger.warning(
                    "[%s] ⚠️ 步骤 %d 连续失败 %d 次，跳过",
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

            logger.info(
                "[%s] 🔄 步骤 %d 失败，LLM 修正 (第 %d 次)",
                lab_id, index + 1, error_count + 1,
            )

            a2a_router.route(
                sender="Operator", receiver="LLM",
                intent=A2AIntent.ERROR_CORRECTION,
                payload={"step": index, "error": feedback.get("detail")},
            )

            current = plan[index] if index < len(plan) else {}
            prompt = _CORRECTION_PROMPT.format(
                skill_name=current.get("skill", "unknown"),
                original_params=json.dumps(
                    current.get("params", {}), ensure_ascii=False,
                ),
                error_detail=feedback.get("detail", "未知错误"),
            )
            response = await llm.ainvoke(prompt)
            raw_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            logger.info("[%s] 🔄 LLM 修正输出:\n%s", lab_id, raw_text)

            try:
                corrected = _extract_json(raw_text)
                if corrected.get("skill") == "skip":
                    logger.info(
                        "[%s] 🔄 LLM 建议跳过: %s", lab_id, corrected.get("reason"),
                    )
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
                return {
                    "current_step": corrected,
                    "fsm_feedback": None,
                    "error_count": error_count + 1,
                }
            except ValueError:
                logger.warning("[%s] 修正 JSON 解析失败，跳过", lab_id)
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

        # ── Case 2: 上一步成功，记录并推进索引 ── #
        if isinstance(feedback, dict) and feedback.get("status") == "success":
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

        # ── Case 3: 提取下一步执行 ── #
        if index < len(plan):
            step = plan[index]
            logger.info(
                "[%s] 📋 Operator: 提取步骤 %d/%d → %s",
                lab_id, index + 1, len(plan), step,
            )
            return {
                "current_step": step,
                "fsm_feedback": None,
            }

        # ── Case 4: 全部完成 ── #
        logger.info("[%s] ✅ Operator: 所有步骤执行完毕", lab_id)

        a2a_router.route(
            sender="Operator", receiver="System",
            intent=A2AIntent.EXECUTION_COMPLETE,
            payload={"total_steps": len(plan), "executed": len(log)},
        )

        return {
            "current_step": None,
            "fsm_feedback": None,
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
        if not step or not isinstance(step, dict):
            return {
                "fsm_feedback": {
                    "status": "error",
                    "skill": "unknown",
                    "detail": "execute_node 收到无效步骤",
                },
            }

        skill_name = step.get("skill", "")
        params = step.get("params", {})

        logger.info(
            "[%s] ⚙️ Execute: '%s' params=%s", lab_id, skill_name, params,
        )

        try:
            result = await gateway.invoke_skill(skill_name, params)
        except Exception as exc:
            logger.exception("[%s] ⚙️ Execute 异常", lab_id)
            result = {
                "skill": skill_name,
                "status": "error",
                "detail": f"Gateway 调用异常: {exc!r}",
            }

        logger.info(
            "[%s] ⚙️ Execute: 结果 → %s", lab_id, result.get("status"),
        )
        return {"fsm_feedback": result}

    # ================================================================== #
    #  Conditional Edge: operator_node → execute_node 或 END              #
    # ================================================================== #

    def should_continue(state: LabGraphState) -> str:
        """条件边判断：继续执行 or 结束。

        三条严格规则（防御 NoneType）：
        1. final_result 非 None → 结束
        2. current_step 是有效 dict → 继续执行
        3. 兜底 → 结束（不会返回 None）
        """
        final = state.get("final_result")
        if final is not None:
            return "done"

        step = state.get("current_step")
        if isinstance(step, dict) and step.get("skill"):
            return "has_step"

        return "done"

    # ================================================================== #
    #  Build & Compile (with MemorySaver for HITL)                        #
    # ================================================================== #

    graph = StateGraph(LabGraphState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("operator_node", operator_node)
    graph.add_node("execute_node", execute_node)

    graph.set_entry_point("planner_node")
    graph.add_edge("planner_node", "operator_node")
    graph.add_conditional_edges(
        "operator_node",
        should_continue,
        {"has_step": "execute_node", "done": END},
    )
    graph.add_edge("execute_node", "operator_node")

    # HITL: MemorySaver 检查点 + execute_node 前中断
    memory = MemorySaver()
    compiled = graph.compile(
        checkpointer=memory,
        interrupt_before=["execute_node"],
    )

    return compiled, memory
