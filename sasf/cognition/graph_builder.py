"""
AstroSASF · Cognition · GraphBuilder (V4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态图工作流构建器。

V4 变化:
- 接受外部注入的 LLM（由 config_loader 工厂创建）
- 使用 A2ARouter 记录节点间通信
- Skill 描述从 SkillRegistry 动态获取（而非硬编码）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

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
) -> Any:
    """构建并编译实验柜的 LangGraph 状态图。

    Parameters
    ----------
    gateway : SpaceMCPGateway
        该实验柜的 Space-MCP 网关。
    llm : BaseChatModel
        LLM 实例（由 config_loader.create_llm 创建）。
    lab_id : str
        实验柜标识。
    a2a_router : A2ARouter
        A2A 消息路由器。
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
        task = state["original_task"]
        logger.info("[%s] 🧠 Planner: 正在规划「%s」...", lab_id, task)

        # A2A: 记录任务请求
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

        # A2A: 记录计划生成
        a2a_router.route(
            sender="Planner", receiver="Operator",
            intent=A2AIntent.PLAN_GENERATED,
            payload={"steps": len(plan), "plan": plan},
        )

        logger.info("[%s] 🧠 Planner: 生成 %d 步计划", lab_id, len(plan))
        return {
            "plan": plan, "current_step_index": 0, "current_step": None,
            "fsm_feedback": None, "execution_log": [], "error_count": 0,
        }

    # ================================================================== #
    #  Node: operator_node                                                #
    # ================================================================== #

    async def operator_node(state: LabGraphState) -> dict[str, Any]:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        feedback = state.get("fsm_feedback")
        error_count = state.get("error_count", 0)
        log = list(state.get("execution_log", []))

        # ── Case 1: FSM 拦截，LLM 修正 ── #
        if feedback and feedback.get("status") == "error":
            if error_count >= _MAX_RETRIES_PER_STEP:
                logger.warning("[%s] ⚠️ 步骤 %d 连续失败 %d 次，跳过", lab_id, index + 1, error_count)
                log.append({
                    "step_index": index, "skill": feedback.get("skill", "unknown"),
                    "params": {}, "result": feedback, "status": "error", "correction": None,
                })
                return {
                    "current_step_index": index + 1, "current_step": None,
                    "fsm_feedback": None, "execution_log": log, "error_count": 0,
                }

            logger.info("[%s] 🔄 步骤 %d 失败，LLM 修正 (第 %d 次)", lab_id, index + 1, error_count + 1)

            # A2A: 错误修正请求
            a2a_router.route(
                sender="Operator", receiver="LLM",
                intent=A2AIntent.ERROR_CORRECTION,
                payload={"step": index, "error": feedback.get("detail")},
            )

            current = plan[index] if index < len(plan) else {}
            prompt = _CORRECTION_PROMPT.format(
                skill_name=current.get("skill", "unknown"),
                original_params=json.dumps(current.get("params", {}), ensure_ascii=False),
                error_detail=feedback.get("detail", "未知错误"),
            )
            response = await llm.ainvoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            logger.info("[%s] 🔄 LLM 修正输出:\n%s", lab_id, raw_text)

            try:
                corrected = _extract_json(raw_text)
                if corrected.get("skill") == "skip":
                    logger.info("[%s] 🔄 LLM 建议跳过: %s", lab_id, corrected.get("reason"))
                    log.append({
                        "step_index": index, "skill": current.get("skill", "unknown"),
                        "params": current.get("params", {}), "result": feedback,
                        "status": "error", "correction": corrected.get("reason"),
                    })
                    return {
                        "current_step_index": index + 1, "current_step": None,
                        "fsm_feedback": None, "execution_log": log, "error_count": 0,
                    }
                else:
                    return {"current_step": corrected, "fsm_feedback": None, "error_count": error_count + 1}
            except ValueError:
                logger.warning("[%s] 修正 JSON 解析失败，跳过", lab_id)
                log.append({
                    "step_index": index, "skill": current.get("skill", "unknown"),
                    "params": current.get("params", {}), "result": feedback,
                    "status": "error", "correction": "LLM 修正 JSON 解析失败",
                })
                return {
                    "current_step_index": index + 1, "current_step": None,
                    "fsm_feedback": None, "execution_log": log, "error_count": 0,
                }

        # ── Case 2: 上一步成功 ── #
        if feedback and feedback.get("status") == "success":
            log.append({
                "step_index": index, "skill": feedback.get("skill", "unknown"),
                "params": {}, "result": feedback, "status": "success", "correction": None,
            })
            return {
                "current_step_index": index + 1, "current_step": None,
                "fsm_feedback": None, "execution_log": log, "error_count": 0,
            }

        # ── Case 3: 提取当前步骤 ── #
        if index < len(plan):
            step = plan[index]
            logger.info("[%s] 📋 Operator: 步骤 %d/%d → %s", lab_id, index + 1, len(plan), step)
            return {"current_step": step}

        # ── Case 4: 全部完成 ── #
        logger.info("[%s] ✅ Operator: 所有步骤执行完毕", lab_id)

        a2a_router.route(
            sender="Operator", receiver="System",
            intent=A2AIntent.EXECUTION_COMPLETE,
            payload={"total_steps": len(plan)},
        )

        return {
            "current_step": None,
            "final_result": {
                "status": "completed", "total_steps": len(plan), "execution_log": log,
            },
        }

    # ================================================================== #
    #  Node: execute_node                                                 #
    # ================================================================== #

    async def execute_node(state: LabGraphState) -> dict[str, Any]:
        step = state.get("current_step")
        if not step:
            return {"fsm_feedback": {"status": "error", "detail": "无有效步骤"}}

        skill_name = step.get("skill", "")
        params = step.get("params", {})
        logger.info("[%s] ⚙️ Execute: '%s' params=%s", lab_id, skill_name, params)

        result = await gateway.invoke_skill(skill_name, params)
        logger.info("[%s] ⚙️ Execute: 结果 → %s", lab_id, result.get("status"))
        return {"fsm_feedback": result}

    # ================================================================== #
    #  Conditional Edge                                                   #
    # ================================================================== #

    def should_continue(state: LabGraphState) -> str:
        if state.get("final_result") is not None:
            return "done"
        if state.get("current_step") is not None:
            return "has_step"
        return "done"

    # ================================================================== #
    #  Build Graph                                                        #
    # ================================================================== #

    graph = StateGraph(LabGraphState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("operator_node", operator_node)
    graph.add_node("execute_node", execute_node)
    graph.set_entry_point("planner_node")
    graph.add_edge("planner_node", "operator_node")
    graph.add_conditional_edges("operator_node", should_continue, {"has_step": "execute_node", "done": END})
    graph.add_edge("execute_node", "operator_node")

    return graph.compile()
