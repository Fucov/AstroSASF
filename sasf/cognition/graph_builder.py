"""
AstroSASF · Cognition · GraphBuilder (V4.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态图工作流构建器。

V4.3 修复：
- operator_node 合并 "成功记录 + 下一步提取" 为单次执行，
  避免 should_continue 在两步之间误判 current_step=None → END
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from sasf.cognition.skill_loader import OpenAISkillCatalog
from sasf.cognition.state import LabGraphState
from sasf.middleware.a2a_protocol import A2AIntent, A2ARouter
from sasf.middleware.gateway import SpaceMCPGateway

logger = logging.getLogger(__name__)

_MAX_RETRIES_PER_STEP: int = 2


# --------------------------------------------------------------------------- #
#  Prompt Templates                                                            #
# --------------------------------------------------------------------------- #

_PLANNER_PROMPT = """你是太空实验柜的规划智能体 (Planner)。
你的任务是将用户的自然语言指令拆解为一系列可执行的 MCP Tool 调用步骤。

## 可用 MCP Tools (底层原子操作)
{tools_description}

{skills_context}

请严格按以下 JSON 格式输出计划（不要添加任何其他文字解释）：
[
  {{"skill": "<tool_name>", "params": {{...}} }},
  ...
]

用户指令: {task}"""

_CORRECTION_PROMPT = """你是太空实验柜的操作智能体 (Operator)。
上一步 MCP Tool 执行失败，FSM 安全护栏返回了错误。

失败的 Tool: {skill_name}
原始参数: {original_params}
错误信息: {error_detail}

请分析错误原因，给出修正后的参数。严格仅输出修正后的 JSON 对象（不要添加任何其他解释）：
{{"skill": "<tool_name>", "params": {{...}} }}

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
    skill_catalog: OpenAISkillCatalog | None = None,
) -> tuple[Any, MemorySaver]:
    """构建并编译实验柜的 LangGraph 状态图。

    Returns
    -------
    tuple[CompiledGraph, MemorySaver]
    """

    # ── MCP Tools 描述（自动反射的 Schema） ── #
    tools_desc = "\n".join(
        f"- `{t['name']}`: {t['description']}  "
        f"Schema: {json.dumps(t['json_schema']['function']['parameters'], ensure_ascii=False)}"
        for t in gateway.list_tools()
    )

    # ── OpenAI Skills SOP 上下文 ── #
    skills_context = ""
    if skill_catalog and skill_catalog.count > 0:
        skills_context = (
            "## 已加载的 OpenAI Skills (标准操作程序 SOP)\n"
            "以下 Skill 告诉你**如何**组合调用 MCP Tools 完成复杂任务，请参考它们规划步骤：\n\n"
            + skill_catalog.get_all_skills_context()
        )

    # ================================================================== #
    #  Node: planner_node                                                 #
    # ================================================================== #

    async def planner_node(state: LabGraphState) -> dict[str, Any]:
        task = state["original_task"]
        logger.info("[%s] 🧠 Planner: 正在规划「%s」...", lab_id, task)

        a2a_router.route(
            sender="System", receiver="Planner",
            intent=A2AIntent.TASK_REQUEST,
            payload={"task": task},
        )

        prompt = _PLANNER_PROMPT.format(
            tools_description=tools_desc,
            skills_context=skills_context,
            task=task,
        )
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
    #                                                                      #
    #  V4.3 核心修复：合并 "成功记录" 和 "下一步提取" 为单次执行。          #
    #  流程：                                                              #
    #    1) 有 error feedback  → LLM 修正或跳过                            #
    #    2) 有 success feedback → 记录日志 + 递增 index                    #
    #    3) 提取 plan[index] 作为 current_step（或标记完成）               #
    #  步骤 2 和 3 合并在一次调用中完成，不会出现                          #
    #  "index 已递增但 current_step=None" 的中间态。                       #
    # ================================================================== #

    async def operator_node(state: LabGraphState) -> dict[str, Any]:
        plan = state.get("plan") or []
        index = state.get("current_step_index", 0)
        feedback = state.get("fsm_feedback")
        error_count = state.get("error_count", 0)
        log = list(state.get("execution_log") or [])

        # ── Phase A: 处理上一步 Error Feedback → LLM 修正 ── #
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
                # 跳过此步 → 递增 index，fall through 到 Phase C
                index += 1
                error_count = 0

            else:
                # 请求 LLM 修正
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
                    response.content if hasattr(response, "content")
                    else str(response)
                )
                logger.info("[%s] 🔄 LLM 修正输出:\n%s", lab_id, raw_text)

                try:
                    corrected = _extract_json(raw_text)
                    if corrected.get("skill") == "skip":
                        logger.info(
                            "[%s] 🔄 LLM 建议跳过: %s",
                            lab_id, corrected.get("reason"),
                        )
                        log.append({
                            "step_index": index,
                            "skill": current.get("skill", "unknown"),
                            "params": current.get("params", {}),
                            "result": feedback,
                            "status": "error",
                            "correction": corrected.get("reason"),
                        })
                        index += 1
                        error_count = 0
                        # fall through 到 Phase C
                    else:
                        # 给出修正后的 step → 直接作为 current_step 返回
                        return {
                            "current_step": corrected,
                            "current_step_index": index,
                            "fsm_feedback": None,
                            "execution_log": log,
                            "error_count": error_count + 1,
                        }
                except ValueError:
                    logger.warning("[%s] 修正 JSON 解析失败，跳过该步骤", lab_id)
                    log.append({
                        "step_index": index,
                        "skill": current.get("skill", "unknown"),
                        "params": current.get("params", {}),
                        "result": feedback,
                        "status": "error",
                        "correction": "LLM 修正 JSON 解析失败",
                    })
                    index += 1
                    error_count = 0
                    # fall through 到 Phase C

        # ── Phase B: 处理上一步 Success Feedback → 记录 + 递增 index ── #
        elif isinstance(feedback, dict) and feedback.get("status") == "success":
            log.append({
                "step_index": index,
                "skill": feedback.get("skill", "unknown"),
                "params": {},
                "result": feedback,
                "status": "success",
                "correction": None,
            })
            index += 1
            error_count = 0
            logger.info(
                "[%s] ✅ 步骤 %d 成功，推进至步骤 %d",
                lab_id, index, index + 1,
            )

        # ── Phase C: 提取下一步 or 标记完成 ── #
        if index < len(plan):
            step = plan[index]
            logger.info(
                "[%s] 📋 Operator: 提取步骤 %d/%d → %s",
                lab_id, index + 1, len(plan), step,
            )
            return {
                "current_step_index": index,
                "current_step": step,
                "fsm_feedback": None,
                "execution_log": log,
                "error_count": error_count,
                "final_result": None,
            }

        # ── 所有步骤执行完毕 ── #
        logger.info("[%s] ✅ Operator: 所有 %d 步执行完毕", lab_id, len(plan))
        a2a_router.route(
            sender="Operator", receiver="System",
            intent=A2AIntent.EXECUTION_COMPLETE,
            payload={"total_steps": len(plan), "executed": len(log)},
        )
        return {
            "current_step_index": index,
            "current_step": None,
            "fsm_feedback": None,
            "execution_log": log,
            "error_count": 0,
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
            result = await gateway.invoke_tool(skill_name, params)
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
    #  Conditional Edge                                                   #
    # ================================================================== #

    def should_continue(state: LabGraphState) -> str:
        """判断 operator_node 的出口：

        - final_result 不为 None → 所有步骤已完成 → END
        - current_step 存在      → 有待执行步骤 → execute_node (HITL)
        - 其他                    → END (安全兜底)
        """
        final = state.get("final_result")
        if final is not None:
            return "done"
        step = state.get("current_step")
        if isinstance(step, dict) and step.get("skill"):
            return "has_step"
        return "done"

    # ================================================================== #
    #  Build & Compile (HITL)                                             #
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

    memory = MemorySaver()
    compiled = graph.compile(
        checkpointer=memory,
        interrupt_before=["execute_node"],
    )

    return compiled, memory
