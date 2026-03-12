"""
AstroSASF · Cognition · GraphBuilder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于 LangGraph 的状态图工作流构建器。

V3.1 改进：
- 修复 operator_node 状态传递 Bug（Case 2 成功后未提取下一步骤）
- 引入 MemorySaver + ``interrupt_before=["execute_node"]``
  实现 Human-in-the-loop 审核机制

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
                              │ ⏸ HITL   │                     │
                              └────┬─────┘                     │
                                   │                           │
                                   └───────────────────────────┘
                                   (回到 operator_node)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from cognition.state import LabGraphState
from middleware.gateway import SpaceMCPGateway

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

_MAX_RETRIES_PER_STEP: int = 2

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
#  JSON 提取                                                                    #
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
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:200]}")


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """安全地从可能为 None 的对象中取值。"""
    if d is None or not isinstance(d, dict):
        return default
    return d.get(key, default)


# --------------------------------------------------------------------------- #
#  Graph Builder                                                               #
# --------------------------------------------------------------------------- #

def build_lab_graph(
    gateway: SpaceMCPGateway,
    llm: ChatOllama,
    lab_id: str,
) -> tuple[Any, MemorySaver]:
    """构建并编译实验柜的 LangGraph 状态图。

    Returns
    -------
    tuple[CompiledGraph, MemorySaver]
        编译后的状态图 + checkpointer（用于 HITL interrupt/resume）。
    """

    # ================================================================== #
    #  Node: planner_node                                                 #
    # ================================================================== #

    async def planner_node(state: LabGraphState) -> dict[str, Any]:
        """调用 LLM 将原始任务拆解为分步 Skill 计划。"""
        task = state.get("original_task", "")
        logger.info("[%s] 🧠 Planner: 正在规划任务「%s」...", lab_id, task)

        prompt = _PLANNER_PROMPT.format(
            skills_description=_SKILLS_DESCRIPTION,
            task=task,
        )

        try:
            response = await llm.ainvoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("[%s] Planner LLM 调用失败: %s", lab_id, exc)
            raw_text = "[]"

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
            "final_result": None,
        }

    # ================================================================== #
    #  Node: operator_node                                                #
    # ================================================================== #

    async def operator_node(state: LabGraphState) -> dict[str, Any]:
        """判断下一步动作：处理反馈 → 提取步骤或标记完成。

        修复 V3 Bug：在处理完反馈后，**立即** 判断是否有下一步骤，
        而非返回空 current_step 导致 should_continue 误判。
        """
        plan = state.get("plan") or []
        index = state.get("current_step_index", 0)
        feedback = state.get("fsm_feedback")
        error_count = state.get("error_count", 0)
        log = list(state.get("execution_log") or [])

        # ── Phase 1: 处理上一步反馈 ── #

        if feedback is not None and isinstance(feedback, dict):
            status = _safe_get(feedback, "status", "")

            if status == "error":
                # ── 错误处理 ── #
                if error_count >= _MAX_RETRIES_PER_STEP:
                    # 超过重试次数 → 记录并跳过
                    logger.warning(
                        "[%s] ⚠️ Operator: 步骤 %d 连续失败 %d 次，跳过",
                        lab_id, index + 1, error_count,
                    )
                    log.append({
                        "step_index": index,
                        "skill": _safe_get(feedback, "skill", "unknown"),
                        "params": {},
                        "result": feedback,
                        "status": "error",
                        "correction": None,
                    })
                    index += 1  # 推进到下一步
                else:
                    # 调用 LLM 修正
                    logger.info(
                        "[%s] 🔄 Operator: 步骤 %d 失败，调用 LLM 修正 (第 %d 次)",
                        lab_id, index + 1, error_count + 1,
                    )
                    current = plan[index] if index < len(plan) else {}
                    correction_prompt = _CORRECTION_PROMPT.format(
                        skill_name=_safe_get(current, "skill", "unknown"),
                        original_params=json.dumps(
                            _safe_get(current, "params", {}), ensure_ascii=False,
                        ),
                        error_detail=_safe_get(feedback, "detail", "未知错误"),
                    )

                    try:
                        response = await llm.ainvoke(correction_prompt)
                        raw_text = (
                            response.content
                            if hasattr(response, "content")
                            else str(response)
                        )
                        logger.info(
                            "[%s] 🔄 Operator LLM 修正输出:\n%s", lab_id, raw_text,
                        )
                        corrected = _extract_json(raw_text)

                        if _safe_get(corrected, "skill") == "skip":
                            logger.info(
                                "[%s] 🔄 LLM 建议跳过: %s",
                                lab_id, _safe_get(corrected, "reason"),
                            )
                            log.append({
                                "step_index": index,
                                "skill": _safe_get(current, "skill", "unknown"),
                                "params": _safe_get(current, "params", {}),
                                "result": feedback,
                                "status": "error",
                                "correction": _safe_get(corrected, "reason"),
                            })
                            index += 1
                        else:
                            # 修正后立即返回新 step，让 execute_node 执行
                            return {
                                "current_step": corrected,
                                "fsm_feedback": None,
                                "error_count": error_count + 1,
                                "execution_log": log,
                            }
                    except (ValueError, Exception) as exc:
                        logger.warning(
                            "[%s] 修正失败 (%s)，跳过此步", lab_id, exc,
                        )
                        log.append({
                            "step_index": index,
                            "skill": _safe_get(current, "skill", "unknown"),
                            "params": _safe_get(current, "params", {}),
                            "result": feedback,
                            "status": "error",
                            "correction": f"修正失败: {exc}",
                        })
                        index += 1

            elif status == "success":
                # ── 成功 → 记录并推进 ── #
                log.append({
                    "step_index": index,
                    "skill": _safe_get(feedback, "skill", "unknown"),
                    "params": {},
                    "result": feedback,
                    "status": "success",
                    "correction": None,
                })
                index += 1

            else:
                # 未知状态 → 视为错误，跳过
                logger.warning(
                    "[%s] ⚠️ 未知反馈状态: %s，跳过", lab_id, status,
                )
                log.append({
                    "step_index": index,
                    "skill": _safe_get(feedback, "skill", "unknown"),
                    "params": {},
                    "result": feedback,
                    "status": "error",
                    "correction": f"未知反馈状态: {status}",
                })
                index += 1

        # ── Phase 2: 提取下一步骤 或 标记完成 ── #

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
                "error_count": 0 if feedback is None or _safe_get(feedback, "status") != "error" else error_count,
            }

        # 全部完成
        logger.info("[%s] ✅ Operator: 所有 %d 个步骤执行完毕", lab_id, len(plan))
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
        """通过 SpaceMCPGateway 调用当前步骤的 Skill。"""
        step = state.get("current_step")
        if step is None or not isinstance(step, dict):
            logger.warning("[%s] ⚙️ Execute: 无有效步骤，返回错误", lab_id)
            return {
                "fsm_feedback": {
                    "skill": "unknown",
                    "status": "error",
                    "detail": "无有效步骤",
                },
            }

        skill_name = _safe_get(step, "skill", "")
        params = _safe_get(step, "params", {})

        logger.info(
            "[%s] ⚙️ Execute: 调用 Skill '%s' params=%s",
            lab_id, skill_name, params,
        )

        try:
            result = await gateway.invoke_skill(skill_name, params)
        except Exception as exc:
            logger.exception("[%s] ⚙️ Execute: 网关异常", lab_id)
            result = {
                "skill": skill_name,
                "status": "error",
                "detail": f"网关异常: {exc!r}",
            }

        # 确保 result 永远是 dict
        if result is None or not isinstance(result, dict):
            result = {
                "skill": skill_name,
                "status": "error",
                "detail": "网关返回无效响应",
            }

        logger.info(
            "[%s] ⚙️ Execute: 结果 → %s",
            lab_id, _safe_get(result, "status", "unknown"),
        )
        return {"fsm_feedback": result}

    # ================================================================== #
    #  Conditional Edge                                                    #
    # ================================================================== #

    def should_continue(state: LabGraphState) -> str:
        """条件边：判断是继续执行还是结束。"""
        if state.get("final_result") is not None:
            return "done"
        if state.get("current_step") is not None:
            return "has_step"
        return "done"

    # ================================================================== #
    #  Build & Compile with MemorySaver + HITL                             #
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

    # MemorySaver + interrupt_before execute_node → Human-in-the-loop
    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_node"],
    )

    return compiled, checkpointer
