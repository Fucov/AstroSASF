"""
AstroSASF · Cognition · GraphBuilder (V4.3 — Headless)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态图工作流构建器。

V4.3 变化：
- 返回 **未编译的 StateGraph**（不含 MemorySaver / interrupt）
- 调用方自行决定编译选项（Headless 或 HITL）
- 保留 V4.3 修复的三段式 operator_node 逻辑
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

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
你的唯一任务：将用户指令拆解为 **底层 MCP Tool 原子调用序列**。

## ⚠️ 严格规则 (违反将导致系统崩溃)

1. **只允许使用以下 MCP Tools** —— 这是完整的白名单，不存在其他工具：
{tools_whitelist}

2. **绝对禁止**创造不在上述列表中的工具名！
   - ❌ 禁止输出 SOP/Skill 的名称作为 tool_name（如 fluid_experiment）
   - ❌ 禁止编造任何不在白名单中的工具
   - ✅ 每个步骤的 "skill" 字段必须精确匹配白名单中的名称

3. 如果用户提到某个 SOP（标准操作程序），你必须**阅读下方 SOP 文档**，
   然后将其中的每一步操作**翻译**为白名单中的底层 MCP Tool 调用。
   SOP 只是操作指南，不是可执行工具！

## 可用 MCP Tools 详细说明
{tools_description}

{skills_context}

## 输出格式 (严格 JSON，不允许任何其他文字)
[
  {{"skill": "<白名单中的tool名>", "params": {{...}} }},
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
    """从 LLM 输出中鲁棒地提取 JSON（容忍代码块、多余文字）。"""
    # 1) 去除 ```json ... ``` 代码块包裹
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2) 直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) 正则提取 JSON 数组或对象（DOTALL 跨行匹配）
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue

    # 4) 最后一搏：ast.literal_eval（容忍单引号等 Python 格式）
    import ast as _ast
    try:
        result = _ast.literal_eval(text.strip())
        if isinstance(result, (list, dict)):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:300]}")


# --------------------------------------------------------------------------- #
#  Graph Builder                                                               #
# --------------------------------------------------------------------------- #

def build_lab_graph(
    gateway: SpaceMCPGateway,
    llm: Any,
    lab_id: str,
    a2a_router: A2ARouter,
    skill_catalog: OpenAISkillCatalog | None = None,
) -> StateGraph:
    """构建实验柜的 LangGraph 状态图（**未编译**）。

    调用方自行编译，可选注入 checkpointer / interrupt::

        # Headless（默认）
        compiled = graph.compile()

        # HITL
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory, interrupt_before=["execute_node"])

    Returns
    -------
    StateGraph
        未编译的状态图。
    """

    # ── MCP Tools 描述 ── #
    tool_names = [t['name'] for t in gateway.list_tools()]
    tools_whitelist = "\n".join(f"   - `{name}`" for name in tool_names)
    tools_desc = "\n".join(
        f"- `{t['name']}`: {t['description']}  "
        f"Schema: {json.dumps(t['json_schema']['function']['parameters'], ensure_ascii=False)}"
        for t in gateway.list_tools()
    )
    tool_name_set = set(tool_names)

    # ── OpenAI Skills SOP 上下文 (V6.0: 动态 RAG 替代全量注入) ── #
    # skills_context 不再在此处静态构建，改为 planner_node 内动态检索

    # ================================================================== #
    #  Node: planner_node (V6.0 — Edge-RAG)                               #
    # ================================================================== #

    async def planner_node(state: LabGraphState) -> dict[str, Any]:
        task = state["original_task"]
        logger.info("[%s] 🧠 Planner: 正在规划「%s」...", lab_id, task)

        a2a_router.route(
            sender="System", receiver="Planner",
            intent=A2AIntent.TASK_REQUEST,
            payload={"task": task},
        )

        # ── V6.0: Edge-RAG 动态检索最相关 SOP ── #
        rag_context = ""
        if skill_catalog and skill_catalog.count > 0:
            retrieved = skill_catalog.retrieve_relevant_skills(task, top_k=1)
            if retrieved:
                hit = retrieved[0]
                logger.info("")
                logger.info("╔" + "═" * 60 + "╗")
                logger.info(
                    "║  🔍 Edge-RAG: 动态检索到相关知识                            ║",
                )
                logger.info("╚" + "═" * 60 + "╝")
                logger.info(
                    "[%s] 🔍 Edge-RAG: '%s' → '%s' (BM25 评分: %.4f)",
                    lab_id, task[:30], hit["name"], hit["score"],
                )
                logger.info(
                    "[%s] 🔍 Edge-RAG: %s",
                    lab_id, hit["description"],
                )
                logger.info("")

                rag_context = (
                    "## 🔍 Edge-RAG 检索到的最相关操作程序 (BM25 自动匹配)\n"
                    f"匹配度: {hit['score']:.4f}\n\n"
                    f"{hit['context']}"
                )

                # 追加 Macro 提示
                if skill_catalog.registry is not None:
                    macro_hint = skill_catalog._build_macro_hint()
                    if macro_hint:
                        rag_context += "\n\n" + macro_hint
            else:
                logger.info("[%s] 🔍 Edge-RAG: 未检索到相关知识", lab_id)

        prompt = _PLANNER_PROMPT.format(
            tools_whitelist=tools_whitelist,
            tools_description=tools_desc,
            skills_context=rag_context,
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

        # ── 后置校验：过滤不在白名单中的虚構工具 ── #
        validated_plan = []
        for step in plan:
            if isinstance(step, dict) and step.get("skill") in tool_name_set:
                validated_plan.append(step)
            else:
                logger.warning(
                    "[%s] ⚠️ Planner 输出了未注册的工具，已过滤: %s", lab_id, step,
                )
        plan = validated_plan

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
    #  Node: operator_node (V4.3 三段式流水线)                             #
    # ================================================================== #

    async def operator_node(state: LabGraphState) -> dict[str, Any]:
        plan = state.get("plan") or []
        index = state.get("current_step_index", 0)
        feedback = state.get("fsm_feedback")
        error_count = state.get("error_count", 0)
        log = list(state.get("execution_log") or [])

        # ── Phase A: Error → LLM 修正 ── #
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
                index += 1
                error_count = 0
            else:
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
                    else:
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

        # ── Phase B: Success → 记录 + 递增 ── #
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

        # ── Phase C: 提取下一步 or 完成 ── #
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
        final = state.get("final_result")
        if final is not None:
            return "done"
        step = state.get("current_step")
        if isinstance(step, dict) and step.get("skill"):
            return "has_step"
        return "done"

    # ================================================================== #
    #  Build (未编译 — 调用方自行 compile)                                  #
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

    return graph
