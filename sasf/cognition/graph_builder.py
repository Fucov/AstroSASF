"""
AstroSASF · Cognition · GraphBuilder (V7.0 — Dual-Track DAG Scheduling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph 状态图工作流构建器，支持理论/实践双轨调度。

V7.0 核心变化：
- 新增 ``dag_planner_node``：LLM 生成 DAG 任务图（带依赖关系）
- 新增 DAG 验证节点：检查循环依赖和工具白名单
- 保留原有线性 planner_node 以兼容 V6.2
- 状态图新增 DAG 执行模式
"""

from __future__ import annotations

import ast as _ast
import json
import logging
import re
from typing import Any

from langgraph.graph import END, StateGraph

from sasf.cognition.skill_loader import OpenAISkillCatalog
from sasf.cognition.state import LabGraphState
from sasf.core.models import (
    DAGTaskGraph,
    DAG_PLANNER_PROMPT_TEMPLATE,
    DAG_VALIDATOR_PROMPT,
    NodeStatus,
    TaskPriority,
)
from sasf.middleware.a2a_protocol import A2AIntent, A2ARouter
from sasf.middleware.gateway import SpaceMCPGateway

logger = logging.getLogger(__name__)

_MAX_RETRIES_PER_STEP: int = 2


# --------------------------------------------------------------------------- #
#  Prompt Templates (V7.0 — DAG Generation)                                    #
# --------------------------------------------------------------------------- #

_ROUTER_PROMPT = """你是太空实验柜的语义路由智能体 (Semantic Router)。

## 任务
分析用户的任务指令，从以下已注册的 SOP (标准操作程序) 列表中选择最匹配的一个。

## 已注册的 SOP 列表
{skills_list}

## 输出规则 (严格遵守)
- **只输出一个 SOP 的 name（不加引号、不加解释）**
- 如果没有任何 SOP 匹配该任务，输出: UNKNOWN

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
#  JSON 提取工具 (V6.2 四层防护)                                                 #
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> Any:
    """从 LLM 输出中鲁棒地提取 JSON。

    四层防护：
    1. 去除 ```json ``` 代码块
    2. 直接 json.loads
    3. 正则提取 [...] 或 {...}
    4. ast.literal_eval 兜底
    """
    # 1) 去除代码块包裹
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2) 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) 正则提取 JSON 数组或对象
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue

    # 4) ast.literal_eval（容忍单引号等 Python 格式）
    try:
        result = _ast.literal_eval(text.strip())
        if isinstance(result, (list, dict)):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:300]}")


# --------------------------------------------------------------------------- #
#  DAG Validation Helper                                                      #
# --------------------------------------------------------------------------- #

def _validate_dag_output(
    llm_output: list[dict[str, Any]],
    tool_name_set: set[str],
) -> tuple[bool, list[str]]:
    """验证 LLM 输出的 DAG 结构。

    Parameters
    ----------
    llm_output : list[dict[str, Any]]
        LLM 解析出的任务列表
    tool_name_set : set[str]
        合法的工具名称集合

    Returns
    -------
    tuple[bool, list[str]]
        (是否有效, 问题列表)
    """
    issues: list[str] = []
    node_ids: set[str] = set()

    for i, item in enumerate(llm_output):
        if not isinstance(item, dict):
            issues.append(f"步骤 {i}: 必须是字典类型")
            continue

        # 检查 ID
        node_id = item.get("id")
        if not node_id:
            issues.append(f"步骤 {i}: 缺少 'id' 字段")
        elif node_id in node_ids:
            issues.append(f"步骤 {i}: 发现重复的 ID '{node_id}'")
        else:
            node_ids.add(node_id)

        # 检查 skill
        skill = item.get("skill")
        if not skill:
            issues.append(f"步骤 {i}: 缺少 'skill' 字段")
        elif skill not in tool_name_set:
            issues.append(f"步骤 {i}: 工具 '{skill}' 不在白名单中")

        # 检查 depends_on
        depends_on = item.get("depends_on", [])
        if not isinstance(depends_on, list):
            issues.append(f"步骤 {i}: 'depends_on' 必须是数组")
        else:
            for dep_id in depends_on:
                if dep_id not in node_ids and dep_id not in [it.get("id") for it in llm_output]:
                    issues.append(f"步骤 {i}: 引用了不存在的依赖 '{dep_id}'")

    return (len(issues) == 0, issues)


def _detect_cycle_in_dag(llm_output: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """检测 DAG 中的循环依赖。

    Parameters
    ----------
    llm_output : list[dict[str, Any]]
        LLM 解析出的任务列表

    Returns
    -------
    tuple[bool, list[str]]
        (是否有循环, 循环节点列表)
    """
    from collections import deque

    # 构建邻接表
    node_map: dict[str, int] = {}
    adj: dict[str, list[str]] = {}

    for i, item in enumerate(llm_output):
        node_id = item.get("id", f"node_{i}")
        node_map[node_id] = i
        adj[node_id] = []

    for item in llm_output:
        node_id = item.get("id")
        if not node_id:
            continue
        for dep_id in item.get("depends_on", []):
            if dep_id in adj:
                adj[dep_id].append(node_id)

    # Kahn 算法
    in_degree = {nid: 0 for nid in adj}
    for node_id in adj:
        for child_id in adj[node_id]:
            in_degree[child_id] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    count = 0

    while queue:
        node_id = queue.popleft()
        count += 1
        for child_id in adj[node_id]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    if count != len(adj):
        cycle_nodes = [nid for nid in adj if in_degree[nid] > 0]
        return True, cycle_nodes

    return False, []


# --------------------------------------------------------------------------- #
#  Graph Builder (V7.0 — Dual-Track)                                          #
# --------------------------------------------------------------------------- #

def build_lab_graph(
    gateway: SpaceMCPGateway,
    llm: Any,
    lab_id: str,
    a2a_router: A2ARouter,
    skill_catalog: OpenAISkillCatalog | None = None,
) -> StateGraph:
    """构建实验柜的 LangGraph 状态图（**未编译**）。

    V7.0 节点流 (DAG 模式):
        router_node → dag_planner_node → dag_validator_node → END

    V7.0 节点流 (兼容模式, V6.2):
        router_node → planner_node → operator_node ⇄ execute_node → END
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

    # ── Skills 列表（供 Router 使用） ── #
    skills_list_str = ""
    if skill_catalog and skill_catalog.count > 0:
        skills_info = skill_catalog.get_skill_names_and_descriptions()
        skills_list_str = "\n".join(
            f"- `{s['name']}`: {s['description']}" for s in skills_info
        )

    # ================================================================== #
    #  Node: router_node (V6.2 — LLM Semantic Router)                     #
    # ================================================================== #

    async def router_node(state: LabGraphState) -> dict[str, Any]:
        task = state["original_task"]

        if not skill_catalog or skill_catalog.count == 0:
            logger.info("[%s] 🧭 Semantic Router: 无可用 SOP，跳过路由", lab_id)
            return {"selected_skill": None}

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🧭 Semantic Router: 正在分析任务意图...                   ║")
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("[%s] 🧭 任务: %s", lab_id, task)
        logger.info("[%s] 🧭 候选 SOP: %s", lab_id,
                     [s["name"] for s in skill_catalog.get_skill_names_and_descriptions()])

        prompt = _ROUTER_PROMPT.format(
            skills_list=skills_list_str,
            task=task,
        )

        try:
            response = await llm.ainvoke(
                prompt,
            )
            raw = response.content if hasattr(response, "content") else str(response)
            selected = raw.strip().strip('"').strip("'").strip('`')
        except Exception as exc:
            logger.warning("[%s] 🧭 Router LLM 调用失败: %s", lab_id, exc)
            selected = "UNKNOWN"

        # 校验：选中的 SOP 是否存在
        if selected != "UNKNOWN" and skill_catalog.get_skill(selected) is not None:
            logger.info(
                "[%s] 🧭 Semantic Router: ✅ 选中 SOP → '%s'",
                lab_id, selected,
            )
            return {"selected_skill": selected}
        else:
            logger.info(
                "[%s] 🧭 Semantic Router: ⚠️ 未匹配到已知 SOP (LLM 输出: '%s')",
                lab_id, selected,
            )
            return {"selected_skill": None}

    # ================================================================== #
    #  Node: dag_planner_node (V7.0 — DAG 任务规划)                        #
    # ================================================================== #

    async def dag_planner_node(state: LabGraphState) -> dict[str, Any]:
        """理论智能体：生成带依赖关系的 DAG 任务图。"""
        task = state["original_task"]
        selected_skill = state.get("selected_skill")

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🧠 DAG Planner (理论智能体): 正在生成任务图...             ║")
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("[%s] 🧠 任务: %s", lab_id, task)

        a2a_router.route(
            sender="System", receiver="Planner",
            intent=A2AIntent.TASK_REQUEST,
            payload={"task": task, "mode": "dag"},
        )

        # ── 构建 SOP 上下文 ── #
        skills_context = ""
        if selected_skill and skill_catalog:
            sop_context = skill_catalog.get_skill_context(selected_skill)
            if sop_context:
                logger.info(
                    "[%s] 🧠 Planner: 注入 SOP '%s' 到提示词",
                    lab_id, selected_skill,
                )
                skills_context = (
                    f"## 🧭 语义路由选中的操作程序\n\n{sop_context}"
                )
                if skill_catalog.registry is not None:
                    macro_hint = skill_catalog._build_macro_hint()
                    if macro_hint:
                        skills_context += "\n\n" + macro_hint

        # ── 使用 DAG Prompt 模板 ── #
        prompt = DAG_PLANNER_PROMPT_TEMPLATE.format(
            tools_whitelist=tools_whitelist,
            tools_description=tools_desc,
            skills_context=skills_context or "（无相关 SOP）",
            task=task,
        )

        try:
            response = await llm.ainvoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("[%s] 🧠 DAG Planner LLM 调用失败: %s", lab_id, exc)
            return {
                "dag_graph": None,
                "dag_error": f"DAG Planner LLM 调用失败: {exc}",
                "plan": [],
                "final_result": {
                    "status": "failed",
                    "reason": f"DAG Planner LLM 调用失败: {exc}",
                },
            }

        logger.info("[%s] 🧠 DAG Planner LLM 输出:\n%s", lab_id, raw_text)

        # ── JSON 提取 ── #
        try:
            dag_plan = _extract_json(raw_text)
        except ValueError:
            logger.warning("[%s] 🧠 DAG Planner: JSON 提取失败", lab_id)
            return {
                "dag_graph": None,
                "dag_error": f"JSON 提取失败: {raw_text[:200]}",
                "plan": [],
                "final_result": {
                    "status": "failed",
                    "reason": f"DAG Planner 无法生成有效计划: {raw_text[:200]}",
                },
            }

        if not isinstance(dag_plan, list):
            dag_plan = [dag_plan]

        # ── DAG 结构验证 ── #
        is_valid, issues = _validate_dag_output(dag_plan, tool_name_set)
        if not is_valid:
            logger.warning("[%s] 🧠 DAG Planner: 结构验证失败 - %s", lab_id, issues)
            return {
                "dag_graph": None,
                "dag_error": f"DAG 结构验证失败: {issues}",
                "plan": [],
                "final_result": {
                    "status": "failed",
                    "reason": f"DAG 结构验证失败: {issues}",
                },
            }

        # ── 循环依赖检测 ── #
        has_cycle, cycle_nodes = _detect_cycle_in_dag(dag_plan)
        if has_cycle:
            logger.warning("[%s] 🧠 DAG Planner: 检测到循环依赖 - %s", lab_id, cycle_nodes)
            return {
                "dag_graph": None,
                "dag_error": f"检测到循环依赖: {cycle_nodes}",
                "plan": [],
                "final_result": {
                    "status": "failed",
                    "reason": f"检测到循环依赖: {cycle_nodes}",
                },
            }

        # ── 构建 DAGTaskGraph ── #
        try:
            dag_graph = DAGTaskGraph.from_llm_output(
                llm_output=dag_plan,
                graph_id=f"{lab_id}_{uuid_short()}",
                lab_id=lab_id,
                default_priority=TaskPriority.NORMAL,
            )
            logger.info(
                "[%s] 🧠 DAG Planner: ✅ 生成 DAG 图 (节点数: %d)",
                lab_id, len(dag_graph.nodes),
            )
        except Exception as exc:
            logger.error("[%s] 🧠 DAG Planner: 图构建失败 - %s", lab_id, exc)
            return {
                "dag_graph": None,
                "dag_error": f"图构建失败: {exc}",
                "plan": [],
                "final_result": {
                    "status": "failed",
                    "reason": f"DAG 图构建失败: {exc}",
                },
            }

        a2a_router.route(
            sender="Planner", receiver="Orchestrator",
            intent=A2AIntent.PLAN_GENERATED,
            payload={
                "graph_id": dag_graph.graph_id,
                "nodes": len(dag_graph.nodes),
                "plan": dag_plan,
            },
        )

        # 打印 DAG 结构
        _log_dag_structure(lab_id, dag_graph)

        return {
            "dag_graph": dag_graph,
            "dag_error": None,
            "plan": dag_plan,
            "final_result": None,
        }

    # ================================================================== #
    #  Node: dag_validator_node (V7.0 — DAG 验证)                          #
    # ================================================================== #

    async def dag_validator_node(state: LabGraphState) -> dict[str, Any]:
        """验证 DAG 计划的有效性（可选的二次验证）。"""
        dag_graph = state.get("dag_graph")

        if dag_graph is None:
            logger.warning("[%s] 🔍 DAG Validator: 无 DAG 图可验证", lab_id)
            return {"final_result": {"status": "failed", "reason": "DAG 图为空"}}

        try:
            dag_graph.validate()
            logger.info(
                "[%s] 🔍 DAG Validator: ✅ 验证通过 (节点: %d)",
                lab_id, len(dag_graph.nodes),
            )
            return {"final_result": None}
        except ValueError as exc:
            logger.error("[%s] 🔍 DAG Validator: ❌ 验证失败 - %s", lab_id, exc)
            return {
                "final_result": {
                    "status": "failed",
                    "reason": f"DAG 验证失败: {exc}",
                },
            }

    # ================================================================== #
    #  Legacy: planner_node (V6.2 — 线性任务规划，保留兼容性)               #
    # ================================================================== #

    async def planner_node(state: LabGraphState) -> dict[str, Any]:
        """线性任务规划器（V6.2 兼容模式）。"""
        task = state["original_task"]
        selected_skill = state.get("selected_skill")

        logger.info("[%s] 🧠 Planner: 正在规划「%s」...", lab_id, task)

        a2a_router.route(
            sender="System", receiver="Planner",
            intent=A2AIntent.TASK_REQUEST,
            payload={"task": task},
        )

        # ── 构建 SOP 上下文 ── #
        skills_context = ""
        if selected_skill and skill_catalog:
            sop_context = skill_catalog.get_skill_context(selected_skill)
            if sop_context:
                skills_context = f"## 🧭 语义路由选中的操作程序\n\n{sop_context}"
                if skill_catalog.registry is not None:
                    macro_hint = skill_catalog._build_macro_hint()
                    if macro_hint:
                        skills_context += "\n\n" + macro_hint

        legacy_prompt = """你是太空实验柜的规划智能体 (Planner)。
你的唯一任务：将用户指令拆解为 **底层 MCP Tool 原子调用序列**。

## ⚠️ 严格规则 (违反将导致系统崩溃)

1. **只允许使用以下 MCP Tools** —— 这是完整的白名单，不存在其他工具：
{tools_whitelist}

2. **绝对禁止**创造不在上述列表中的工具名！

## 可用 MCP Tools 详细说明
{tools_description}

{skills_context}

## 输出格式 (严格 JSON，不允许任何其他文字)
[
  {{"skill": "<tool_name>", "params": {{...}} }},
  ...
]

用户指令: {task}"""

        prompt = legacy_prompt.format(
            tools_whitelist=tools_whitelist,
            tools_description=tools_desc,
            skills_context=skills_context,
            task=task,
        )

        try:
            response = await llm.ainvoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("[%s] 🧠 Planner LLM 调用失败: %s", lab_id, exc)
            return {
                "plan": [],
                "error_msg": f"Planner LLM 调用失败: {exc}",
                "final_result": {
                    "status": "failed",
                    "reason": f"Planner LLM 调用失败: {exc}",
                },
            }

        logger.info("[%s] 🧠 Planner LLM 输出:\n%s", lab_id, raw_text)

        # ── 健壮 JSON 提取 ── #
        try:
            plan = _extract_json(raw_text)
            if not isinstance(plan, list):
                plan = [plan]
        except ValueError:
            logger.warning("[%s] 🧠 Planner: JSON 提取失败", lab_id)
            return {
                "plan": [],
                "error_msg": raw_text[:500],
                "final_result": {
                    "status": "failed",
                    "reason": f"Planner 无法生成有效计划: {raw_text[:200]}",
                },
            }

        # ── 白名单校验 ── #
        validated_plan = []
        for step in plan:
            if isinstance(step, dict) and step.get("skill") in tool_name_set:
                validated_plan.append(step)
            else:
                logger.warning(
                    "[%s] ⚠️ Planner 输出了未注册的工具，已过滤: %s", lab_id, step,
                )
        plan = validated_plan

        logger.info("[%s] 🧠 Planner: 生成 %d 步计划", lab_id, len(plan))
        for i, step in enumerate(plan):
            logger.info("[%s]    步骤 %d: %s", lab_id, i + 1, step)

        return {
            "plan": plan,
            "error_msg": None,
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
    #  Build (V7.0: DAG 模式优先)                                          #
    # ================================================================== #

    graph = StateGraph(LabGraphState)

    # V7.0 DAG 节点
    graph.add_node("router_node", router_node)
    graph.add_node("dag_planner_node", dag_planner_node)
    graph.add_node("dag_validator_node", dag_validator_node)

    # V6.2 兼容节点
    graph.add_node("planner_node", planner_node)
    graph.add_node("operator_node", operator_node)
    graph.add_node("execute_node", execute_node)

    # V7.0 DAG 流：router → dag_planner → dag_validator → END
    graph.set_entry_point("router_node")

    # Router 后的模式路由（互斥选择 DAG 模式 或 Legacy 线性模式）
    graph.add_conditional_edges(
        "router_node",
        _router_mode_decider,
        {"dag_mode": "dag_planner_node", "legacy_mode": "planner_node"},
    )
    graph.add_edge("dag_planner_node", "dag_validator_node")
    graph.add_conditional_edges(
        "dag_validator_node",
        _dag_validator_continue,
        {"has_dag": END, "failed": END},
    )

    # V6.2 兼容流
    graph.add_edge("planner_node", "operator_node")
    graph.add_conditional_edges(
        "operator_node",
        should_continue,
        {"has_step": "execute_node", "done": END},
    )
    graph.add_edge("execute_node", "operator_node")

    return graph


def _router_mode_decider(state: LabGraphState) -> str:
    """Router 后的模式选择器：V7.0 DAG 模式 vs V6.2 兼容模式。

    如果 router_node 选中了 SOP，则使用 DAG 模式；否则回退到 legacy 线性模式。
    """
    selected_skill = state.get("selected_skill")
    # lab_id 不可用，使用通用日志
    if selected_skill:
        logger.info("🧭 Router: 选中 SOP '%s'，进入 DAG 模式", selected_skill)
        return "dag_mode"
    logger.info("🧭 Router: 无可用 SOP，回退到 Legacy 线性模式")
    return "legacy_mode"


def _dag_validator_continue(state: LabGraphState) -> str:
    """DAG 验证后的条件路由。"""
    final = state.get("final_result")
    if final is not None and final.get("status") == "failed":
        return "failed"
    dag_graph = state.get("dag_graph")
    if dag_graph is not None:
        return "has_dag"
    return "failed"


def _log_dag_structure(lab_id: str, dag_graph: DAGTaskGraph) -> None:
    """打印 DAG 结构到日志。"""
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║  📊 DAG 任务图结构                                          ║")
    logger.info("╠" + "═" * 60 + "╣")

    for node_id, node in dag_graph.nodes.items():
        deps_str = ", ".join(node.dependencies) if node.dependencies else "（无依赖）"
        logger.info(
            "║  [%s] %-15s → %-20s 依赖: %-20s ║",
            node.status.name[:3],
            node_id[:15],
            node.skill_name[:20],
            deps_str[:20],
        )

    logger.info("╚" + "═" * 60 + "╝")
    logger.info("")


def uuid_short() -> str:
    """生成短 UUID。"""
    import uuid
    return uuid.uuid4().hex[:12]
