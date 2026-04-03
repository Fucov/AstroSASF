"""
AstroSASF · Demo · Agents · Planner Agent
=========================================
规划 Agent：负责将任务分解为 DAG 节点序列，提交给 Executor。
Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from demo.agents.base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)


_PLANNER_SYSTEM_PROMPT = """你是一个专业的实验任务规划 Agent。
你的职责是将一个高层任务分解为严格有序的 MCP Tool 调用序列。

已知 MCP Tools（白名单）：
{tool_list}

约束：
1. 每步只能调用一个 Tool
2. 必须严格按顺序执行
3. 关注 FSM 状态转换
4. 遵循联锁规则

输出格式（JSON数组，每项为 {{"step": 1, "tool": "xxx", "params": {{...}}}}）：
"""


# ── Mock 模式计划生成（无 LLM 时的降级方案）────────────────────────────────────

def _mock_generate_plan(task_desc: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """基于关键词匹配生成执行计划（mock/demo 模式）。"""
    task_lower = task_desc.lower()
    plan: list[dict[str, Any]] = []
    used = set()

    def add_step(tool_name: str, params: dict[str, Any]) -> None:
        if tool_name not in used:
            plan.append({"step": len(plan) + 1, "tool": tool_name, "params": params})
            used.add(tool_name)

    tool_names = {t.get("name", "").lower() for t in tools}
    tool_map = {t.get("name", "").lower(): t.get("name", "") for t in tools}

    # 温度相关
    if any(k in task_lower for k in ["温度", "temperature", "加热", "heat", "培养", "culture", "37", "设置温度"]):
        t = tool_map.get("set_temperature") or tool_map.get("control_heater")
        if t:
            temp = 37.0
            for word in task_desc.split():
                try:
                    val = float(word.replace("°C", "").replace("C", "").strip())
                    if 20 <= val <= 80:
                        temp = val
                        break
                except ValueError:
                    pass
            if "control_heater" in tool_names:
                add_step("control_heater", {"temperature": temp, "duration": 3600})
            else:
                add_step("set_temperature", {"temperature": temp})

    # 营养液/注入
    if any(k in task_lower for k in ["营养液", "nutrient", "注入", "inject", "ml", "毫升"]):
        vol = 50
        for word in task_desc.split():
            try:
                val = float(word.replace("ml", "").replace("ML", "").replace("毫升", "").strip())
                if 0 < val <= 500:
                    vol = val
                    break
            except ValueError:
                pass
        add_step("inject_nutrient", {"volume": vol})

    # 真空泵
    if any(k in task_lower for k in ["真空", "vacuum", "泵", "pump"]):
        activate = True
        if any(k in task_lower for k in ["关闭", "stop", "关", "off", "false"]):
            activate = False
        add_step("toggle_vacuum_pump", {"activate": activate})

    # 阀门
    if any(k in task_lower for k in ["阀门", "valve", "mixing", "混合", "切换"]):
        position = "MIXING"
        if any(k in task_lower for k in ["open_a", "a口"]):
            position = "OPEN_A"
        elif any(k in task_lower for k in ["open_b", "b口"]):
            position = "OPEN_B"
        elif any(k in task_lower for k in ["关闭", "close", "closed"]):
            position = "CLOSED"
        add_step("control_valve", {"position": position})

    # 机械臂
    if any(k in task_lower for k in ["机械臂", "arm", "移动", "move", "位置"]):
        pos = "center"
        if any(k in task_lower for k in ["左", "left"]):
            pos = "left"
        elif any(k in task_lower for k in ["右", "right"]):
            pos = "right"
        elif any(k in task_lower for k in ["存储", "storage"]):
            pos = "storage"
        add_step("move_robotic_arm", {"target_position": pos})

    # 离心机
    if any(k in task_lower for k in ["离心", "centrifuge", "离心机", "spin"]):
        if any(k in task_lower for k in ["启动", "start", "spin"]):
            add_step("control_centrifuge", {"action": "START", "rpm": 3000})
        elif any(k in task_lower for k in ["停止", "stop", "brake"]):
            add_step("control_centrifuge", {"action": "STOP", "rpm": 0})

    # 压力传感器
    if any(k in task_lower for k in ["压力", "pressure", "传感器", "sensor"]):
        add_step("read_pressure_sensor", {"channel": 1})

    # 如果没匹配到任何工具，至少返回一个读取传感器步骤
    if not plan and tool_map:
        default = tool_map.get("read_sensor") or tool_map.get("read_pressure_sensor")
        if default:
            if "read_sensor" in default.lower():
                add_step(default, {"channel": 1})
            else:
                add_step(default, {"channel": 1})

    # 重新编号
    for i, step in enumerate(plan):
        step["step"] = i + 1

    return plan


class PlannerAgent(BaseAgent):
    """规划 Agent (V7.2)。

    接收来自 Q&A Agent 的任务请求，
    调用 LLM 将任务分解为有序的工具调用步骤，
    然后将完整规划发送给 Executor Agent 执行。
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        super().__init__(agent_id="Planner-Agent", base_url=base_url)
        self._pending_plans: dict[str, Any] = {}

    async def think(self, message: AgentMessage | None) -> AgentMessage | None:
        """接收 Q&A 请求，生成执行计划。"""
        if message is None:
            return None

        if message.metadata.get("type") != "plan_request":
            return None

        lab_id = message.metadata["lab_id"]
        task_desc = message.content

        # ── Step 2: 获取工具列表 ── #
        try:
            meta = await self.get_lab_meta(lab_id)
            tools = meta.get("tools", [])
        except Exception as exc:
            logger.warning("[Planner] 获取工具列表失败: %s", exc)
            tools = []

        tool_list = "\n".join(
            f"- {t['name']}: {t['description']}"
            for t in tools
        ) if tools else "(无法获取工具列表)"

        # ── Step 3: LLM 生成工具调用计划 ── #
        system_prompt = _PLANNER_SYSTEM_PROMPT.format(tool_list=tool_list)
        user_prompt = (
            f"实验舱: {lab_id}\n"
            f"任务: {task_desc}\n\n"
            f"请将上述任务分解为具体的工具调用步骤。"
        )

        try:
            llm_resp = await self.call_llm(
                self.make_messages(system_prompt, user_prompt),
                temperature=0.2,
                max_tokens=1024,
            )
            llm_text = llm_resp.get("content", "")

            # 检测是否为 mock 响应
            is_mock = "[FACADE] LLM Gateway 未配置" in llm_text or not llm_text

            # ── Step 4: 解析规划 ── #
            if is_mock:
                # Mock/Demo 模式：使用关键词匹配降级生成计划
                plan_steps = _mock_generate_plan(task_desc, tools)
                logger.info(
                    "[Planner] 使用 Mock 计划生成器，为 %s 生成了 %d 步",
                    lab_id, len(plan_steps),
                )
            else:
                plan_steps = self._parse_plan(llm_text)

            if not plan_steps:
                return AgentMessage(
                    sender=self.agent_id,
                    content=f"[Planner] 无法生成有效计划：\n{llm_text}",
                    metadata={"type": "error", "lab_id": lab_id},
                )

            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            self._pending_plans[plan_id] = {
                "lab_id": lab_id,
                "steps": plan_steps,
                "original_task": task_desc,
            }

            steps_summary = "\n".join(
                f"  {i+1}. {s['tool']}({s.get('params', {})})"
                for i, s in enumerate(plan_steps)
            )

            logger.info(
                "[Planner] ✅ 生成了 %d 步执行计划 [%s] for %s",
                len(plan_steps), plan_id, lab_id,
            )

            return AgentMessage(
                sender=self.agent_id,
                content=(
                    f"[Planner] 已生成 {len(plan_steps)} 步执行计划：\n"
                    f"{steps_summary}"
                ),
                metadata={
                    "type": "execution_request",
                    "lab_id": lab_id,
                    "plan_id": plan_id,
                    "steps": plan_steps,
                    "priority": message.metadata.get("priority", 2),
                },
            )

        except Exception as exc:
            logger.exception("[Planner] 规划失败: %s", exc)
            return AgentMessage(
                sender=self.agent_id,
                content=f"[Planner] 规划失败: {exc}",
                metadata={"type": "error", "lab_id": lab_id},
            )

    @staticmethod
    def _parse_plan(text: str) -> list[dict[str, Any]]:
        """从 LLM 输出中解析出工具调用计划。"""
        import json, re

        # 尝试提取 JSON 数组
        json_match = re.search(r"\[[\s\S]*\]", text)
        if json_match:
            try:
                steps = json.loads(json_match.group())
                if isinstance(steps, list) and all("tool" in s for s in steps):
                    return steps
            except Exception:
                pass

        # 降级：逐行解析 "step N: tool_name(params)"
        steps: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 匹配 "1. tool_name" 或 "tool_name" 格式
            m = re.match(r"(?:\d+[\.\)]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if m:
                tool_name = m.group(1)
                steps.append({"tool": tool_name, "params": {}})

        return steps

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        """获取指定计划。"""
        return self._pending_plans.get(plan_id)
