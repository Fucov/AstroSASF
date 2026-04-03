"""
AstroSASF · Demo · Agents · Q&A Agent
======================================
问答 Agent：负责理解用户需求，向 Planner 发起规划请求。
Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from demo.agents.base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)


_QA_SYSTEM_PROMPT = """你是一个专业的科学实验顾问 Agent。
你的职责是：
1. 理解用户的实验需求
2. 选择合适的实验舱
3. 将任务清晰地转交给 Planner Agent

可用的实验舱：{labs_info}

输出格式（纯文本，供下游 Agent 解析）：
LAB_ID: <实验舱ID>
TASK: <任务描述>
PRIORITY: <priority_0-3>
"""


class QAAgent(BaseAgent):
    """问答 Agent (V7.2)。

    接收用户自然语言请求，解析出实验舱 ID、任务描述和优先级，
    然后向 Planner Agent 发起规划请求。
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        super().__init__(agent_id="Q&A-Agent", base_url=base_url)

    async def think(self, message: AgentMessage | None) -> AgentMessage | None:
        """处理用户请求，生成规划请求。"""
        if message is None:
            return None

        # ── Step 1: 获取可用实验舱信息 ── #
        try:
            labs = await self.list_labs()
            labs_info = "\n".join(
                f"- {lab['lab_id']}: {lab['name']} — {lab['description']}"
                for lab in labs
            )
        except Exception as exc:
            logger.warning("[Q&A] 获取实验舱列表失败: %s", exc)
            labs_info = "(无法获取实验舱信息)"

        # ── Step 2: LLM 解析用户意图 ── #
        user_text = message.content
        system_prompt = _QA_SYSTEM_PROMPT.format(labs_info=labs_info)
        user_prompt = f"用户请求：{user_text}"

        try:
            llm_resp = await self.call_llm(
                self.make_messages(system_prompt, user_prompt),
                temperature=0.3,
                max_tokens=300,
            )
            llm_text = llm_resp.get("content", "")

            # ── Step 3: 解析 LLM 输出 ── #
            lab_id = self._extract_field(llm_text, "LAB_ID:")
            task_desc = self._extract_field(llm_text, "TASK:")
            priority_str = self._extract_field(llm_text, "PRIORITY:")
            priority = int(priority_str) if priority_str else 2

            if not lab_id or not task_desc:
                return AgentMessage(
                    sender=self.agent_id,
                    content=f"[Q&A] 无法解析任务：\n{llm_text}",
                    metadata={"type": "error"},
                )

            return AgentMessage(
                sender=self.agent_id,
                content=task_desc,
                metadata={
                    "type": "plan_request",
                    "lab_id": lab_id,
                    "priority": priority,
                    "original_request": user_text,
                    "llm_raw": llm_text,
                },
            )

        except Exception as exc:
            logger.exception("[Q&A] 处理失败: %s", exc)
            return AgentMessage(
                sender=self.agent_id,
                content=f"[Q&A] 处理失败: {exc}",
                metadata={"type": "error"},
            )

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        """从 LLM 输出中提取指定字段。"""
        for line in text.splitlines():
            if line.startswith(field_name):
                return line[len(field_name):].strip()
        return ""
