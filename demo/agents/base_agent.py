"""
AstroSASF · Demo · Agents
==========================
协作智能体包 — Q&A Agent、Planner Agent、Executor Agent。

每个 Agent 通过 HTTP 调用 interface/server.py 的 REST API 实现协作。
所有 Agent 均支持 asyncio 并发运行。

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """智能体消息。"""
    sender: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """智能体基类 (V7.2)。

    所有 Agent 均通过 HTTP 客户端向 interface/server.py 发送请求，
    不直接依赖内核组件，实现真正的 C/S 解耦。

    生命周期：
        start() → [receive/run loop] → stop()
    """

    def __init__(
        self,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self.agent_id = agent_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._running = False
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._subscribers: list[asyncio.Queue[AgentMessage]] = []

    # --------------------------------------------------------------------------- #
    #  HTTP 代理方法（透传到 API Facade）                                         #
    # --------------------------------------------------------------------------- #

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """调用 LLM（经由 /api/v1/llm/chat）。"""
        payload: dict[str, Any] = {
            "messages": messages,
            "agent_id": self.agent_id,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = await self._http.post(
            f"{self.base_url}/api/v1/llm/chat",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def execute_tool(
        self,
        lab_id: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
        priority: int = 2,
    ) -> dict[str, Any]:
        """执行 MCP Tool（经由 /api/v1/labs/{lab_id}/execute）。"""
        payload = {
            "tool_name": tool_name,
            "params": params or {},
            "agent_id": self.agent_id,
            "priority": priority,
        }
        resp = await self._http.post(
            f"{self.base_url}/api/v1/labs/{lab_id}/execute",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def list_labs(self) -> list[dict[str, str]]:
        """列出所有实验舱。"""
        resp = await self._http.get(f"{self.base_url}/api/v1/labs")
        resp.raise_for_status()
        return resp.json()

    async def get_lab_meta(self, lab_id: str) -> dict[str, Any]:
        """获取实验舱元数据。"""
        resp = await self._http.get(f"{self.base_url}/api/v1/labs/{lab_id}/meta")
        resp.raise_for_status()
        return resp.json()

    async def health_check(self) -> dict[str, Any]:
        """健康检查。"""
        resp = await self._http.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------------- #
    #  消息总线 (Pub/Sub)                                                         #
    # --------------------------------------------------------------------------- #

    async def send_to(self, target: "BaseAgent", content: str, metadata: dict[str, Any] | None = None) -> None:
        """向另一 Agent 发送消息。"""
        msg = AgentMessage(sender=self.agent_id, content=content, metadata=metadata or {})
        await target._inbox.put(msg)
        logger.info("[%s] → [%s]: %s", self.agent_id, target.agent_id, content[:80])

    def subscribe(self, queue: asyncio.Queue[AgentMessage]) -> None:
        """订阅来自本 Agent 的消息。"""
        self._subscribers.append(queue)

    # --------------------------------------------------------------------------- #
    #  生命周期                                                                  #
    # --------------------------------------------------------------------------- #

    @abstractmethod
    async def think(self, message: AgentMessage | None) -> AgentMessage | None:
        """Agent 核心思维逻辑——由子类实现。"""

    async def run(self) -> None:
        """Agent 主循环。"""
        logger.info("[%s] Agent 启动", self.agent_id)
        self._running = True

        while self._running:
            try:
                message = await asyncio.wait_for(self._inbox.get(), timeout=0.5)
                response = await self.think(message)
                if response is not None:
                    for sub in self._subscribers:
                        await sub.put(response)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.exception("[%s] 运行异常: %s", self.agent_id, exc)
                await asyncio.sleep(1)

        logger.info("[%s] Agent 已停止", self.agent_id)

    async def start(self) -> None:
        """启动 Agent 后台任务。"""
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """停止 Agent。"""
        self._running = False
        if hasattr(self, "_task"):
            await self._task
        await self._http.aclose()

    # --------------------------------------------------------------------------- #
    #  工具辅助                                                                  #
    # --------------------------------------------------------------------------- #

    def make_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict[str, str]]:
        """构建 OpenAI 格式消息列表。"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
