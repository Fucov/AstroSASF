"""
AstroSASF V7.1 · Demo · External Agent Client
============================================
外部多智能体系统的 HTTP 客户端示例。

⚠️ 注意：此文件是纯 HTTP 客户端，**不导入**任何 sasf.core 模块。

启动方式：
1. 终端 1: uvicorn server:app --reload --host 0.0.0.0 --port 8000
2. 终端 2: python demo_integration.py

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0  # 请求超时（秒）


# --------------------------------------------------------------------------- #
#  HTTP Client Wrapper                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class AstroOSClient:
    """AstroSASF 服务端 HTTP 客户端。

    封装所有 API 调用，提供简洁的 Python 接口。
    """

    base_url: str = API_BASE_URL
    timeout: float = TIMEOUT

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        await self._client.aclose()

    # ── 健康检查 ── #

    async def health_check(self) -> dict[str, Any]:
        """检查服务健康状态。"""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # ── 实验室 API ── #

    async def list_labs(self) -> list[dict[str, Any]]:
        """获取所有实验舱列表。"""
        response = await self._client.get("/api/v1/labs")
        response.raise_for_status()
        return response.json()

    async def get_lab_metadata(self, lab_id: str) -> dict[str, Any]:
        """获取实验舱完整元数据。"""
        response = await self._client.get(f"/api/v1/labs/{lab_id}/meta")
        response.raise_for_status()
        return response.json()

    # ── 执行 API ── #

    async def execute_tool(
        self,
        lab_id: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
        agent_id: str = "external_agent",
        priority: int = 2,
    ) -> dict[str, Any]:
        """执行工具调用。

        Parameters
        ----------
        lab_id : str
            实验舱 ID
        tool_name : str
            工具名称
        params : dict, optional
            工具参数
        agent_id : str
            调用者 ID
        priority : int
            优先级 (0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW)

        Returns
        -------
        dict
            执行结果
        """
        payload = {
            "tool_name": tool_name,
            "params": params or {},
            "agent_id": agent_id,
            "priority": priority,
        }

        response = await self._client.post(
            f"/api/v1/labs/{lab_id}/execute",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    # ── LLM API ── #

    async def chat(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "external_agent",
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 推理。

        Parameters
        ----------
        messages : list[dict[str, str]]
            消息列表
        agent_id : str
            调用者 ID
        temperature : float, optional
            采样温度

        Returns
        -------
        dict
            LLM 响应
        """
        payload = {
            "messages": messages,
            "agent_id": agent_id,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = await self._client.post(
            "/api/v1/llm/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------- #
#  External Agent Simulation                                                  #
# --------------------------------------------------------------------------- #

class ExternalAgent:
    """模拟外部多智能体框架中的 Agent。

    通过 HTTP API 与 AstroSASF 服务端交互。
    """

    def __init__(self, agent_id: str, client: AstroOSClient):
        self.agent_id = agent_id
        self.client = client
        self.labs: list[dict] = []
        self.current_lab: dict | None = None
        self.available_tools: list[dict] = []

    async def discover_services(self) -> None:
        """发现服务 — 从服务端拉取所有实验舱信息。"""
        logger.info("[%s] 正在发现服务...", self.agent_id)

        # 1. 获取实验舱列表
        self.labs = await self.client.list_labs()
        logger.info("[%s] 发现 %d 个实验舱:", self.agent_id, len(self.labs))
        for lab in self.labs:
            logger.info("  - %s: %s", lab["lab_id"], lab["description"])

        # 2. 获取第一个舱的完整元数据
        if self.labs:
            lab_id = self.labs[0]["lab_id"]
            self.current_lab = await self.client.get_lab_metadata(lab_id)
            self.available_tools = self.current_lab.get("tools", [])

            logger.info("")
            logger.info("[%s] 已加载实验舱: %s", self.agent_id, lab_id)
            logger.info("[%s] 发现 %d 个工具:", self.agent_id, len(self.available_tools))
            for tool in self.available_tools:
                guard_info = ""
                if tool.get("guard"):
                    guard_info = " [Guard]"
                logger.info("  - %s%s: %s", tool["name"], guard_info, tool.get("description", "")[:40])

            # 显示 FSM 信息
            fsm = self.current_lab.get("fsm", {})
            logger.info("")
            logger.info("[%s] FSM 子系统: %s", self.agent_id, list(fsm.get("subsystems", {}).keys()))
            logger.info("[%s] 联锁规则数: %d", self.agent_id, len(fsm.get("interlocks", [])))

            # 显示 Macros
            macros = self.current_lab.get("macros", [])
            if macros:
                logger.info("")
                logger.info("[%s] 可用宏:", self.agent_id)
                for macro in macros:
                    logger.info("  - %s → %s(%s)", macro["name"], macro["target"], macro["preset"])

    async def think(self, user_message: str) -> str:
        """调用 LLM 进行思考。

        这是外部 Agent 的"大脑"，用于自然语言理解。
        """
        logger.info("")
        logger.info("[%s] 💭 请求 LLM 思考...", self.agent_id)

        messages = [
            {"role": "system", "content": "你是一个太空实验舱的智能助手。"},
            {"role": "user", "content": user_message},
        ]

        response = await self.client.chat(
            messages=messages,
            agent_id=self.agent_id,
        )

        if response.get("error"):
            logger.error("[%s] LLM 错误: %s", self.agent_id, response["error"])
            return ""

        content = response.get("content", "")
        logger.info("[%s] 💡 LLM 回复: %s", self.agent_id, content[:100] + "..." if len(content) > 100 else content)
        return content

    async def execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行工具调用。"""
        if self.current_lab is None:
            raise RuntimeError("未选择实验舱")

        lab_id = self.current_lab["lab_id"]
        result = await self.client.execute_tool(
            lab_id=lab_id,
            tool_name=tool_name,
            params=params or {},
            agent_id=self.agent_id,
        )
        return result


# --------------------------------------------------------------------------- #
#  Demo Scenarios                                                             #
# --------------------------------------------------------------------------- #

async def demo_basic_operations(client: AstroOSClient) -> None:
    """演示 1：基础工具操作。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("演示 1: 基础工具操作")
    logger.info("=" * 70)

    agent = ExternalAgent(agent_id="demo_agent", client=client)
    await agent.discover_services()

    # 执行一个简单的工具调用
    logger.info("")
    logger.info("[演示] 执行 read_sensor 工具...")

    result = await agent.execute_tool("read_sensor", {"channel": 1})
    logger.info("[结果] status=%s, execution_id=%s", result["status"], result["execution_id"])

    if result["result"]:
        logger.info("[结果] 返回数据: %s", result["result"])


async def demo_llm_thinking(client: AstroOSClient) -> None:
    """演示 2：LLM 思考能力。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("演示 2: LLM 思考能力")
    logger.info("=" * 70)

    agent = ExternalAgent(agent_id="llm_agent", client=client)
    await agent.discover_services()

    # 使用 LLM 分析任务
    question = "在生物实验舱中，培养细胞需要哪些传感器数据？"
    await agent.think(question)


async def demo_fsm_blocking(client: AstroOSClient) -> None:
    """演示 3：FSM 联锁拦截。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("演示 3: FSM 联锁拦截")
    logger.info("=" * 70)

    agent = ExternalAgent(agent_id="fsm_agent", client=client)
    await agent.discover_services()

    # 先检查 FSM 状态
    fsm = agent.current_lab.get("fsm", {})
    logger.info("[FSM] 当前状态: %s", fsm.get("current_states", {}))

    # 执行需要特定状态的操作
    logger.info("")
    logger.info("[演示] 尝试执行 control_heater (需要 vacuum == IDLE)...")

    result = await agent.execute_tool("control_heater", {"temperature": 37.0, "duration": 1000})

    logger.info("[结果] status=%s", result["status"])
    if result.get("detail"):
        logger.info("[详情] %s", result["detail"])


async def demo_tool_discovery_and_planning(client: AstroOSClient) -> None:
    """演示 4：工具发现与规划。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("演示 4: 工具发现与规划")
    logger.info("=" * 70)

    agent = ExternalAgent(agent_id="planner_agent", client=client)
    await agent.discover_services()

    # 模拟 Agent 根据工具信息生成了执行计划
    logger.info("")
    logger.info("[规划] 基于可用工具，生成执行计划:")

    plan = [
        {"step": 1, "tool": "get_system_status", "reason": "获取当前系统状态"},
        {"step": 2, "tool": "read_sensor", "params": {"channel": 1}, "reason": "读取温度传感器"},
        {"step": 3, "tool": "control_heater", "params": {"temperature": 37.0, "duration": 5000}, "reason": "设置培养温度"},
    ]

    for step in plan:
        logger.info("  步骤 %d: %s(%s) — %s",
                   step["step"], step["tool"], step.get("params", {}), step["reason"])

    # 尝试执行计划
    logger.info("")
    logger.info("[执行] 开始执行计划...")

    for step in plan:
        result = await agent.execute_tool(
            step["tool"],
            step.get("params"),
        )
        status_icon = "✅" if result["status"] == "SUCCESS" else "⚠️"
        logger.info("  %s 步骤 %d: %s → %s",
                   status_icon, step["step"], step["tool"], result["status"])


# --------------------------------------------------------------------------- #
#  Main Entry Point                                                          #
# --------------------------------------------------------------------------- #

async def run_demos():
    """运行所有演示场景。"""
    logger.info("")
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  AstroSASF V7.1 — 外部 Agent 客户端演示                          ║")
    logger.info("║  纯 HTTP 客户端，不依赖 sasf.core                                  ║")
    logger.info("╚" + "═" * 70 + "╝")

    client = AstroOSClient()

    try:
        # ── 健康检查 ── #
        logger.info("")
        logger.info("[客户端] 连接服务端...")
        health = await client.health_check()
        logger.info("[客户端] ✅ 服务健康: version=%s, labs=%s",
                   health["version"], health["loaded_labs"])

        # ── 运行演示 ── #
        await demo_basic_operations(client)
        await demo_llm_thinking(client)
        await demo_fsm_blocking(client)
        await demo_tool_discovery_and_planning(client)

        logger.info("")
        logger.info("╔" + "═" * 70 + "╗")
        logger.info("║  所有演示完成 ✓                                                   ║")
        logger.info("╚" + "═" * 70 + "╝")

    except httpx.ConnectError:
        logger.error("")
        logger.error("❌ 无法连接到服务端 (http://localhost:8000)")
        logger.error("请确保服务端已启动：uvicorn server:app --reload")
        sys.exit(1)

    except httpx.HTTPStatusError as exc:
        logger.error("")
        logger.error("❌ HTTP 错误: %s %s", exc.response.status_code, exc.response.text)
        sys.exit(1)

    except Exception as exc:
        logger.exception("演示异常: %s", exc)
        sys.exit(1)

    finally:
        await client.close()


def main():
    """入口点。"""
    try:
        asyncio.run(run_demos())
    except KeyboardInterrupt:
        logger.info("演示被用户中断")


if __name__ == "__main__":
    main()
