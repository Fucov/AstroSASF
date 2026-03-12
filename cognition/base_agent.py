"""
AstroSASF · Cognition · BaseAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
智能体抽象基类 —— 封装 LLM 调用接口与生命周期。

Version 1 内置 Mock LLM 实现，使用 ``asyncio.sleep`` 模拟推理延迟，
返回预设的 JSON 结构。后续版本替换为真实 LLM API 调用即可。
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cognition.agent_nexus import AgentNexus

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  LLM 推理延迟模拟参数                                                         #
# --------------------------------------------------------------------------- #

_MOCK_LLM_DELAY_SECONDS: float = 0.3  # 模拟大模型推理延迟


# --------------------------------------------------------------------------- #
#  BaseAgent                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class BaseAgent(ABC):
    """智能体抽象基类。

    Parameters
    ----------
    agent_id : str
        智能体唯一标识（通常为 ``{lab_id}::{role}``）。
    nexus : AgentNexus
        所属环境的 A2A 通信总线。
    """

    agent_id: str
    nexus: AgentNexus
    _running: bool = field(default=False, init=False, repr=False)

    # ---- 生命周期 --------------------------------------------------------- #

    @abstractmethod
    async def run(self) -> None:
        """智能体主循环，由子类实现。"""
        ...

    async def stop(self) -> None:
        """优雅关闭信号。"""
        self._running = False
        logger.info("[%s] 收到停止信号", self.agent_id)

    # ---- Mock LLM 调用 ---------------------------------------------------- #

    async def _call_llm(
        self,
        prompt: str,
        preset_response: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """模拟 LLM 推理调用。

        Parameters
        ----------
        prompt : str
            发送给 LLM 的 prompt（V1 仅记录日志）。
        preset_response : dict | list | None
            预设返回值；如为 ``None`` 则返回空 dict。

        Returns
        -------
        dict | list
            模拟的 LLM 响应。
        """
        logger.info("[%s] → LLM 请求 (模拟): %s", self.agent_id, prompt[:80])
        await asyncio.sleep(_MOCK_LLM_DELAY_SECONDS)
        response = preset_response if preset_response is not None else {}
        logger.info(
            "[%s] ← LLM 响应 (模拟): %s",
            self.agent_id, json.dumps(response, ensure_ascii=False)[:120],
        )
        return response
