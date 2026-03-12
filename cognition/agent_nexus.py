"""
AstroSASF · Cognition · AgentNexus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于 ``asyncio.Queue`` 的进程内 A2A（Agent-to-Agent）通信总线。

支持 Topic 级别的发布/订阅，实现 PlannerAgent ↔ OperatorAgent 之间
的异步解耦通信。每个 LaboratoryEnvironment 持有独立的 Nexus 实例，
确保跨环境消息隔离。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Topics                                                                      #
# --------------------------------------------------------------------------- #

class Topic(Enum):
    """Nexus 消息主题枚举。"""
    PLANNING_RESULT = auto()       # Planner → Operator：分步计划
    EXECUTION_FEEDBACK = auto()    # Operator → Planner：执行反馈
    SYSTEM_ALERT = auto()          # 广播：系统级告警
    TASK_REQUEST = auto()          # 外部 → Planner：新任务请求


# --------------------------------------------------------------------------- #
#  Message Envelope                                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Message:
    """Nexus 消息信封。

    Attributes
    ----------
    topic : Topic
        消息所属主题。
    sender : str
        发送方标识。
    payload : Any
        消息正文（通常为 dict/JSON 兼容结构）。
    timestamp : float
        UNIX 时间戳。
    """
    topic: Topic
    sender: str
    payload: Any
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
#  AgentNexus                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class AgentNexus:
    """进程内 A2A 消息路由器。

    每个 Topic 维护一组独立的 ``asyncio.Queue``，一个订阅者对应一个
    Queue，发布时扇出（fan-out）到该 Topic 的所有订阅者队列。
    """

    lab_id: str
    _subscriptions: dict[Topic, list[asyncio.Queue[Message]]] = field(
        default_factory=lambda: {t: [] for t in Topic},
        init=False,
    )

    # ---- 订阅 ------------------------------------------------------------- #

    def subscribe(self, topic: Topic) -> asyncio.Queue[Message]:
        """订阅指定 Topic，返回该订阅者专属的消息队列。"""
        queue: asyncio.Queue[Message] = asyncio.Queue()
        self._subscriptions[topic].append(queue)
        logger.debug(
            "[%s] Nexus: 新订阅者加入 Topic.%s (当前共 %d)",
            self.lab_id, topic.name, len(self._subscriptions[topic]),
        )
        return queue

    # ---- 发布 ------------------------------------------------------------- #

    async def publish(self, message: Message) -> int:
        """向指定 Topic 的所有订阅者扇出消息。

        Returns
        -------
        int
            成功投递的订阅者数量。
        """
        queues = self._subscriptions.get(message.topic, [])
        for q in queues:
            await q.put(message)
        logger.info(
            "[%s] Nexus: Topic.%s  发送方=%s  投递 %d 个订阅者",
            self.lab_id, message.topic.name, message.sender, len(queues),
        )
        return len(queues)

    # ---- 辅助 ------------------------------------------------------------- #

    def subscriber_count(self, topic: Topic) -> int:
        """返回指定 Topic 的当前订阅者数量。"""
        return len(self._subscriptions.get(topic, []))
