"""
AstroSASF · Middleware · A2A Protocol (V4.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agent-to-Agent 通信协议标准 —— 消息信封、路由器与发布/订阅接口。

V4.3 变化：
- 新增 ``subscribe(intent, callback)`` / ``unsubscribe()`` 接口
- ``route()`` 时自动通知所有订阅者
- ``A2ASubscriber`` Protocol 抽象，未来可对接 Redis / MQTT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Intent — 消息意图枚举                                                        #
# --------------------------------------------------------------------------- #

class A2AIntent(Enum):
    """A2A 消息意图类型。"""
    TASK_REQUEST = auto()        # 外部 → Planner: 新任务提交
    PLAN_GENERATED = auto()      # Planner → Operator: 规划完成
    SKILL_INVOCATION = auto()    # Operator → Gateway: 技能调用请求
    SKILL_RESULT = auto()        # Gateway → Operator: 技能执行结果
    ERROR_CORRECTION = auto()    # Operator → LLM: 错误修正请求
    EXECUTION_COMPLETE = auto()  # Operator → System: 执行完毕


# --------------------------------------------------------------------------- #
#  A2AMessage — 标准消息信封                                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class A2AMessage:
    """A2A 标准消息信封。"""
    sender: str
    receiver: str
    intent: A2AIntent
    payload: Any
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


# --------------------------------------------------------------------------- #
#  A2ASubscriber — 订阅者协议 (抽象接口)                                        #
# --------------------------------------------------------------------------- #

class A2ASubscriber(Protocol):
    """A2A 消息订阅者协议。

    实现此接口即可接入 A2ARouter 的 Pub/Sub 机制。
    未来可实现 Redis / MQTT 版本的 Subscriber。
    """

    def on_message(self, message: A2AMessage) -> None:
        """收到匹配消息时的回调。"""
        ...


# 简单回调函数也可以作为订阅者
A2ACallback = Callable[[A2AMessage], None]


# --------------------------------------------------------------------------- #
#  A2ARouter — 消息路由器 + Pub/Sub                                            #
# --------------------------------------------------------------------------- #

@dataclass
class A2ARouter:
    """A2A 消息路由器 —— 通信记录 + 发布/订阅。

    Pub/Sub 接口:
        - ``subscribe(intent, callback)`` — 订阅指定意图的消息
        - ``subscribe_all(callback)``     — 订阅所有消息
        - ``unsubscribe(intent, callback)`` — 取消订阅
        - ``clear_subscriptions()``       — 清空所有订阅

    ``route()`` 在记录日志后，自动通知所有匹配的订阅者。
    """

    lab_id: str
    _sequence_counter: int = field(default=0, init=False)
    _message_log: list[A2AMessage] = field(default_factory=list, init=False)

    # Pub/Sub: intent → list of callbacks
    _subscribers: dict[A2AIntent | None, list[A2ACallback]] = field(
        default_factory=dict, init=False,
    )

    # ---- Pub/Sub ---------------------------------------------------------- #

    def subscribe(self, intent: A2AIntent, callback: A2ACallback) -> None:
        """订阅指定意图类型的消息。"""
        self._subscribers.setdefault(intent, []).append(callback)
        logger.debug(
            "[%s] A2A: 订阅 %s → %s", self.lab_id, intent.name, callback,
        )

    def subscribe_all(self, callback: A2ACallback) -> None:
        """订阅所有意图类型的消息（使用 None 作为通配键）。"""
        self._subscribers.setdefault(None, []).append(callback)

    def unsubscribe(self, intent: A2AIntent, callback: A2ACallback) -> None:
        """取消订阅。"""
        subs = self._subscribers.get(intent, [])
        if callback in subs:
            subs.remove(callback)

    def clear_subscriptions(self) -> None:
        """清空所有订阅。"""
        self._subscribers.clear()

    def _notify_subscribers(self, msg: A2AMessage) -> None:
        """通知所有匹配的订阅者。"""
        # 精确匹配
        for cb in self._subscribers.get(msg.intent, []):
            try:
                cb(msg)
            except Exception as exc:
                logger.warning(
                    "[%s] A2A 订阅者回调异常: %s", self.lab_id, exc,
                )
        # 通配订阅
        for cb in self._subscribers.get(None, []):
            try:
                cb(msg)
            except Exception as exc:
                logger.warning(
                    "[%s] A2A 通配订阅者回调异常: %s", self.lab_id, exc,
                )

    # ---- 路由 ------------------------------------------------------------- #

    def route(
        self,
        sender: str,
        receiver: str,
        intent: A2AIntent,
        payload: Any,
    ) -> A2AMessage:
        """创建、记录、通知订阅者并返回 A2A 消息。"""
        self._sequence_counter += 1

        full_sender = f"{self.lab_id}::{sender}"
        full_receiver = f"{self.lab_id}::{receiver}"

        msg = A2AMessage(
            sender=full_sender,
            receiver=full_receiver,
            intent=intent,
            payload=payload,
            sequence=self._sequence_counter,
        )

        self._message_log.append(msg)

        logger.info(
            "[%s] 📨 A2A #%04d │ %s → %s │ %s",
            self.lab_id,
            msg.sequence,
            sender,
            receiver,
            intent.name,
        )

        # 通知订阅者
        self._notify_subscribers(msg)

        return msg

    # ---- 审计 ------------------------------------------------------------- #

    @property
    def message_count(self) -> int:
        return len(self._message_log)

    @property
    def message_log(self) -> list[A2AMessage]:
        return list(self._message_log)

    def get_messages_by_intent(self, intent: A2AIntent) -> list[A2AMessage]:
        return [m for m in self._message_log if m.intent == intent]

    @property
    def stats(self) -> dict[str, Any]:
        intent_counts: dict[str, int] = {}
        for msg in self._message_log:
            key = msg.intent.name
            intent_counts[key] = intent_counts.get(key, 0) + 1

        return {
            "total_messages": self._sequence_counter,
            "intent_distribution": intent_counts,
            "active_subscriptions": sum(
                len(cbs) for cbs in self._subscribers.values()
            ),
        }
