"""
AstroSASF · Middleware · A2A Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agent-to-Agent 通信协议标准 —— 中间件定义的消息信封与路由器。

即使上层使用 LangGraph 管理工作流，智能体间的每一条消息都必须封装为
中间件定义的 ``A2AMessage`` 标准结构。这确保了：

1. **可观测性**：中间件统一记录所有 A2A 通信日志。
2. **可审计性**：每条消息携带递增序列号和时间戳。
3. **可扩展性**：未来可在此层注入鉴权、限流、路由规则等。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

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
    """A2A 标准消息信封。

    所有智能体间通信必须封装为此结构，由 ``A2ARouter`` 统一路由和记录。

    Attributes
    ----------
    sender : str
        发送方标识（如 ``"Lab-Alpha::Planner"``）。
    receiver : str
        接收方标识（如 ``"Lab-Alpha::Operator"``）。
    intent : A2AIntent
        消息意图类型。
    payload : Any
        消息正文（JSON 兼容结构）。
    timestamp : float
        UNIX 时间戳。
    sequence : int
        消息序列号（由 Router 分配）。
    """
    sender: str
    receiver: str
    intent: A2AIntent
    payload: Any
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


# --------------------------------------------------------------------------- #
#  A2ARouter — 消息路由器                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class A2ARouter:
    """A2A 消息路由器 —— 中间件级别的通信记录与路由。

    每个 LaboratoryEnvironment 持有独立实例，负责：
    - 分配递增消息序列号
    - 记录完整的通信日志
    - 提供消息审计能力

    Parameters
    ----------
    lab_id : str
        所属实验柜标识。
    """

    lab_id: str
    _sequence_counter: int = field(default=0, init=False)
    _message_log: list[A2AMessage] = field(default_factory=list, init=False)

    # ---- 路由 ------------------------------------------------------------- #

    def route(
        self,
        sender: str,
        receiver: str,
        intent: A2AIntent,
        payload: Any,
    ) -> A2AMessage:
        """创建、记录并返回一条 A2A 消息。

        Parameters
        ----------
        sender : str
            发送方标识（如 ``"Planner"``，会自动添加 lab_id 前缀）。
        receiver : str
            接收方标识。
        intent : A2AIntent
            消息意图。
        payload : Any
            消息正文。

        Returns
        -------
        A2AMessage
            带有序列号的标准消息。
        """
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

        return msg

    # ---- 审计 ------------------------------------------------------------- #

    @property
    def message_count(self) -> int:
        """已路由的消息总数。"""
        return len(self._message_log)

    @property
    def message_log(self) -> list[A2AMessage]:
        """完整的消息日志（只读副本）。"""
        return list(self._message_log)

    def get_messages_by_intent(self, intent: A2AIntent) -> list[A2AMessage]:
        """按意图类型筛选消息。"""
        return [m for m in self._message_log if m.intent == intent]

    @property
    def stats(self) -> dict[str, Any]:
        """路由器统计信息。"""
        intent_counts: dict[str, int] = {}
        for msg in self._message_log:
            key = msg.intent.name
            intent_counts[key] = intent_counts.get(key, 0) + 1

        return {
            "total_messages": self._sequence_counter,
            "intent_distribution": intent_counts,
        }
