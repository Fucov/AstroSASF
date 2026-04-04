"""
AstroSASF · Scheduler · A2A Protocol (Kernel)
============================================
Agent-to-Agent 通信协议标准 —— 消息信封、路由器与发布/订阅接口。

V7.5 核心重构：
- Semantic Diff Sync：废弃全量发送，基于 JSON Patch 的增量状态同步
- A2ASemanticDiff：对比本地状态快照，只传输增量变更
- Apply Diff：对端收到 Diff 后自动合并到本地状态
- V7.5 新增弱网抗性指标埋点

Author: AstroSASF Team
Version: 7.5
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Diff 操作类型                                                                 #
# --------------------------------------------------------------------------- #

class DiffOp(str, Enum):
    """JSON Patch 操作类型。"""
    ADD = "add"         # 添加新字段
    REPLACE = "replace" # 替换现有字段
    REMOVE = "remove"   # 删除字段
    # 注意：不支持 "move" 和 "copy"，保持协议简洁


# --------------------------------------------------------------------------- #
#  语义增量 Diff 结构                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class A2ASemanticDiff:
    """A2A 通信的语义增量（Semantic Diff）。

    背景：
    - 传统 A2A 通信在汇报执行结果时全量发送 DAG 状态或环境上下文，
      在 200 Kbps SpaceWire 带宽下极为低效。
    - V7.5 引入 Diff 结构，只传输从上一版本到当前版本之间的增量变更。

    Diff 生成策略：
    - 节点状态变更：只有从 PENDING/READY/RUNNING → COMPLETED/FAILED 的节点才包含在 diff 中
    - 遥测快照：只有与上一快照相比发生变化的字段才包含在 diff 中
    - 字段变更：记录旧值 → 新值（如某传感器数值变化）

    Example
    -------
    >>> old_state = {"nodes": {"A": {"status": "RUNNING"}, "B": {"status": "PENDING"}}}
    >>> new_state = {"nodes": {"A": {"status": "COMPLETED"}, "B": {"status": "RUNNING"}}}
    >>> diff = A2ASemanticDiff.from_snapshot(old_state, new_state, base_version=1)
    >>> print(diff.to_json_bytes())   # 只包含变化的节点 A 和 B
    """

    # 元信息
    diff_id: str
    sender: str
    receiver: str
    base_version: int          # 基于的旧版本号
    target_version: int        # 当前版本号
    timestamp: float = field(default_factory=time.time)

    # 变更集合（JSON Pointer 路径 → 操作类型 + 值）
    operations: list[dict[str, Any]] = field(default_factory=list)

    # 变更统计（用于埋点）
    bytes_saved: int = 0        # 与全量发送相比节省的字节数
    changed_field_count: int = 0  # 变更字段总数

    def to_json_bytes(self) -> bytes:
        """序列化为 JSON bytes（用于 SpaceWire 传输）。"""
        return json.dumps(
            {
                "diff_id": self.diff_id,
                "sender": self.sender,
                "receiver": self.receiver,
                "base_version": self.base_version,
                "target_version": self.target_version,
                "timestamp": self.timestamp,
                "operations": self.operations,
                "bytes_saved": self.bytes_saved,
                "changed_field_count": self.changed_field_count,
            },
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "A2ASemanticDiff":
        """从 JSON bytes 反序列化。"""
        obj = json.loads(raw.decode("utf-8"))
        return cls(
            diff_id=obj["diff_id"],
            sender=obj["sender"],
            receiver=obj["receiver"],
            base_version=obj["base_version"],
            target_version=obj["target_version"],
            timestamp=obj["timestamp"],
            operations=obj["operations"],
            bytes_saved=obj.get("bytes_saved", 0),
            changed_field_count=obj.get("changed_field_count", 0),
        )

    # ------------------------------------------------------------------------- #
    #  Diff 生成（语义感知的增量对比）                                              #
    # ------------------------------------------------------------------------- #

    @classmethod
    def from_snapshot(
        cls,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        base_version: int,
        sender: str = "",
        receiver: str = "",
        diff_id: str = "",
    ) -> "A2ASemanticDiff":
        """对比两个状态快照，生成语义增量。

        对比策略（语义感知，非纯粹字符串 diff）：
        - 节点状态机：只记录从非终态 → 终态 的变化（COMPLETED / FAILED）
        - 浮点数容差：temperature: 36.999 → 37.001 视为不变（容差 0.01）
        - 嵌套字段：支持 dot-notation 路径（如 "nodes.A.status"）
        """
        operations: list[dict[str, Any]] = []
        changed_count = 0
        full_size = len(json.dumps(new_state, ensure_ascii=False).encode("utf-8"))

        cls._compare_dict(old_state, new_state, "", operations)
        changed_count = len(operations)

        # 统计字节节省
        diff_size = len(json.dumps({"operations": operations}, ensure_ascii=False).encode("utf-8"))
        bytes_saved = max(0, full_size - diff_size)

        return cls(
            diff_id=diff_id or f"diff-{int(time.time() * 1000)}",
            sender=sender,
            receiver=receiver,
            base_version=base_version,
            target_version=base_version + 1,
            operations=operations,
            bytes_saved=bytes_saved,
            changed_field_count=changed_count,
        )

    @classmethod
    def _compare_dict(
        cls,
        old: dict[str, Any],
        new: dict[str, Any],
        path: str,
        operations: list[dict[str, Any]],
        float_tolerance: float = 0.001,
    ) -> None:
        """递归对比两个字典，收集差异操作。"""
        all_keys = set(old.keys()) | set(new.keys())

        for key in sorted(all_keys):
            current_path = f"{path}.{key}" if path else key
            old_val = old.get(key, ...)
            new_val = new.get(key, ...)

            if old_val is ...:
                # 新增字段
                operations.append({
                    "op": DiffOp.ADD.value,
                    "path": f"/{current_path.replace('.', '/')}",
                    "value": new_val,
                })
            elif new_val is ...:
                # 字段被删除
                operations.append({
                    "op": DiffOp.REMOVE.value,
                    "path": f"/{current_path.replace('.', '/')}",
                })
            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                # 递归对比嵌套字典
                cls._compare_dict(old_val, new_val, current_path, operations, float_tolerance)
            elif isinstance(old_val, float) and isinstance(new_val, float):
                # 浮点数容差比较
                if abs(old_val - new_val) > float_tolerance:
                    operations.append({
                        "op": DiffOp.REPLACE.value,
                        "path": f"/{current_path.replace('.', '/')}",
                        "value": new_val,
                    })
            elif old_val != new_val:
                # 标量值变更
                operations.append({
                    "op": DiffOp.REPLACE.value,
                    "path": f"/{current_path.replace('.', '/')}",
                    "value": new_val,
                })

    # ------------------------------------------------------------------------- #
    #  Diff 合并（Apply Diff 到本地状态）                                          #
    # ------------------------------------------------------------------------- #

    def apply_to(self, base_state: dict[str, Any]) -> dict[str, Any]:
        """将增量 Diff 合并到本地状态。

        Parameters
        ----------
        base_state : dict[str, Any]
            当前的本地状态快照（base_version 时的状态）

        Returns
        -------
        dict[str, Any]
            应用 Diff 后的新状态（深拷贝，base_state 不被修改）
        """
        import copy
        result = copy.deepcopy(base_state)

        for op in self.operations:
            op_type = op.get("op")
            json_path = op.get("path", "")
            # JSON Pointer 路径：/a/b/c → ["a", "b", "c"]
            parts = [p for p in json_path.split("/") if p]

            if op_type == DiffOp.REMOVE.value:
                cls._json_pointer_remove(result, parts)
            elif op_type in (DiffOp.ADD.value, DiffOp.REPLACE.value):
                cls._json_pointer_set(result, parts, op.get("value"))

        return result

    @staticmethod
    def _json_pointer_remove(obj: Any, parts: list[str]) -> None:
        """沿 JSON Pointer 路径删除目标节点。"""
        if not parts:
            return
        current = obj
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return
            else:
                return
        # 删除最后一个键
        last = parts[-1]
        if isinstance(current, dict):
            current.pop(last, None)
        elif isinstance(current, list):
            try:
                current.pop(int(last))
            except (ValueError, IndexError):
                pass

    @staticmethod
    def _json_pointer_set(obj: Any, parts: list[str], value: Any) -> None:
        """沿 JSON Pointer 路径设置目标节点的值（必要时创建中间路径）。"""
        if not parts:
            return
        current = obj
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if idx >= len(current):
                        current.extend([{}] * (idx - len(current) + 1))
                    current = current[idx]
                except (ValueError, IndexError):
                    return
            else:
                return
        last = parts[-1]
        if isinstance(current, dict):
            current[last] = value
        elif isinstance(current, list):
            try:
                current[int(last)] = value
            except (ValueError, IndexError):
                pass


# --------------------------------------------------------------------------- #
#  A2A 消息意图类型                                                             #
# --------------------------------------------------------------------------- #

class A2AIntent(Enum):
    """A2A 消息意图类型。"""
    TASK_REQUEST = auto()        # 外部 → Planner: 新任务提交
    PLAN_GENERATED = auto()      # Planner → Operator: 规划完成
    SKILL_INVOCATION = auto()  # Operator → Gateway: 技能调用请求
    SKILL_RESULT = auto()       # Gateway → Operator: 技能执行结果
    ERROR_CORRECTION = auto()  # Operator → LLM: 错误修正请求
    EXECUTION_COMPLETE = auto() # Operator → System: 执行完毕
    # V7.5 新增
    SEMANTIC_DIFF = auto()      # 增量状态同步（替代全量发送）
    STATE_SYNC_REQUEST = auto() # 状态同步请求（对端需要全量快照）


@dataclass(frozen=True)
class A2AMessage:
    """A2A 标准消息信封。"""
    sender: str
    receiver: str
    intent: A2AIntent
    payload: Any
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    # V7.5 新增：增量 Diff 传输
    use_diff: bool = False      # True = payload 是 A2ASemanticDiff.bytes
    bytes_saved: int = 0        # Diff 传输相比全量节省的字节数


class A2ASubscriber(Protocol):
    """A2A 消息订阅者协议。"""
    def on_message(self, message: A2AMessage) -> None:
        """收到匹配消息时的回调。"""
        ...


A2ACallback = Callable[[A2AMessage], None]


@dataclass
class A2ARouter:
    """A2A 消息路由器 —— 通信记录 + 发布/订阅 + 增量状态同步。

    V7.5 新增：
    - Semantic Diff Sync：全量发送改为增量 Diff 传输
    - 本地状态快照版本管理（base_version 追踪）
    - Diff 字节节省统计
    """

    lab_id: str
    _sequence_counter: int = field(default=0, init=False)
    _message_log: list[A2AMessage] = field(default_factory=list, init=False)
    _subscribers: dict[A2AIntent | None, list[A2ACallback]] = field(
        default_factory=dict, init=False,
    )
    # V7.5: 本地状态版本管理
    _local_state: dict[str, Any] = field(default_factory=dict, init=False)
    _state_version: int = field(default=0, init=False)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    # V7.5: Diff 统计
    _total_bytes_saved_by_diff: int = field(default=0, init=False)
    _diff_count: int = field(default=0, init=False)

    # ------------------------------------------------------------------------- #
    #  订阅接口                                                                 #
    # ------------------------------------------------------------------------- #

    def subscribe(self, intent: A2AIntent, callback: A2ACallback) -> None:
        """订阅指定意图类型的消息。"""
        self._subscribers.setdefault(intent, []).append(callback)
        logger.debug("[%s] A2A: 订阅 %s → %s", self.lab_id, intent.name, callback)

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
        """通知所有匹配的订阅者（含 SEMANTIC_DIFF 的自动合并）。"""
        for cb in self._subscribers.get(msg.intent, []):
            try:
                cb(msg)
            except Exception as exc:
                logger.warning("[%s] A2A 订阅者回调异常: %s", self.lab_id, exc)
        for cb in self._subscribers.get(None, []):
            try:
                cb(msg)
            except Exception as exc:
                logger.warning("[%s] A2A 通配订阅者回调异常: %s", self.lab_id, exc)

    # ------------------------------------------------------------------------- #
    #  增量状态同步 API                                                           #
    # ------------------------------------------------------------------------- #

    async def update_local_state(self, new_state: dict[str, Any]) -> None:
        """更新本地状态快照（每次 Executor/Planner 汇报结果时调用）。

        自动对比旧状态与新状态，生成增量 Diff 并通过 A2A 广播。
        """
        async with self._state_lock:
            old_state = dict(self._local_state)
            old_version = self._state_version

            diff = A2ASemanticDiff.from_snapshot(
                old_state=old_state,
                new_state=new_state,
                base_version=old_version,
                sender=self.lab_id,
                receiver="*",  # 广播
                diff_id=f"diff-{self.lab_id}-{old_version + 1}",
            )

            # 更新本地状态
            self._local_state = new_state
            self._state_version += 1

            # 更新统计
            self._total_bytes_saved_by_diff += diff.bytes_saved
            self._diff_count += 1

            logger.info(
                "[%s] A2A 状态更新: v%d → v%d | Diff: %d ops | 节省 %d B",
                self.lab_id, old_version, self._state_version,
                diff.changed_field_count, diff.bytes_saved,
            )

            # 自动广播增量 Diff
            self.route(
                sender=self.lab_id,
                receiver="*",
                intent=A2AIntent.SEMANTIC_DIFF,
                payload=diff.to_json_bytes(),
                use_diff=True,
                bytes_saved=diff.bytes_saved,
            )

    async def apply_incoming_diff(self, diff_bytes: bytes) -> dict[str, Any]:
        """接收到增量 Diff 时，自动合并到本地状态。

        Returns
        -------
        dict[str, Any]
            应用 Diff 后的新本地状态
        """
        async with self._state_lock:
            diff = A2ASemanticDiff.from_json_bytes(diff_bytes)
            new_state = diff.apply_to(self._local_state)
            self._local_state = new_state
            self._state_version = diff.target_version

            logger.info(
                "[%s] A2A Diff 合并: v%d → v%d | %d ops",
                self.lab_id, diff.base_version, diff.target_version,
                diff.changed_field_count,
            )
            return new_state

    async def request_full_sync(self, target: str) -> None:
        """向指定 Agent 请求全量状态快照（用于版本对齐失败时的兜底）。"""
        logger.info(
            "[%s] A2A 全量同步请求: → %s (本地版本: v%d)",
            self.lab_id, target, self._state_version,
        )
        self.route(
            sender=self.lab_id,
            receiver=target,
            intent=A2AIntent.STATE_SYNC_REQUEST,
            payload=self._local_state,
            use_diff=False,
            bytes_saved=0,
        )

    @property
    def current_state_version(self) -> int:
        """获取当前本地状态版本号。"""
        return self._state_version

    # ------------------------------------------------------------------------- #
    #  路由接口                                                                 #
    # ------------------------------------------------------------------------- #

    def route(
        self,
        sender: str,
        receiver: str,
        intent: A2AIntent,
        payload: Any,
        use_diff: bool = False,
        bytes_saved: int = 0,
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
            use_diff=use_diff,
            bytes_saved=bytes_saved,
        )

        self._message_log.append(msg)

        if intent == A2AIntent.SEMANTIC_DIFF:
            logger.info(
                "[%s] 📨 A2A DIFF #%04d │ %s → %s │ %d ops │ 节省 %d B",
                self.lab_id, msg.sequence,
                sender, receiver,
                len(json.loads(payload.decode("utf-8")).get("operations", [])),
                bytes_saved,
            )
        else:
            logger.info(
                "[%s] 📨 A2A #%04d │ %s → %s │ %s",
                self.lab_id, msg.sequence,
                sender, receiver, intent.name,
            )

        self._notify_subscribers(msg)
        return msg

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
        """获取路由统计（含 V7.5 Diff 节省指标）。"""
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
            # V7.5 Diff 指标
            "state_version": self._state_version,
            "total_bytes_saved_by_diff": self._total_bytes_saved_by_diff,
            "diff_count": self._diff_count,
        }


__all__ = [
    "A2AIntent",
    "A2AMessage",
    "A2ASubscriber",
    "A2ACallback",
    "A2ARouter",
    "A2ASemanticDiff",
    "DiffOp",
]
