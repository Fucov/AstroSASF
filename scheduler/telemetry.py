"""
AstroSASF · Scheduler · Telemetry Bus (Kernel)
==============================================
模拟 1553B 总线 —— 通用遥测数据存储 + 硬件级抢占报警系统。

V7.2 整合：
- 遥测数据读写（通用 1553B 总线影子）
- 硬件报警注册与评估（联锁安全求值）
- 报警监控协程（后台轮询 + 回调通知调度器）

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from scheduler.models import HardwareAlarm, HardwareInterruptTask, TaskPriority

logger = logging.getLogger(__name__)


AlarmCallback = Callable[[HardwareInterruptTask], None]


def _safe_eval_bool(expression: str, env: dict[str, Any]) -> bool:
    """安全布尔表达式求值（仅允许数学比较运算符）。

    允许的操作数：
    - 变量引用（从 env 中查找）
    - 数字（int/float）
    - 比较运算符：==, !=, <, >, <=, >=, in, not in
    - 逻辑运算符：and, or, not
    - 括号

    严格禁止：
    - 函数调用（abs, len 等）
    - 属性访问（.）
    - 字符串操作
    - 列表/字典构造
    """
    import ast
    import operator as op

    # 支持的运算符映射
    _ops = {
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
        ast.Not: lambda a: not a,
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    _bool_ops = {ast.And, ast.Or, ast.Not}

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 兼容
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(f"未知变量: {node.id}")
        elif isinstance(node, ast.BinOp):
            return _ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not _eval(node.operand)
            return _ops[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                result = values[0]
                for v in values[1:]:
                    result = result and v
                return result
            else:
                result = values[0]
                for v in values[1:]:
                    result = result or v
                return result
        elif isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                if not _ops[type(op_node)](left, right):
                    return False
                left = right
            return True
        else:
            raise ValueError(f"不支持的 AST 节点类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return bool(_eval(tree.body))
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        raise ValueError(f"表达式求值错误: '{expression}' -> {e}")


class SecurityGuardrailException(Exception):
    """安全联锁拦截异常。"""
    pass


@dataclass
class TelemetryBus:
    """实例级遥测总线 —— 通用 1553B 总线影子（内核模块）。

    V7.2 整合功能：
    - 遥测数据读写
    - 硬件级抢占报警系统
    """

    lab_id: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    alarm_poll_interval: float = 0.5

    _state: dict[str, Any] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    _alarms: dict[str, HardwareAlarm] = field(default_factory=dict, init=False)
    _alarm_callbacks: list[AlarmCallback] = field(default_factory=list, init=False)
    _alarm_monitor_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _alarm_active: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _shutdown_flag: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._state = dict(self.initial_state)
        self._alarm_active.set()

    # ── 遥测读写 API ── #

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return deepcopy(self._state)

    async def read(self, key: str) -> Any:
        async with self._lock:
            if key not in self._state:
                raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
            return self._state[key]

    async def write(self, key: str, value: Any) -> None:
        async with self._lock:
            old = self._state.get(key)
            self._state[key] = value
            logger.info("[%s] 遥测更新: %s  %r → %r", self.lab_id, key, old, value)

    async def batch_write(self, updates: dict[str, Any]) -> None:
        async with self._lock:
            for key, value in updates.items():
                self._state[key] = value
            logger.info("[%s] 遥测批量更新: %s", self.lab_id, list(updates.keys()))

    # ── 硬件报警系统 ── #

    def register_alarm(
        self,
        alarm_id: str,
        condition_expr: str,
        interrupt_action_skill: str,
        interrupt_action_params: dict[str, Any] | None = None,
        severity: TaskPriority = TaskPriority.CRITICAL,
    ) -> HardwareAlarm:
        """注册硬件报警条件。

        当 condition_expr 求值为 True 时，自动触发 interrupt_action。
        """
        alarm = HardwareAlarm(
            alarm_id=alarm_id,
            condition_expr=condition_expr,
            interrupt_action_skill=interrupt_action_skill,
            interrupt_action_params=interrupt_action_params or {},
            severity=severity,
        )
        self._alarms[alarm_id] = alarm
        logger.info(
            "[%s] 🔔 硬件报警注册: %s → %s (%s)",
            self.lab_id, alarm_id, condition_expr, severity.name,
        )
        return alarm

    def unregister_alarm(self, alarm_id: str) -> bool:
        """注销硬件报警。"""
        if alarm_id in self._alarms:
            del self._alarms[alarm_id]
            logger.info("[%s] 🔕 硬件报警注销: %s", self.lab_id, alarm_id)
            return True
        return False

    def register_alarm_callback(self, callback: AlarmCallback) -> None:
        """注册报警触发回调函数。"""
        self._alarm_callbacks.append(callback)
        logger.info("[%s] 📞 报警回调注册: %s", self.lab_id, callback)

    def start_alarm_monitor(self) -> None:
        """启动报警监控协程（后台运行）。"""
        if self._alarm_monitor_task is not None and not self._alarm_monitor_task.done():
            logger.warning("[%s] 报警监控协程已在运行", self.lab_id)
            return

        self._shutdown_flag = False
        self._alarm_monitor_task = asyncio.create_task(self._alarm_monitor_loop())
        logger.info("[%s] 🔍 报警监控协程启动 (轮询间隔: %.1fs)", self.lab_id, self.alarm_poll_interval)

    async def stop_alarm_monitor(self) -> None:
        """停止报警监控协程。"""
        self._shutdown_flag = True
        self._alarm_active.set()

        if self._alarm_monitor_task is not None:
            try:
                await asyncio.wait_for(self._alarm_monitor_task, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("[%s] 报警监控协程未能在 3s 内退出，强制取消", self.lab_id)
                self._alarm_monitor_task.cancel()
            finally:
                self._alarm_monitor_task = None
        logger.info("[%s] 🔕 报警监控协程已停止", self.lab_id)

    async def _alarm_monitor_loop(self) -> None:
        """报警监控协程主循环。"""
        logger.info("[%s] 🔍 报警监控协程开始监控 (共 %d 条规则)", self.lab_id, len(self._alarms))

        while not self._shutdown_flag:
            try:
                await asyncio.wait_for(
                    self._alarm_active.wait(),
                    timeout=self.alarm_poll_interval,
                )
                self._alarm_active.clear()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            if self._shutdown_flag:
                break

            await self._evaluate_alarms()

        logger.info("[%s] 🔍 报警监控协程退出", self.lab_id)

    async def _evaluate_alarms(self) -> None:
        """评估所有报警条件，触发匹配项。"""
        if not self._alarms:
            return

        telemetry = await self.snapshot()

        for alarm_id, alarm in self._alarms.items():
            if not alarm.enabled:
                continue

            try:
                triggered = _safe_eval_bool(alarm.condition_expr, telemetry)
            except (SyntaxError, NameError, ValueError) as exc:
                logger.warning(
                    "[%s] 🔔 报警 '%s' 求值失败: %s",
                    self.lab_id, alarm_id, exc,
                )
                continue

            if triggered:
                alarm.trigger_count += 1
                alarm.last_trigger_time = time.monotonic()

                logger.warning(
                    "[%s] 🚨 硬件报警触发! [%s] %s",
                    self.lab_id, alarm.severity.name, alarm.condition_expr,
                )

                interrupt = HardwareInterruptTask(
                    interrupt_id=f"{alarm.alarm_id}_{int(alarm.last_trigger_time * 1000)}",
                    description=f"硬件报警: {alarm.condition_expr} (触发了 {alarm.trigger_count} 次)",
                    action_skill=alarm.interrupt_action_skill,
                    action_params=alarm.interrupt_action_params,
                    lab_id=self.lab_id,
                    source_condition=alarm.condition_expr,
                    timestamp=alarm.last_trigger_time,
                )

                for callback in self._alarm_callbacks:
                    try:
                        callback(interrupt)
                    except Exception as exc:
                        logger.exception(
                            "[%s] 🔔 报警回调执行失败: %s → %s",
                            self.lab_id, callback, exc,
                        )

    def trigger_alarm_manually(self, alarm_id: str) -> bool:
        """手动触发指定报警（用于测试）。"""
        if alarm_id not in self._alarms:
            return False

        alarm = self._alarms[alarm_id]
        interrupt = HardwareInterruptTask(
            interrupt_id=f"{alarm.alarm_id}_manual_{int(time.monotonic() * 1000)}",
            description=f"手动触发: {alarm.condition_expr}",
            action_skill=alarm.interrupt_action_skill,
            action_params=alarm.interrupt_action_params,
            lab_id=self.lab_id,
            source_condition=alarm.condition_expr,
        )

        for callback in self._alarm_callbacks:
            try:
                callback(interrupt)
            except Exception as exc:
                logger.exception("[%s] 手动报警回调失败: %s", self.lab_id, exc)

        return True

    def get_alarm_status(self) -> dict[str, Any]:
        """获取所有报警状态。"""
        return {
            alarm_id: {
                "condition": alarm.condition_expr,
                "severity": alarm.severity.name,
                "enabled": alarm.enabled,
                "trigger_count": alarm.trigger_count,
                "last_trigger": alarm.last_trigger_time,
            }
            for alarm_id, alarm in self._alarms.items()
        }


# ── 导出联锁安全求值函数 ── #
safe_eval_bool = _safe_eval_bool


__all__ = [
    "TelemetryBus",
    "AlarmCallback",
    "safe_eval_bool",
    "SecurityGuardrailException",
]
