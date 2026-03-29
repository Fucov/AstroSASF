"""
AstroSASF · Physics · TelemetryBus (V5 + Hardware Preemption)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模拟 1553B 总线 —— 通用遥测数据存储。

V5.1 新增硬件级抢占：
- register_alarm: 注册硬件报警条件
- 监控协程实时评估条件表达式
- 触发时通过回调通知 Orchestrator

Author: AstroSASF Team
Version: 5.1
"""

from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from sasf.core.models import HardwareAlarm, HardwareInterruptTask, TaskPriority
from sasf.physics.interlock_engine import safe_eval_bool, SecurityGuardrailException

logger = logging.getLogger(__name__)


# ── 报警回调类型 ── #
AlarmCallback = Callable[[HardwareInterruptTask], None]


@dataclass
class TelemetryBus:
    """实例级遥测总线 —— 通用 1553B 总线影子。

    V5.1 新增：硬件级抢占报警系统

    Parameters
    ----------
    lab_id : str
    initial_state : dict, optional
        初始遥测数据，由应用层注入。
    alarm_poll_interval : float
        报警检测轮询间隔（秒），默认 0.5s
    """

    lab_id: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    alarm_poll_interval: float = 0.5

    # ── 遥测状态 ── #
    _state: dict[str, Any] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # ── 硬件报警系统 ── #
    _alarms: dict[str, HardwareAlarm] = field(default_factory=dict, init=False)
    _alarm_callbacks: list[AlarmCallback] = field(default_factory=list, init=False)
    _alarm_monitor_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _alarm_active: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _shutdown_flag: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._state = dict(self.initial_state)
        self._alarm_active.set()  # 初始时激活报警系统

    # ────────────────────────────────────────────────────────────────────────── #
    #  遥测读写 API                                                                #
    # ────────────────────────────────────────────────────────────────────────── #

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

    # ────────────────────────────────────────────────────────────────────────── #
    #  硬件报警系统 (Hardware Preemption)                                          #
    # ────────────────────────────────────────────────────────────────────────── #

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

        Parameters
        ----------
        alarm_id : str
            报警唯一标识符
        condition_expr : str
            布尔条件表达式，支持遥测变量，如 "temperature >= 80"
        interrupt_action_skill : str
            触发时执行的 MCP Tool 名称
        interrupt_action_params : dict, optional
            触发时执行的工具参数
        severity : TaskPriority
            报警严重程度

        Returns
        -------
        HardwareAlarm
            注册的报警条目

        Example
        -------
        >>> bus.register_alarm(
        ...     alarm_id="temp_overheat",
        ...     condition_expr="temperature >= 80",
        ...     interrupt_action_skill="emergency_cooling",
        ...     interrupt_action_params={"mode": "rapid"},
        ...     severity=TaskPriority.CRITICAL,
        ... )
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
        self._alarm_active.set()  # 唤醒监控循环以便退出

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
        """报警监控协程主循环。

        以固定频率对所有启用的报警条件进行求值，
        一旦触发，立即通过回调通知 Orchestrator。
        """
        logger.info("[%s] 🔍 报警监控协程开始监控 (共 %d 条规则)", self.lab_id, len(self._alarms))

        while not self._shutdown_flag:
            try:
                # 等待轮询间隔或显式唤醒
                await asyncio.wait_for(
                    self._alarm_active.wait(),
                    timeout=self.alarm_poll_interval,
                )
                self._alarm_active.clear()
            except asyncio.TimeoutError:
                pass  # 正常轮询超时
            except asyncio.CancelledError:
                break

            if self._shutdown_flag:
                break

            # ── 评估所有启用的报警条件 ── #
            await self._evaluate_alarms()

        logger.info("[%s] 🔍 报警监控协程退出", self.lab_id)

    async def _evaluate_alarms(self) -> None:
        """评估所有报警条件，触发匹配项。"""
        if not self._alarms:
            return

        # 获取当前遥测快照（带锁）
        telemetry = await self.snapshot()

        for alarm_id, alarm in self._alarms.items():
            if not alarm.enabled:
                continue

            try:
                # 使用 InterlockEngine 的安全求值器
                triggered = safe_eval_bool(alarm.condition_expr, telemetry)
            except (SecurityGuardrailException, SyntaxError, KeyError) as exc:
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

                # ── 构建中断任务 ── #
                interrupt = HardwareInterruptTask(
                    interrupt_id=f"{alarm.alarm_id}_{int(alarm.last_trigger_time * 1000)}",
                    description=f"硬件报警: {alarm.condition_expr} (触发了 {alarm.trigger_count} 次)",
                    action_skill=alarm.interrupt_action_skill,
                    action_params=alarm.interrupt_action_params,
                    lab_id=self.lab_id,
                    source_condition=alarm.condition_expr,
                    timestamp=alarm.last_trigger_time,
                )

                # ── 通知所有回调 ── #
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

