"""
AstroSASF · Physics · ShadowFSM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
影子设备有限状态机 —— 拦截 LLM 幻觉指令的**绝对护栏**。

本模块独立于 LLM 推理链路，仅依赖有限状态机的确定性转移表。
任何违背物理安全的动作在此处被拦截，抛出不可忽略的
``SecurityGuardrailException``。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Exception                                                                   #
# --------------------------------------------------------------------------- #

class SecurityGuardrailException(Exception):
    """FSM 安全护栏异常 —— 绝对不可被静默忽略。"""


# --------------------------------------------------------------------------- #
#  Device States                                                               #
# --------------------------------------------------------------------------- #

class DeviceState(Enum):
    """实验柜设备的有限状态集。"""
    IDLE = auto()
    HEATING = auto()
    COOLING = auto()
    ROBOTIC_ARM_MOVING = auto()
    VACUUM_ACTIVE = auto()
    EMERGENCY_STOP = auto()


# --------------------------------------------------------------------------- #
#  Action definitions                                                          #
# --------------------------------------------------------------------------- #

class Action(Enum):
    """可对实验柜执行的操作动词。"""
    START_HEATING = auto()
    STOP_HEATING = auto()
    START_COOLING = auto()
    STOP_COOLING = auto()
    MOVE_ROBOTIC_ARM = auto()
    STOP_ROBOTIC_ARM = auto()
    ACTIVATE_VACUUM = auto()
    DEACTIVATE_VACUUM = auto()
    EMERGENCY_STOP = auto()
    RESET = auto()


# --------------------------------------------------------------------------- #
#  Transition Table                                                            #
# --------------------------------------------------------------------------- #

# (current_state, action) → next_state
# 仅出现在此表中的转移才是合法的。
_TRANSITION_TABLE: dict[tuple[DeviceState, Action], DeviceState] = {
    # 从 IDLE 出发
    (DeviceState.IDLE, Action.START_HEATING):    DeviceState.HEATING,
    (DeviceState.IDLE, Action.START_COOLING):    DeviceState.COOLING,
    (DeviceState.IDLE, Action.MOVE_ROBOTIC_ARM): DeviceState.ROBOTIC_ARM_MOVING,
    (DeviceState.IDLE, Action.ACTIVATE_VACUUM):  DeviceState.VACUUM_ACTIVE,
    (DeviceState.IDLE, Action.EMERGENCY_STOP):   DeviceState.EMERGENCY_STOP,

    # 从 HEATING 出发
    (DeviceState.HEATING, Action.STOP_HEATING):  DeviceState.IDLE,
    (DeviceState.HEATING, Action.EMERGENCY_STOP): DeviceState.EMERGENCY_STOP,

    # 从 COOLING 出发
    (DeviceState.COOLING, Action.STOP_COOLING):  DeviceState.IDLE,
    (DeviceState.COOLING, Action.EMERGENCY_STOP): DeviceState.EMERGENCY_STOP,

    # 从 ROBOTIC_ARM_MOVING 出发
    (DeviceState.ROBOTIC_ARM_MOVING, Action.STOP_ROBOTIC_ARM): DeviceState.IDLE,
    (DeviceState.ROBOTIC_ARM_MOVING, Action.EMERGENCY_STOP):   DeviceState.EMERGENCY_STOP,

    # 从 VACUUM_ACTIVE 出发
    (DeviceState.VACUUM_ACTIVE, Action.DEACTIVATE_VACUUM): DeviceState.IDLE,
    (DeviceState.VACUUM_ACTIVE, Action.EMERGENCY_STOP):    DeviceState.EMERGENCY_STOP,

    # 从 EMERGENCY_STOP 恢复
    (DeviceState.EMERGENCY_STOP, Action.RESET): DeviceState.IDLE,
}


# --------------------------------------------------------------------------- #
#  Safety Constraints (beyond transition legality)                             #
# --------------------------------------------------------------------------- #

# 物理安全约束：某些参数值超出安全范围时，对应动作必须被拦截。
_SAFETY_CONSTRAINTS: dict[Action, list[tuple[str, str, float]]] = {
    # (遥测指标, 比较运算, 阈值)  —— 满足条件时拒绝动作
    Action.START_HEATING: [
        ("temperature", ">=", 80.0),   # 温度 ≥ 80 ℃ 时禁止继续加热
    ],
    Action.MOVE_ROBOTIC_ARM: [
        ("pressure", "<", 50.0),       # 气压 < 50 kPa 时禁止移动机械臂（低气压环境不稳定）
    ],
}

_OP_MAP = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">":  lambda v, t: v > t,
    "<":  lambda v, t: v < t,
    "==": lambda v, t: v == t,
}


# --------------------------------------------------------------------------- #
#  ShadowFSM                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class ShadowFSM:
    """影子设备有限状态机。

    Parameters
    ----------
    lab_id : str
        所属实验柜标识，用于日志前缀。
    """

    lab_id: str
    _state: DeviceState = field(default=DeviceState.IDLE, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # ---- 只读属性 --------------------------------------------------------- #

    @property
    def current_state(self) -> DeviceState:
        return self._state

    # ---- 核心方法 --------------------------------------------------------- #

    async def validate_and_transition(
        self,
        action: Action,
        params: dict[str, Any] | None = None,
        telemetry_snapshot: dict[str, Any] | None = None,
    ) -> DeviceState:
        """校验动作的合法性和物理安全性，通过则执行状态迁移。

        Parameters
        ----------
        action : Action
            要执行的操作。
        params : dict, optional
            操作参数（如 ``{"target_angle": 45.0}``）。
        telemetry_snapshot : dict, optional
            当前遥测快照，用于安全约束校验。

        Returns
        -------
        DeviceState
            迁移后的新状态。

        Raises
        ------
        SecurityGuardrailException
            当动作违背 FSM 转移合法性或物理安全约束时。
        """
        params = params or {}
        telemetry_snapshot = telemetry_snapshot or {}

        async with self._lock:
            # 1) 转移合法性校验
            transition_key = (self._state, action)
            if transition_key not in _TRANSITION_TABLE:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] FSM 拒绝: 状态 {self._state.name} "
                    f"下不允许执行 {action.name}"
                )

            # 2) 物理安全约束校验
            constraints = _SAFETY_CONSTRAINTS.get(action, [])
            for metric_key, op_str, threshold in constraints:
                current_value = telemetry_snapshot.get(metric_key)
                if current_value is not None:
                    comparator = _OP_MAP[op_str]
                    if comparator(current_value, threshold):
                        raise SecurityGuardrailException(
                            f"[{self.lab_id}] 安全拦截: {metric_key}={current_value} "
                            f"{op_str} {threshold}，禁止执行 {action.name}"
                        )

            # 3) 执行状态迁移
            old_state = self._state
            self._state = _TRANSITION_TABLE[transition_key]
            logger.info(
                "[%s] FSM 状态迁移: %s -[%s]-> %s",
                self.lab_id, old_state.name, action.name, self._state.name,
            )
            return self._state
