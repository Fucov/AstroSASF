"""
AstroSASF · Physics · ShadowFSM (V4.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
影子设备有限状态机 —— 拦截 LLM 幻觉指令的**绝对护栏**。

V4.2：使用 ``@registry.mcp_tool`` 声明式装饰器注册 MCP Tools，
自动反射 Type Hints 生成 JSON Schema，零手写 Schema。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sasf.middleware.mcp_registry import MCPToolContext, MCPToolRegistry
    from sasf.physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Exception                                                                   #
# --------------------------------------------------------------------------- #

class SecurityGuardrailException(Exception):
    """FSM 安全护栏异常。"""


# --------------------------------------------------------------------------- #
#  Device States                                                               #
# --------------------------------------------------------------------------- #

class DeviceState(Enum):
    IDLE = auto()
    HEATING = auto()
    COOLING = auto()
    ROBOTIC_ARM_MOVING = auto()
    VACUUM_ACTIVE = auto()
    EMERGENCY_STOP = auto()


class Action(Enum):
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

_TRANSITION_TABLE: dict[tuple[DeviceState, Action], DeviceState] = {
    (DeviceState.IDLE, Action.START_HEATING):    DeviceState.HEATING,
    (DeviceState.IDLE, Action.START_COOLING):    DeviceState.COOLING,
    (DeviceState.IDLE, Action.MOVE_ROBOTIC_ARM): DeviceState.ROBOTIC_ARM_MOVING,
    (DeviceState.IDLE, Action.ACTIVATE_VACUUM):  DeviceState.VACUUM_ACTIVE,
    (DeviceState.IDLE, Action.EMERGENCY_STOP):   DeviceState.EMERGENCY_STOP,
    (DeviceState.HEATING, Action.STOP_HEATING):  DeviceState.IDLE,
    (DeviceState.HEATING, Action.EMERGENCY_STOP): DeviceState.EMERGENCY_STOP,
    (DeviceState.COOLING, Action.STOP_COOLING):  DeviceState.IDLE,
    (DeviceState.COOLING, Action.EMERGENCY_STOP): DeviceState.EMERGENCY_STOP,
    (DeviceState.ROBOTIC_ARM_MOVING, Action.STOP_ROBOTIC_ARM): DeviceState.IDLE,
    (DeviceState.ROBOTIC_ARM_MOVING, Action.EMERGENCY_STOP):   DeviceState.EMERGENCY_STOP,
    (DeviceState.VACUUM_ACTIVE, Action.DEACTIVATE_VACUUM): DeviceState.IDLE,
    (DeviceState.VACUUM_ACTIVE, Action.EMERGENCY_STOP):    DeviceState.EMERGENCY_STOP,
    (DeviceState.EMERGENCY_STOP, Action.RESET): DeviceState.IDLE,
}

# --------------------------------------------------------------------------- #
#  Safety Constraints                                                          #
# --------------------------------------------------------------------------- #

_SAFETY_CONSTRAINTS: dict[Action, list[tuple[str, str, float]]] = {
    Action.START_HEATING: [("temperature", ">=", 80.0)],
    Action.MOVE_ROBOTIC_ARM: [("pressure", "<", 50.0)],
}

_OP_MAP = {
    ">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t,
    ">":  lambda v, t: v > t,  "<":  lambda v, t: v < t,
    "==": lambda v, t: v == t,
}


# --------------------------------------------------------------------------- #
#  ShadowFSM                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class ShadowFSM:
    """影子设备有限状态机。"""

    lab_id: str
    _state: DeviceState = field(default=DeviceState.IDLE, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def current_state(self) -> DeviceState:
        return self._state

    async def validate_and_transition(
        self,
        action: Action,
        params: dict[str, Any] | None = None,
        telemetry_snapshot: dict[str, Any] | None = None,
    ) -> DeviceState:
        params = params or {}
        telemetry_snapshot = telemetry_snapshot or {}

        async with self._lock:
            transition_key = (self._state, action)
            if transition_key not in _TRANSITION_TABLE:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] FSM 拒绝: 状态 {self._state.name} "
                    f"下不允许执行 {action.name}"
                )

            constraints = _SAFETY_CONSTRAINTS.get(action, [])
            for metric_key, op_str, threshold in constraints:
                current_value = telemetry_snapshot.get(metric_key)
                if current_value is not None:
                    if _OP_MAP[op_str](current_value, threshold):
                        raise SecurityGuardrailException(
                            f"[{self.lab_id}] 安全拦截: {metric_key}={current_value} "
                            f"{op_str} {threshold}，禁止执行 {action.name}"
                        )

            old_state = self._state
            self._state = _TRANSITION_TABLE[transition_key]
            logger.info(
                "[%s] FSM 状态迁移: %s -[%s]-> %s",
                self.lab_id, old_state.name, action.name, self._state.name,
            )
            return self._state


# --------------------------------------------------------------------------- #
#  MCP Tool Registration — 使用 @registry.mcp_tool 声明式注册                    #
# --------------------------------------------------------------------------- #

def register_default_tools(
    registry: MCPToolRegistry,
    fsm: ShadowFSM,
    bus: TelemetryBus,
) -> None:
    """物理设备层通过 ``@registry.mcp_tool`` 装饰器注册底层硬件操作。

    V4.2 优势：装饰器自动反射 Type Hints，生成 JSON Schema。
    物理开发者只需写函数签名和 Docstring，零手工 Schema。
    """
    from sasf.middleware.mcp_registry import MCPToolContext

    @registry.mcp_tool
    async def set_temperature(ctx: MCPToolContext, target: float) -> dict[str, Any]:
        """设置舱内温度目标值（℃）"""
        target = float(target)
        current_temp = await ctx.bus.read("temperature")
        action = Action.START_HEATING if target > current_temp else Action.START_COOLING

        snapshot = await ctx.bus.snapshot()
        new_state = await ctx.fsm.validate_and_transition(
            action=action, params={"target": target}, telemetry_snapshot=snapshot,
        )
        await ctx.bus.write("temperature", target)

        stop = Action.STOP_HEATING if action == Action.START_HEATING else Action.STOP_COOLING
        await ctx.fsm.validate_and_transition(action=stop)

        return {
            "skill": "set_temperature",
            "status": "success",
            "detail": f"温度已设置为 {target}℃",
            "fsm_state": new_state.name,
        }

    @registry.mcp_tool
    async def move_robotic_arm(ctx: MCPToolContext, target_angle: float) -> dict[str, Any]:
        """移动机械臂至指定角度（°）"""
        target_angle = float(target_angle)
        snapshot = await ctx.bus.snapshot()
        new_state = await ctx.fsm.validate_and_transition(
            action=Action.MOVE_ROBOTIC_ARM,
            params={"target_angle": target_angle},
            telemetry_snapshot=snapshot,
        )
        await ctx.bus.write("robotic_arm_angle", target_angle)
        await ctx.fsm.validate_and_transition(action=Action.STOP_ROBOTIC_ARM)

        return {
            "skill": "move_robotic_arm",
            "status": "success",
            "detail": f"机械臂已移动至 {target_angle}°",
            "fsm_state": new_state.name,
        }

    @registry.mcp_tool
    async def toggle_vacuum_pump(ctx: MCPToolContext, activate: bool) -> dict[str, Any]:
        """切换真空泵开关"""
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")
        
        action = Action.ACTIVATE_VACUUM if activate else Action.DEACTIVATE_VACUUM
        snapshot = await ctx.bus.snapshot()
        new_state = await ctx.fsm.validate_and_transition(
            action=action, params={"activate": activate}, telemetry_snapshot=snapshot,
        )
        await ctx.bus.write("vacuum_pump_active", activate)

        verb = "启动" if activate else "关闭"
        return {
            "skill": "toggle_vacuum_pump",
            "status": "success",
            "detail": f"真空泵已{verb}",
            "fsm_state": new_state.name,
        }

    logger.info(
        "[%s] 物理设备层: 已注册 %d 个 MCP Tools (自动 Schema)",
        fsm.lab_id, registry.count,
    )
