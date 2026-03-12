"""
AstroSASF · Middleware · MCPGateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP 技能分发中心 —— 将底层硬件操作抽象为标准的 Model Context Protocol
工具接口（Skills）。

工作流：Operator JSON 请求 → FSM 校验 → 修改 TelemetryBus → 返回结果。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from physics.shadow_fsm import Action, SecurityGuardrailException, ShadowFSM
from physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Skill 类型别名                                                              #
# --------------------------------------------------------------------------- #

# 一个 Skill 的签名：接收 (gateway, params) → 返回结果字典
SkillHandler = Callable[["MCPGateway", dict[str, Any]], Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------------- #
#  Skill Registry — 装饰器注册机制                                              #
# --------------------------------------------------------------------------- #

# 全局注册表：skill_name → (handler, description)
_SKILL_REGISTRY: dict[str, tuple[SkillHandler, str]] = {}


def skill(name: str, description: str = "") -> Callable[[SkillHandler], SkillHandler]:
    """装饰器：将一个 async 函数注册为 MCP Skill。

    Example
    -------
    >>> @skill("set_temperature", "设置舱内温度")
    ... async def _set_temperature(gw: MCPGateway, params: dict) -> dict:
    ...     ...
    """
    def decorator(func: SkillHandler) -> SkillHandler:
        _SKILL_REGISTRY[name] = (func, description)
        return func
    return decorator


# --------------------------------------------------------------------------- #
#  Built-in Skills                                                             #
# --------------------------------------------------------------------------- #

@skill("set_temperature", "设置舱内温度目标值（℃）")
async def _set_temperature(gw: MCPGateway, params: dict[str, Any]) -> dict[str, Any]:
    target: float = params["target"]
    current_temp = await gw.bus.read("temperature")

    if target > current_temp:
        action = Action.START_HEATING
    else:
        action = Action.START_COOLING

    snapshot = await gw.bus.snapshot()
    new_state = await gw.fsm.validate_and_transition(
        action=action, params=params, telemetry_snapshot=snapshot,
    )
    await gw.bus.write("temperature", target)

    # 加热 / 冷却完成后回到 IDLE
    stop_action = Action.STOP_HEATING if action == Action.START_HEATING else Action.STOP_COOLING
    await gw.fsm.validate_and_transition(action=stop_action)

    return {
        "skill": "set_temperature",
        "status": "success",
        "detail": f"温度已设置为 {target}℃",
        "fsm_state": new_state.name,
    }


@skill("move_robotic_arm", "移动机械臂至指定角度（°）")
async def _move_robotic_arm(gw: MCPGateway, params: dict[str, Any]) -> dict[str, Any]:
    target_angle: float = params["target_angle"]

    snapshot = await gw.bus.snapshot()
    new_state = await gw.fsm.validate_and_transition(
        action=Action.MOVE_ROBOTIC_ARM, params=params, telemetry_snapshot=snapshot,
    )
    await gw.bus.write("robotic_arm_angle", target_angle)

    # 移动完成 → 回到 IDLE
    await gw.fsm.validate_and_transition(action=Action.STOP_ROBOTIC_ARM)

    return {
        "skill": "move_robotic_arm",
        "status": "success",
        "detail": f"机械臂已移动至 {target_angle}°",
        "fsm_state": new_state.name,
    }


@skill("toggle_vacuum_pump", "切换真空泵开关")
async def _toggle_vacuum_pump(gw: MCPGateway, params: dict[str, Any]) -> dict[str, Any]:
    activate: bool = params.get("activate", True)

    action = Action.ACTIVATE_VACUUM if activate else Action.DEACTIVATE_VACUUM
    snapshot = await gw.bus.snapshot()
    new_state = await gw.fsm.validate_and_transition(
        action=action, params=params, telemetry_snapshot=snapshot,
    )
    await gw.bus.write("vacuum_pump_active", activate)

    verb = "启动" if activate else "关闭"
    return {
        "skill": "toggle_vacuum_pump",
        "status": "success",
        "detail": f"真空泵已{verb}",
        "fsm_state": new_state.name,
    }


# --------------------------------------------------------------------------- #
#  MCPGateway                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class MCPGateway:
    """MCP 技能分发中心。

    每个 LaboratoryEnvironment 持有一个独立实例，绑定该环境的
    FSM 与 TelemetryBus。
    """

    lab_id: str
    fsm: ShadowFSM
    bus: TelemetryBus
    _skills: dict[str, tuple[SkillHandler, str]] = field(init=False)

    def __post_init__(self) -> None:
        # 从全局注册表拷贝一份，避免运行时交叉污染
        self._skills = dict(_SKILL_REGISTRY)

    # ---- 技能目录 --------------------------------------------------------- #

    def list_skills(self) -> list[dict[str, str]]:
        """返回当前可用 Skills 的清单（供 LLM function_calling 使用）。"""
        return [
            {"name": name, "description": desc}
            for name, (_, desc) in self._skills.items()
        ]

    # ---- 技能调用入口 ----------------------------------------------------- #

    async def invoke_skill(
        self,
        skill_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """调用指定 Skill，统一错误封装。

        Returns
        -------
        dict
            包含 ``status`` 字段（"success" | "error"）的响应。
        """
        if skill_name not in self._skills:
            return {
                "skill": skill_name,
                "status": "error",
                "detail": f"未知 Skill: {skill_name}",
            }

        handler, _ = self._skills[skill_name]
        try:
            result = await handler(self, params)
            logger.info("[%s] Skill '%s' 执行成功", self.lab_id, skill_name)
            return result
        except SecurityGuardrailException as exc:
            logger.warning(
                "[%s] Skill '%s' 被 FSM 拦截: %s",
                self.lab_id, skill_name, exc,
            )
            return {
                "skill": skill_name,
                "status": "error",
                "detail": str(exc),
            }
        except Exception as exc:
            logger.exception(
                "[%s] Skill '%s' 执行异常",
                self.lab_id, skill_name,
            )
            return {
                "skill": skill_name,
                "status": "error",
                "detail": f"内部异常: {exc!r}",
            }
