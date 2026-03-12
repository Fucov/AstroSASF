"""
AstroSASF · Middleware · SpaceMCPGateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Space-MCP 协议转换网关 —— Agent 与底层物理世界的**唯一通信桥梁**。

V2 全链路数据流
~~~~~~~~~~~~~~~
::

  Operator Agent ──JSON──▶ Gateway.invoke_skill()
       │                      │
       │  1. 记录 JSON 字节长度
       │  2. Codec.encode() → bytearray
       │  3. 记录 Binary 长度 + 打印压缩率
       │  4. VirtualSpaceWire.transmit()
       │  5. Codec.decode() → 还原 JSON
       │  6. FSM.validate_and_transition()
       │  7. TelemetryBus.write()
       │  8. 响应 encode → SpaceWire → decode
       ◀──JSON──────────────────┘

对上层暴露与 V1 完全兼容的 ``invoke_skill(skill_name, params)`` 接口，
内部透明地完成 JSON ↔ Binary ↔ SpaceWire 全套协议转换。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from middleware.codec import SpaceMCPCodec
from middleware.virtual_bus import VirtualSpaceWire
from physics.shadow_fsm import Action, SecurityGuardrailException, ShadowFSM
from physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Skill 类型别名                                                              #
# --------------------------------------------------------------------------- #

SkillHandler = Callable[["SpaceMCPGateway", dict[str, Any]], Awaitable[dict[str, Any]]]

# 全局技能注册表
_SKILL_REGISTRY: dict[str, tuple[SkillHandler, str]] = {}


def skill(name: str, description: str = "") -> Callable[[SkillHandler], SkillHandler]:
    """装饰器：注册 MCP Skill。"""
    def decorator(func: SkillHandler) -> SkillHandler:
        _SKILL_REGISTRY[name] = (func, description)
        return func
    return decorator


# --------------------------------------------------------------------------- #
#  Built-in Skills（保留 V1 逻辑，绑定新 Gateway 类型）                          #
# --------------------------------------------------------------------------- #

@skill("set_temperature", "设置舱内温度目标值（℃）")
async def _set_temperature(gw: SpaceMCPGateway, params: dict[str, Any]) -> dict[str, Any]:
    target: float = params["target"]
    current_temp = await gw.bus.read("temperature")
    action = Action.START_HEATING if target > current_temp else Action.START_COOLING

    snapshot = await gw.bus.snapshot()
    new_state = await gw.fsm.validate_and_transition(
        action=action, params=params, telemetry_snapshot=snapshot,
    )
    await gw.bus.write("temperature", target)

    stop = Action.STOP_HEATING if action == Action.START_HEATING else Action.STOP_COOLING
    await gw.fsm.validate_and_transition(action=stop)

    return {
        "skill": "set_temperature",
        "status": "success",
        "detail": f"温度已设置为 {target}℃",
        "fsm_state": new_state.name,
    }


@skill("move_robotic_arm", "移动机械臂至指定角度（°）")
async def _move_robotic_arm(gw: SpaceMCPGateway, params: dict[str, Any]) -> dict[str, Any]:
    target_angle: float = params["target_angle"]
    snapshot = await gw.bus.snapshot()
    new_state = await gw.fsm.validate_and_transition(
        action=Action.MOVE_ROBOTIC_ARM, params=params, telemetry_snapshot=snapshot,
    )
    await gw.bus.write("robotic_arm_angle", target_angle)
    await gw.fsm.validate_and_transition(action=Action.STOP_ROBOTIC_ARM)

    return {
        "skill": "move_robotic_arm",
        "status": "success",
        "detail": f"机械臂已移动至 {target_angle}°",
        "fsm_state": new_state.name,
    }


@skill("toggle_vacuum_pump", "切换真空泵开关")
async def _toggle_vacuum_pump(gw: SpaceMCPGateway, params: dict[str, Any]) -> dict[str, Any]:
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
#  SpaceMCPGateway                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class SpaceMCPGateway:
    """Space-MCP 协议转换网关。

    对上层暴露标准的 ``invoke_skill(skill_name, params)`` 异步 JSON
    接口，在内部透明完成：

    1. JSON → ``SpaceMCPCodec.encode()`` → 二进制帧
    2. 二进制帧 → ``VirtualSpaceWire.transmit()`` → 模拟物理传输
    3. 二进制帧 → ``SpaceMCPCodec.decode()`` → 还原 JSON
    4. FSM 校验 → TelemetryBus 写入
    5. 响应反向编码 → SpaceWire → 解码 → 返回 Agent
    """

    lab_id: str
    fsm: ShadowFSM
    bus: TelemetryBus
    codec: SpaceMCPCodec
    space_wire: VirtualSpaceWire
    _skills: dict[str, tuple[SkillHandler, str]] = field(init=False)

    def __post_init__(self) -> None:
        self._skills = dict(_SKILL_REGISTRY)

    # ---- 技能目录 --------------------------------------------------------- #

    def list_skills(self) -> list[dict[str, str]]:
        """返回可用 Skills 清单。"""
        return [
            {"name": name, "description": desc}
            for name, (_, desc) in self._skills.items()
        ]

    # ---- 技能调用入口（含 Space-MCP 全链路） -------------------------------- #

    async def invoke_skill(
        self,
        skill_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """调用指定 Skill —— 经过完整的 Space-MCP 协议转换链路。

        Returns
        -------
        dict
            包含 ``status``（"success" | "error"）的 JSON 响应。
        """
        if skill_name not in self._skills:
            return {
                "skill": skill_name,
                "status": "error",
                "detail": f"未知 Skill: {skill_name}",
            }

        # ============================================================ #
        #  Phase 1 · 下行链路：Agent JSON → Binary → SpaceWire         #
        # ============================================================ #

        request_json = {"skill": skill_name, "params": params}
        json_text = json.dumps(request_json, ensure_ascii=False)
        json_bytes = len(json_text.encode("utf-8"))

        # 编码
        binary_frame = self.codec.encode(request_json)
        binary_bytes = len(binary_frame)
        ratio = SpaceMCPCodec.calculate_compression_ratio(json_bytes, binary_bytes)

        # 🔥 高亮压缩日志
        logger.info("")
        logger.info(
            "[%s] ┌─── Space-MCP 下行链路 ───────────────────────────",
            self.lab_id,
        )
        logger.info(
            "[%s] │ 📦 JSON 原文: %s", self.lab_id, json_text,
        )
        logger.info(
            "[%s] │ 📐 JSON %d Bytes → Space-MCP %d Bytes │ 🗜️  压缩率: %.1f%%",
            self.lab_id, json_bytes, binary_bytes, ratio,
        )
        logger.info(
            "[%s] │ 🔢 Binary Hex: %s",
            self.lab_id, binary_frame.hex(" "),
        )

        # SpaceWire 传输
        wire_data = await self.space_wire.transmit(binary_frame)

        # 解码（FSM 端接收）
        decoded_request = self.codec.decode(wire_data)
        logger.info(
            "[%s] │ ✅ FSM 端解码还原: %s", self.lab_id, decoded_request,
        )

        # ============================================================ #
        #  Phase 2 · 执行：FSM 校验 → TelemetryBus 写入                 #
        # ============================================================ #

        handler, _ = self._skills[skill_name]
        try:
            result = await handler(self, decoded_request["params"])
        except SecurityGuardrailException as exc:
            logger.warning(
                "[%s] │ 🛡️  FSM 安全拦截: %s", self.lab_id, exc,
            )
            result = {
                "skill": skill_name,
                "status": "error",
                "detail": str(exc),
            }
        except Exception as exc:
            logger.exception(
                "[%s] │ ❌ Skill '%s' 执行异常", self.lab_id, skill_name,
            )
            result = {
                "skill": skill_name,
                "status": "error",
                "detail": f"内部异常: {exc!r}",
            }

        # ============================================================ #
        #  Phase 3 · 上行链路：响应 JSON → Binary → SpaceWire → JSON    #
        # ============================================================ #

        resp_json_text = json.dumps(result, ensure_ascii=False)
        resp_json_bytes = len(resp_json_text.encode("utf-8"))

        resp_binary = self.codec.encode_response(result)
        resp_binary_bytes = len(resp_binary)
        resp_ratio = SpaceMCPCodec.calculate_compression_ratio(
            resp_json_bytes, resp_binary_bytes,
        )

        logger.info(
            "[%s] │ 📡 响应: JSON %d B → Binary %d B │ 🗜️  压缩率: %.1f%%",
            self.lab_id, resp_json_bytes, resp_binary_bytes, resp_ratio,
        )

        # 响应经 SpaceWire 回传
        resp_wire = await self.space_wire.transmit(resp_binary)
        decoded_resp = self.codec.decode_response(resp_wire)

        logger.info(
            "[%s] └─── 链路完成 ─ 状态: %s ─────────────────────────",
            self.lab_id, decoded_resp.get("status", "N/A"),
        )
        logger.info("")

        return decoded_resp
