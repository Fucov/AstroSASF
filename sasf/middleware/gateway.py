"""
AstroSASF · Middleware · SpaceMCPGateway (V4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Space-MCP 协议转换网关 —— Agent 与底层物理世界的**唯一通信桥梁**。

V4 核心变化：网关**不再硬编码任何 Skill**。所有 Skill 由物理设备层
通过 ``SkillRegistry`` 注册，网关仅做纯粹的协议透传。

全链路数据流
~~~~~~~~~~~~~~~
::

  Agent ──JSON──▶ Gateway.invoke_skill()
       │           │
       │  1. A2A 消息记录
       │  2. Codec.encode() → 二进制帧
       │  3. VirtualSpaceWire.transmit()
       │  4. Codec.decode() → 还原 JSON
       │  5. SkillRegistry.invoke() → FSM 校验 + Bus 写入
       │  6. 响应 encode → SpaceWire → decode
       ◀──JSON──────┘
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sasf.middleware.a2a_protocol import A2AIntent, A2ARouter
from sasf.middleware.codec import SpaceMCPCodec
from sasf.middleware.skill_registry import SkillContext, SkillRegistry
from sasf.middleware.virtual_bus import VirtualSpaceWire
from sasf.physics.shadow_fsm import SecurityGuardrailException, ShadowFSM
from sasf.physics.telemetry_bus import TelemetryBus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  SpaceMCPGateway                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class SpaceMCPGateway:
    """Space-MCP 协议转换网关 (V4)。

    不包含任何业务逻辑 —— 通过 ``SkillRegistry`` 查找并委托执行。
    """

    lab_id: str
    fsm: ShadowFSM
    bus: TelemetryBus
    codec: SpaceMCPCodec
    space_wire: VirtualSpaceWire
    registry: SkillRegistry
    a2a_router: A2ARouter

    def list_skills(self) -> list[dict[str, Any]]:
        """返回可用 Skills 清单（委托给 Registry）。"""
        return self.registry.list_skills()

    async def invoke_skill(
        self,
        skill_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """调用指定 Skill —— 经过完整的 Space-MCP 协议转换链路。"""

        if not self.registry.has_skill(skill_name):
            return {
                "skill": skill_name,
                "status": "error",
                "detail": f"SkillRegistry 中未注册: {skill_name}",
            }

        # ── A2A: 记录调用请求 ── #
        self.a2a_router.route(
            sender="Operator",
            receiver="Gateway",
            intent=A2AIntent.SKILL_INVOCATION,
            payload={"skill": skill_name, "params": params},
        )

        # ============================================================ #
        #  Phase 1 · 下行链路：Agent JSON → Binary → SpaceWire         #
        # ============================================================ #

        request_json = {"skill": skill_name, "params": params}
        json_text = json.dumps(request_json, ensure_ascii=False)
        json_bytes = len(json_text.encode("utf-8"))

        binary_frame = self.codec.encode(request_json)
        binary_bytes = len(binary_frame)
        ratio = SpaceMCPCodec.calculate_compression_ratio(json_bytes, binary_bytes)

        logger.info("")
        logger.info(
            "[%s] ┌─── Space-MCP 下行链路 ───────────────────────────",
            self.lab_id,
        )
        logger.info("[%s] │ 📦 JSON 原文: %s", self.lab_id, json_text)
        logger.info(
            "[%s] │ 📐 JSON %d B → Space-MCP %d B │ 🗜️  压缩率: %.1f%%",
            self.lab_id, json_bytes, binary_bytes, ratio,
        )
        logger.info("[%s] │ 🔢 Hex: %s", self.lab_id, binary_frame.hex(" "))

        wire_data = await self.space_wire.transmit(binary_frame)
        decoded_request = self.codec.decode(wire_data)
        logger.info("[%s] │ ✅ FSM 端解码: %s", self.lab_id, decoded_request)

        # ============================================================ #
        #  Phase 2 · 执行：SkillRegistry.invoke()                      #
        # ============================================================ #

        context = SkillContext(fsm=self.fsm, bus=self.bus, lab_id=self.lab_id)
        try:
            result = await self.registry.invoke(
                name=skill_name,
                params=decoded_request["params"],
                context=context,
            )
        except SecurityGuardrailException as exc:
            logger.warning("[%s] │ 🛡️  FSM 安全拦截: %s", self.lab_id, exc)
            result = {"skill": skill_name, "status": "error", "detail": str(exc)}
        except Exception as exc:
            logger.exception("[%s] │ ❌ Skill '%s' 异常", self.lab_id, skill_name)
            result = {"skill": skill_name, "status": "error", "detail": f"内部异常: {exc!r}"}

        # ── A2A: 记录执行结果 ── #
        self.a2a_router.route(
            sender="Gateway",
            receiver="Operator",
            intent=A2AIntent.SKILL_RESULT,
            payload=result,
        )

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

        resp_wire = await self.space_wire.transmit(resp_binary)
        decoded_resp = self.codec.decode_response(resp_wire)

        logger.info(
            "[%s] └─── 链路完成 ─ 状态: %s ─────────────────────────",
            self.lab_id, decoded_resp.get("status", "N/A"),
        )
        logger.info("")

        return decoded_resp
