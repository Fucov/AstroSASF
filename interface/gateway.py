"""
AstroSASF · Interface · Space-MCP Gateway
========================================
Space-MCP 协议转换网关 —— Agent 与底层物理世界的唯一通信桥梁。

V7.2 整合了以下组件：
- InterlockEngine（联锁校验）
- TelemetryBus（遥测读写）
- MCPToolRegistry（工具调用）
- SpaceMCPCodec（编解码）
- VirtualSpaceWire（传输模拟）
- A2ARouter（A2A 消息路由）

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from scheduler.a2a_protocol import A2AIntent, A2ARouter

logger = logging.getLogger(__name__)


@dataclass
class SpaceMCPGateway:
    """Space-MCP 协议转换网关（内核接口模块）。

    完整链路：
    1. 下行链路: JSON → Space-MCP Binary → VirtualSpaceWire
    2. 联锁检查: InterlockEngine.check_interlocks()
    3. 执行: Registry.invoke()
    4. 上行链路: Binary Response → VirtualSpaceWire → JSON
    """

    lab_id: str
    registry: Any
    engine: Any
    bus: Any
    codec: Any
    space_wire: Any
    a2a_router: A2ARouter
    device_runtime: Any = None  # V8.0: 物理设备统一调用层
    metrics: Any = None  # V8.0: 指标采集器

    def list_tools(self) -> list[dict[str, Any]]:
        return self.registry.list_tools()

    async def invoke_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """调用指定 MCP Tool —— 经过完整的 Space-MCP 协议转换链路。"""

        if not self.registry.has_tool(tool_name):
            return {
                "skill": tool_name,
                "status": "error",
                "detail": f"MCPToolRegistry 中未注册: {tool_name}",
            }

        self.a2a_router.route(
            sender="Operator",
            receiver="Gateway",
            intent=A2AIntent.SKILL_INVOCATION,
            payload={"skill": tool_name, "params": params},
        )

        # ── Phase 1: 下行链路 ── #
        request_json = {"skill": tool_name, "params": params}
        json_text = json.dumps(request_json, ensure_ascii=False)
        json_bytes = len(json_text.encode("utf-8"))

        binary_frame = self.codec.encode(request_json)
        binary_bytes = len(binary_frame)
        ratio = self.codec.calculate_compression_ratio(json_bytes, binary_bytes)

        logger.info("")
        logger.info(
            "[%s] ┌─── Space-MCP 下行链路 ───────────────────────────",
            self.lab_id,
        )
        logger.info("[%s] │ 📦 JSON 原文: %s", self.lab_id, json_text)
        logger.info(
            "[%s] │ 📐 JSON %d B → Space-MCP %d B │ 压缩率: %.1f%%",
            self.lab_id, json_bytes, binary_bytes, ratio,
        )
        logger.info("[%s] │ 🔢 Hex: %s", self.lab_id, binary_frame.hex(" "))

        wire_data = await self.space_wire.transmit(binary_frame)
        decoded_request = self.codec.decode(wire_data)
        logger.info("[%s] │ ✅ 解码: %s", self.lab_id, decoded_request)

        # ── Phase 2: 联锁检查 + 执行 ── #
        from labs.interlock_engine import SecurityGuardrailException
        from labs.mcp_registry import MCPToolContext
        context = MCPToolContext(
            engine=self.engine,
            bus=self.bus,
            lab_id=self.lab_id,
            device_runtime=self.device_runtime,
            metrics=self.metrics,
        )

        try:
            telemetry = await self.bus.snapshot()
            await self.engine.check_interlocks(
                tool_name=tool_name, telemetry=telemetry,
            )
            result = await self.registry.invoke(
                name=tool_name,
                params=decoded_request["params"],
                context=context,
            )
        except SecurityGuardrailException as exc:
            logger.warning("[%s] │ 🛡️  安全拦截: %s", self.lab_id, exc)
            result = {"skill": tool_name, "status": "error", "detail": str(exc)}
        except Exception as exc:
            logger.exception("[%s] │ ❌ Tool '%s' 异常", self.lab_id, tool_name)
            result = {"skill": tool_name, "status": "error", "detail": f"内部异常: {exc!r}"}

        self.a2a_router.route(
            sender="Gateway",
            receiver="Operator",
            intent=A2AIntent.SKILL_RESULT,
            payload=result,
        )

        # ── Phase 3: 上行链路 ── #
        resp_json_text = json.dumps(result, ensure_ascii=False)
        resp_json_bytes = len(resp_json_text.encode("utf-8"))
        resp_binary = self.codec.encode_response(result)
        resp_binary_bytes = len(resp_binary)
        resp_ratio = self.codec.calculate_compression_ratio(
            resp_json_bytes, resp_binary_bytes,
        )

        logger.info(
            "[%s] │ 📡 响应: JSON %d B → Binary %d B │ 压缩率: %.1f%%",
            self.lab_id, resp_json_bytes, resp_binary_bytes, resp_ratio,
        )

        resp_wire = await self.space_wire.transmit(resp_binary)
        decoded_resp = self.codec.decode_response(resp_wire)

        logger.info(
            "[%s] └─── 链路完成 ─ 状态: %s",
            self.lab_id, decoded_resp.get("status", "N/A"),
        )
        logger.info("")

        return decoded_resp


__all__ = ["SpaceMCPGateway"]
