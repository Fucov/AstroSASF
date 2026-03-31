"""
Lab-Fluid 专属 MCP Tools — 流体实验舱专用工具。
"""

import asyncio

from sasf.middleware.mcp_registry import MCPToolContext


async def control_pump(ctx: MCPToolContext, direction: str, flow_rate: int = 50) -> dict:
    """控制泵的方向和流速。"""
    if direction not in ["FORWARD", "REVERSE", "OFF"]:
        return {"status": "error", "detail": f"无效方向: {direction}"}

    if flow_rate < 0 or flow_rate > 100:
        return {"status": "error", "detail": "流速必须在 0-100 之间"}

    state = "OFF" if direction == "OFF" else direction
    await ctx.engine.set_subsystem_state("pump", state)
    await ctx.bus.write("pump_direction", direction)
    await ctx.bus.write("flow_rate", flow_rate)

    return {
        "status": "success",
        "direction": direction,
        "flow_rate": flow_rate,
    }


async def control_valve(ctx: MCPToolContext, position: str) -> dict:
    """控制阀门位置。"""
    valid_positions = ["CLOSED", "OPEN_A", "OPEN_B", "MIXING"]
    if position not in valid_positions:
        return {"status": "error", "detail": f"无效阀门位置: {position}"}

    await ctx.engine.set_subsystem_state("valve", position)
    await ctx.bus.write("valve_position", position)

    return {"status": "success", "valve_position": position}


async def start_centrifuge(ctx: MCPToolContext, rpm: int = 3000) -> dict:
    """启动离心机。"""
    if rpm < 0 or rpm > 10000:
        return {"status": "error", "detail": "RPM 必须在 0-10000 之间"}

    await ctx.engine.set_subsystem_state("centrifuge", "SPINNING")
    await ctx.bus.write("centrifuge_rpm", rpm)

    return {
        "status": "success",
        "action": "centrifuge_started",
        "rpm": rpm,
    }


async def stop_centrifuge(ctx: MCPToolContext) -> dict:
    """停止离心机（带刹车）。"""
    await ctx.engine.set_subsystem_state("centrifuge", "BRAKING")
    await asyncio.sleep(2)  # 模拟刹车时间
    await ctx.engine.set_subsystem_state("centrifuge", "STOPPED")
    await ctx.bus.write("centrifuge_rpm", 0)

    return {"status": "success", "action": "centrifuge_stopped"}
