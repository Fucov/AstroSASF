"""
Shared MCP Tools — 可跨多个实验舱使用的公共工具。

这些工具定义了基础传感器读写、通用设备控制等操作。
"""

from sasf.middleware.mcp_registry import MCPToolContext


async def read_sensor(ctx: MCPToolContext, channel: int) -> dict:
    """读取传感器通道值（通用）。"""
    telemetry = await ctx.bus.snapshot()
    key = f"sensor_channel_{channel}"
    value = telemetry.get(key, 20.0 + channel * 0.5)
    return {"status": "success", "channel": channel, "value": value}


async def emergency_stop(ctx: MCPToolContext) -> dict:
    """紧急停止所有设备（公共安全操作）。"""
    # 停止所有活跃子系统
    await ctx.engine.set_subsystem_state("heater", "IDLE")
    await ctx.engine.set_subsystem_state("vacuum", "IDLE")
    await ctx.engine.set_subsystem_state("pump", "OFF")
    return {"status": "success", "action": "emergency_stop", "message": "All systems stopped"}


async def get_system_status(ctx: MCPToolContext) -> dict:
    """获取当前系统状态快照。"""
    states = ctx.engine.current_states
    telemetry = await ctx.bus.snapshot()
    return {
        "status": "success",
        "fsm_states": states,
        "telemetry": telemetry,
    }
