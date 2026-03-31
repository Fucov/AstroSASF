"""
Lab-Bio 专属 MCP Tools — 生物实验舱专用工具。

基于 FSM 规则定义前置/禁止条件，确保实验安全。
"""

from sasf.middleware.mcp_registry import MCPToolContext


async def control_heater(ctx: MCPToolContext, temperature: float, duration: int) -> dict:
    """控制加热器设置温度和持续时间（毫秒）。"""
    if temperature < 20 or temperature > 80:
        return {"status": "error", "detail": f"温度 {temperature}°C 超范围 (20-80°C)"}

    await ctx.engine.set_subsystem_state("heater", "HEATING")
    await ctx.bus.write("target_temperature", temperature)
    await ctx.bus.write("heater_active", True)

    return {
        "status": "success",
        "temperature": temperature,
        "duration_ms": duration,
        "message": f"加热器已设置至 {temperature}°C，持续 {duration}ms"
    }


async def move_robotic_arm(ctx: MCPToolContext, position: str) -> dict:
    """移动机械臂到指定位置。"""
    positions = ["home", "left", "right", "center", "storage"]
    if position not in positions:
        return {"status": "error", "detail": f"未知位置: {position}"}

    await ctx.engine.set_subsystem_state("arm", "MOVING")
    await ctx.bus.write("arm_position", position)
    await asyncio.sleep(0.5)  # 模拟移动
    await ctx.engine.set_subsystem_state("arm", "IDLE")

    return {"status": "success", "position": position}


async def start_uv_sterilization(ctx: MCPToolContext, duration: int = 300) -> dict:
    """启动 UV 灭菌程序。"""
    await ctx.engine.set_subsystem_state("uv_lamp", "STERILIZING")
    await ctx.bus.write("uv_active", True)
    await ctx.bus.write("uv_duration", duration)

    return {
        "status": "success",
        "action": "uv_sterilization",
        "duration_seconds": duration,
        "message": f"UV 灭菌已开始，持续 {duration} 秒"
    }


async def open_chamber_door(ctx: MCPToolContext) -> dict:
    """打开培养舱门。"""
    return {"status": "error", "detail": "请先关闭 UV 灭菌程序"}
