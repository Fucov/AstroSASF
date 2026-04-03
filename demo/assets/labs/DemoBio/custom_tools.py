"""
AstroSASF · Demo · Custom Tools — 生物实验舱
============================================
生物实验舱的 MCP 工具实现（演示用）。
这些工具模拟了真实的生物培养设备交互。
Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import random
from labs.mcp_registry import MCPToolContext


# ── 温度控制 ────────────────────────────────────────────────────────────────── #

async def set_temperature(ctx: MCPToolContext, temperature: float) -> dict:
    """设置培养舱温度（单位：℃）。"""
    await asyncio.sleep(0.1)
    await ctx.bus.write("temperature", temperature)
    return {
        "status": "ok",
        "detail": f"温度已设置为 {temperature}℃",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 加热器控制 ────────────────────────────────────────────────────────────── #

async def control_heater(ctx: MCPToolContext, temperature: float, duration: int) -> dict:
    """控制加热器：设置目标温度和持续时间（秒）。"""
    await asyncio.sleep(0.1)
    await ctx.engine.set_subsystem_state("heater", "HEATING")
    await ctx.bus.write("temperature", temperature)
    return {
        "status": "ok",
        "detail": f"加热器: 目标 {temperature}℃，持续 {duration}s",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 真空泵 ────────────────────────────────────────────────────────────────── #

async def toggle_vacuum_pump(ctx: MCPToolContext, activate: bool) -> dict:
    """切换真空泵开关状态。"""
    await asyncio.sleep(0.05)
    new_state = "ACTIVE" if activate else "IDLE"
    await ctx.engine.set_subsystem_state("vacuum", new_state)
    await ctx.bus.write("vacuum_pump", new_state)
    return {
        "status": "ok",
        "detail": f"真空泵状态: {new_state}",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 营养液注入 ───────────────────────────────────────────────────────────── #

async def inject_nutrient(ctx: MCPToolContext, volume: float) -> dict:
    """注入营养液（单位：ml）。"""
    await asyncio.sleep(0.1)
    await ctx.bus.write("nutrient_level", min(100.0, 80.0 + volume))
    return {
        "status": "ok",
        "detail": f"已注入营养液 {volume}ml",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 传感器读取 ───────────────────────────────────────────────────────────── #

async def read_sensor(ctx: MCPToolContext, channel: int = 1) -> dict:
    """读取传感器通道值。"""
    await asyncio.sleep(0.05)
    value = round(random.uniform(20.0, 40.0), 2)
    await ctx.bus.write(f"sensor_ch{channel}", value)
    return {
        "status": "ok",
        "detail": f"通道 {channel} 传感器读数: {value}",
        "value": value,
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 机械臂 ───────────────────────────────────────────────────────────────── #

async def move_robotic_arm(ctx: MCPToolContext, target_position: str) -> dict:
    """移动机械臂到指定位置（度或预设标签 HOME/90/45）。"""
    await asyncio.sleep(0.15)
    angle_map = {"HOME": 0, "90": 90, "45": 45, "30": 30}
    angle = angle_map.get(target_position, float(target_position) if target_position.replace(".", "").isdigit() else 0.0)
    ctx.engine.set_subsystem_state("arm", "MOVING")
    await ctx.bus.write("arm_angle", angle)
    await ctx.engine.set_subsystem_state("arm", "IDLE")
    return {
        "status": "ok",
        "detail": f"机械臂移动至 {target_position} ({angle}°)",
        "telemetry": await ctx.bus.snapshot(),
    }
