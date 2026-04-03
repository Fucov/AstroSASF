"""
AstroSASF · Demo · Custom Tools — 流体实验舱
============================================
流体实验舱的 MCP 工具实现（演示用）。
这些工具模拟了真实的物理设备交互。
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


# ── 真空泵 ────────────────────────────────────────────────────────────────── #

async def toggle_vacuum_pump(ctx: MCPToolContext, activate: bool) -> dict:
    """切换真空泵开关状态（流体实验舱版本）。"""
    await asyncio.sleep(0.05)
    new_state = "FORWARD" if activate else "IDLE"
    await ctx.engine.set_subsystem_state("pump", new_state)
    await ctx.bus.write("vacuum_pump", new_state)
    return {
        "status": "ok",
        "detail": f"真空泵状态: {new_state}",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 阀门控制 ──────────────────────────────────────────────────────────────── #

async def control_valve(ctx: MCPToolContext, position: str) -> dict:
    """控制阀门位置：CLOSED | OPEN_A | OPEN_B | MIXING。"""
    await asyncio.sleep(0.05)
    await ctx.engine.set_subsystem_state("valve", position)
    await ctx.bus.write("valve_position", position)
    return {
        "status": "ok",
        "detail": f"阀门位置: {position}",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 压力传感器 ───────────────────────────────────────────────────────────── #

async def read_pressure_sensor(ctx: MCPToolContext, channel: int = 1) -> dict:
    """读取压力传感器通道值（单位：kPa）。"""
    await asyncio.sleep(0.05)
    # 模拟传感器读数
    pressure = round(random.uniform(80.0, 120.0), 2)
    await ctx.bus.write(f"pressure_ch{channel}", pressure)
    return {
        "status": "ok",
        "detail": f"通道 {channel} 压力: {pressure} kPa",
        "value": pressure,
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 机械臂 ───────────────────────────────────────────────────────────────── #

async def move_robotic_arm(ctx: MCPToolContext, target_position: str) -> dict:
    """移动机械臂到指定位置（单位：度）。"""
    await asyncio.sleep(0.15)
    # 解析角度
    angle = float(target_position) if target_position.replace(".", "").isdigit() else 0.0
    await ctx.bus.write("arm_angle", angle)
    return {
        "status": "ok",
        "detail": f"机械臂移动至 {target_position}°",
        "telemetry": await ctx.bus.snapshot(),
    }


# ── 离心机 ───────────────────────────────────────────────────────────────── #

async def control_centrifuge(ctx: MCPToolContext, action: str, rpm: int = 3000) -> dict:
    """控制离心机：START | STOP | BRAKE。"""
    await asyncio.sleep(0.1)
    state_map = {"START": "SPINNING", "STOP": "STOPPED", "BRAKE": "BRAKING"}
    new_state = state_map.get(action, "STOPPED")
    await ctx.engine.set_subsystem_state("centrifuge", new_state)
    await ctx.bus.write("centrifuge_rpm", rpm if new_state == "SPINNING" else 0)
    return {
        "status": "ok",
        "detail": f"离心机: {action} → {new_state}",
        "telemetry": await ctx.bus.snapshot(),
    }
