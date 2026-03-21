#!/usr/bin/env python3
"""
AstroSASF · Space Station Demo (V6.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
全逻辑验证演示：

1. 多领域知识库 (流体实验 / 生物培养 / 材料合成)
2. Edge-RAG 动态 SOP 检索 (BM25-lite, 零第三方依赖)
3. 优先级抢占调度 + InterlockEngine + Guard + Macro
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# --- 新增代码：手动挂载根目录到 sys.path ---
# 获取当前文件 (examples/space_station_demo.py) 的父目录的父目录 (AstroSASF)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# ---------------------------------------
from sasf.core.config_loader import load_config
from sasf.core.orchestrator import Orchestrator, TaskPriority
from sasf.middleware.mcp_registry import MCPToolContext, MCPToolRegistry
from sasf.physics.interlock_engine import InterlockEngine
from sasf.physics.telemetry_bus import TelemetryBus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


# ============================================================================ #
#  业务层: 初始遥测                                                              #
# ============================================================================ #

INITIAL_TELEMETRY: dict[str, Any] = {
    "temperature": 22.0,
    "pressure": 101.3,
    "robotic_arm_angle": 0.0,
    "vacuum_pump_active": False,
    "heater_active": False,
    "coolant_flow_rate": 0.0,
    "nutrient_injected": False,
    "laser_active": False,
}


# ============================================================================ #
#  业务层: MCP Tools 注册 (含 Guard + 多领域工具)                                  #
# ============================================================================ #


def register_tools(
    registry: MCPToolRegistry,
    engine: InterlockEngine,
    bus: TelemetryBus,
) -> None:
    """注册太空实验柜 MCP Tools。"""

    @registry.mcp_tool(
        require_states={"thermal": "IDLE"},
        telemetry_rules=["temperature < 80"],
    )
    async def set_temperature(ctx: MCPToolContext, target: float) -> dict[str, Any]:
        """设置舱内温度目标值（℃）"""
        target = float(target)
        current = await ctx.bus.read("temperature")
        action = "HEATING" if target > current else "COOLING"

        await ctx.engine.set_subsystem_state("thermal", action)
        await ctx.bus.write("temperature", target)
        await ctx.engine.set_subsystem_state("thermal", "IDLE")

        return {
            "skill": "set_temperature",
            "status": "success",
            "detail": f"温度已设置为 {target}℃",
        }

    @registry.mcp_tool(
        require_states={"arm": "IDLE"},
        forbid_states={"vacuum": "ACTIVE"},
        telemetry_rules=["pressure >= 50"],
    )
    async def move_robotic_arm(
        ctx: MCPToolContext, target_angle: float
    ) -> dict[str, Any]:
        """移动机械臂至指定角度（°）"""
        target_angle = float(target_angle)

        await ctx.engine.set_subsystem_state("arm", "MOVING")
        await ctx.bus.write("robotic_arm_angle", target_angle)
        await ctx.engine.set_subsystem_state("arm", "IDLE")

        return {
            "skill": "move_robotic_arm",
            "status": "success",
            "detail": f"机械臂已移至 {target_angle}°",
        }

    @registry.mcp_tool(require_states={"arm": "IDLE"})
    async def toggle_vacuum_pump(ctx: MCPToolContext, activate: bool) -> dict[str, Any]:
        """切换真空泵开关"""
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        new_state = "ACTIVE" if activate else "IDLE"
        await ctx.engine.set_subsystem_state("vacuum", new_state)
        await ctx.bus.write("vacuum_pump_active", activate)

        return {
            "skill": "toggle_vacuum_pump",
            "status": "success",
            "detail": f"真空泵已{'激活' if activate else '关闭'}",
        }

    # ── 多领域通用工具 ── #

    @registry.mcp_tool
    async def inject_nutrient(
        ctx: MCPToolContext, volume_ml: float = 10.0
    ) -> dict[str, Any]:
        """注入培养基营养液（mL）"""
        volume_ml = float(volume_ml)
        await ctx.bus.write("nutrient_injected", True)

        return {
            "skill": "inject_nutrient",
            "status": "success",
            "detail": f"已注入 {volume_ml}mL 营养液",
        }

    @registry.mcp_tool
    async def turn_on_laser(
        ctx: MCPToolContext, activate: bool = True
    ) -> dict[str, Any]:
        """控制激光烧结设备开关"""
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        await ctx.bus.write("laser_active", activate)

        return {
            "skill": "turn_on_laser",
            "status": "success",
            "detail": f"激光设备已{'开启' if activate else '关闭'}",
        }

    logger.info(
        "[%s] 物理设备层: 已注册 %d 个 MCP Tools (含多领域)",
        registry.lab_id,
        registry.count,
    )


# ============================================================================ #
#  业务层: Macro 绑定                                                            #
# ============================================================================ #


def register_macros(registry: MCPToolRegistry) -> None:
    """注册参数预绑定的 Macro。"""
    registry.bind_macro(
        "heat_to_50",
        "set_temperature",
        {"target": 50.0},
        description="快速加热到 50℃",
    )
    registry.bind_macro(
        "heat_to_37",
        "set_temperature",
        {"target": 37.0},
        description="加热到 37℃（细胞培养温度）",
    )
    registry.bind_macro(
        "arm_to_observation",
        "move_robotic_arm",
        {"target_angle": 45.0},
        description="机械臂移至观测位（45°）",
    )
    registry.bind_macro(
        "arm_to_dock",
        "move_robotic_arm",
        {"target_angle": 90.0},
        description="机械臂移至对接位（90°）",
    )
    registry.bind_macro(
        "arm_home",
        "move_robotic_arm",
        {"target_angle": 0.0},
        description="机械臂归零",
    )
    registry.bind_macro(
        "vacuum_on",
        "toggle_vacuum_pump",
        {"activate": True},
        description="打开真空泵",
    )
    registry.bind_macro(
        "vacuum_off",
        "toggle_vacuum_pump",
        {"activate": False},
        description="关闭真空泵",
    )


# ============================================================================ #
#  主函数 — V6.0 多领域 Edge-RAG + 优先级调度                                      #
# ============================================================================ #


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ███████╗████████╗██████╗  ██████╗                 ║
║    ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗                ║
║    ███████║███████╗   ██║   ██████╔╝██║   ██║                ║
║    ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║                ║
║    ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝                ║
║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝                 ║
║                                                              ║
║    S A S F  v6.0  ·  Lightweight Edge-RAG                    ║
║    BM25-lite (Zero Deps) + Multi-Domain Knowledge            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── 1) 配置 ── #
    config = load_config(PROJECT_ROOT / "config.yaml")

    # ── 2) InterlockEngine ── #
    engine = InterlockEngine.from_yaml(
        path=PROJECT_ROOT / "fsm_rules.yaml",
        lab_id="Lab-Alpha",
    )

    # ── 3) 创建调度器 ── #
    scheduler = Orchestrator(config=config, max_workers=1)

    # ── 4) 创建实验柜 ── #
    env = scheduler.spawn_laboratory(
        lab_id="Lab-Alpha",
        engine=engine,
        tool_registrar=register_tools,
        macro_registrar=register_macros,
        initial_telemetry=INITIAL_TELEMETRY,
    )

    # ── 5) 展示初始化信息 ── #
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📖 动态字典 (含 Macro + 多领域 Tools)              ║")
    logger.info("╚" + "═" * 60 + "╝")
    for word, tid in env.codec_dictionary.items():
        logger.info("    0x%02X  ←  '%s'", tid, word)
    logger.info("    共 %d 个映射词条", len(env.codec_dictionary))

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📚 已加载 OpenAI Skills (多领域知识库)              ║")
    logger.info("╚" + "═" * 60 + "╝")
    for s in env.loaded_skills:
        logger.info("    ✅ %-25s — %s", s["name"], s["description"])

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          🔗 已注册 Macro                                    ║")
    logger.info("╚" + "═" * 60 + "╝")
    for name, info in env.registry.get_macros().items():
        logger.info("    🔗 %-20s → %s(%s)", name, info["target"], info["preset"])

    # ── 6) 多领域任务 — Edge-RAG 动态上下文切换验证 ── #
    tasks = [
        ("请执行流体实验的环境准备工作", TaskPriority.NORMAL),
        ("开始进行太空生物细胞培养", TaskPriority.NORMAL),
        ("执行微重力合金材料合成", TaskPriority.NORMAL),
    ]

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     🧪 多领域 Edge-RAG 验证 (3 个不同领域任务)              ║")
    logger.info("╚" + "═" * 60 + "╝")
    for desc, prio in tasks:
        logger.info("    📋 [%s] %s", prio.name, desc)

    # ── 7) 启动调度器 + 提交任务 ── #
    await scheduler.start()

    for desc, prio in tasks:
        await scheduler.submit_task(
            lab_id="Lab-Alpha",
            description=desc,
            priority=prio,
        )

    # ── 8) 等待完成 ── #
    await scheduler._queue.join()
    completed = await scheduler.shutdown()

    # ── 9) 结果汇总 ── #
    final_telemetry = await env.get_telemetry()

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     📊 AstroSASF V6.0 结果汇总                             ║")
    logger.info("╚" + "═" * 60 + "╝")
    logger.info("")

    logger.info("  ┌─── Lab-Alpha ────────────────────────────────────")
    logger.info("  │  🔒 正交状态       : %s", env.engine_states)
    logger.info("  │")
    logger.info("  │  📡 遥测终态:")
    for k, v in final_telemetry.items():
        logger.info("  │     %-25s = %s", k, v)
    logger.info("  │")

    cs = env.codec_stats
    logger.info("  │  🗜️  编解码器:")
    logger.info("  │     编码次数      : %s", cs.get("encode_count", 0))
    logger.info(
        "  │     词条数        : %s (含 Macro + 多领域)", cs.get("dictionary_size", 0)
    )
    logger.info("  │")

    logger.info("  │  📋 调度统计:")
    logger.info("  │     总完成任务    : %d", len(completed))
    for t in completed:
        logger.info(
            "  │     [%s] %-8s %s → %s",
            t.task_id[:8],
            t.priority.name,
            t.description[:25],
            t.status,
        )
    logger.info("  │")

    logger.info("  │  📨 A2A: %s", env.a2a_stats)
    logger.info("  └──────────────────────────────────────────────────")

    logger.info("")
    logger.info("AstroSASF V6.0 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
