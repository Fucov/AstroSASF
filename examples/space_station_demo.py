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
from sasf.core.orchestrator import DAGOrchestrator, TaskPriority
from sasf.core.models import DAGTaskGraph, DAGNode, NodeStatus
from sasf.middleware.mcp_registry import MCPToolContext, MCPToolRegistry
from sasf.physics.interlock_engine import InterlockEngine
from sasf.physics.telemetry_bus import TelemetryBus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


# ============================================================================ #
#  业务层: 初始遥测                                                              #
# ============================================================================ #

INITIAL_TELEMETRY: dict[str, Any] = {
    # 原有遥测
    "temperature": 22.0,
    "pressure": 101.3,
    "robotic_arm_angle": 0.0,
    "vacuum_pump_active": False,
    "heater_active": False,
    "coolant_flow_rate": 0.0,
    "nutrient_injected": False,
    "laser_active": False,
    # V7.1 新增遥测
    "co2_level": 450.0,          # CO2 浓度 (ppm)
    "smoke_level": 0.0,          # 烟雾浓度 (0-1)
    "flame_detected": 0,         # 火焰检测 (0/1)
    "soil_moisture": 35.0,       # 土壤湿度 (%)
    "grow_light_active": False,  # 生长灯状态
    "ventilation_active": False, # 通风系统状态
}


# ============================================================================ #
#  业务层: MCP Tools 注册 (含 Guard + 多领域工具)                                  #
# ============================================================================ #


# ============================================================================ #
#  V7.1 物理模拟参数                                                               #
# ============================================================================ #

PHYSICS_PARAMS = {
    "robotic_arm": {
        "max_speed_deg_per_sec": 15.0,
        "max_angle": 180.0,
    },
    "temperature": {
        "rate_deg_per_sec": 2.0,
    },
    "nutrient_pump": {
        "flow_rate_ml_per_sec": 20.0,
    },
    "vacuum_pump": {
        "fixed_duration_sec": 4.0,
    },
}

# ============================================================================ #
#  业务层: MCP Tools 注册 (含物理模拟 + 紧急制动)                                  #
# ============================================================================ #


def register_tools(
    registry: MCPToolRegistry,
    engine: InterlockEngine,
    bus: TelemetryBus,
) -> None:
    """注册太空实验柜 MCP Tools (V7.1 物理模拟层)。"""

    # ── 温度控制 ── #
    @registry.mcp_tool(
        require_states={"thermal": "IDLE"},
        telemetry_rules=["temperature < 80"],
    )
    async def set_temperature(ctx: MCPToolContext, target: float) -> dict[str, Any]:
        """设置舱内温度目标值（℃）。

        V7.1 物理模拟：
        - 升/降温速率: 2.0℃/秒
        - 耗时计算: abs(target - current) / 2.0 秒
        - 优雅响应 asyncio.CancelledError，支持硬件级紧急制动
        """
        target = float(target)
        rate = PHYSICS_PARAMS["temperature"]["rate_deg_per_sec"]

        current = await ctx.bus.read("temperature")
        delta = abs(target - current)
        duration = delta / rate
        action = "HEATING" if target > current else "COOLING"

        await ctx.engine.set_subsystem_state("thermal", action)
        logger.info("[%s] 物理模拟: %s 温度 %.1f→%.1f℃ (耗时 %.2fs)",
                    ctx.lab_id, action, current, target, duration)

        interrupted = False
        try:
            await asyncio.sleep(duration)
            await ctx.bus.write("temperature", target)
        except asyncio.CancelledError:
            interrupted = True
            # 计算被打断瞬间的实际温度
            import time as time_module
            elapsed = time_module.monotonic()
            # 估算已完成的温度变化比例
            # 注意：asyncio.sleep 被取消时无法精确知道已过时间
            # 这里用 start_time 记录，finally 中重新读取
        finally:
            await ctx.engine.set_subsystem_state("thermal", "IDLE")
            if interrupted:
                # 读取当前温度（停在半途）
                actual = await ctx.bus.read("temperature")
                logger.warning(
                    "[%s] ⚠️ 温度调节被紧急制动打断！停在 %.1f℃ (目标 %.1f℃)",
                    ctx.lab_id, actual, target
                )

        return {
            "skill": "set_temperature",
            "status": "success" if not interrupted else "preempted",
            "detail": f"温度已设置为 {target}℃",
            "interrupted": interrupted,
        }

    # ── 机械臂控制 ── #
    @registry.mcp_tool(
        require_states={"arm": "IDLE"},
        forbid_states={"vacuum": "ACTIVE"},
        telemetry_rules=["pressure >= 50", "smoke_level <= 0.1", "flame_detected == 0"],
    )
    async def move_robotic_arm(
        ctx: MCPToolContext, target_angle: float
    ) -> dict[str, Any]:
        """移动机械臂至指定角度（°）。

        V7.1 物理模拟：
        - 速度限制: 15.0 度/秒
        - 耗时计算: abs(target - current) / 15.0 秒
        - 优雅响应 asyncio.CancelledError，记录中断位置
        """
        import time as time_module

        target_angle = float(target_angle)
        max_speed = PHYSICS_PARAMS["robotic_arm"]["max_speed_deg_per_sec"]

        current_angle = await ctx.bus.read("robotic_arm_angle")
        delta = abs(target_angle - current_angle)
        duration = delta / max_speed

        await ctx.engine.set_subsystem_state("arm", "MOVING")
        logger.info("[%s] 物理模拟: 机械臂 %.1f→%.1f° (耗时 %.2fs @ %.1f°/s)",
                    ctx.lab_id, current_angle, target_angle, duration, max_speed)

        interrupted = False
        start_time = time_module.monotonic()
        try:
            await asyncio.sleep(duration)
            await ctx.bus.write("robotic_arm_angle", target_angle)
        except asyncio.CancelledError:
            interrupted = True
            elapsed = time_module.monotonic() - start_time
            # 计算被打断瞬间的实际位置
            actual_angle = current_angle + (target_angle - current_angle) * (elapsed / duration)
            actual_angle = max(0.0, min(180.0, actual_angle))  # 边界限制
            # ★ 关键：记录物理操作的真实中断位置
            await ctx.bus.write("robotic_arm_angle", actual_angle)
        finally:
            await ctx.engine.set_subsystem_state("arm", "IDLE")
            if interrupted:
                final_angle = await ctx.bus.read("robotic_arm_angle")
                logger.warning(
                    "[%s] ⚠️ 机械臂运动被紧急制动打断！停在 %.1f° (目标 %.1f°)",
                    ctx.lab_id, final_angle, target_angle
                )

        return {
            "skill": "move_robotic_arm",
            "status": "success" if not interrupted else "preempted",
            "detail": f"机械臂已移至 {target_angle}°",
            "interrupted": interrupted,
        }

    # ── 真空泵控制 ── #
    @registry.mcp_tool(require_states={"arm": "IDLE"})
    async def toggle_vacuum_pump(ctx: MCPToolContext, activate: bool) -> dict[str, Any]:
        """切换真空泵开关。

        V7.1 物理模拟：
        - 抽真空或恢复常压，固定耗时: 4.0 秒
        - 优雅响应 asyncio.CancelledError
        """
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        duration = PHYSICS_PARAMS["vacuum_pump"]["fixed_duration_sec"]
        new_state = "ACTIVE" if activate else "IDLE"
        action_desc = "抽真空" if activate else "恢复常压"

        logger.info("[%s] 物理模拟: 真空泵 %s (耗时 %.1fs)",
                    ctx.lab_id, action_desc, duration)

        interrupted = False
        try:
            await ctx.engine.set_subsystem_state("vacuum", new_state)
            await asyncio.sleep(duration)
            await ctx.bus.write("vacuum_pump_active", activate)
        except asyncio.CancelledError:
            interrupted = True
        finally:
            if interrupted:
                # 记录真空泵状态停在半途
                await ctx.bus.write("vacuum_pump_active", activate)
                logger.warning(
                    "[%s] ⚠️ 真空泵操作被紧急制动打断！状态: %s",
                    ctx.lab_id, new_state
                )

        return {
            "skill": "toggle_vacuum_pump",
            "status": "success" if not interrupted else "preempted",
            "detail": f"真空泵已{'激活' if activate else '关闭'}",
            "interrupted": interrupted,
        }

    # ── 营养液注入 ── #
    @registry.mcp_tool(
        forbid_states={"greenhouse": "LIGHTING"},
        telemetry_rules=["soil_moisture < 80.0"],
    )
    async def inject_nutrient(
        ctx: MCPToolContext, volume_ml: float = 10.0
    ) -> dict[str, Any]:
        """注入培养基营养液（mL）。

        V7.1 物理模拟：
        - 泵速: 20.0 mL/秒
        - 耗时计算: volume_ml / 20.0 秒
        - 优雅响应 asyncio.CancelledError
        """
        volume_ml = float(volume_ml)
        flow_rate = PHYSICS_PARAMS["nutrient_pump"]["flow_rate_ml_per_sec"]
        duration = volume_ml / flow_rate

        logger.info("[%s] 物理模拟: 注入营养液 %.1fmL (耗时 %.2fs @ %.1fmL/s)",
                    ctx.lab_id, volume_ml, duration, flow_rate)

        interrupted = False
        try:
            await asyncio.sleep(duration)
            await ctx.bus.write("nutrient_injected", True)
            # 更新土壤湿度
            current_moisture = await ctx.bus.read("soil_moisture")
            new_moisture = min(100.0, current_moisture + (volume_ml / 100.0) * 10)
            await ctx.bus.write("soil_moisture", new_moisture)
        except asyncio.CancelledError:
            interrupted = True
            # 记录已注入的部分
            import time as time_module
            elapsed = getattr(inject_nutrient, '_last_elapsed', 0)
            injected = min(volume_ml, elapsed * flow_rate)
            logger.warning(
                "[%s] ⚠️ 营养液注入被紧急制动打断！已注入 %.1fmL / %.1fmL",
                ctx.lab_id, injected, volume_ml
            )

        return {
            "skill": "inject_nutrient",
            "status": "success" if not interrupted else "preempted",
            "detail": f"已注入 {volume_ml}mL 营养液",
            "interrupted": interrupted,
        }

    # ── 激光控制 ── #
    @registry.mcp_tool(
        telemetry_rules=["co2_level < 1000.0", "flame_detected == 0"],
    )
    async def turn_on_laser(
        ctx: MCPToolContext, activate: bool = True
    ) -> dict[str, Any]:
        """控制激光烧结设备开关。

        V7.1 联锁规则：
        - CO2 >= 1000.0ppm 时禁止燃烧实验
        - 火焰检测到时禁止使用激光
        """
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        await ctx.bus.write("laser_active", activate)

        return {
            "skill": "turn_on_laser",
            "status": "success",
            "detail": f"激光设备已{'开启' if activate else '关闭'}",
        }

    # ── 温室灯光控制 ── #
    @registry.mcp_tool(
        require_states={"greenhouse": "IDLE"},
    )
    async def set_greenhouse_lighting(
        ctx: MCPToolContext, activate: bool = True
    ) -> dict[str, Any]:
        """控制温室植物生长灯开关。

        V7.1 物理模拟：
        - 开灯/关灯固定耗时: 1.0 秒
        """
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        if activate:
            await ctx.engine.set_subsystem_state("greenhouse", "LIGHTING")
            await asyncio.sleep(1.0)
            await ctx.bus.write("grow_light_active", True)
        else:
            await ctx.engine.set_subsystem_state("greenhouse", "IDLE")
            await asyncio.sleep(1.0)
            await ctx.bus.write("grow_light_active", False)

        return {
            "skill": "set_greenhouse_lighting",
            "status": "success",
            "detail": f"生长灯已{'开启' if activate else '关闭'}",
        }

    # ── 生命支持系统控制 ── #
    @registry.mcp_tool(
        forbid_states={"safety": "ALERT"},
    )
    async def toggle_ventilation(
        ctx: MCPToolContext, activate: bool = True
    ) -> dict[str, Any]:
        """控制生命支持系统通风/除碳。

        V7.1 物理模拟：
        - 启动/关闭固定耗时: 2.0 秒
        - 通风运行时实时降低 CO2 浓度
        """
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        if activate:
            await ctx.engine.set_subsystem_state("life_support", "VENTILATING")
            logger.info("[%s] 物理模拟: 通风系统启动 (降低 CO2)", ctx.lab_id)
            try:
                # 通风过程中逐步降低 CO2
                for _ in range(4):
                    await asyncio.sleep(0.5)
                    current_co2 = await ctx.bus.read("co2_level")
                    new_co2 = max(400.0, current_co2 - 50.0)
                    await ctx.bus.write("co2_level", new_co2)
            except asyncio.CancelledError:
                logger.warning("[%s] ⚠️ 通风系统被紧急制动打断！", ctx.lab_id)
                raise
            await ctx.bus.write("ventilation_active", True)
        else:
            await ctx.engine.set_subsystem_state("life_support", "IDLE")
            await asyncio.sleep(2.0)
            await ctx.bus.write("ventilation_active", False)

        return {
            "skill": "toggle_ventilation",
            "status": "success",
            "detail": f"通风系统已{'启动' if activate else '关闭'}",
        }

    logger.info(
        "[%s] 物理设备层: 已注册 %d 个 MCP Tools (含 V7.1 物理模拟)",
        registry.lab_id,
        registry.count,
    )


# ============================================================================ #
#  业务层: Macro 绑定                                                            #
# ============================================================================ #


def register_macros(registry: MCPToolRegistry) -> None:
    """注册参数预绑定的 Macro。"""
    # 温度控制宏
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
        "cool_to_20",
        "set_temperature",
        {"target": 20.0},
        description="降温到 20℃（室温）",
    )

    # 机械臂宏
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

    # 真空泵宏
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

    # V7.1 温室宏
    registry.bind_macro(
        "start_lighting",
        "set_greenhouse_lighting",
        {"activate": True},
        description="开启温室生长灯",
    )
    registry.bind_macro(
        "stop_lighting",
        "set_greenhouse_lighting",
        {"activate": False},
        description="关闭温室生长灯",
    )

    # V7.1 生命支持宏
    registry.bind_macro(
        "start_ventilation",
        "toggle_ventilation",
        {"activate": True},
        description="启动通风除碳",
    )
    registry.bind_macro(
        "stop_ventilation",
        "toggle_ventilation",
        {"activate": False},
        description="关闭通风系统",
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
║    S A S F  v7.0  ·  Dual-Track DAG Scheduling               ║
║    Theory Agent (Planner) + Practice Agent (Worker)          ║
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
    scheduler = DAGOrchestrator(config=config, max_workers=1)

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
    logger.info("║          📖 动态字典 (含 Macro + 多领域 Tools)             ║")
    logger.info("╚" + "═" * 60 + "╝")
    for word, tid in env.codec_dictionary.items():
        logger.info("    0x%02X  ←  '%s'", tid, word)
    logger.info("    共 %d 个映射词条", len(env.codec_dictionary))

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📚 已加载 OpenAI Skills (多领域知识库)             ║")
    logger.info("╚" + "═" * 60 + "╝")
    for s in env.loaded_skills:
        logger.info("    ✅ %-25s — %s", s["name"], s["description"])

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          🔗 已注册 Macro                                   ║")
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
    logger.info("║     🧪 多领域 Edge-RAG 验证 (3 个不同领域任务)             ║")
    logger.info("╚" + "═" * 60 + "╝")
    for desc, prio in tasks:
        logger.info("    📋 [%s] %s", prio.name, desc)

    # ── 7) 构建 DAG 图 + 提交任务 ── #
    import uuid
    dag_graph = DAGTaskGraph(
        graph_id=str(uuid.uuid4()),
        name="Space-Station-Multi-Domain"
    )
    for i, (desc, prio) in enumerate(tasks):
        node = DAGNode(
            node_id=f"task-{i}",
            skill_name=desc,
            lab_id="Lab-Alpha",
            priority=prio,
        )
        dag_graph.add_node(node)

    await scheduler.submit_dag(dag_graph)

    # ── 8) DAG 执行 (等待完成) ── #
    result = await scheduler.run_dag(dag_graph)

    # ── 9) 结果汇总 ── #
    final_telemetry = await env.get_telemetry()

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     📊 AstroSASF V7.0 DAG 执行结果                           ║")
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

    logger.info("  │  📋 DAG 执行统计:")
    logger.info("  │     图 ID         : %s", result.graph_id)
    logger.info("  │     状态          : %s", result.status)
    logger.info("  │     总节点数      : %d", result.total_nodes)
    logger.info("  │     完成节点数    : %d", result.completed_nodes)
    logger.info("  │     失败节点数    : %d", result.failed_nodes)
    logger.info("  │     总耗时        : %.2fs", result.total_time)
    logger.info("  │     执行层级      : %d", result.execution_levels)
    logger.info("  │")

    logger.info("  │  📨 A2A: %s", env.a2a_stats)
    logger.info("  └──────────────────────────────────────────────────")

    logger.info("")
    logger.info("AstroSASF V7.0 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
