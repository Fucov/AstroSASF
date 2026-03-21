#!/usr/bin/env python3
"""
AstroSASF · Space Station Demo (V5.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
全逻辑验证演示 — 优先级调度 + 抢占式调度内核。

场景设计：
1. 提交 NORMAL 任务（常规流体实验）
2. 延迟 2 秒后提交 CRITICAL 任务（紧急安全复位）
3. 终端日志展示：挂起 → 抢占 → 恢复 完整过程
"""

from __future__ import annotations

import asyncio
import json
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
}


# ============================================================================ #
#  业务层: MCP Tools 注册 (含 Guard)                                              #
# ============================================================================ #


def register_tools(
    registry: MCPToolRegistry,
    engine: InterlockEngine,
    bus: TelemetryBus,
) -> None:
    """注册太空实验柜 MCP Tools（含声明式 Guard）。"""

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

    @registry.mcp_tool(
        require_states={"arm": "IDLE"},
    )
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

    logger.info(
        "[%s] 物理设备层: 已注册 %d 个 MCP Tools",
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
#  HITL 循环 (保留，供单任务模式使用)                                               #
# ============================================================================ #


async def hitl_loop(compiled, initial_state, config, lab_id):
    state = await compiled.ainvoke(initial_state, config)
    snapshot = compiled.get_state(config)

    while snapshot.next:
        step = state.get("current_step")
        if step and isinstance(step, dict):
            logger.info("")
            logger.info("[%s] ⏸️  HITL — 即将执行:", lab_id)
            logger.info("[%s]    Tool  : %s", lab_id, step.get("skill"))
            logger.info(
                "[%s]    Params: %s",
                lab_id,
                json.dumps(step.get("params", {}), ensure_ascii=False),
            )

        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: input(
                f"\n{'─' * 60}\n"
                f"🛡️ HITL | {step.get('skill', '?')}({step.get('params', {})})\n"
                f"  [y/回车] 批准  |  [n] 中止  |  [JSON] 修正\n"
                f"{'─' * 60}\n>>> "
            ),
        )
        raw = raw.strip()

        if raw.lower() in ("n", "no"):
            return {
                "status": "aborted_by_user",
                "execution_log": list(state.get("execution_log") or []),
            }

        if raw and raw not in ("", "y", "yes"):
            corrected = None
            try:
                corrected = json.loads(raw)
            except json.JSONDecodeError:
                pass
            if corrected is None:
                try:
                    import ast as _ast

                    corrected = _ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    logger.warning("[%s] 无法解析输入，按原计划继续", lab_id)

            if isinstance(corrected, dict):
                if "skill" in corrected and "params" in corrected:
                    new_step = corrected
                else:
                    new_step = dict(step) if step else {}
                    new_step.setdefault("params", {}).update(corrected)
                compiled.update_state(config, {"current_step": new_step})
                logger.info("[%s] ✏️  参数修正: %s", lab_id, new_step)

        state = await compiled.ainvoke(None, config)
        snapshot = compiled.get_state(config)

    final = state.get("final_result")
    return (
        final
        if isinstance(final, dict)
        else {
            "status": "completed",
            "execution_log": list(state.get("execution_log") or []),
        }
    )


# ============================================================================ #
#  主函数 — V5.1 优先级调度 + 抢占演示                                             #
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
║    S A S F  v5.1  ·  Priority Scheduling Kernel              ║
║    Preemptive Task Scheduler + Interlock Engine              ║
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
    scheduler = Orchestrator(config=config, max_workers=2)

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
    logger.info("║          📖 动态字典 (自动握手 — 含 Macro)                  ║")
    logger.info("╚" + "═" * 60 + "╝")
    for word, tid in env.codec_dictionary.items():
        logger.info("    0x%02X  ←  '%s'", tid, word)

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          🔗 已注册 Macro                                    ║")
    logger.info("╚" + "═" * 60 + "╝")
    for name, info in env.registry.get_macros().items():
        logger.info("    🔗 %s → %s(%s)", name, info["target"], info["preset"])

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          🔒 联锁规则                                        ║")
    logger.info("╚" + "═" * 60 + "╝")
    for rule in engine.interlocks:
        logger.info("    [%s] %s → %s", rule.scope or "*", rule.condition, rule.message)

    # ── 6) 启动调度器 ── #
    await scheduler.start()

    # ── 7) 提交 NORMAL 任务（常规实验） ── #
    _normal_task = await scheduler.submit_task(
        lab_id="Lab-Alpha",
        description="执行流体实验的环境准备和样品装载流程",
        priority=TaskPriority.NORMAL,
    )

    # ── 8) 延迟后提交 CRITICAL 紧急任务（模拟异常响应） ── #
    async def inject_critical_task():
        await asyncio.sleep(2)
        logger.info("")
        logger.info("🚨" * 30)
        logger.info("🚨 [模拟] 检测到异常！提交 CRITICAL 紧急任务...")
        logger.info("🚨" * 30)
        logger.info("")
        await scheduler.submit_task(
            lab_id="Lab-Alpha",
            description="紧急安全复位: 机械臂归零并关闭真空泵",
            priority=TaskPriority.CRITICAL,
        )

    # 并发：NORMAL 任务执行 + 2s 后注入 CRITICAL
    critical_injector = asyncio.create_task(inject_critical_task())

    # ── 9) 等待队列清空 ── #
    await scheduler._queue.join()
    await critical_injector

    # ── 10) 关闭调度器 ── #
    completed = await scheduler.shutdown()

    # ── 11) 结果汇总 ── #
    final_telemetry = await env.get_telemetry()

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     📊 AstroSASF V5.1 结果汇总                             ║")
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
        "  │     综合压缩率    : %s", cs.get("overall_compression_ratio", "N/A")
    )
    logger.info("  │")

    logger.info("  │  📋 调度统计:")
    logger.info("  │     总完成任务    : %d", len(completed))
    for t in completed:
        logger.info(
            "  │     [%s] %-8s %s → %s",
            t.task_id[:8],
            t.priority.name,
            t.description,
            t.status,
        )
    logger.info("  │")

    logger.info("  │  📨 A2A: %s", env.a2a_stats)
    logger.info("  └──────────────────────────────────────────────────")

    logger.info("")
    logger.info("AstroSASF V5.1 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
