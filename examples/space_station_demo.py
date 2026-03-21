#!/usr/bin/env python3
"""
AstroSASF · Space Station Demo (V5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
全逻辑验证演示：

1. 正交子系统状态 + InterlockEngine
2. Guard 装饰器 (require_states / forbid_states / telemetry_rules)
3. Macro 参数预绑定
4. Codec 词表自动握手（含 Macro 名）
5. Skill Loader Macro 感知
6. HITL 外部注入
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# --- 新增代码：手动挂载根目录到 sys.path ---
# 获取当前文件 (examples/space_station_demo.py) 的父目录的父目录 (AstroSASF)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# ---------------------------------------

from sasf.core.config_loader import load_config
from sasf.core.environment import LaboratoryEnvironment
from sasf.middleware.a2a_protocol import A2AIntent
from sasf.middleware.mcp_registry import MCPToolContext, MCPToolRegistry
from sasf.physics.interlock_engine import InterlockEngine
from sasf.physics.telemetry_bus import TelemetryBus

from langgraph.checkpoint.memory import MemorySaver

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
#  HITL 循环                                                                    #
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
            # 尝试 JSON 解析
            try:
                corrected = json.loads(raw)
            except json.JSONDecodeError:
                pass
            # 退级：ast.literal_eval（容忍单引号 Python 字典）
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
#  主函数                                                                        #
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
║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝                ║
║                                                              ║
║    S A S F  v5.0  ·  Interlock Engine + Guard + Macro        ║
║    Astro Scientific Agent Scheduling Framework               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── 1) 配置 ── #
    config = load_config(PROJECT_ROOT / "config.yaml")

    # ── 2) InterlockEngine from YAML ── #
    engine = InterlockEngine.from_yaml(
        path=PROJECT_ROOT / "fsm_rules.yaml",
        lab_id="Lab-Alpha",
    )

    # ── 3) 构建环境（注入 Tools + Macros） ── #
    env = LaboratoryEnvironment(
        lab_id="Lab-Alpha",
        config=config,
        engine=engine,
        tool_registrar=register_tools,
        macro_registrar=register_macros,
        initial_telemetry=INITIAL_TELEMETRY,
    )

    # ── 4) A2A 订阅 ── #
    env.a2a_router.subscribe(
        A2AIntent.SKILL_RESULT,
        lambda msg: logger.info(
            "📬 [订阅] SKILL_RESULT: %s",
            msg.payload.get("status") if isinstance(msg.payload, dict) else msg.payload,
        ),
    )

    # ── 5) 展示信息 ── #
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📖 动态字典 (自动握手 — 含 Macro)                  ║")
    logger.info("╚" + "═" * 60 + "╝")
    for word, tid in env.codec_dictionary.items():
        logger.info("    0x%02X  ←  '%s'", tid, word)
    logger.info("    共 %d 个映射词条", len(env.codec_dictionary))

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

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📚 已加载 OpenAI Skills                            ║")
    logger.info("╚" + "═" * 60 + "╝")
    for s in env.loaded_skills:
        logger.info("    ✅ %s — %s", s["name"], s["description"])

    # ── 6) HITL 模式运行 ── #
    logger.info("")
    logger.info("💡 HITL: y/回车 = 批准 | n = 中止 | JSON = 修正参数")

    tasks = ["请执行流体实验的环境准备和样品装载流程"]

    memory = MemorySaver()
    compiled = env.graph.compile(
        checkpointer=memory,
        interrupt_before=["execute_node"],
    )

    logger.info("")
    logger.info("=" * 64)
    logger.info("[Lab-Alpha] 🚀 V5 启动 (HITL 模式)")
    logger.info(
        "[Lab-Alpha] 🔧 Tools (含 Macro): %s",
        [t["name"] for t in env.available_tools],
    )
    logger.info("=" * 64)

    all_results = []
    for task_desc in tasks:
        logger.info("")
        logger.info("[Lab-Alpha] 📥 任务: %s", task_desc)

        initial_state = {
            "original_task": task_desc,
            "plan": [],
            "current_step_index": 0,
            "current_step": None,
            "fsm_feedback": None,
            "execution_log": [],
            "error_count": 0,
            "final_result": None,
        }

        thread_config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        result = await hitl_loop(compiled, initial_state, thread_config, "Lab-Alpha")
        all_results.append(result)

    # ── 7) 结果汇总 ── #
    final_telemetry = await env.get_telemetry()

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     📊 AstroSASF V5 结果汇总                               ║")
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
    logger.info("  │     词条数        : %s (含 Macro)", cs.get("dictionary_size", 0))
    logger.info("  │")

    logger.info("  │  📨 A2A: %s", env.a2a_stats)
    logger.info("  │")

    for i, r in enumerate(all_results, 1):
        logger.info("  │  🧠 任务 %d: %s", i, r.get("status", "N/A"))
    logger.info("  └──────────────────────────────────────────────────")

    logger.info("")
    logger.info("AstroSASF V5 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
