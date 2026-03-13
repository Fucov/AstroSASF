#!/usr/bin/env python3
"""
AstroSASF · Space Station Demo (V4.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
演示如何在 **应用层** 完成：

1. 从 ``fsm_rules.yaml`` 加载 FSM 业务规则
2. 注册业务 MCP Tools（set_temperature 等）
3. 注入 HITL（MemorySaver + interrupt_before）
4. 订阅 A2A 消息
5. 全链路运行并输出结果

框架层（sasf/）不包含任何业务词汇。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 框架导入 ── #
from sasf.core.config_loader import load_config
from sasf.core.environment import LaboratoryEnvironment
from sasf.middleware.a2a_protocol import A2AIntent
from sasf.middleware.mcp_registry import MCPToolContext, MCPToolRegistry
from sasf.physics.shadow_fsm import ShadowFSM
from sasf.physics.telemetry_bus import TelemetryBus

# HITL 所需
from langgraph.checkpoint.memory import MemorySaver
import uuid

logger = logging.getLogger(__name__)


# ============================================================================ #
#  业务层: 初始遥测数据                                                          #
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
#  业务层: MCP Tools 注册函数                                                    #
# ============================================================================ #


def register_space_station_tools(
    registry: MCPToolRegistry,
    fsm: ShadowFSM,
    bus: TelemetryBus,
) -> None:
    """将太空实验柜的物理操作注册为 MCP Tools。

    此函数在应用层定义，框架层不包含任何业务代码。
    """

    @registry.mcp_tool
    async def set_temperature(ctx: MCPToolContext, target: float) -> dict[str, Any]:
        """设置舱内温度目标值（℃）"""
        target = float(target)
        current_temp = await ctx.bus.read("temperature")
        action = "START_HEATING" if target > current_temp else "START_COOLING"

        snapshot = await ctx.bus.snapshot()
        await ctx.fsm.validate_and_transition(
            action=action,
            params={"target": target},
            telemetry_snapshot=snapshot,
        )

        await ctx.bus.write("temperature", target)

        stop_action = "STOP_HEATING" if action == "START_HEATING" else "STOP_COOLING"
        await ctx.fsm.validate_and_transition(
            action=stop_action,
            params={},
            telemetry_snapshot=await ctx.bus.snapshot(),
        )

        return {
            "skill": "set_temperature",
            "status": "success",
            "fsm_state": ctx.fsm.current_state,
            "detail": f"温度已设置为 {target}℃",
        }

    @registry.mcp_tool
    async def move_robotic_arm(
        ctx: MCPToolContext, target_angle: float
    ) -> dict[str, Any]:
        """移动机械臂至指定角度（°）"""
        target_angle = float(target_angle)
        snapshot = await ctx.bus.snapshot()
        await ctx.fsm.validate_and_transition(
            action="MOVE_ROBOTIC_ARM",
            params={"target_angle": target_angle},
            telemetry_snapshot=snapshot,
        )

        await ctx.bus.write("robotic_arm_angle", target_angle)

        await ctx.fsm.validate_and_transition(
            action="STOP_ROBOTIC_ARM",
            params={},
            telemetry_snapshot=await ctx.bus.snapshot(),
        )

        return {
            "skill": "move_robotic_arm",
            "status": "success",
            "fsm_state": ctx.fsm.current_state,
            "detail": f"机械臂已移至 {target_angle}°",
        }

    @registry.mcp_tool
    async def toggle_vacuum_pump(ctx: MCPToolContext, activate: bool) -> dict[str, Any]:
        """切换真空泵开关"""
        if isinstance(activate, str):
            activate = activate.lower() in ("true", "1", "yes")

        action = "ACTIVATE_VACUUM" if activate else "DEACTIVATE_VACUUM"
        snapshot = await ctx.bus.snapshot()
        await ctx.fsm.validate_and_transition(
            action=action,
            params={"activate": activate},
            telemetry_snapshot=snapshot,
        )

        await ctx.bus.write("vacuum_pump_active", activate)

        return {
            "skill": "toggle_vacuum_pump",
            "status": "success",
            "fsm_state": ctx.fsm.current_state,
            "detail": f"真空泵已{'激活' if activate else '关闭'}",
        }

    logger.info(
        "[%s] 物理设备层: 已注册 %d 个 MCP Tools (自动 Schema)",
        registry.lab_id,
        len(registry.all_tool_names()),
    )


# ============================================================================ #
#  HITL 循环 (应用层实现)                                                        #
# ============================================================================ #


async def hitl_loop(
    compiled_graph: Any,
    initial_state: dict[str, Any],
    config: dict[str, Any],
    lab_id: str,
) -> dict[str, Any]:
    """在应用层实现 Human-in-the-Loop 循环。"""
    state = await compiled_graph.ainvoke(initial_state, config)
    snapshot = compiled_graph.get_state(config)

    while snapshot.next:
        step = state.get("current_step")
        if step and isinstance(step, dict):
            logger.info("")
            logger.info("[%s] ⏸️  HITL 中断 — 即将执行:", lab_id)
            logger.info("[%s]    Tool  : %s", lab_id, step.get("skill"))
            logger.info(
                "[%s]    Params: %s",
                lab_id,
                json.dumps(step.get("params", {}), ensure_ascii=False),
            )

        user_input = await _get_human_approval(step, lab_id)

        if user_input == "abort":
            logger.warning("[%s] ❌ 用户中止执行", lab_id)
            return {
                "status": "aborted_by_user",
                "total_steps": len(state.get("plan", [])),
                "execution_log": list(state.get("execution_log") or []),
            }

        if user_input == "approve":
            state = await compiled_graph.ainvoke(None, config)
        else:
            try:
                corrected = json.loads(user_input)
                if isinstance(corrected, dict):
                    if "skill" in corrected and "params" in corrected:
                        new_step = corrected
                    else:
                        new_step = dict(step) if step else {}
                        new_params = new_step.get("params", {})
                        if isinstance(new_params, dict):
                            new_params.update(corrected)
                        else:
                            new_params = corrected
                        new_step["params"] = new_params

                    compiled_graph.update_state(
                        config,
                        {"current_step": new_step},
                    )
                    logger.info("[%s] ✏️  用户修正: %s", lab_id, new_step)
            except (json.JSONDecodeError, TypeError):
                logger.warning("[%s] 无法解析用户输入，按原计划继续", lab_id)
            state = await compiled_graph.ainvoke(None, config)

        snapshot = compiled_graph.get_state(config)

    final = state.get("final_result")
    if final and isinstance(final, dict):
        return final
    return {
        "status": "completed",
        "total_steps": len(state.get("plan", [])),
        "execution_log": list(state.get("execution_log") or []),
    }


async def _get_human_approval(step: dict[str, Any] | None, lab_id: str) -> str:
    """获取人类审批。"""
    if step:
        skill = step.get("skill", "unknown")
        params = json.dumps(step.get("params", {}), ensure_ascii=False)
        prompt = (
            f"\n{'─' * 60}\n"
            f"🛡️ HITL | 即将执行 MCP Tool: {skill}({params})\n"
            f"  [y/回车] 批准  |  [n] 中止  |  [JSON] 修正参数\n"
            f"{'─' * 60}\n"
            f">>> "
        )
    else:
        prompt = "\n>>> 继续? [y/n]: "

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, lambda: input(prompt))

    raw = raw.strip()
    if raw == "" or raw.lower() in ("y", "yes"):
        return "approve"
    if raw.lower() in ("n", "no"):
        return "abort"
    return raw


# ============================================================================ #
#  主函数                                                                        #
# ============================================================================ #


async def main() -> None:
    # ── 日志 ── #
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.INFO)

    # ── Banner ── #
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
║    S A S F  v4.4  ·  Generic Engine + Strict Planner         ║
║    Astro Scientific Agent Scheduling Framework               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── 1) 加载配置 ── #
    config = load_config(PROJECT_ROOT / "config.yaml")

    # ── 2) 从 fsm_rules.yaml 加载 FSM 规则 ── #
    fsm = ShadowFSM.from_yaml(
        path=PROJECT_ROOT / "fsm_rules.yaml",
        lab_id="Lab-Alpha",
    )

    # ── 3) 构建环境（传入 FSM + 业务 Tool 注册函数 + 初始遥测） ── #
    env = LaboratoryEnvironment(
        lab_id="Lab-Alpha",
        config=config,
        fsm=fsm,
        tool_registrar=register_space_station_tools,
        initial_telemetry=INITIAL_TELEMETRY,
    )

    # ── 4) A2A 订阅演示 ── #
    def on_skill_result(msg):
        logger.info(
            "📬 [订阅者] 收到 SKILL_RESULT: %s → %s",
            msg.sender,
            msg.payload.get("status", "N/A")
            if isinstance(msg.payload, dict)
            else msg.payload,
        )

    env.a2a_router.subscribe(A2AIntent.SKILL_RESULT, on_skill_result)

    # ── 5) 展示动态字典 ── #
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📖 动态字典协商结果（启动期自动生成）              ║")
    logger.info("╚" + "═" * 60 + "╝")
    for word, tid in env.codec_dictionary.items():
        logger.info("    0x%02X  ←  '%s'", tid, word)
    logger.info("    共 %d 个映射词条", len(env.codec_dictionary))

    # ── 6) 展示已加载 Skills ── #
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║          📚 已加载 OpenAI Skills (SOP 知识套件)            ║")
    logger.info("╚" + "═" * 60 + "╝")
    for s in env.loaded_skills:
        logger.info("    ✅ %s — %s", s["name"], s["description"])

    # ── 7) HITL 模式运行 ── #
    logger.info("")
    logger.info("💡 HITL: 每个 MCP Tool 执行前会暂停等待人类审批")
    logger.info("   y/回车 = 批准 | n = 中止 | JSON = 修正参数")

    tasks = ["进行流体实验"]

    # 应用层注入 HITL (MemorySaver + interrupt_before)
    memory = MemorySaver()
    compiled = env.graph.compile(
        checkpointer=memory,
        interrupt_before=["execute_node"],
    )

    logger.info("")
    logger.info("=" * 64)
    logger.info("[Lab-Alpha] 🚀 实验柜启动 (V4.3 HITL 模式)")
    logger.info(
        "[Lab-Alpha] 🔧 已注册 MCP Tools: %s",
        [t["name"] for t in env.available_tools],
    )
    logger.info(
        "[Lab-Alpha] 📚 已加载 OpenAI Skills: %s",
        [s["name"] for s in env.loaded_skills],
    )
    logger.info("=" * 64)

    all_results = []
    for task_desc in tasks:
        logger.info("")
        logger.info("[Lab-Alpha] 📥 提交任务: %s", task_desc)

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

        logger.info(
            "[Lab-Alpha] 📤 任务完成: %s",
            result.get("status", "N/A"),
        )

    # ── 8) 结果汇总 ── #
    final_telemetry = await env.get_telemetry()
    logger.info("")
    logger.info("-" * 64)
    logger.info("[Lab-Alpha] ✅ 环境关闭  FSM=%s", env.fsm_state)
    logger.info("[Lab-Alpha] 最终遥测: %s", final_telemetry)
    logger.info("[Lab-Alpha] A2A 统计: %s", env.a2a_stats)
    logger.info("-" * 64)

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     📊 AstroSASF V4.3 (Generic Engine + HITL) 结果汇总    ║")
    logger.info("╚" + "═" * 60 + "╝")

    logger.info("")
    logger.info("  ┌─── Lab-Alpha ────────────────────────────────────")
    logger.info("  │")
    logger.info("  │  🔧 FSM 终态          : %s", env.fsm_state)
    logger.info("  │")
    logger.info("  │  📡 遥测终态:")
    for k, v in final_telemetry.items():
        logger.info("  │     %-25s = %s", k, v)
    logger.info("  │")

    codec_stats = env.codec_stats
    logger.info("  │  🗜️  编解码器统计:")
    logger.info("  │     编码次数           : %s", codec_stats.get("encode_count", 0))
    logger.info(
        "  │     JSON 总字节        : %s B", codec_stats.get("total_json_bytes", 0)
    )
    logger.info(
        "  │     Binary 总字节      : %s B", codec_stats.get("total_binary_bytes", 0)
    )
    logger.info(
        "  │     综合压缩率         : %s",
        codec_stats.get("overall_compression_ratio", "N/A"),
    )
    logger.info(
        "  │     动态字典词条数     : %s", codec_stats.get("dictionary_size", 0)
    )
    logger.info("  │")

    bus_stats = env.bus_stats
    logger.info("  │  🛰️  SpaceWire 总线统计:")
    logger.info("  │     传输帧数           : %s", bus_stats.get("total_frames", 0))
    logger.info("  │     传输总字节         : %s B", bus_stats.get("total_bytes", 0))
    logger.info(
        "  │     累计传输延迟       : %s ms", bus_stats.get("total_latency_ms", 0)
    )
    logger.info("  │")

    a2a_stats = env.a2a_stats
    logger.info("  │  📨 A2A 路由器统计:")
    logger.info("  │     总消息数           : %s", a2a_stats.get("total_messages", 0))
    logger.info(
        "  │     活跃订阅           : %s", a2a_stats.get("active_subscriptions", 0)
    )
    for intent_name, count in a2a_stats.get("intent_distribution", {}).items():
        logger.info("  │     %-20s : %s", intent_name, count)
    logger.info("  │")

    logger.info("  │  🧠 LangGraph 任务结果:")
    for i, r in enumerate(all_results, 1):
        logger.info(
            "  │     任务 %d: %s (%d 步)",
            i,
            r.get("status", "N/A"),
            r.get("total_steps", 0),
        )
    logger.info("  │")
    logger.info("  └──────────────────────────────────────────────────")

    logger.info("")
    logger.info("AstroSASF V4.3 (Generic Engine + External HITL) 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
