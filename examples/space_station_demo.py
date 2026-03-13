"""
AstroSASF · Examples · Space Station Demo (V4.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
演示 V4.1 全链路工作流，包含 HITL (Human-in-the-Loop)。

运行 Lab-Alpha 实验柜，交互流程：
  config.yaml 加载 → SkillRegistry 注册 → LLM 规划
  → HITL 中断（每步等待人类 y/n/JSON）
  → Space-MCP 压缩 → SpaceWire 传输 → FSM 校验
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    if not root.handlers:
        root.addHandler(handler)


# --------------------------------------------------------------------------- #
#  Banner                                                                      #
# --------------------------------------------------------------------------- #

_BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ███████╗████████╗██████╗  ██████╗                 ║
║    ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗                ║
║    ███████║███████╗   ██║   ██████╔╝██║   ██║                ║
║    ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║                ║
║    ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝                ║
║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝                 ║
║                                                              ║
║    S A S F  v4.1  ·  HITL + Middleware-First                 ║
║    Astro Scientific Agent Scheduling Framework               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# --------------------------------------------------------------------------- #
#  Result Printer                                                              #
# --------------------------------------------------------------------------- #


def _print_results(results: list[dict], logger: logging.Logger) -> None:
    """统一打印结果汇总 —— 无论成功还是崩溃都能展示统计。"""

    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║      📊 AstroSASF V4.1 (HITL + MW-First) 结果汇总       ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    for r in results:
        if not isinstance(r, dict):
            logger.error("  异常结果: %s", r)
            continue

        lab = r.get("lab_id", "Unknown")
        is_error = r.get("status") == "error"
        fsm = r.get("fsm_state", "N/A")
        telemetry = r.get("final_telemetry", {})
        codec = r.get("codec_stats", {})
        bus = r.get("bus_stats", {})
        a2a = r.get("a2a_stats", {})
        task_results = r.get("task_results", [])

        logger.info("")
        status_tag = "💥 CRASHED" if is_error else "✅ OK"
        logger.info(
            "  ┌─── %s [%s] ────────────────────────────────────",
            lab,
            status_tag,
        )

        if is_error:
            logger.info("  │  ❌ 异常详情: %s", r.get("detail", ""))

        logger.info("  │")
        logger.info("  │  🔧 FSM 终态          : %s", fsm)
        logger.info("  │")

        if telemetry:
            logger.info("  │  📡 遥测终态:")
            for k, v in telemetry.items():
                logger.info("  │     %-25s = %s", k, v)
            logger.info("  │")

        logger.info("  │  🗜️  编解码器统计:")
        logger.info(
            "  │     编码次数           : %s",
            codec.get("encode_count", 0),
        )
        logger.info(
            "  │     JSON 总字节        : %s B",
            codec.get("total_json_bytes", 0),
        )
        logger.info(
            "  │     Binary 总字节      : %s B",
            codec.get("total_binary_bytes", 0),
        )
        logger.info(
            "  │     综合压缩率         : %s",
            codec.get("overall_compression_ratio", "N/A"),
        )
        logger.info("  │")
        logger.info("  │  🛰️  SpaceWire 总线统计:")
        logger.info(
            "  │     传输帧数           : %s",
            bus.get("total_frames", 0),
        )
        logger.info(
            "  │     传输总字节         : %s B",
            bus.get("total_bytes", 0),
        )
        logger.info(
            "  │     累计传输延迟       : %s ms",
            bus.get("total_latency_ms", 0),
        )
        logger.info("  │")
        logger.info("  │  📨 A2A 路由器统计:")
        logger.info(
            "  │     总消息数           : %s",
            a2a.get("total_messages", 0),
        )
        intent_dist = a2a.get("intent_distribution", {})
        for intent_name, count in intent_dist.items():
            logger.info("  │     %-22s: %d", intent_name, count)

        if task_results:
            logger.info("  │")
            logger.info("  │  🧠 LangGraph 任务结果:")
            for i, tr in enumerate(task_results):
                if isinstance(tr, dict):
                    status = tr.get("status", "N/A")
                    total = tr.get("total_steps", 0)
                    logger.info(
                        "  │     任务 %d: %s (%d 步)",
                        i + 1,
                        status,
                        total,
                    )

        logger.info("  │")
        logger.info("  └──────────────────────────────────────────────────")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #


async def main() -> None:
    """主入口 —— 演示 V4.1 HITL + Middleware-First 全链路。"""
    from sasf.core.config_loader import load_config
    from sasf.core.orchestrator import Orchestrator

    _setup_logging()
    logger = logging.getLogger(__name__)

    print(_BANNER)

    # ── 加载配置 ── #
    config = load_config()
    logger.info(
        "LLM Provider: %s | Model: %s",
        config.llm.provider,
        config.llm.model_name,
    )
    logger.info(
        "SpaceWire 带宽: %.0f Kbps",
        config.middleware.spacewire_bandwidth_kbps,
    )

    # ── Orchestrator ── #
    orchestrator = Orchestrator(config=config)

    orchestrator.spawn_laboratory(
        lab_id="Lab-Alpha",
        tasks=["请将实验柜温度升高到50℃，然后将机械臂移动到45度位置"],
    )

    # ── 运行 (HITL: 每步等待人类输入 y/n/JSON) ── #
    logger.info("")
    logger.info("💡 提示: 每个 Skill 执行前会暂停等待人类审批")
    logger.info("   y/回车 = 批准 | n = 中止 | JSON = 修正参数")
    logger.info("")

    results = await orchestrator.run_all()

    # ── 结果汇总 ── #
    _print_results(results, logger)

    logger.info("")
    logger.info("AstroSASF V4.1 (HITL + Middleware-First) 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
