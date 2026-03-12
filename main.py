"""
AstroSASF · Main Entry Point (V3.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
演示 LangGraph + Ollama + Space-MCP + Human-in-the-loop 全链路工作流。

运行 Lab-Alpha 实验柜，在每一步 Skill 执行前暂停，等待科学家在控制台
确认后才调用底层的 Space-MCP 网关。
"""

from __future__ import annotations

import asyncio
import logging
import sys


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
║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝                ║
║                                                              ║
║    S A S F  v3.1  ·  LangGraph + HITL + Space-MCP            ║
║    Astro Scientific Agent Scheduling Framework               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #


async def main() -> None:
    """主入口 —— 演示 HITL 审核驱动的实验柜工作流。"""
    from core.orchestrator import Orchestrator

    _setup_logging()
    logger = logging.getLogger(__name__)

    print(_BANNER)

    # ── Orchestrator ── #
    orchestrator = Orchestrator()

    orchestrator.spawn_laboratory(
        lab_id="Lab-Alpha",
        tasks=["请将实验柜温度升高到50℃，然后将机械臂移动到45度位置"],
    )

    # ── 运行 ── #
    results = await orchestrator.run_all()

    # ── 结果汇总 ── #
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║       📊 AstroSASF V3.1 (LangGraph + HITL) 运行结果      ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    for r in results:
        if isinstance(r, dict) and "lab_id" in r:
            lab = r["lab_id"]
            fsm = r.get("fsm_state", "N/A")
            telemetry = r.get("final_telemetry", {})
            codec = r.get("codec_stats", {})
            bus = r.get("bus_stats", {})
            task_results = r.get("task_results", [])

            logger.info("")
            logger.info(
                "  ┌─── %s ────────────────────────────────────────", lab
            )
            logger.info("  │")
            logger.info("  │  🔧 FSM 终态          : %s", fsm)
            logger.info("  │")
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
            logger.info(
                "  │     总线带宽           : %s Kbps",
                bus.get("bandwidth_kbps", 0),
            )
            logger.info("  │")
            logger.info("  │  🧠 LangGraph 任务结果:")
            for i, tr in enumerate(task_results):
                if isinstance(tr, dict):
                    status = tr.get("status", "N/A")
                    total = tr.get("total_steps", 0)
                    logger.info(
                        "  │     任务 %d: %s (%d 步)", i + 1, status, total
                    )
                else:
                    logger.info("  │     任务 %d: %s", i + 1, tr)
            logger.info("  │")
            logger.info(
                "  └──────────────────────────────────────────────────"
            )
        else:
            logger.error("  实验柜运行异常: %s", r)

    logger.info("")
    logger.info("AstroSASF V3.1 (LangGraph + HITL + Space-MCP) 运行完毕。🚀")


# --------------------------------------------------------------------------- #
#  Script Entry                                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    asyncio.run(main())
