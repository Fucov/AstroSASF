"""
AstroSASF · Examples · Space Station Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
演示 V4 Middleware-First Architecture 全链路工作流。

运行 Lab-Alpha 实验柜，展示：
  config.yaml 配置加载 → SkillRegistry 注册 → A2A 消息路由
  → LLM 规划 → Space-MCP 压缩 → SpaceWire 传输 → FSM 校验
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
║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝                ║
║                                                              ║
║    S A S F  v4.0  ·  Middleware-First Architecture           ║
║    Astro Scientific Agent Scheduling Framework               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

async def main() -> None:
    """主入口 —— 演示 V4 Middleware-First 全链路。"""
    from sasf.core.config_loader import load_config
    from sasf.core.orchestrator import Orchestrator

    _setup_logging()
    logger = logging.getLogger(__name__)

    print(_BANNER)

    # ── 加载配置 ── #
    config = load_config()
    logger.info("LLM Provider: %s | Model: %s", config.llm.provider, config.llm.model_name)
    logger.info("SpaceWire 带宽: %.0f Kbps", config.middleware.spacewire_bandwidth_kbps)

    # ── Orchestrator ── #
    orchestrator = Orchestrator(config=config)

    orchestrator.spawn_laboratory(
        lab_id="Lab-Alpha",
        tasks=["请将实验柜温度升高到50℃，然后将机械臂移动到45度位置"],
    )

    # ── 运行 ── #
    results = await orchestrator.run_all()

    # ── 结果汇总 ── #
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║       📊 AstroSASF V4 (Middleware-First) 运行结果汇总      ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    for r in results:
        if isinstance(r, dict) and "lab_id" in r:
            lab = r["lab_id"]
            fsm = r.get("fsm_state", "N/A")
            telemetry = r.get("final_telemetry", {})
            codec = r.get("codec_stats", {})
            bus = r.get("bus_stats", {})
            a2a = r.get("a2a_stats", {})
            task_results = r.get("task_results", [])

            logger.info("")
            logger.info("  ┌─── %s ────────────────────────────────────────", lab)
            logger.info("  │")
            logger.info("  │  🔧 FSM 终态          : %s", fsm)
            logger.info("  │")
            logger.info("  │  📡 遥测终态:")
            for k, v in telemetry.items():
                logger.info("  │     %-25s = %s", k, v)
            logger.info("  │")
            logger.info("  │  🗜️  编解码器统计:")
            logger.info("  │     编码次数           : %s", codec.get("encode_count", 0))
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
            logger.info("  │     传输帧数           : %s", bus.get("total_frames", 0))
            logger.info("  │     传输总字节         : %s B", bus.get("total_bytes", 0))
            logger.info(
                "  │     累计传输延迟       : %s ms",
                bus.get("total_latency_ms", 0),
            )
            logger.info("  │")
            logger.info("  │  📨 A2A 路由器统计:")
            logger.info("  │     总消息数           : %s", a2a.get("total_messages", 0))
            intent_dist = a2a.get("intent_distribution", {})
            for intent_name, count in intent_dist.items():
                logger.info("  │     %-22s: %d", intent_name, count)
            logger.info("  │")
            logger.info("  │  🧠 LangGraph 任务结果:")
            for i, tr in enumerate(task_results):
                status = tr.get("status", "N/A") if isinstance(tr, dict) else "N/A"
                total = tr.get("total_steps", 0) if isinstance(tr, dict) else 0
                logger.info("  │     任务 %d: %s (%d 步)", i + 1, status, total)
            logger.info("  │")
            logger.info("  └──────────────────────────────────────────────────")
        else:
            logger.error("  实验柜运行异常: %s", r)

    logger.info("")
    logger.info("AstroSASF V4 (Middleware-First Architecture) 运行完毕。🚀")


if __name__ == "__main__":
    asyncio.run(main())
