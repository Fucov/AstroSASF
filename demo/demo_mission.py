"""
AstroSASF · Demo · demo_mission
================================
V7.2 演示任务入口 —— 三智能体协作演示。

演示场景：用户请求"在生物实验舱执行细胞培养任务"，
三个 Agent（Q&A、Planner、Executor）通过 FastAPI REST API 协作完成。

┌─────────────────────────────────────────────────────────────┐
│                    demo_mission.py                           │
│                                                             │
│   User  →  [Q&A Agent]  →  [Planner Agent]                  │
│              ↑                         ↓                     │
│              └────  Execution Report  ←  [Executor Agent]    │
│                                                             │
│   所有 Agent 通过 HTTP (httpx) 调用 interface/server.py     │
│   interface/server.py 由 uvicorn 在后台线程启动              │
└─────────────────────────────────────────────────────────────┘

运行方式：
    cd /root/AstroSASF
    python demo/demo_mission.py

    # 或带参数：
    python demo/demo_mission.py --lab DemoBio --mission "在37°C下启动细胞培养流程"

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

# ── 日志配置 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_mission")


# ── 导入必须在任何运行代码之前完成 ──────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_ROOT))

from interface.server import app  # noqa: E402
from interface.state import state  # noqa: E402
from interface.facade import APIFacade  # noqa: E402
from labs.lab_loader import LabLoader, create_loader  # noqa: E402
from infra.llm.config_loader import load_config  # noqa: E402


# ── Agent 导入（延迟导入避免循环依赖）──────────────────────────────────────────
def _import_agents():
    from demo.agents.qa_agent import QAAgent  # noqa: F401, E402
    from demo.agents.planner_agent import PlannerAgent  # noqa: F401, E402
    from demo.agents.executor_agent import ExecutorAgent  # noqa: F401, E402
    from demo.agents.base_agent import AgentMessage  # noqa: F401, E402


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    banner = r"""
     ██████╗██╗  ██╗ █████╗ ██████╗  █████╗     ███████╗
    ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗    ██╔════╝
    ██║     ███████║███████║██████╔╝███████║    ███████╗
    ██║     ██╔══██║██╔══██║██╔═══╝ ██╔══██║    ╚════██║
    ╚██████╗██║  ██║██║  ██║██║     ██║  ██║    ███████║
     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝    ╚══════╝
      ┌──────────────────────────────────────────────────┐
      │  AstroSASF V7.2 — Scientific Agent Scheduling   │
      │  Demo Mission: 3-Agent Collaborative Workflow    │
      └──────────────────────────────────────────────────┘
    """
    print(banner)


def _run_server_in_thread(host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    """在后台线程中启动 uvicorn FastAPI 服务器。"""

    def _target():
        import uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    t = threading.Thread(target=_target, name="uvicorn-server", daemon=True)
    t.start()
    logger.info("[Server] FastAPI 服务启动于 http://%s:%d", host, port)
    return t


async def _wait_for_server(base_url: str, timeout: float = 10.0) -> bool:
    """等待服务器就绪（健康检查轮询）。"""
    import httpx
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() - t0 < timeout:
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        "[Server] ✅ 就绪 — 已加载实验舱: %s",
                        data.get("loaded_labs", []),
                    )
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    logger.error("[Server] 等待超时，服务器未响应")
    return False


# ── 协作任务主流程 ─────────────────────────────────────────────────────────────

async def run_collaborative_mission(
    base_url: str,
    user_request: str,
    *,
    demo_lab: str | None = None,
) -> dict[str, Any]:
    """运行三智能体协作任务。

    协作流程：
        1. Q&A Agent 接收用户请求，解析出实验舱和任务描述
        2. Q&A → Planner：发送 plan_request
        3. Planner Agent 调用 LLM 将任务分解为有序工具调用步骤
        4. Planner → Executor：发送 execution_request（含步骤列表）
        5. Executor Agent 按顺序执行每个工具调用
        6. Executor → Q&A：发送执行报告

    Parameters
    ----------
    base_url : str
        FastAPI 服务地址
    user_request : str
        用户自然语言请求
    demo_lab : str, optional
        强制指定实验舱 ID（默认自动选择）

    Returns
    -------
    dict
        包含各阶段结果的字典
    """
    from demo.agents.qa_agent import QAAgent
    from demo.agents.planner_agent import PlannerAgent
    from demo.agents.executor_agent import ExecutorAgent
    from demo.agents.base_agent import AgentMessage

    # ── 阶段 0: 初始化 ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 60)
    logger.info("  🚀 任务开始: %s", user_request)
    logger.info("─" * 60)

    qa = QAAgent(base_url=base_url)
    planner = PlannerAgent(base_url=base_url)
    executor = ExecutorAgent(base_url=base_url)

    results: dict[str, Any] = {
        "user_request": user_request,
        "qa_result": None,
        "planner_result": None,
        "executor_result": None,
        "success": False,
    }

    try:
        # ── 阶段 1: Q&A Agent ────────────────────────────────────────────────
        logger.info("")
        logger.info("  ┌─────────────────────────────────────────────────┐")
        logger.info("  │  [1/3] Q&A Agent: 理解用户意图...                  │")
        logger.info("  └─────────────────────────────────────────────────┘")

        if demo_lab:
            # 强制指定实验舱模式（跳过 LLM 意图解析）
            plan_msg = AgentMessage(
                sender="user",
                content=user_request,
                metadata={
                    "type": "plan_request",
                    "lab_id": demo_lab,
                    "priority": 2,
                    "original_request": user_request,
                },
            )
            results["qa_result"] = {
                "lab_id": demo_lab,
                "task": user_request,
                "note": "demo_lab 强制指定，跳过 LLM 解析",
            }
        else:
            # 正常模式：Q&A Agent 调用 LLM 解析意图
            init_msg = AgentMessage(sender="user", content=user_request)
            qa_resp = await qa.think(init_msg)
            if qa_resp is None:
                raise RuntimeError("Q&A Agent 返回空响应")

            results["qa_result"] = {
                "lab_id": qa_resp.metadata.get("lab_id"),
                "task": qa_resp.content,
                "priority": qa_resp.metadata.get("priority", 2),
                "raw_llm": qa_resp.metadata.get("llm_raw", ""),
            }

            if qa_resp.metadata.get("type") == "error":
                logger.error("  ❌ Q&A 失败: %s", qa_resp.content)
                return results

            plan_msg = qa_resp

        logger.info("  ✅ Q&A 完成 — 实验舱: %s", plan_msg.metadata.get("lab_id"))
        logger.info("       任务描述: %s", plan_msg.content[:100])

        # ── 阶段 2: Planner Agent ────────────────────────────────────────────
        logger.info("")
        logger.info("  ┌─────────────────────────────────────────────────┐")
        logger.info("  │  [2/3] Planner Agent: 生成执行计划...             │")
        logger.info("  └─────────────────────────────────────────────────┘")

        planner_resp = await planner.think(plan_msg)
        if planner_resp is None:
            raise RuntimeError("Planner Agent 返回空响应")

        results["planner_result"] = {
            "plan_id": planner_resp.metadata.get("plan_id"),
            "steps": planner_resp.metadata.get("steps", []),
            "response": planner_resp.content,
        }

        if planner_resp.metadata.get("type") == "error":
            logger.error("  ❌ Planner 失败: %s", planner_resp.content)
            return results

        logger.info("  ✅ Planner 完成 — 计划ID: %s", planner_resp.metadata.get("plan_id"))
        logger.info("       %s", planner_resp.content[:200])

        # ── 阶段 3: Executor Agent ───────────────────────────────────────────
        logger.info("")
        logger.info("  ┌─────────────────────────────────────────────────┐")
        logger.info("  │  [3/3] Executor Agent: 按计划执行工具调用...      │")
        logger.info("  └─────────────────────────────────────────────────┘")

        executor_resp = await executor.think(planner_resp)
        if executor_resp is None:
            raise RuntimeError("Executor Agent 返回空响应")

        report_meta = executor_resp.metadata
        results["executor_result"] = {
            "plan_id": report_meta.get("plan_id"),
            "success_count": report_meta.get("success_count", 0),
            "failure_count": report_meta.get("failure_count", 0),
            "total_time_ms": report_meta.get("total_time_ms", 0),
        }

        # ── 完成 ───────────────────────────────────────────────────────────
        logger.info("")
        logger.info("  ╔" + "═" * 58 + "╗")
        logger.info("  ║  ✅ 任务完成!                                      ║")
        logger.info(
            "  ║  成功率: %d/%d (%.0f%%)  |  耗时: %.1fms                ║"
            % (
                report_meta.get("success_count", 0),
                report_meta.get("success_count", 0) + report_meta.get("failure_count", 0),
                100 * report_meta.get("success_count", 0) / max(
                    1,
                    report_meta.get("success_count", 0) + report_meta.get("failure_count", 0),
                ),
                report_meta.get("total_time_ms", 0),
            )
        )
        logger.info("  ╚" + "═" * 58 + "╝")

        print("\n" + executor_resp.content + "\n")
        results["success"] = True

    finally:
        await qa.stop()
        await planner.stop()
        await executor.stop()

    return results


# ── 并发演示：同时运行多个任务 ─────────────────────────────────────────────────

async def run_concurrent_missions(
    base_url: str,
    missions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """并发运行多个任务（展示 asyncio 并发调度能力）。"""
    logger.info("")
    logger.info("  ╔" + "═" * 58 + "╗")
    logger.info("  ║  🚀 并发模式: 同时启动 %d 个任务                      ║" % len(missions))
    logger.info("  ╚" + "═" * 58 + "╝")

    async def _single(idx: int, lab: str, task: str):
        t0 = time.monotonic()
        result = await run_collaborative_mission(
            base_url=base_url,
            user_request=task,
            demo_lab=lab,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "  [并发 %d/%d] ✅ 完成 (%.1fs) — %s",
            idx + 1, len(missions), elapsed, task[:50],
        )
        return result

    tasks = [
        _single(i, m["lab"], m["task"])
        for i, m in enumerate(missions)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ── 入口点 ───────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    """主入口。"""
    _print_banner()

    base_url = f"http://{args.host}:{args.port}"
    catalog_dir = Path(args.catalog).resolve()
    skills_dir = Path(args.skills).resolve()

    logger.info("[Demo] 配置:")
    logger.info("  - Catalog: %s", catalog_dir)
    logger.info("  - Skills:  %s", skills_dir)
    logger.info("  - API:     %s", base_url)

    # ── Step 1: 替换 LabLoader 的 catalog 路径（适配 demo 资源）─────────────
    import labs.lab_loader as ll_module
    _original_loader = ll_module.create_loader
    ll_module.create_loader = lambda c=None: LabLoader(catalog_dir=catalog_dir)

    # ── Step 2: 初始化服务器 ───────────────────────────────────────────────
    server_thread = _run_server_in_thread(host=args.host, port=args.port)

    # 等待服务器就绪
    if not await _wait_for_server(base_url):
        logger.error("服务器启动失败，退出")
        return

    # 等待 LabLoader 完成加载
    await asyncio.sleep(1.0)

    # ── Step 3: 显示可用实验舱 ──────────────────────────────────────────────
    import httpx
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{base_url}/api/v1/labs")
        labs = resp.json()
        logger.info("[Demo] 已注册的实验舱:")
        for lab in labs:
            logger.info("  - %s: %s", lab["lab_id"], lab["description"])

    # ── Step 4: 执行演示任务 ────────────────────────────────────────────────
    if args.mission:
        # 单任务模式
        await run_collaborative_mission(
            base_url=base_url,
            user_request=args.mission,
            demo_lab=args.lab,
        )
    else:
        # 默认演示：并发执行多个任务
        default_missions = [
            {
                "lab": "DemoBio",
                "task": "将培养舱温度设置为37°C，然后注入50ml营养液",
            },
            {
                "lab": "DemoFluid",
                "task": "启动真空泵并将阀门切换到MIXING模式",
            },
        ]
        await run_concurrent_missions(base_url=base_url, missions=default_missions)

    logger.info("")
    logger.info("[Demo] 运行结束。服务器线程仍在后台运行（daemon），程序退出后将自动终止。")
    logger.info("[Demo] 访问 http://%s:%d/docs 查看 API 文档", args.host, args.port)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="AstroSASF V7.2 — 三智能体协作演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="FastAPI 服务地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="FastAPI 服务端口 (默认: 8000)",
    )
    parser.add_argument(
        "--catalog",
        default=str(_ROOT / "demo" / "assets" / "labs"),
        help="实验舱配置目录",
    )
    parser.add_argument(
        "--skills",
        default=str(_ROOT / "demo" / "assets" / "skills"),
        help="技能 SOP 目录",
    )
    parser.add_argument(
        "--mission",
        default="",
        help="自然语言任务描述（不提供则运行默认并发演示）",
    )
    parser.add_argument(
        "--lab",
        default=None,
        help="强制指定实验舱 ID（配合 --mission 使用）",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("[Demo] 用户中断")


if __name__ == "__main__":
    _cli()
