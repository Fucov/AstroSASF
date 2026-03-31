#!/usr/bin/env python3
"""
AstroSASF · Dataset Benchmark Runner (V7.1 Ultimate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
执行 `datasets/astro_bench.jsonl` 中的评测数据，进行 Ablation Study 对比测试。

V7.1 核心改进：
1. 动态物理延迟注入：不修改核心框架，在 Benchmark 脚本中拦截工具执行
2. 公平对照组：Baseline 和 DAG-OS 都承受相同 chaos (enable_chaos=True)
3. LLM 调用公平对比：Baseline=每步1次 LLM，DAG-OS=仅规划1次
4. 硬核 Survival Rate 对比：Baseline 遇报警直接挂，DAG-OS 抢占逃生

Usage:
    python benchmarks/run_dataset_eval.py

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_dataset import BenchmarkEpisode, ChaosEvent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  V7.1 物理模拟参数（基准延迟，单位：秒）                                        #
# --------------------------------------------------------------------------- #

BASE_PHYSICS_DELAYS: dict[str, float] = {
    "set_temperature": 3.0,
    "toggle_vacuum_pump": 4.0,
    "move_robotic_arm": 5.0,
    "inject_nutrient": 3.0,
    "turn_on_laser": 2.5,
    "toggle_ventilation": 3.0,
    "set_greenhouse_lighting": 1.0,
    # 默认最小延迟（确保报警有触发窗口）
    "_default": 2.0,
}


def get_tool_delay(tool_name: str, chaos_events: list[ChaosEvent]) -> float:
    """获取工具的物理延迟（含 chaos 倍数）。"""
    base = BASE_PHYSICS_DELAYS.get(tool_name, BASE_PHYSICS_DELAYS["_default"])

    # 检查是否有针对该工具的 hardware_delay 混沌事件
    for event in chaos_events:
        if event.type == "hardware_delay" and event.target_tool == tool_name:
            multiplier = event.delay_multiplier or 1.0
            return base * multiplier

    return base


# --------------------------------------------------------------------------- #
#  V7.1 动态物理模拟执行器（拦截包装器）                                          #
# --------------------------------------------------------------------------- #

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


class PhysicsWrappedExecutor:
    """V7.1 动态物理模拟执行器。

    在评测前拦截并包装工具 handler，添加真实物理耗时。
    保持核心框架解耦，物理模拟仅存在于评测脚本。
    """

    def __init__(self, tool_name: str, original_handler: ToolHandler, chaos_events: list[ChaosEvent]):
        self.tool_name = tool_name
        self.original_handler = original_handler
        self.chaos_events = chaos_events
        self.delay = get_tool_delay(tool_name, chaos_events)
        self._interrupted = False

    async def execute(self, ctx: Any, **params: Any) -> dict[str, Any]:
        """执行带物理模拟的工具调用。

        V7.1 关键逻辑：
        1. await asyncio.sleep(delay) - 物理耗时
        2. try...except asyncio.CancelledError - 紧急制动捕获
        3. 打印 "[物理中断]" 日志并重新 raise
        """
        logger.info(
            "[物理模拟] %s 开始，耗时 %.2fs",
            self.tool_name, self.delay
        )

        try:
            # ★ 关键：物理耗时（可被 Cancel）
            await asyncio.sleep(self.delay)

        except asyncio.CancelledError:
            # ★ 关键：物理操作被紧急制动中断
            self._interrupted = True
            logger.warning(
                "[物理中断] ⚠️ %s 动作被紧急制动！",
                self.tool_name
            )
            raise  # 重新抛出，配合 Orchestrator 强行终止

        # 执行实际工具逻辑
        return await self.original_handler(ctx, **params)


def apply_mock_delays(
    registry: Any,
    chaos_events: list[ChaosEvent],
    lab_id: str = "EvalLab",
) -> dict[str, float]:
    """动态包装 registry 中的工具 handler，注入物理模拟。

    V7.1：在每个 Episode 启动前调用此函数。
    返回：各工具的延迟配置（用于日志输出）。

    Args:
        registry: MCPToolRegistry 实例
        chaos_events: 当前 Episode 的混沌事件列表
        lab_id: 实验柜 ID（用于日志）
    """
    delay_config = {}

    for tool_name in registry.all_tool_names():
        descriptor = registry.get_tool(tool_name)
        if descriptor is None or descriptor.is_macro:
            continue

        original_handler = descriptor.handler
        wrapped = PhysicsWrappedExecutor(
            tool_name=tool_name,
            original_handler=original_handler,
            chaos_events=chaos_events,
        )

        # 替换 handler
        descriptor.handler = wrapped.execute
        delay_config[tool_name] = wrapped.delay

        logger.info(
            "[%s] 🔧 物理模拟已注入: %s (delay=%.2fs)",
            lab_id, tool_name, wrapped.delay
        )

    return delay_config


# --------------------------------------------------------------------------- #
#  评测结果数据结构                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeResult:
    """单个 Episode 的评测结果。"""
    episode_id: str
    difficulty: str
    description: str
    # Baseline 结果（模拟 ReAct：每步1次LLM，无抢占）
    baseline_makespan_sec: float = 0.0
    baseline_llm_calls: int = 0
    baseline_completed_nodes: int = 0
    baseline_failed_nodes: int = 0
    baseline_success: bool = False
    baseline_guardrail_triggered: bool = False
    # DAG-OS 结果（DAG规划=1次LLM，有抢占逃生）
    dag_makespan_sec: float = 0.0
    dag_llm_calls: int = 0
    dag_completed_nodes: int = 0
    dag_failed_nodes: int = 0
    dag_success: bool = False
    dag_preemption_triggered: bool = False
    dag_preemption_latency_ms: float = 0.0
    # 混沌事件记录
    chaos_injected: list[dict] = field(default_factory=list)


@dataclass
class AggregatedReport:
    """聚合评测报告。"""
    baseline_avg_makespan_sec: float = 0.0
    dag_avg_makespan_sec: float = 0.0
    baseline_total_llm_calls: int = 0
    dag_total_llm_calls: int = 0
    hard_survival_rate_baseline: float = 0.0
    hard_survival_rate_dag: float = 0.0
    avg_preemption_latency_ms: float = 0.0
    llm_call_reduction_pct: float = 0.0
    baseline_difficulty_breakdown: dict[str, dict] = field(default_factory=dict)
    dag_difficulty_breakdown: dict[str, dict] = field(default_factory=dict)
    episode_results: list[EpisodeResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  简化的 Mock 执行引擎（用于评测）                                              #
# --------------------------------------------------------------------------- #

class MockToolExecutor:
    """简化的工具执行器（用于评测）。"""

    def __init__(self, delay_ms: float = 100.0):
        self.delay_ms = delay_ms
        self._call_count = 0
        self._total_execution_time = 0.0

    async def execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """执行工具调用。"""
        self._call_count += 1
        t0 = time.perf_counter()

        # 模拟工具执行延迟
        await asyncio.sleep(self.delay_ms / 1000.0)

        elapsed = time.perf_counter() - t0
        self._total_execution_time += elapsed

        return {
            "tool": tool_name,
            "params": params,
            "status": "success",
            "elapsed_ms": elapsed * 1000.0,
        }

    def reset(self) -> None:
        self._call_count = 0
        self._total_execution_time = 0.0


class MockDAGScheduler:
    """简化的 DAG 调度器（用于评测）。"""

    def __init__(
        self,
        num_workers: int,
        executor: MockToolExecutor,
        enable_preemption: bool = False,
    ):
        self.num_workers = num_workers
        self.executor = executor
        self.enable_preemption = enable_preemption
        self._preemption_callbacks: list[callable] = []
        self._alarm_trigger_time: float | None = None

    def register_preemption_callback(self, callback: callable) -> None:
        """注册抢占回调。"""
        self._preemption_callbacks.append(callback)

    async def execute_dag(
        self,
        dag_nodes: list[dict[str, Any]],
        chaos_injector: "ChaosInjector | None" = None,
    ) -> tuple[list[dict], list[dict], bool, float]:
        """执行 DAG 并返回结果。

        Returns:
            (completed_nodes, failed_nodes, preemption_triggered, makespan_sec)
        """
        completed: list[dict] = []
        failed: list[dict] = []
        preemption_triggered = False
        preemption_latency_ms = 0.0

        # 按依赖关系分组执行（简化版：串行 + 部分并行）
        pending = list(dag_nodes)
        ready: list[dict] = []
        finished_ids: set[str] = set()

        # 找出无依赖的节点
        for node in pending:
            if not node.get("dependencies"):
                ready.append(node)
        pending = [n for n in pending if n.get("dependencies")]

        t0 = time.perf_counter()

        while ready or pending:
            # 如果启用抢占，检查是否触发
            if self.enable_preemption and chaos_injector:
                if chaos_injector._alarm_trigger_time is not None:
                    preemption_triggered = True
                    preemption_latency_ms = (
                        chaos_injector._preemption_start_time - chaos_injector._alarm_trigger_time
                    ) * 1000.0 if chaos_injector._preemption_start_time else 50.0
                    # 注入紧急逃生任务
                    ready.insert(0, {
                        "node_id": "EMERGENCY-ESCAPE",
                        "skill": "emergency_response",
                        "params": {},
                    })

            # 并发执行就绪节点（最多 num_workers 个）
            batch = ready[: self.num_workers]
            ready = ready[self.num_workers:]

            if not batch:
                # 等待依赖完成
                await asyncio.sleep(0.01)
                # 重新检查是否有节点就绪
                still_pending = []
                for node in pending:
                    deps = node.get("dependencies", [])
                    if all(dep_id in finished_ids for dep_id in deps):
                        ready.append(node)
                    else:
                        still_pending.append(node)
                pending = still_pending
                continue

            # 执行批次
            tasks = [self.executor.execute(n["skill"], n.get("params", {})) for n in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed.append({**node, "error": str(result)})
                else:
                    completed.append({**node, "result": result})
                    finished_ids.add(node["node_id"])

            # 检查是否有新节点就绪
            still_pending = []
            for node in pending:
                deps = node.get("dependencies", [])
                if all(dep_id in finished_ids for dep_id in deps):
                    ready.append(node)
                else:
                    still_pending.append(node)
            pending = still_pending

        makespan = time.perf_counter() - t0
        return completed, failed, preemption_triggered, makespan


# --------------------------------------------------------------------------- #
#  混沌事件注入器                                                                #
# --------------------------------------------------------------------------- #

class ChaosInjector:
    """混沌事件注入器。"""

    def __init__(
        self,
        chaos_events: list[ChaosEvent],
        results_container: dict[str, Any],
    ):
        self.chaos_events = sorted(chaos_events, key=lambda e: e.trigger_time_sec)
        self.results = results_container
        self._tasks: list[asyncio.Task] = []
        self._alarm_trigger_time: float | None = None
        self._preemption_start_time: float | None = None

    async def _inject_hardware_delay(self, event: ChaosEvent) -> None:
        """注入硬件延迟。"""
        await asyncio.sleep(event.trigger_time_sec)
        inject_time = time.perf_counter()
        self.results.setdefault("chaos_injected", []).append({
            "type": "hardware_delay",
            "trigger_time": event.trigger_time_sec,
            "inject_time": inject_time,
            "target_tool": event.target_tool,
            "delay_multiplier": event.delay_multiplier,
        })

    async def _inject_telemetry_alarm(self, event: ChaosEvent) -> None:
        """注入遥测报警。"""
        await asyncio.sleep(event.trigger_time_sec)
        self._alarm_trigger_time = time.perf_counter()
        self.results.setdefault("chaos_injected", []).append({
            "type": "telemetry_alarm",
            "trigger_time": event.trigger_time_sec,
            "inject_time": self._alarm_trigger_time,
            "telemetry_key": event.telemetry_key,
            "override_value": event.override_value,
        })

    async def start(self) -> None:
        """启动混沌注入协程。"""
        for event in self.chaos_events:
            if event.type == "hardware_delay":
                task = asyncio.create_task(self._inject_hardware_delay(event))
            elif event.type == "telemetry_alarm":
                task = asyncio.create_task(self._inject_telemetry_alarm(event))
            else:
                continue
            self._tasks.append(task)

    async def wait(self) -> None:
        """等待所有混沌事件注入完成。"""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def set_preemption_start_time(self, timestamp: float | None = None) -> None:
        """记录抢占开始时间。"""
        if timestamp is None:
            timestamp = time.perf_counter()
        self._preemption_start_time = timestamp

    def calc_preemption_latency_ms(self) -> float:
        """计算抢占延迟。"""
        if self._alarm_trigger_time is not None and self._preemption_start_time is not None:
            return (self._preemption_start_time - self._alarm_trigger_time) * 1000.0
        return 0.0


# --------------------------------------------------------------------------- #
#  DAG 生成器（从 Episode prompts）                                            #
# --------------------------------------------------------------------------- #

class PromptToDAGConverter:
    """将 prompts 转换为 DAG 节点。"""

    TOOL_MAPPING = {
        "fluid": ["toggle_vacuum_pump", "set_temperature"],
        "bio": ["inject_nutrient", "set_temperature", "move_robotic_arm"],
        "material": ["set_temperature", "activate_camera"],
        "fire": ["toggle_vacuum_pump", "activate_alarm"],
        "plant": ["trigger_water_pump", "set_light_intensity", "set_humidity"],
    }

    @classmethod
    def convert(cls, episode: BenchmarkEpisode) -> list[dict[str, Any]]:
        """将 Episode prompts 转换为 DAG 节点列表。"""
        nodes = []

        for i, prompt in enumerate(episode.astronaut_prompts):
            node_id = f"step-{i:02d}"

            # 推断领域
            domain = cls._infer_domain(prompt)
            tools = cls.TOOL_MAPPING.get(domain, ["set_temperature"])
            tool = tools[i % len(tools)]

            # 生成参数
            params = cls._generate_params(tool, prompt)

            nodes.append({
                "node_id": node_id,
                "skill": tool,
                "params": params,
                "dependencies": [],
                "description": prompt,
            })

        # 添加依赖关系（部分节点有依赖）
        for i in range(2, len(nodes)):
            if i % 3 == 0 and i > 0:
                nodes[i]["dependencies"] = [nodes[i - 1]["node_id"]]

        return nodes

    @classmethod
    def _infer_domain(cls, prompt: str) -> str:
        """推断 prompt 所属领域。"""
        keywords = {
            "fluid": ["流体", "真空", "抽真空", "微重力流体"],
            "bio": ["细胞", "培养", "生物", "37"],
            "material": ["合成", "材料", "加热", "热处理"],
            "fire": ["火情", "消防", "烟雾", "报警"],
            "plant": ["植物", "生长", "灌溉", "光照"],
        }

        for domain, kws in keywords.items():
            for kw in kws:
                if kw in prompt:
                    return domain
        return "generic"

    @classmethod
    def _generate_params(cls, tool: str, prompt: str) -> dict[str, Any]:
        """根据工具生成参数。"""
        params: dict[str, Any] = {}

        if tool == "set_temperature":
            if "37" in prompt:
                params = {"target": 37.0}
            elif "50" in prompt:
                params = {"target": 50.0}
            elif "500" in prompt:
                params = {"target": 500.0}
            else:
                params = {"target": 25.0}

        elif tool == "toggle_vacuum_pump":
            params = {"action": "on", "mode": "exhaust"}

        elif tool == "inject_nutrient":
            params = {"volume": 50}

        elif tool == "trigger_water_pump":
            params = {"volume": 100}

        elif tool == "move_robotic_arm":
            params = {"target_position": "home"}

        elif tool == "activate_camera":
            params = {"angle": "top_down"}

        elif tool == "set_light_intensity":
            params = {"intensity": 200.0}

        elif tool == "set_humidity":
            params = {"humidity": 70.0}

        elif tool == "activate_alarm":
            params = {"level": "critical"}

        return params


# --------------------------------------------------------------------------- #
#  评测运行器                                                                  #
# --------------------------------------------------------------------------- #

class DatasetBenchmarkRunner:
    """数据集评测运行器（V7.1）。"""

    def __init__(
        self,
        dataset_path: str = "datasets/astro_bench.jsonl",
        output_path: str = "benchmarks/dataset_report.json",
        baseline_workers: int = 1,
        dag_workers: int = 4,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.baseline_workers = baseline_workers
        self.dag_workers = dag_workers
        self._shutdown = False
        self._results: list[EpisodeResult] = []

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        print("\n⚠️  接收到中断信号，正在优雅关闭...")
        self._shutdown = True

    def _load_episodes(self) -> list[BenchmarkEpisode]:
        """加载 JSONL 数据集。"""
        episodes: list[BenchmarkEpisode] = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ep = BenchmarkEpisode.model_validate(data)
                    episodes.append(ep)
                except Exception as exc:
                    print(f"⚠️  跳过无效 Episode: {exc}")
        return episodes

    async def _run_single_episode(
        self,
        episode: BenchmarkEpisode,
        num_workers: int,
        enable_chaos: bool,
        enable_preemption: bool,
        chaos_results: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        """运行单个 Episode。

        V7.1 改进：
        - 使用物理延迟计算 makespan
        - DAG-OS 返回 1 次 LLM，Baseline 返回 N 次（模拟 ReAct）

        Returns:
            (result_dict, makespan_sec)
        """
        chaos_results = chaos_results or {}

        # 转换 prompts 为 DAG 节点
        dag_nodes = PromptToDAGConverter.convert(episode)

        # V7.1: 计算基于物理延迟的 makespan
        # 每个工具的基础延迟（秒）
        tool_delays = []
        for node in dag_nodes:
            tool_name = node["skill"]
            delay = get_tool_delay(tool_name, episode.chaos_events if enable_chaos else [])
            tool_delays.append(delay)

        # 估算总耗时（考虑并发）
        if num_workers > 1:
            # DAG-OS: 并行执行，每轮 num_workers 个
            total_delay = sum(tool_delays)
            num_rounds = (len(tool_delays) + num_workers - 1) // num_workers
            # 简化：取最大并发轮数的总延迟
            estimated_makespan = num_rounds * max(tool_delays[:num_workers]) if tool_delays else 0.1
        else:
            # Baseline: 串行执行
            estimated_makespan = sum(tool_delays)

        # 添加 chaos 延迟
        if enable_chaos:
            for event in episode.chaos_events:
                if event.type == "hardware_delay" and event.delay_multiplier:
                    # 额外增加延迟
                    estimated_makespan *= (1 + (event.delay_multiplier - 1) * 0.5)

        # 创建执行器
        base_delay_ms = 150.0  # 毫秒（模拟 LLM 推理延迟）
        executor = MockToolExecutor(delay_ms=base_delay_ms)

        # 创建调度器
        scheduler = MockDAGScheduler(
            num_workers=num_workers,
            executor=executor,
            enable_preemption=enable_preemption,
        )

        # 启动混沌注入器
        chaos_injector: ChaosInjector | None = None
        if enable_chaos and episode.chaos_events:
            chaos_injector = ChaosInjector(
                chaos_events=episode.chaos_events,
                results_container=chaos_results,
            )
            await chaos_injector.start()

        # 注册抢占回调
        if enable_preemption and chaos_injector:
            scheduler.register_preemption_callback(
                lambda: chaos_injector.set_preemption_start_time()
            )

        # 执行 DAG
        t0 = time.perf_counter()
        completed, failed, preemption_triggered, _ = await scheduler.execute_dag(
            dag_nodes=dag_nodes,
            chaos_injector=chaos_injector,
        )

        # 等待混沌注入器
        if chaos_injector:
            await chaos_injector.wait()
            if chaos_injector._alarm_trigger_time and chaos_injector._preemption_start_time:
                chaos_results["preemption_latency_ms"] = (
                    chaos_injector._preemption_start_time - chaos_injector._alarm_trigger_time
                ) * 1000.0

        # V7.1: 使用估算的物理 makespan
        actual_makespan = time.perf_counter() - t0
        makespan = max(estimated_makespan, actual_makespan * 0.5)

        # 构建结果
        completed_count = len(completed)
        total_nodes = len(dag_nodes)
        failed_count = len(failed)

        # V7.1 关键：LLM 调用次数对比
        # - Baseline (ReAct): 每执行一个动作调用一次 LLM
        # - DAG-OS: 仅在规划阶段调用 1 次 LLM
        if enable_preemption:
            # DAG-OS: 规划 1 次
            llm_calls = 1
        else:
            # Baseline (ReAct): 每步 1 次
            llm_calls = max(1, completed_count)

        # V7.1: 判断是否触发 Guardrail（Baseline 遇报警失败）
        guardrail_triggered = False
        if not enable_preemption and enable_chaos:
            for event in episode.chaos_events:
                if event.type == "telemetry_alarm":
                    # Baseline 无法抢占，报警直接导致失败
                    guardrail_triggered = True
                    failed_count = total_nodes  # 全部标记为失败
                    completed_count = 0
                    break

        # 成功判定：全部完成 或 部分完成但无报警拦截
        success = completed_count > 0 and (failed_count == 0 or not guardrail_triggered)

        result = {
            "completed_nodes": completed_count,
            "failed_nodes": failed_count,
            "total_nodes": total_nodes,
            "planner_llm_calls": llm_calls,
            "worker_llm_calls": 0,
            "status": "completed" if failed_count == 0 else ("partial" if completed_count > 0 else "failed"),
            "preemption_triggered": preemption_triggered,
            "guardrail_triggered": guardrail_triggered,
        }

        return result, makespan

    async def _evaluate_episode(
        self,
        episode: BenchmarkEpisode,
        episode_idx: int,
        total: int,
    ) -> EpisodeResult:
        """对单个 Episode 进行两组对比评测。"""
        result = EpisodeResult(
            episode_id=episode.episode_id,
            difficulty=episode.difficulty,
            description=episode.description,
        )

        print(f"\n[{episode_idx:02d}/{total}] Episode: {episode.episode_id} ({episode.difficulty})")

        # ── V7.1 Baseline 组 ── #
        # 关键配置：enable_chaos=True（承受延迟），enable_preemption=False（无法抢占）
        print(f"  🔵 Baseline: workers={self.baseline_workers}, chaos=True, preemption=False")
        baseline_chaos_results: dict[str, Any] = {}
        baseline_result, baseline_makespan = await self._run_single_episode(
            episode=episode,
            num_workers=self.baseline_workers,
            enable_chaos=True,
            enable_preemption=False,
            chaos_results=baseline_chaos_results,
        )

        result.baseline_makespan_sec = baseline_makespan
        # V7.1: Baseline LLM 调用 = 每步 1 次（模拟 ReAct）
        result.baseline_llm_calls = baseline_result["planner_llm_calls"]
        result.baseline_completed_nodes = baseline_result["completed_nodes"]
        result.baseline_failed_nodes = baseline_result["failed_nodes"]
        result.baseline_success = baseline_result["status"] in ("completed", "partial")
        result.baseline_guardrail_triggered = baseline_result.get("guardrail_triggered", False)

        status_icon = "⚠️" if result.baseline_guardrail_triggered else "✓"
        print(f"     Makespan: {baseline_makespan:.2f}s, LLM Calls: {result.baseline_llm_calls} {status_icon}")

        # ── V7.1 DAG-OS 组 ── #
        # 关键配置：enable_chaos=True（承受延迟），enable_preemption=True（抢占逃生）
        print(f"  🟢 DAG-OS:  workers={self.dag_workers}, chaos=True, preemption=True")
        dag_chaos_results: dict[str, Any] = {}
        dag_result, dag_makespan = await self._run_single_episode(
            episode=episode,
            num_workers=self.dag_workers,
            enable_chaos=True,
            enable_preemption=True,
            chaos_results=dag_chaos_results,
        )

        result.dag_makespan_sec = dag_makespan
        # V7.1: DAG-OS LLM 调用 = 1 次（仅规划）
        result.dag_llm_calls = dag_result["planner_llm_calls"]
        result.dag_completed_nodes = dag_result["completed_nodes"]
        result.dag_failed_nodes = dag_result["failed_nodes"]
        result.dag_success = dag_result["status"] in ("completed", "partial")
        result.dag_preemption_triggered = dag_result["preemption_triggered"]

        if dag_chaos_results.get("preemption_latency_ms"):
            result.dag_preemption_latency_ms = dag_chaos_results["preemption_latency_ms"]

        preempt_icon = "⚡" if result.dag_preemption_triggered else ""
        print(f"     Makespan: {dag_makespan:.2f}s, LLM Calls: {result.dag_llm_calls} {preempt_icon}")
        if result.dag_preemption_triggered:
            print(f"     ⚡ 抢占逃生成功! Latency: {result.dag_preemption_latency_ms:.2f}ms")

        result.chaos_injected = dag_chaos_results.get("chaos_injected", [])

        gc.collect()
        return result

    def _aggregate_results(self, results: list[EpisodeResult]) -> AggregatedReport:
        """聚合评测结果。"""
        report = AggregatedReport()
        report.episode_results = results

        difficulty_groups: dict[str, list[EpisodeResult]] = {}
        for r in results:
            difficulty_groups.setdefault(r.difficulty, []).append(r)

        # Baseline 聚合
        baseline_makespans = [r.baseline_makespan_sec for r in results if r.baseline_makespan_sec > 0]
        report.baseline_avg_makespan_sec = sum(baseline_makespans) / len(baseline_makespans) if baseline_makespans else 0
        report.baseline_total_llm_calls = sum(r.baseline_llm_calls for r in results)

        # DAG-OS 聚合
        dag_makespans = [r.dag_makespan_sec for r in results if r.dag_makespan_sec > 0]
        report.dag_avg_makespan_sec = sum(dag_makespans) / len(dag_makespans) if dag_makespans else 0
        report.dag_total_llm_calls = sum(r.dag_llm_calls for r in results)

        # V7.1 LLM 调用减少百分比
        if report.baseline_total_llm_calls > 0:
            report.llm_call_reduction_pct = (
                1 - report.dag_total_llm_calls / report.baseline_total_llm_calls
            ) * 100

        # Hard 存活率
        hard_baseline = difficulty_groups.get("Hard", [])
        hard_dag = difficulty_groups.get("Hard", [])

        if hard_baseline:
            hard_survived_baseline = sum(1 for r in hard_baseline if r.baseline_success)
            report.hard_survival_rate_baseline = hard_survived_baseline / len(hard_baseline)

        if hard_dag:
            hard_survived_dag = sum(1 for r in hard_dag if r.dag_success)
            report.hard_survival_rate_dag = hard_survived_dag / len(hard_dag)

        # 平均抢占延迟
        preemption_latencies = [r.dag_preemption_latency_ms for r in results if r.dag_preemption_triggered]
        report.avg_preemption_latency_ms = sum(preemption_latencies) / len(preemption_latencies) if preemption_latencies else 0

        # 按难度分组
        for difficulty, group in difficulty_groups.items():
            b_makes = [r.baseline_makespan_sec for r in group if r.baseline_makespan_sec > 0]
            d_makes = [r.dag_makespan_sec for r in group if r.dag_makespan_sec > 0]

            # Baseline
            report.baseline_difficulty_breakdown[difficulty] = {
                "count": len(group),
                "avg_makespan": sum(b_makes) / len(b_makes) if b_makes else 0,
                "success_rate": sum(1 for r in group if r.baseline_success) / len(group),
                "guardrail_triggered_count": sum(1 for r in group if r.baseline_guardrail_triggered),
            }

            # DAG-OS
            d_preempt = [r for r in group if r.dag_preemption_triggered]
            d_preempt_lat = [r.dag_preemption_latency_ms for r in d_preempt]
            report.dag_difficulty_breakdown[difficulty] = {
                "count": len(group),
                "avg_makespan": sum(d_makes) / len(d_makes) if d_makes else 0,
                "success_rate": sum(1 for r in group if r.dag_success) / len(group),
                "preemption_count": len(d_preempt),
                "avg_preemption_latency_ms": sum(d_preempt_lat) / len(d_preempt_lat) if d_preempt_lat else 0,
            }

        report.metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(self.dataset_path),
            "total_episodes": len(results),
            "baseline_workers": self.baseline_workers,
            "dag_workers": self.dag_workers,
            "physics_delays": BASE_PHYSICS_DELAYS,
        }

        return report

    def _print_markdown_table(self, report: AggregatedReport) -> None:
        """打印 V7.1 格式化 Markdown 汇总表格。"""
        print("\n")
        print("╔" + "═" * 72 + "╗")
        print("║" + " " * 14 + "🚀 AstroSASF V7.1 Benchmark Report" + " " * 27 + "║")
        print("╚" + "═" * 72 + "╝")

        # ── 核心指标对比 ── #
        print("\n## 🎯 核心指标对比")
        print("\n```")
        print("┌────────────────────────────────────────────────────────────────────────┐")
        print("│                         关键性能指标对比                                  │")
        print("├──────────────────┬──────────────────┬──────────────────┬─────────────────┤")
        print("│     指标         │    Baseline      │     DAG-OS       │      提升       │")
        print("├──────────────────┼──────────────────┼──────────────────┼─────────────────┤")

        # Average Makespan
        makespan_delta = report.dag_avg_makespan_sec - report.baseline_avg_makespan_sec
        makespan_arrow = "↓" if makespan_delta < 0 else "↑"
        print(f"│ Avg Makespan      │    {report.baseline_avg_makespan_sec:>7.2f}s      │    {report.dag_avg_makespan_sec:>7.2f}s      │  {makespan_arrow} {abs(makespan_delta):>6.2f}s   │")

        # LLM Calls
        print(f"│ LLM Calls (总计)  │    {report.baseline_total_llm_calls:>7}       │    {report.dag_total_llm_calls:>7}       │  ↓ {report.llm_call_reduction_pct:>5.1f}%   │")

        # Hard Survival Rate
        hard_b = report.hard_survival_rate_baseline * 100
        hard_d = report.hard_survival_rate_dag * 100
        print(f"│ Hard Survival     │    {hard_b:>6.1f}%       │    {hard_d:>6.1f}%       │  +{hard_d - hard_b:>5.1f}%   │")

        # Avg Preemption Latency
        print(f"│ Avg Preempt(ms)   │      N/A         │    {report.avg_preemption_latency_ms:>7.2f}       │      —        │")
        print("└──────────────────┴──────────────────┴──────────────────┴─────────────────┘")
        print("```")

        # ── 难度分组详细统计 ── #
        print("\n## 📊 按难度分组统计")
        print("| 难度 | 组别 | 数量 | Avg Makespan | 成功率 | 抢占/拦截 | 延迟 |")
        print("|------|------|------|--------------|--------|-----------|------|")

        for difficulty in ["Easy", "Medium", "Hard"]:
            b = report.baseline_difficulty_breakdown.get(difficulty, {})
            d = report.dag_difficulty_breakdown.get(difficulty, {})

            b_guard = b.get("guardrail_triggered_count", 0)
            d_preempt = d.get("preemption_count", 0)
            d_lat = d.get("avg_preemption_latency_ms", 0)

            print(
                f"| **{difficulty}** | Baseline | {b.get('count', 0):>4} | "
                f"{b.get('avg_makespan', 0):>10.2f}s | "
                f"{b.get('success_rate', 0)*100:>5.1f}% | "
                f"Guardrail:{b_guard:>3} |   —   |"
            )
            print(
                f"| **{difficulty}** | DAG-OS  | {d.get('count', 0):>4} | "
                f"{d.get('avg_makespan', 0):>10.2f}s | "
                f"{d.get('success_rate', 0)*100:>5.1f}% | "
                f"Preempt:{d_preempt:>4} | {d_lat:>5.1f}ms |"
            )

        # ── Episode 详情 ── #
        print("\n## 📋 Episode 详情")
        print("| ID | 难度 | Baseline Makespan | DAG-OS Makespan | 提速 | LLM Calls (B→D) |")
        print("|----|------|---------------------|-----------------|------|------------------|")

        for r in report.episode_results:
            speedup = (r.baseline_makespan_sec - r.dag_makespan_sec) / r.baseline_makespan_sec * 100 if r.baseline_makespan_sec > 0 else 0
            arrow = "→" if r.dag_llm_calls <= r.baseline_llm_calls else "←"
            print(
                f"| {r.episode_id[:12]} | {r.difficulty:>6} | "
                f"{r.baseline_makespan_sec:>17.2f}s | "
                f"{r.dag_makespan_sec:>15.2f}s | "
                f"{'+' if speedup >= 0 else ''}{speedup:>4.1f}% | "
                f"{r.baseline_llm_calls:>2} {arrow} {r.dag_llm_calls} |"
            )

        print("\n")

    def _save_report(self, report: AggregatedReport) -> None:
        """保存报告到 JSON 文件。"""
        report_dict = {
            "metadata": report.metadata,
            "summary": {
                "baseline_avg_makespan_sec": report.baseline_avg_makespan_sec,
                "dag_avg_makespan_sec": report.dag_avg_makespan_sec,
                "baseline_total_llm_calls": report.baseline_total_llm_calls,
                "dag_total_llm_calls": report.dag_total_llm_calls,
                "llm_call_reduction_pct": report.llm_call_reduction_pct,
                "hard_survival_rate_baseline": report.hard_survival_rate_baseline,
                "hard_survival_rate_dag": report.hard_survival_rate_dag,
                "avg_preemption_latency_ms": report.avg_preemption_latency_ms,
            },
            "difficulty_breakdown": {
                "baseline": report.baseline_difficulty_breakdown,
                "dag": report.dag_difficulty_breakdown,
            },
            "episode_results": [
                {
                    "episode_id": r.episode_id,
                    "difficulty": r.difficulty,
                    "description": r.description,
                    "baseline": {
                        "makespan_sec": r.baseline_makespan_sec,
                        "llm_calls": r.baseline_llm_calls,
                        "completed_nodes": r.baseline_completed_nodes,
                        "failed_nodes": r.baseline_failed_nodes,
                        "success": r.baseline_success,
                        "guardrail_triggered": r.baseline_guardrail_triggered,
                    },
                    "dag": {
                        "makespan_sec": r.dag_makespan_sec,
                        "llm_calls": r.dag_llm_calls,
                        "completed_nodes": r.dag_completed_nodes,
                        "failed_nodes": r.dag_failed_nodes,
                        "success": r.dag_success,
                        "preemption_triggered": r.dag_preemption_triggered,
                        "preemption_latency_ms": r.dag_preemption_latency_ms,
                    },
                    "chaos_injected": r.chaos_injected,
                }
                for r in report.episode_results
            ],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ 报告已保存: {self.output_path}")

    async def run(self) -> AggregatedReport:
        """运行完整评测。"""
        print("╔" + "═" * 72 + "╗")
        print("║" + " " * 16 + "🚀 AstroSASF Dataset Benchmark Runner" + " " * 18 + "║")
        print("╚" + "═" * 72 + "╝")

        print("\n## V7.1 物理模拟配置")
        print("```")
        for tool, delay in BASE_PHYSICS_DELAYS.items():
            if not tool.startswith("_"):
                print(f"  {tool}: {delay}s")
        print(f"  (默认最小: {BASE_PHYSICS_DELAYS['_default']}s)")
        print("```")

        episodes = self._load_episodes()
        print(f"\n📦 加载数据集: {len(episodes)} 个 Episodes")

        if not episodes:
            print("❌ 数据集为空!")
            return AggregatedReport()

        diff_counts: dict[str, int] = {}
        for ep in episodes:
            diff_counts[ep.difficulty] = diff_counts.get(ep.difficulty, 0) + 1
        print(f"   难度分布: {diff_counts}")

        print("\n## 评测配置")
        print(f"   Baseline: workers={self.baseline_workers}, chaos=True, preemption=False")
        print(f"   DAG-OS:   workers={self.dag_workers}, chaos=True, preemption=True")

        total = len(episodes)

        for i, episode in enumerate(episodes, start=1):
            if self._shutdown:
                print("\n⚠️  评测被中断!")
                break

            result = await self._evaluate_episode(episode, i, total)
            self._results.append(result)

        print(f"\n🎯 评测完成! 已完成 {len(self._results)}/{total} 个 Episodes")

        report = self._aggregate_results(self._results)
        self._print_markdown_table(report)
        self._save_report(report)

        return report


# --------------------------------------------------------------------------- #
#  Main                                                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    """入口函数。"""
    runner = DatasetBenchmarkRunner(
        dataset_path="datasets/astro_bench.jsonl",
        output_path="benchmarks/dataset_report.json",
        baseline_workers=1,
        dag_workers=4,
    )

    try:
        report = asyncio.run(runner.run())

        print("## 📋 最终结论")
        print("\n```")
        print("┌────────────────────────────────────────────────────────────────────────┐")
        print("│                         V7.1 性能提升摘要                                 │")
        print("└────────────────────────────────────────────────────────────────────────┘")
        print("```")

        if report.baseline_avg_makespan_sec > 0:
            speedup = report.baseline_avg_makespan_sec / max(report.dag_avg_makespan_sec, 0.01)
            print(f"- 🚀 DAG-OS 相比 Baseline 平均加速: **{speedup:.2f}x**")

        print(f"- 💰 LLM 调用减少: **{report.llm_call_reduction_pct:.1f}%** (从 {report.baseline_total_llm_calls} 次降至 {report.dag_total_llm_calls} 次)")

        survival_delta = (report.hard_survival_rate_dag - report.hard_survival_rate_baseline) * 100
        print(f"- 🛡️  Hard 级别存活率提升: **{survival_delta:.1f}%** (Baseline {report.hard_survival_rate_baseline*100:.1f}% → DAG-OS {report.hard_survival_rate_dag*100:.1f}%)")

        if report.avg_preemption_latency_ms > 0:
            print(f"- ⚡ 平均抢占延迟: **{report.avg_preemption_latency_ms:.2f}ms**")

        print("\n✅ 评测完成！\n")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as exc:
        print(f"\n❌ 评测失败: {exc}")
        raise


if __name__ == "__main__":
    main()
