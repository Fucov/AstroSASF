#!/usr/bin/env python3
"""
AstroSASF · Dataset Benchmark Runner (V7.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
执行 `datasets/astro_bench.jsonl` 中的评测数据，进行 Ablation Study 对比测试。

对比维度：
- Baseline 组：num_workers=1, enable_chaos=False, enable_preemption=False
- DAG-OS 组：num_workers=4, enable_chaos=True, enable_preemption=True

输出：
- benchmarks/dataset_report.json（详细报告）
- 终端打印 Markdown 表格

Usage:
    python benchmarks/run_dataset_eval.py

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import asyncio
import gc
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_dataset import BenchmarkEpisode, ChaosEvent


# --------------------------------------------------------------------------- #
#  评测结果数据结构                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeResult:
    """单个 Episode 的评测结果。"""
    episode_id: str
    difficulty: str
    description: str
    # Baseline 结果
    baseline_makespan_sec: float = 0.0
    baseline_llm_calls: int = 0
    baseline_completed_nodes: int = 0
    baseline_failed_nodes: int = 0
    baseline_success: bool = False
    # DAG-OS 结果
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
    baseline_difficulty_breakdown: dict[str, dict] = field(default_factory=dict)
    dag_difficulty_breakdown: dict[str, dict] = field(default_factory=dict)
    episode_results: list[EpisodeResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  简化的 Mock 执行引擎（用于评测）                                            #
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
                    # 注入紧急任务
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
#  混沌事件注入器                                                            #
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
#  评测运行器                                                                #
# --------------------------------------------------------------------------- #

class DatasetBenchmarkRunner:
    """数据集评测运行器。"""

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

        Returns:
            (result_dict, makespan_sec)
        """
        chaos_results = chaos_results or {}

        # 转换 prompts 为 DAG 节点
        dag_nodes = PromptToDAGConverter.convert(episode)

        # 计算执行延迟（根据 chaos 调整）
        base_delay = 150.0  # 毫秒

        if enable_chaos:
            for event in episode.chaos_events:
                if event.type == "hardware_delay" and event.delay_multiplier:
                    base_delay *= event.delay_multiplier

        # 创建执行器
        executor = MockToolExecutor(delay_ms=base_delay)

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
        if enable_preemption:
            scheduler.register_preemption_callback(
                lambda: chaos_injector.set_preemption_start_time() if chaos_injector else None
            )

        # 执行 DAG
        t0 = time.perf_counter()
        completed, failed, preemption_triggered, makespan = await scheduler.execute_dag(
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

        # 构建结果
        result = {
            "completed_nodes": len(completed),
            "failed_nodes": len(failed),
            "total_nodes": len(dag_nodes),
            "planner_llm_calls": 1,  # 理论智能体固定调用 1 次
            "worker_llm_calls": 0,    # 实践智能体强控为 0
            "status": "completed" if len(failed) == 0 else ("partial" if completed else "failed"),
            "preemption_triggered": preemption_triggered,
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

        # ── Baseline 组 ── #
        print(f"  🔵 Baseline: workers={self.baseline_workers}, chaos=False, preemption=False")
        baseline_chaos_results: dict[str, Any] = {}
        baseline_result, baseline_makespan = await self._run_single_episode(
            episode=episode,
            num_workers=self.baseline_workers,
            enable_chaos=False,
            enable_preemption=False,
            chaos_results=baseline_chaos_results,
        )

        result.baseline_makespan_sec = baseline_makespan
        result.baseline_llm_calls = baseline_result["planner_llm_calls"]
        result.baseline_completed_nodes = baseline_result["completed_nodes"]
        result.baseline_failed_nodes = baseline_result["failed_nodes"]
        result.baseline_success = baseline_result["status"] in ("completed", "partial")

        print(f"     Makespan: {baseline_makespan:.2f}s, LLM Calls: {result.baseline_llm_calls}")

        # ── DAG-OS 组 ── #
        print(f"  🟢 DAG-OS: workers={self.dag_workers}, chaos=True, preemption=True")
        dag_chaos_results: dict[str, Any] = {}
        dag_result, dag_makespan = await self._run_single_episode(
            episode=episode,
            num_workers=self.dag_workers,
            enable_chaos=True,
            enable_preemption=True,
            chaos_results=dag_chaos_results,
        )

        result.dag_makespan_sec = dag_makespan
        result.dag_llm_calls = dag_result["planner_llm_calls"]
        result.dag_completed_nodes = dag_result["completed_nodes"]
        result.dag_failed_nodes = dag_result["failed_nodes"]
        result.dag_success = dag_result["status"] in ("completed", "partial")
        result.dag_preemption_triggered = dag_result["preemption_triggered"]

        if dag_chaos_results.get("preemption_latency_ms"):
            result.dag_preemption_latency_ms = dag_chaos_results["preemption_latency_ms"]

        print(f"     Makespan: {dag_makespan:.2f}s, LLM Calls: {result.dag_llm_calls}")
        if result.dag_preemption_triggered:
            print(f"     ⚡ Preemption Triggered! Latency: {result.dag_preemption_latency_ms:.2f}ms")

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

            report.baseline_difficulty_breakdown[difficulty] = {
                "count": len(group),
                "avg_makespan": sum(b_makes) / len(b_makes) if b_makes else 0,
                "success_rate": sum(1 for r in group if r.baseline_success) / len(group),
            }
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
        }

        return report

    def _print_markdown_table(self, report: AggregatedReport) -> None:
        """打印 Markdown 汇总表格。"""
        print("\n")
        print("╔" + "═" * 70 + "╗")
        print("║" + " " * 15 + "AstroSASF Benchmark Report (V7.1)" + " " * 23 + "║")
        print("╚" + "═" * 70 + "╝")

        print("\n## 📊 宏观对比指标")
        print("| 指标 | Baseline | DAG-OS | 提升 |")
        print("|------|----------|--------|------|")

        makespan_delta = (
            (report.baseline_avg_makespan_sec - report.dag_avg_makespan_sec)
            / report.baseline_avg_makespan_sec * 100
            if report.baseline_avg_makespan_sec > 0 else 0
        )
        print(f"| Average Makespan (秒) | {report.baseline_avg_makespan_sec:.2f} | {report.dag_avg_makespan_sec:.2f} | {'+' if makespan_delta > 0 else ''}{makespan_delta:.1f}% |")

        llm_improvement = (
            (report.baseline_total_llm_calls - report.dag_total_llm_calls)
            / report.baseline_total_llm_calls * 100
            if report.baseline_total_llm_calls > 0 else 0
        )
        print(f"| Overall LLM Calls | {report.baseline_total_llm_calls} | {report.dag_total_llm_calls} | {'+' if llm_improvement > 0 else ''}{llm_improvement:.1f}% |")

        hard_baseline_pct = report.hard_survival_rate_baseline * 100
        hard_dag_pct = report.hard_survival_rate_dag * 100
        hard_delta = hard_dag_pct - hard_baseline_pct
        print(f"| Hard Survival Rate | {hard_baseline_pct:.1f}% | {hard_dag_pct:.1f}% | {'+' if hard_delta > 0 else ''}{hard_delta:.1f}% |")

        print(f"| Avg Preemption Latency (ms) | N/A | {report.avg_preemption_latency_ms:.2f} | — |")

        print("\n## 📈 按难度分组统计")
        print("| 难度 | 组别 | 数量 | Avg Makespan | Success Rate | 抢占次数 | Avg 抢占延迟 |")
        print("|------|------|------|--------------|--------------|----------|--------------|")

        for difficulty in ["Easy", "Medium", "Hard"]:
            b = report.baseline_difficulty_breakdown.get(difficulty, {})
            d = report.dag_difficulty_breakdown.get(difficulty, {})

            print(f"| **{difficulty}** | Baseline | {b.get('count', 0)} | {b.get('avg_makespan', 0):.2f}s | {b.get('success_rate', 0)*100:.1f}% | — | — |")
            print(f"| **{difficulty}** | DAG-OS | {d.get('count', 0)} | {d.get('avg_makespan', 0):.2f}s | {d.get('success_rate', 0)*100:.1f}% | {d.get('preemption_count', 0)} | {d.get('avg_preemption_latency_ms', 0):.2f}ms |")

        print("\n## 🔬 Episode 详情")
        print("| Episode ID | 难度 | Baseline | DAG-OS | 提升 |")
        print("|------------|------|---------|--------|------|")

        for r in report.episode_results:
            if r.baseline_makespan_sec > 0 and r.dag_makespan_sec > 0:
                improvement = (r.baseline_makespan_sec - r.dag_makespan_sec) / r.baseline_makespan_sec * 100
                print(f"| {r.episode_id[:16]} | {r.difficulty} | {r.baseline_makespan_sec:.2f}s | {r.dag_makespan_sec:.2f}s | {'+' if improvement > 0 else ''}{improvement:.1f}% |")
            else:
                print(f"| {r.episode_id[:16]} | {r.difficulty} | {r.baseline_makespan_sec:.2f}s | {r.dag_makespan_sec:.2f}s | — |")

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
        print("╔" + "═" * 70 + "╗")
        print("║" + " " * 18 + "AstroSASF Dataset Benchmark Runner" + " " * 16 + "║")
        print("╚" + "═" * 70 + "╝")

        episodes = self._load_episodes()
        print(f"\n📦 加载数据集: {len(episodes)} 个 Episodes")

        if not episodes:
            print("❌ 数据集为空!")
            return AggregatedReport()

        diff_counts: dict[str, int] = {}
        for ep in episodes:
            diff_counts[ep.difficulty] = diff_counts.get(ep.difficulty, 0) + 1
        print(f"   难度分布: {diff_counts}")

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
        if report.baseline_avg_makespan_sec > 0:
            speedup = report.baseline_avg_makespan_sec / report.dag_avg_makespan_sec if report.dag_avg_makespan_sec > 0 else 1.0
            print(f"- DAG-OS 相比 Baseline 平均加速: **{speedup:.2f}x**")
        if report.dag_total_llm_calls > 0:
            llm_reduction = (1 - report.dag_total_llm_calls / max(report.baseline_total_llm_calls, 1)) * 100
            print(f"- LLM 调用减少: **{llm_reduction:.1f}%**")
        print(f"- Hard 级别存活率提升: **{(report.hard_survival_rate_dag - report.hard_survival_rate_baseline) * 100:.1f}%**")
        if report.avg_preemption_latency_ms > 0:
            print(f"- 平均抢占延迟: **{report.avg_preemption_latency_ms:.2f}ms**")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as exc:
        print(f"\n❌ 评测失败: {exc}")
        raise


if __name__ == "__main__":
    main()
