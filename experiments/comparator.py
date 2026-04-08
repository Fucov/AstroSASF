"""
AstroSASF · Experiments · Comparator
================================
对比实验运行器 —— 6 个 baseline × N 个 episodes，输出对比表格。

Baseline 列表：
1. Sequential   — 严格顺序推进，物理动作阻塞
2. Async-only  — 异步提交但无乱序恢复
3. OoO-lite    — 事件驱动恢复，无完整空间/时间维度
4. OoO-proposed — 完整乱序调度框架
5. Lock-only   — 只加资源锁，无乱序
6. Resume-only — 只做事件恢复，无乱序

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── 动态项目根路径（支持 uv run / 直接 python / 任意 cwd）──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_generator import BenchmarkGenerator, BenchmarkEpisode, DifficultyLevel
from benchmarks.bench_suite import BenchmarkResult, BenchmarkSuite, SchedulerMode

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  Experiment Config                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class ExperimentConfig:
    baselines: list[SchedulerMode]
    episodes: list[BenchmarkEpisode]
    max_workers: int = 3
    seed: int = 42
    repeat: int = 1  # 每组重复次数（用于抖动分析）
    output_dir: Path = field(default_factory=lambda: Path("results"))


# ────────────────────────────────────────────────────────────────────────────── #
#  Comparator                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

class Comparator:
    """对比实验运行器。"""

    ALL_BASELINES: list[tuple[str, SchedulerMode]] = [
        ("Sequential",   SchedulerMode.SEQUENTIAL),
        ("Async-only",   SchedulerMode.ASYNC_ONLY),
        ("OoO-lite",     SchedulerMode.OOO_LITE),
        ("OoO-proposed", SchedulerMode.OOO_PROPOSED),
        ("Lock-only",    SchedulerMode.LOCK_ONLY),
        ("Resume-only",  SchedulerMode.RESUME_ONLY),
    ]

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._all_results: list[dict[str, Any]] = []

    async def run(self) -> dict[str, Any]:
        """运行完整对比实验。"""
        baselines = self.config.baselines
        episodes = self.config.episodes
        repeats = self.config.repeat
        output_dir = self.config.output_dir

        print(f"\n{'='*70}")
        print(f"  AstroSASF 对比实验")
        print(f"  Baselines: {[b.value for b in baselines]}")
        print(f"  Episodes:  {len(episodes)}")
        print(f"  Repeats:   {repeats}")
        print(f"{'='*70}\n")

        total_runs = len(baselines) * len(episodes) * repeats
        run_idx = 0

        for baseline in baselines:
            print(f"\n{'─'*70}")
            print(f"  ▶ Baseline: {baseline.value} ({baselines.index(baseline)+1}/{len(baselines)})")
            print(f"{'─'*70}")

            suite = BenchmarkSuite(
                scheduler_mode=baseline,
                seed=self.config.seed,
                max_workers=self.config.max_workers,
                verbose=True,
            )

            for rep in range(repeats):
                for ep in episodes:
                    run_idx += 1
                    print(f"  [{run_idx}/{total_runs}] {ep.episode_id} (rep={rep+1})")

                    result = await suite.run_episode(ep)

                    # 记录结果
                    row = {
                        "baseline": baseline.value,
                        "episode_id": ep.episode_id,
                        "repeat": rep + 1,
                        "scenario_type": ep.scenario_type.value,
                        "difficulty": ep.difficulty.value,
                        "success": result.success,
                        "makespan_s": result.metrics.get("makespan_s", 0),
                        "scheduling_overhead": result.metrics.get("scheduling_overhead", 0),
                        "success_rate": result.metrics.get("success_rate", 0),
                        "physical_device_utilization": result.metrics.get("physical_device_utilization", 0),
                        "cpu_busy_ratio": result.metrics.get("cpu_busy_ratio", 0),
                        "avg_task_wait_time_ms": result.metrics.get("avg_task_wait_time_ms", 0),
                        "overlap_ratio": result.metrics.get("overlap_ratio", 0),
                        "conflict_stall_time_ms": result.metrics.get("conflict_stall_time_ms", 0),
                        "resume_latency_ms": result.metrics.get("resume_latency_ms", 0),
                        "prefix_routing_hit_rate": result.metrics.get("prefix_routing_hit_rate", 0),
                        "alarm_response_latency_ms": result.metrics.get("alarm_response_latency_ms", 0),
                        "safety_rejection_rate": result.metrics.get("safety_rejection_rate", 0),
                        "jains_fairness": result.metrics.get("jains_fairness", 0),
                        "jitter_ms": result.metrics.get("jitter_ms", 0),
                        "ooo_promotion_count": result.metrics.get("ooo_promotion_count", 0),
                        "error": result.error or "",
                    }
                    self._all_results.append(row)

                    # 增量写入
                    self._append_csv_row(output_dir / "all_results.csv", row)

        # 生成汇总表格
        summary = self._compute_summary()
        self._save_summary(output_dir / "summary.json", summary)
        self._print_summary_table(summary)

        return summary

    def _compute_summary(self) -> dict[str, Any]:
        """按 baseline × scenario_type 汇总。"""
        from collections import defaultdict

        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for row in self._all_results:
            key = (row["baseline"], row["scenario_type"])
            for metric in [
                "makespan_s", "success_rate", "overlap_ratio",
                "conflict_stall_time_ms", "ooo_promotion_count",
                "cpu_busy_ratio", "avg_task_wait_time_ms",
            ]:
                groups[key][metric].append(row[metric])

        summary = {}
        for (baseline, scenario), metrics in groups.items():
            if baseline not in summary:
                summary[baseline] = {}
            summary[baseline][scenario] = {
                m: (sum(vals) / len(vals) if vals else 0)
                for m, vals in metrics.items()
            }

        return summary

    def _append_csv_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            keys = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_summary(self, path: Path, summary: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 汇总已保存 → {path}")

    def _print_summary_table(self, summary: dict) -> None:
        """打印汇总表格（可直接复制到论文中）。"""
        print(f"\n{'='*80}")
        print(f"  对比实验汇总表")
        print(f"{'='*80}")
        print(f"{'Baseline':<20} {'Scenario':<20} {'Makespan':>10} {'SuccRate':>10} "
              f"{'Overlap':>10} {'Conflict':>12} {'OoO#':>8}")
        print(f"{'-'*80}")

        for baseline, scenarios in summary.items():
            first = True
            for scenario, metrics in scenarios.items():
                label = baseline if first else ""
                first = False
                print(
                    f"{label:<20} {scenario:<20} "
                    f"{metrics.get('makespan_s', 0):>10.3f} "
                    f"{metrics.get('success_rate', 0):>10.1%} "
                    f"{metrics.get('overlap_ratio', 0):>10.1%} "
                    f"{metrics.get('conflict_stall_time_ms', 0):>12.1f} "
                    f"{metrics.get('ooo_promotion_count', 0):>8.0f}"
                )

        print(f"{'='*80}")


# ────────────────────────────────────────────────────────────────────────────── #
#  便捷入口                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

async def run_comparison(
    baselines: list[SchedulerMode] | None = None,
    tiers: list[str] | None = None,
    difficulty: str | None = None,
    episodes_per_baseline: int = 5,
    output_dir: str = "results/comparison",
) -> dict[str, Any]:
    """快速运行对比实验的便捷入口。"""
    if baselines is None:
        baselines = [m for _, m in Comparator.ALL_BASELINES]

    # 生成 benchmark
    gen = BenchmarkGenerator(seed=42)
    all_eps = gen.generate_full_suite()

    # 过滤
    filtered = []
    for ep in all_eps:
        if tiers and ep.scenario_type.value not in tiers:
            continue
        if difficulty and ep.difficulty.value != difficulty:
            continue
        filtered.append(ep)
        if len(filtered) >= episodes_per_baseline * len(tiers or ["all"]):
            break

    config = ExperimentConfig(
        baselines=baselines,
        episodes=filtered[:episodes_per_baseline],
        seed=42,
        repeat=1,
        output_dir=Path(output_dir),
    )

    comparator = Comparator(config)
    return await comparator.run()


async def _quick_test() -> None:
    """快速验证（开发调试用）。"""
    eps = BenchmarkGenerator(seed=42).generate_tier1(
        count=2, difficulty=DifficultyLevel.EASY,
    )
    config = ExperimentConfig(
        baselines=[SchedulerMode.SEQUENTIAL, SchedulerMode.OOO_PROPOSED],
        episodes=eps,
        seed=42,
        repeat=1,
        output_dir=Path("results/quick_test"),
    )
    await Comparator(config).run()


async def main() -> None:
    """可作为 console_scripts 入口的 main 函数。"""
    import argparse

    parser = argparse.ArgumentParser(description="AstroSASF 对比实验")
    parser.add_argument("--tiers", nargs="+", default=None,
                        choices=["no_conflict", "light_conflict", "heavy_conflict", "alarm_recovery"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="results/comparison")
    args = parser.parse_args()

    await run_comparison(
        tiers=args.tiers,
        episodes_per_baseline=args.episodes,
        output_dir=args.output_dir,
    )
    print(f"\n实验完成，结果保存至 {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
