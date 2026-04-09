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
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class TeeWriter:
    """同时输出到 stdout 和日志文件的 writer（用于 print 重定向）。"""
    def __init__(self, stdout, log_path: Path):
        self.stdout = stdout
        self.log_path = log_path

    def write(self, text):
        self.stdout.write(text)
        self.stdout.flush()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    def flush(self):
        self.stdout.flush()

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
    use_dated_dir: bool = True  # 是否使用日期后缀区分实验
    physical_delay_scale: float = 1.0  # 物理延迟缩放因子（0.0-1.0，越小越快）


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

        # 确保 output_dir 是 Path 对象
        base_dir = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir

        # 构建带日期的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.use_dated_dir:
            output_dir = base_dir / f"comparison_{timestamp}"
        else:
            output_dir = base_dir

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志重定向到文件（同时捕获 root logger 的所有日志）
        log_file = output_dir / f"experiment_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        # 同时将 print 输出重定向到日志文件
        original_stdout = sys.stdout
        sys.stdout = TeeWriter(original_stdout, log_file)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        print(f"\n{'='*70}")
        print(f"  AstroSASF 对比实验")
        print(f"  Baselines: {[b.value for b in baselines]}")
        print(f"  Episodes:  {len(episodes)}")
        print(f"  Repeats:   {repeats}")
        print(f"  输出目录:   {output_dir}")
        print(f"  日志文件:   {log_file}")
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
                physical_delay_scale=self.config.physical_delay_scale,
                ablated_dims=set(),  # 对比实验不消融任何机制
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

        # 移除临时日志文件处理器
        sys.stdout = original_stdout
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()

        print(f"\n日志已保存: {log_file}")
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
    use_dated_dir: bool = True,
    physical_delay_scale: float = 0.1,
    benchmark_file: str | None = None,  # 新增：从文件加载 benchmark
) -> dict[str, Any]:
    """快速运行对比实验的便捷入口。
    
    采样策略：
    - tiers=None: 从所有场景类型均匀采样（每个场景类型 episodes_per_baseline 条）
    - tiers=['xxx']': 从指定场景类型均匀采样
    - difficulty 指定时：只从指定难度采样
    - benchmark_file 指定时：从文件加载 benchmark，忽略 tiers 参数
    """
    if baselines is None:
        baselines = [m for _, m in Comparator.ALL_BASELINES]

    # 加载 benchmark
    from benchmarks.bench_generator import BenchmarkGenerator
    if benchmark_file:
        # 从文件加载
        all_eps = BenchmarkGenerator.load(Path(benchmark_file))
        print(f"[加载] 从文件 {benchmark_file} 加载了 {len(all_eps)} 条 benchmark")
    else:
        # 动态生成
        gen = BenchmarkGenerator(seed=42)
        all_eps = gen.generate_full_suite()

    # 按场景类型分组
    from collections import defaultdict
    by_scenario: dict[str, list] = defaultdict(list)
    for ep in all_eps:
        by_scenario[ep.scenario_type.value].append(ep)

    # 过滤：根据场景类型和难度分组采样
    filtered: list = []
    
    if tiers is None:
        # 默认：从所有场景类型均匀采样
        target_tiers = list(by_scenario.keys())
    else:
        target_tiers = tiers

    for tier in target_tiers:
        tier_eps = by_scenario.get(tier, [])
        if not tier_eps:
            continue
        
        # 按难度分组
        by_difficulty: dict[str, list] = defaultdict(list)
        for ep in tier_eps:
            by_difficulty[ep.difficulty.value].append(ep)
        
        if difficulty:
            # 只取指定难度
            selected = by_difficulty.get(difficulty, [])
        else:
            # 均匀从各难度采样
            selected = []
            diffs = list(by_difficulty.keys())
            per_diff = max(1, episodes_per_baseline // len(diffs))
            for diff_eps in by_difficulty.values():
                # 打乱顺序保证随机性
                import random
                random.seed(42)
                shuffled = diff_eps.copy()
                random.shuffle(shuffled)
                selected.extend(shuffled[:per_diff])
        
        # 限制数量
        filtered.extend(selected[:episodes_per_baseline])
    
    # 打印采样信息
    scenario_counts = defaultdict(int)
    for ep in filtered:
        scenario_counts[ep.scenario_type.value] += 1
    print(f"[采样信息] 共 {len(filtered)} 条: {dict(scenario_counts)}")

    config = ExperimentConfig(
        baselines=baselines,
        episodes=filtered,
        seed=42,
        repeat=1,
        output_dir=Path(output_dir),
        use_dated_dir=use_dated_dir,
        physical_delay_scale=physical_delay_scale,
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
                        choices=["no_conflict", "light_conflict", "heavy_conflict", "alarm_recovery", 
                                 "global_shared", "diamond_deep"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="results/comparison")
    parser.add_argument("--no-dated", action="store_true", help="禁用日期后缀目录")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="物理延迟缩放因子（0.0-1.0），越小实验越快，默认0.1")
    parser.add_argument("--benchmark-file", type=str, default=None,
                        help="从指定文件加载 benchmark（JSONL 格式）")
    args = parser.parse_args()

    summary = await run_comparison(
        tiers=args.tiers,
        episodes_per_baseline=args.episodes,
        output_dir=args.output_dir,
        use_dated_dir=not args.no_dated,
        physical_delay_scale=args.speed,
        benchmark_file=args.benchmark_file,
    )
    print(f"\n实验完成！")


if __name__ == "__main__":
    asyncio.run(main())
