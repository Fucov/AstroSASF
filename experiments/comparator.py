"""
AstroSASF · Experiments · Comparator
====================================
主实验运行器 —— 3 个主方法 × N 个场景，输出论文级对比表格。

主实验（Main Comparison）——回答"我们的方案比传统基线好多少"：
1. Sequential       — 严格顺序推进，物理动作阻塞（理论下界）
2. Traditional DAG — DAG 层并发推进（asyncio.gather），无乱序/事件驱动/资源感知
3. OoO-proposed     — 完整乱序调度框架

实验设计原则：
- 每个场景类型均匀采样，确保可复现性
- 输出适合直接写入论文的汇总表格
- 物理延迟缩放因子 speed=0.1（中等速度，调度开销占比合理）

Author: AstroSASF Team
Version: 9.0
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ── 动态项目根路径（支持 uv run / 直接 python / 任意 cwd）──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_generator import BenchmarkGenerator, BenchmarkEpisode, ScenarioType
from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  Experiment Config                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class ExperimentConfig:
    """主实验配置。"""
    baselines: list[SchedulerMode]
    episodes: list[BenchmarkEpisode]
    max_workers: int = 3
    seed: int = 42
    repeat: int = 1
    output_dir: Path = field(default_factory=lambda: Path("results"))
    use_dated_dir: bool = True
    physical_delay_scale: float = 0.1


# 主实验默认场景（聚焦最有区分度的核心场景）
MAIN_DEFAULT_SCENARIOS = [
    "heavy_conflict",   # 重度资源竞争（最能体现锁机制差异）
    "global_shared",    # 全局共享设备竞争（OoO 越级调度的核心场景）
    "diamond_deep",     # 深层钻石形依赖（乱序越级极限测试）
]

# 场景分组：哪些在普通 suite，哪些在 extreme suite
EXTREME_SCENARIOS = {"global_shared", "diamond_deep"}


# ────────────────────────────────────────────────────────────────────────────── #
#  Comparator                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

class Comparator:
    """主实验运行器（Main Comparison）。"""

    # 主实验三方法（与论文 Table 1 对应）
    MAIN_BASELINES: list[tuple[str, SchedulerMode]] = [
        ("Sequential",       SchedulerMode.SEQUENTIAL),
        ("Traditional DAG",  SchedulerMode.TRADITIONAL_DAG),
        ("OoO-proposed",     SchedulerMode.OOO_PROPOSED),
    ]

    # 主实验默认场景（聚焦最有区分度的核心场景）
    MAIN_DEFAULT_SCENARIOS = [
        "heavy_conflict",   # 重度资源竞争（最能体现锁机制差异）
        "global_shared",    # 全局共享设备竞争（OoO 越级调度的核心场景）
        "diamond_deep",     # 深层钻石形依赖（乱序越级极限测试）
    ]

    # 全量场景（用于全面实验）
    ALL_SCENARIOS = [
        "no_conflict", "light_conflict", "heavy_conflict",
        "alarm_recovery", "ooo_stress", "global_shared", "diamond_deep",
    ]

    # 场景分组：哪些在普通 suite，哪些在 extreme suite
    EXTREME_SCENARIOS = {"global_shared", "diamond_deep"}

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._all_results: list[dict[str, Any]] = []

    async def run(self) -> dict[str, Any]:
        """运行主实验。"""
        baselines = self.config.baselines
        episodes = self.config.episodes
        repeats = self.config.repeat

        output_dir = self._setup_output_dir()

        print(f"\n{'='*72}")
        print(f"  AstroSASF 主实验 (Main Comparison)")
        print(f"  方法: {[b.display_name() for b in baselines]}")
        print(f"  场景数: {len(episodes)}")
        print(f"  每组重复: {repeats}")
        print(f"  Speed: {self.config.physical_delay_scale}")
        print(f"  输出: {output_dir}")
        print(f"{'='*72}\n")

        total_runs = len(baselines) * len(episodes) * repeats
        run_idx = 0

        for baseline in baselines:
            bs_name = baseline.display_name()
            print(f"\n{'─'*72}")
            print(f"  ▶ {bs_name} ({baselines.index(baseline)+1}/{len(baselines)})")
            print(f"{'─'*72}")

            suite = BenchmarkSuite(
                scheduler_mode=baseline,
                seed=self.config.seed,
                max_workers=self.config.max_workers,
                verbose=True,
                physical_delay_scale=self.config.physical_delay_scale,
                ablated_dims=set(),
            )

            for rep in range(repeats):
                for ep in episodes:
                    run_idx += 1
                    print(f"  [{run_idx}/{total_runs}] {ep.episode_id} (rep={rep+1})")

                    result = await suite.run_episode(ep)

                    row = {
                        "baseline": baseline.value,
                        "baseline_display": bs_name,
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
                    self._append_csv_row(output_dir / "all_results.csv", row)

        summary = self._compute_summary()
        self._save_summary(output_dir / "summary.json", summary)
        self._print_paper_table(summary)

        print(f"\n✅ 主实验完成 → {output_dir}")
        return summary

    def _setup_output_dir(self) -> Path:
        base = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir
        if self.config.use_dated_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = base / f"main_{ts}"
        else:
            out = base
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _compute_summary(self) -> dict[str, Any]:
        """按 baseline × scenario_type 汇总（均值）。"""
        from collections import defaultdict
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for row in self._all_results:
            key = (row["baseline_display"], row["scenario_type"])
            for m in [
                "makespan_s", "success_rate", "overlap_ratio",
                "conflict_stall_time_ms", "ooo_promotion_count",
                "cpu_busy_ratio", "avg_task_wait_time_ms",
                "alarm_response_latency_ms",
            ]:
                groups[key][m].append(row[m])

        summary = {}
        for (bs, sc), metrics in groups.items():
            if bs not in summary:
                summary[bs] = {}
            summary[bs][sc] = {m: (sum(v) / len(v) if v else 0) for m, v in metrics.items()}
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

    def _print_paper_table(self, summary: dict) -> None:
        """
        打印论文级汇总表格（Table 1）。
        格式设计：按场景分行，每个 baseline 一组指标，清晰展示 OoO-proposed 的优势。
        """
        all_scenarios = set()
        for bs_data in summary.values():
            all_scenarios.update(bs_data.keys())
        scenarios_sorted = sorted(all_scenarios)

        print(f"\n{'='*90}")
        print(f"  Table 1: 主实验结果 (Main Comparison, speed={self.config.physical_delay_scale})")
        print(f"{'='*90}")

        # 表头
        print(f"\n{'Scenario':<22} {'Method':<18} {'Makespan':>9} {'SuccRate':>9} "
              f"{'Overlap':>9} {'Conf.Stall':>12} {'Promo#':>7}")
        print(f"{'-'*90}")

        for sc in scenarios_sorted:
            first = True
            sc_span = len([bs for bs in summary if sc in summary[bs]])
            count = 0
            for bs, bs_data in summary.items():
                if sc not in bs_data:
                    continue
                metrics = bs_data[sc]
                count += 1
                sc_label = sc if first else ""
                bs_label = bs if count == 1 else bs
                print(
                    f"{sc_label:<22} {bs_label:<18} "
                    f"{metrics.get('makespan_s', 0):>9.3f}s "
                    f"{metrics.get('success_rate', 0):>8.1%} "
                    f"{metrics.get('overlap_ratio', 0):>8.1%} "
                    f"{metrics.get('conflict_stall_time_ms', 0):>11.1f} "
                    f"{metrics.get('ooo_promotion_count', 0):>7.0f}"
                )
                if count == 1:
                    first = False

        print(f"{'='*90}")

        # 打印 OoO-proposed vs Traditional DAG 的加速比
        print(f"\n{'─'*90}")
        print(f"  OoO-proposed 相对于 Traditional DAG 的加速比（Makespan 降低）:")
        print(f"{'─'*90}")
        for sc in scenarios_sorted:
            td = summary.get("Traditional DAG", {}).get(sc, {}).get("makespan_s", 0)
            oo = summary.get("OoO-proposed", {}).get(sc, {}).get("makespan_s", 0)
            if td > 0 and oo > 0:
                improvement = (td - oo) / td * 100
                print(f"  {sc:<22}: {improvement:>+.1f}%  ({td:.3f}s → {oo:.3f}s)")
        print(f"{'─'*90}")


# ────────────────────────────────────────────────────────────────────────────── #
#  便捷入口                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

async def run_main_comparison(
    scenarios: list[str] | None = None,
    episodes_per_scenario: int = 3,
    output_dir: str = "results/main_comparison",
    physical_delay_scale: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """
    运行主实验的便捷入口。

    参数：
        scenarios: 要测试的场景类型列表（默认: heavy_conflict, global_shared, diamond_deep）
        episodes_per_scenario: 每个场景采样的 episode 数量
        physical_delay_scale: 物理延迟缩放因子（默认 0.1）
        seed: 随机种子（默认 42，保证可复现）
    """
    from collections import defaultdict

    # 默认场景：主实验聚焦最有区分度的场景
    if scenarios is None:
        scenarios = MAIN_DEFAULT_SCENARIOS

    gen = BenchmarkGenerator(seed=seed)

    # 分别从普通 suite 和 extreme suite 中采样
    # global_shared 和 diamond_deep 在 generate_extreme_full_suite 中
    normal_scenarios = [s for s in scenarios if s not in EXTREME_SCENARIOS]
    extreme_scenarios = [s for s in scenarios if s in EXTREME_SCENARIOS]

    # 加载普通 suite（tier1~tier4）
    normal_eps = gen.generate_full_suite()

    # 加载 extreme suite（global_shared, diamond_deep）
    extreme_eps = gen.generate_extreme_full_suite()

    # 按场景类型分组
    by_scenario: dict[str, list] = defaultdict(list)
    for ep in normal_eps:
        by_scenario[ep.scenario_type.value].append(ep)
    for ep in extreme_eps:
        by_scenario[ep.scenario_type.value].append(ep)

    # 采样：每个场景均匀采样
    filtered: list[BenchmarkEpisode] = []
    for sc in scenarios:
        sc_eps = by_scenario.get(sc, [])
        if not sc_eps:
            print(f"[警告] 场景 '{sc}' 无 episode，跳过")
            continue
        # 打乱 + 取前 N 条（固定 seed 保证可复现）
        import random
        rng = random.Random(seed)
        shuffled = sc_eps.copy()
        rng.shuffle(shuffled)
        filtered.extend(shuffled[:episodes_per_scenario])

    if not filtered:
        raise ValueError("没有找到有效的 episode，请检查场景类型名称。")

    # 打印采样信息
    from collections import Counter
    counts = Counter(ep.scenario_type.value for ep in filtered)
    print(f"[采样] 共 {len(filtered)} 条: {dict(counts)}")

    config = ExperimentConfig(
        baselines=[m for _, m in Comparator.MAIN_BASELINES],
        episodes=filtered,
        seed=seed,
        repeat=1,
        output_dir=Path(output_dir),
        use_dated_dir=True,
        physical_delay_scale=physical_delay_scale,
    )

    comparator = Comparator(config)
    return await comparator.run()


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AstroSASF 主实验")
    parser.add_argument("--scenarios", nargs="+",
                        default=["global_shared", "diamond_deep", "heavy_conflict"],
                        choices=["no_conflict", "light_conflict", "heavy_conflict",
                                "alarm_recovery", "ooo_stress", "global_shared", "diamond_deep"])
    parser.add_argument("--episodes", type=int, default=3,
                        help="每个场景采样的 episode 数量")
    parser.add_argument("--output-dir", default="results/main_comparison")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="物理延迟缩放因子，默认 0.1")
    args = parser.parse_args()

    await run_main_comparison(
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes,
        output_dir=args.output_dir,
        physical_delay_scale=args.speed,
    )


if __name__ == "__main__":
    asyncio.run(main())
