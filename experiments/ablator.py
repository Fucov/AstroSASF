"""
AstroSASF · Experiments · Ablator
=============================
消融实验运行器 —— 验证空间/时间维度各机制对最终指标的贡献。

消融维度：
- 时间维度：去掉 checkpoint / prefix routing / event-driven wake-up / speculative subgraph
- 空间维度：去掉资源正交判定 / priority-aware lock / ordered lock acquisition / ART 生命周期

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import csv
import json
import asyncio
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 动态项目根路径 ────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_generator import BenchmarkGenerator, BenchmarkEpisode, DifficultyLevel
from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode, BenchmarkResult

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  Ablation Dimension                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class AblationDim:
    name: str
    description: str
    removes: list[str]  # 去掉的机制描述


# 时间维度消融
TEMPORAL_ABLATIONS = [
    AblationDim(
        name="no_checkpoint",
        description="去掉 checkpoint save/load",
        removes=["checkpoint_save", "checkpoint_load"],
    ),
    AblationDim(
        name="no_prefix_routing",
        description="去掉 prefix routing 缓存",
        removes=["prefix_routing"],
    ),
    AblationDim(
        name="no_event_wakeup",
        description="改为轮询+ sleep，不使用 event-driven wake-up",
        removes=["event_wakeup"],
    ),
    AblationDim(
        name="no_speculation",
        description="去掉 speculative subgraph generation",
        removes=["speculative_subgraph"],
    ),
]

# 空间维度消融
SPATIAL_ABLATIONS = [
    AblationDim(
        name="no_orthogonality_check",
        description="去掉资源正交判定（所有节点并发竞争设备）",
        removes=["orthogonality_check"],
    ),
    AblationDim(
        name="no_priority_lock",
        description="去掉 priority-aware lock，改为 FIFO 锁",
        removes=["priority_lock"],
    ),
    AblationDim(
        name="no_ordered_acquisition",
        description="去掉 ordered lock acquisition（随机顺序）",
        removes=["ordered_lock"],
    ),
    AblationDim(
        name="no_art_tracking",
        description="去掉 ActiveResourceTable 生命周期跟踪",
        removes=["art_lifecycle"],
    ),
]

FULL_BASELINE = AblationDim(
    name="full_proposed",
    description="完整算法基准（不做任何消融）",
    removes=[],
)

FULL_ABLATION = AblationDim(
    name="full_ablated",
    description="去掉全部空间+时间维度机制",
    removes=["checkpoint", "prefix_routing", "event_wakeup", "speculation",
             "orthogonality_check", "priority_lock", "ordered_lock", "art_lifecycle"],
)


# ────────────────────────────────────────────────────────────────────────────── #
#  Ablator Config                                                                #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class AblationExperiment:
    name: str
    ablated_dims: list[str]  # 被消融的维度名列表
    description: str


@dataclass
class AblatorConfig:
    base_episodes: list[BenchmarkEpisode]
    ablation_experiments: list[AblationExperiment]
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/ablation"))
    use_dated_dir: bool = True  # 是否使用日期后缀区分实验
    physical_delay_scale: float = 0.1  # 物理延迟缩放因子（0.0-1.0，越小越快）
    include_baseline: bool = True  # 是否同步跑完整原算法作为基准对照


# ────────────────────────────────────────────────────────────────────────────── #
#  Ablator                                                                      #
# ────────────────────────────────────────────────────────────────────────────── #

class Ablator:
    """消融实验运行器。"""

    def __init__(self, config: AblatorConfig) -> None:
        self.config = config
        self._results: list[dict[str, Any]] = []

    async def run(self) -> dict[str, Any]:
        """运行完整消融实验。"""
        episodes = self.config.base_episodes
        experiments = self.config.ablation_experiments
        include_baseline = self.config.include_baseline

        # 构建带日期的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir
        if self.config.use_dated_dir:
            output_dir = base_dir / f"ablation_{timestamp}"
        else:
            output_dir = base_dir

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志重定向到文件
        log_file = output_dir / f"experiment_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        print(f"\n{'='*70}")
        print(f"  AstroSASF 消融实验")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Ablation 实验: {len(experiments)}")
        print(f"  包含基准对照: {'是' if include_baseline else '否'}")
        print(f"  输出目录: {output_dir}")
        print(f"  日志文件: {log_file}")
        print(f"{'='*70}\n")

        # ── 第一步：跑完整原算法基准（仅当 include_baseline=True）─────────────────
        baseline_results: list[dict[str, Any]] = []
        if include_baseline:
            print(f"\n{'─'*70}")
            print(f"  ▶ Baseline: full_proposed (完整原算法，不做任何消融)")
            print(f"{'─'*70}")

            # 直接使用 BenchmarkSuite，不传 ablated_dims 表示完整原算法
            baseline_suite = BenchmarkSuite(
                scheduler_mode=SchedulerMode.OOO_PROPOSED,
                seed=self.config.seed,
                max_workers=3,
                verbose=False,
                physical_delay_scale=self.config.physical_delay_scale,
                ablated_dims=set(),  # 空集 = 不消融任何机制
            )

            for ep in episodes:
                print(f"  [Baseline] {ep.episode_id}")
                result = await baseline_suite.run_episode(ep)
                row = {
                    "ablation": "full_proposed",
                    "ablated_dims": "",
                    "episode_id": ep.episode_id,
                    "scenario_type": ep.scenario_type.value,
                    "difficulty": ep.difficulty.value,
                    "success": result.success,
                    "makespan_s": result.metrics.get("makespan_s", 0),
                    "success_rate": result.metrics.get("success_rate", 0),
                    "overlap_ratio": result.metrics.get("overlap_ratio", 0),
                    "conflict_stall_time_ms": result.metrics.get("conflict_stall_time_ms", 0),
                    "ooo_promotion_count": result.ooo_promotion_count,
                    "cpu_busy_ratio": result.metrics.get("cpu_busy_ratio", 0),
                    "avg_task_wait_time_ms": result.metrics.get("avg_task_wait_time_ms", 0),
                    "alarm_response_latency_ms": result.metrics.get("alarm_response_latency_ms", 0),
                    "safety_rejection_rate": result.metrics.get("safety_rejection_rate", 0),
                    "jains_fairness": result.metrics.get("jains_fairness", 0),
                    "error": result.error or "",
                }
                baseline_results.append(row)
                self._append_csv_row(output_dir / "ablation_results.csv", row)

            print(f"\n  ✅ Baseline 完成 ({len(episodes)} 条)")

        # ── 第二步：跑各消融实验 ───────────────────────────────────────────────
        total_runs = len(experiments) * len(episodes)
        run_idx = 0

        for exp in experiments:
            print(f"\n{'─'*70}")
            print(f"  ▶ Ablation: {exp.name}")
            print(f"     {exp.description}")
            print(f"     消融: {exp.ablated_dims}")
            print(f"{'─'*70}")

            # 直接使用 BenchmarkSuite，传入消融维度
            suite = BenchmarkSuite(
                scheduler_mode=SchedulerMode.OOO_PROPOSED,
                seed=self.config.seed,
                max_workers=3,
                verbose=False,
                physical_delay_scale=self.config.physical_delay_scale,
                ablated_dims=set(exp.ablated_dims),  # 传入消融维度
            )

            for ep in episodes:
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] {ep.episode_id}")

                result = await suite.run_episode(ep)

                row = {
                    "ablation": exp.name,
                    "ablated_dims": ",".join(exp.ablated_dims),
                    "episode_id": ep.episode_id,
                    "scenario_type": ep.scenario_type.value,
                    "difficulty": ep.difficulty.value,
                    "success": result.success,
                    "makespan_s": result.metrics.get("makespan_s", 0),
                    "success_rate": result.metrics.get("success_rate", 0),
                    "overlap_ratio": result.metrics.get("overlap_ratio", 0),
                    "conflict_stall_time_ms": result.metrics.get("conflict_stall_time_ms", 0),
                    "ooo_promotion_count": result.ooo_promotion_count,
                    "cpu_busy_ratio": result.metrics.get("cpu_busy_ratio", 0),
                    "avg_task_wait_time_ms": result.metrics.get("avg_task_wait_time_ms", 0),
                    "alarm_response_latency_ms": result.metrics.get("alarm_response_latency_ms", 0),
                    "safety_rejection_rate": result.metrics.get("safety_rejection_rate", 0),
                    "jains_fairness": result.metrics.get("jains_fairness", 0),
                    "error": result.error or "",
                }
                self._results.append(row)
                self._append_csv_row(output_dir / "ablation_results.csv", row)

        # 合并基准结果到 _results（用于汇总计算）
        self._results.extend(baseline_results)

        summary = self._compute_summary()
        self._save_summary(output_dir / "ablation_summary.json", summary)
        self._print_ablation_table(summary)

        # 移除临时日志文件处理器
        root_logger.removeHandler(file_handler)
        file_handler.close()

        print(f"\n日志已保存: {log_file}")
        return summary

    def _compute_summary(self) -> dict[str, Any]:
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        metric_keys = [
            "makespan_s", "success_rate", "overlap_ratio",
            "conflict_stall_time_ms", "ooo_promotion_count",
            "cpu_busy_ratio", "avg_task_wait_time_ms",
            "alarm_response_latency_ms",
        ]

        for row in self._results:
            key = row["ablation"]
            for m in metric_keys:
                groups[key][m].append(row[m])

        summary = {}
        for name, metrics in groups.items():
            summary[name] = {
                m: (sum(vals) / len(vals) if vals else 0)
                for m, vals in metrics.items()
            }
            summary[name]["_count"] = len(next(iter(metrics.values())))

        # ── 计算各消融维度相对于基准的 delta ──────────────────────────────
        baseline_metrics = summary.get("full_proposed", {})
        if baseline_metrics:
            for name, metrics in summary.items():
                if name == "full_proposed":
                    metrics["_is_baseline"] = True
                    continue
                for m in metric_keys:
                    base_val = baseline_metrics.get(m, 0)
                    ablated_val = metrics.get(m, 0)
                    if base_val != 0:
                        # delta 用百分比表示（正=变差，负=变好）
                        metrics[f"{m}_delta_pct"] = ((ablated_val - base_val) / base_val) * 100
                    else:
                        metrics[f"{m}_delta_pct"] = 0.0

        return summary

    def _append_csv_row(self, path: Path, row: dict) -> None:
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
        print(f"\n✅ 消融汇总已保存 → {path}")

    def _print_ablation_table(self, summary: dict) -> None:
        """打印消融实验汇总表（含基准对照和相对 delta）。"""
        print(f"\n{'='*120}")
        print(f"  消融实验汇总表（相对于 full_proposed 基准）")
        print(f"{'='*120}")
        print(f"{'Ablation':<22} {'Makespan':>10} {'ΔMakespan':>11} {'SuccRate':>9} "
              f"{'Overlap':>9} {'ΔOverlap':>10} {'OoO#':>6} {'CPU%':>7} {'Wait(ms)':>10}")
        print(f"{'-'*120}")

        baseline = summary.get("full_proposed", {})
        has_baseline = bool(baseline)

        for name, metrics in summary.items():
            is_baseline = metrics.get("_is_baseline", False)
            prefix = "★ " if is_baseline else "  "

            makespan = metrics.get("makespan_s", 0)
            succ = metrics.get("success_rate", 0)
            overlap = metrics.get("overlap_ratio", 0)
            ooo = metrics.get("ooo_promotion_count", 0)
            cpu = metrics.get("cpu_busy_ratio", 0)
            wait = metrics.get("avg_task_wait_time_ms", 0)

            if has_baseline and not is_baseline:
                delta_ms = metrics.get("makespan_s_delta_pct", 0)
                delta_ov = metrics.get("overlap_ratio_delta_pct", 0)
                delta_str = f"{delta_ms:>+10.1f}%"
                delta_ov_str = f"{delta_ov:>+9.1f}%"
            else:
                delta_str = f"{'':>11}"
                delta_ov_str = f"{'':>10}"

            print(
                f"{prefix}{name:<20} "
                f"{makespan:>10.3f}s "
                f"{delta_str} "
                f"{succ:>8.1%} "
                f"{overlap:>8.1%} "
                f"{delta_ov_str} "
                f"{ooo:>6.0f} "
                f"{cpu * 100:>6.1f}% "
                f"{wait:>10.1f}"
            )

        print(f"{'='*120}")
        if has_baseline:
            print("\n★ = full_proposed 基准行（完整算法，不做任何消融）")
            print("ΔMakespan: 消融相对于基准的变化（正值=变慢/变差，负值=变快/变好）")
            print("ΔOverlap: 重叠率变化（正值=并行度提升，负值=并行度下降）")
        print("\n解读指南：")
        print("  - no_checkpoint: 去掉后 SuccRate 下降说明 Tier-4 告警恢复能力受损")
        print("  - no_event_wakeup: 去掉后 CPU Busy Ratio 上升，Overlap Ratio 下降")
        print("  - no_orthogonality_check: 去掉后 Conflict Stall Time 暴增")
        print("  - no_priority_lock: 去掉后 Jain's Fairness 下降，高优先级任务等待增加")


# ────────────────────────────────────────────────────────────────────────────── #
#  便捷入口                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

async def run_temporal_ablation(
    tiers: list[str] | None = None,
    output_dir: str = "results/temporal_ablation",
    use_dated_dir: bool = True,
    physical_delay_scale: float = 0.1,
) -> dict[str, Any]:
    """运行时间维度消融实验（自动包含 full_proposed 基准）。"""

    gen = BenchmarkGenerator(seed=42)
    all_eps = gen.generate_full_suite()

    # 按场景类型分组
    from collections import defaultdict
    by_scenario: dict[str, list] = defaultdict(list)
    for ep in all_eps:
        by_scenario[ep.scenario_type.value].append(ep)

    # 从所有场景类型均匀采样
    filtered: list = []
    if tiers is None:
        target_tiers = list(by_scenario.keys())
    else:
        target_tiers = tiers

    for tier in target_tiers:
        tier_eps = by_scenario.get(tier, [])
        if not tier_eps:
            continue
        filtered.extend(tier_eps[:3])

    scenario_counts = defaultdict(int)
    for ep in filtered:
        scenario_counts[ep.scenario_type.value] += 1
    print(f"[采样信息] 共 {len(filtered)} 条: {dict(scenario_counts)}")

    # full_proposed 基准由 include_baseline=True 自动运行，不重复加入
    experiments = [
        AblationExperiment(
            name=dim.name,
            description=dim.description,
            ablated_dims=dim.removes,
        )
        for dim in TEMPORAL_ABLATIONS
    ]

    config = AblatorConfig(
        base_episodes=filtered,
        ablation_experiments=experiments,
        seed=42,
        output_dir=Path(output_dir),
        use_dated_dir=use_dated_dir,
        physical_delay_scale=physical_delay_scale,
        include_baseline=True,  # 自动跑 full_proposed 基准
    )

    ablator = Ablator(config)
    return await ablator.run()


async def run_spatial_ablation(
    tiers: list[str] | None = None,
    output_dir: str = "results/spatial_ablation",
    use_dated_dir: bool = True,
    physical_delay_scale: float = 0.1,
) -> dict[str, Any]:
    """运行空间维度消融实验（自动包含 full_proposed 基准）。"""

    gen = BenchmarkGenerator(seed=42)
    all_eps = gen.generate_full_suite()

    from collections import defaultdict
    by_scenario: dict[str, list] = defaultdict(list)
    for ep in all_eps:
        by_scenario[ep.scenario_type.value].append(ep)

    filtered: list = []
    if tiers is None:
        target_tiers = list(by_scenario.keys())
    else:
        target_tiers = tiers

    for tier in target_tiers:
        tier_eps = by_scenario.get(tier, [])
        if not tier_eps:
            continue
        filtered.extend(tier_eps[:3])

    scenario_counts = defaultdict(int)
    for ep in filtered:
        scenario_counts[ep.scenario_type.value] += 1
    print(f"[采样信息] 共 {len(filtered)} 条: {dict(scenario_counts)}")

    # full_proposed 基准由 include_baseline=True 自动运行，不重复加入
    experiments = [
        AblationExperiment(
            name=dim.name,
            description=dim.description,
            ablated_dims=dim.removes,
        )
        for dim in SPATIAL_ABLATIONS
    ]

    config = AblatorConfig(
        base_episodes=filtered,
        ablation_experiments=experiments,
        seed=42,
        output_dir=Path(output_dir),
        use_dated_dir=use_dated_dir,
        physical_delay_scale=physical_delay_scale,
        include_baseline=True,  # 自动跑 full_proposed 基准
    )

    ablator = Ablator(config)
    return await ablator.run()


async def _quick_ablation() -> None:
    """快速消融验证（开发调试用）。"""
    eps = BenchmarkGenerator(seed=42).generate_tier2(
        count=3, difficulty=DifficultyLevel.EASY,
    )
    experiments = [
        AblationExperiment(
            name="no_event_wakeup",
            description="去掉 event-driven wake-up",
            ablated_dims=["event_wakeup"],
        ),
    ]
    config = AblatorConfig(
        base_episodes=eps,
        ablation_experiments=experiments,
        seed=42,
        output_dir=Path("results/quick_ablation"),
    )
    await Ablator(config).run()


async def main() -> None:
    """可作为 console_scripts 入口的 main 函数。"""
    import argparse

    parser = argparse.ArgumentParser(description="AstroSASF 消融实验")
    parser.add_argument("--tiers", nargs="+", default=None,
                        choices=["no_conflict", "light_conflict", "heavy_conflict", "alarm_recovery"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--no-dated", action="store_true", help="禁用日期后缀目录")
    parser.add_argument("--spatial", action="store_true", help="运行空间维度消融实验（默认时间维度）")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="物理延迟缩放因子（0.0-1.0），越小实验越快，默认0.1")
    args = parser.parse_args()

    if args.spatial:
        await run_spatial_ablation(
            tiers=args.tiers,
            output_dir=args.output_dir,
            use_dated_dir=not args.no_dated,
            physical_delay_scale=args.speed,
        )
    else:
        await run_temporal_ablation(
            tiers=args.tiers,
            output_dir=args.output_dir,
            use_dated_dir=not args.no_dated,
            physical_delay_scale=args.speed,
        )
    print(f"\n消融实验完成！")


if __name__ == "__main__":
    asyncio.run(main())
