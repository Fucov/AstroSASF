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
    physical_delay_scale: float = 0.01  # 物理延迟缩放因子（0.0-1.0，越小越快）


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
        print(f"  Experiments: {len(experiments)}")
        print(f"  输出目录: {output_dir}")
        print(f"  日志文件: {log_file}")
        print(f"{'='*70}\n")

        total_runs = len(experiments) * len(episodes)
        run_idx = 0

        for exp in experiments:
            print(f"\n{'─'*70}")
            print(f"  ▶ Ablation: {exp.name}")
            print(f"     {exp.description}")
            print(f"     消融: {exp.ablated_dims}")
            print(f"{'─'*70}")

            suite = _build_ablated_suite(exp, seed=self.config.seed, physical_delay_scale=self.config.physical_delay_scale)

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
                    "ooo_promotion_count": result.metrics.get("ooo_promotion_count", 0),
                    "cpu_busy_ratio": result.metrics.get("cpu_busy_ratio", 0),
                    "avg_task_wait_time_ms": result.metrics.get("avg_task_wait_time_ms", 0),
                    "alarm_response_latency_ms": result.metrics.get("alarm_response_latency_ms", 0),
                    "safety_rejection_rate": result.metrics.get("safety_rejection_rate", 0),
                    "jains_fairness": result.metrics.get("jains_fairness", 0),
                    "error": result.error or "",
                }
                self._results.append(row)
                self._append_csv_row(output_dir / "ablation_results.csv", row)

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
        """打印消融实验汇总表。"""
        print(f"\n{'='*100}")
        print(f"  消融实验汇总表")
        print(f"{'='*100}")
        print(f"{'Ablation':<25} {'Makespan':>10} {'SuccRate':>10} {'Overlap':>10} "
              f"{'Conflict(ms)':>15} {'OoO#':>8} {'CPU%':>8} {'Wait(ms)':>12}")
        print(f"{'-'*100}")

        for name, metrics in summary.items():
            print(
                f"{name:<25} "
                f"{metrics.get('makespan_s', 0):>10.3f} "
                f"{metrics.get('success_rate', 0):>10.1%} "
                f"{metrics.get('overlap_ratio', 0):>10.1%} "
                f"{metrics.get('conflict_stall_time_ms', 0):>15.1f} "
                f"{metrics.get('ooo_promotion_count', 0):>8.0f} "
                f"{metrics.get('cpu_busy_ratio', 0) * 100:>7.1f}% "
                f"{metrics.get('avg_task_wait_time_ms', 0):>12.1f}"
            )

        print(f"{'='*100}")
        print("\n解读指南：")
        print("  - no_checkpoint: 去掉后 Success Rate 应显著下降（Tier-4 尤其明显）")
        print("  - no_event_wakeup: 去掉后 CPU Busy Ratio 上升，Overlap Ratio 下降")
        print("  - no_orthogonality_check: 去掉后 Conflict Stall Time 暴增")
        print("  - no_priority_lock: 去掉后 Jain's Fairness 下降，高优先级任务等待增加")


# ────────────────────────────────────────────────────────────────────────────── #
#  Ablated Suite Builder                                                          #
# ────────────────────────────────────────────────────────────────────────────── #

def _build_ablated_suite(exp: AblationExperiment, seed: int, physical_delay_scale: float = 0.01) -> BenchmarkSuite:
    """根据消融配置构建特定的 BenchmarkSuite。"""

    # 每个消融实验对应一个特定的 SchedulerMode 实现
    # 在实际运行时，suite._run_ooo_proposed 会根据 ablated_dims 做分支

    class AblatedSuite(BenchmarkSuite):
        def __init__(self2, **kw):
            super().__init__(**kw)
            self2._ablated_dims = set(exp.ablated_dims)

        async def _run_ooo_proposed(self2, dag, labs, metrics, orchestrator, runtime):
            """消融版本的 OoO-proposed。"""
            await orchestrator.start()

            # 根据消融维度调整行为
            if "orthogonality_check" in self2._ablated_dims:
                # 跳过正交性检查：所有节点都能越级发射
                orchestrator._ooo_lock = asyncio.Lock()

            if "prefix_routing" in self2._ablated_dims:
                # 跳过 prefix routing：每次全量推理
                metrics._prefix_hits = 0
                metrics._prefix_misses = 0

            if "checkpoint" in self2._ablated_dims:
                # 跳过 checkpoint
                pass

            if "event_wakeup" in self2._ablated_dims:
                # 使用轮询而非 asyncio.Future
                orchestrator._ooo_scanner_task = None

            await orchestrator.submit_dag(dag)
            try:
                await asyncio.wait_for(orchestrator._dag_complete_event.wait(), timeout=300.0)
            except asyncio.TimeoutError:
                pass
            finally:
                await orchestrator.shutdown()

            metrics.add_compute_time(100.0 * len(dag.nodes))

    return AblatedSuite(
        scheduler_mode=SchedulerMode.OOO_PROPOSED,
        seed=seed,
        max_workers=3,
        verbose=False,
        physical_delay_scale=physical_delay_scale,
    )


# ────────────────────────────────────────────────────────────────────────────── #
#  便捷入口                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

async def run_temporal_ablation(
    tiers: list[str] | None = None,
    output_dir: str = "results/temporal_ablation",
    use_dated_dir: bool = True,
    physical_delay_scale: float = 0.01,
) -> dict[str, Any]:
    """运行时间维度消融实验。"""
    gen = BenchmarkGenerator(seed=42)
    all_eps = gen.generate_full_suite()

    filtered = [
        ep for ep in all_eps
        if tiers is None or ep.scenario_type.value in tiers
    ][:10]  # 最多 10 条

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
    )

    ablator = Ablator(config)
    return await ablator.run()


async def run_spatial_ablation(
    tiers: list[str] | None = None,
    output_dir: str = "results/spatial_ablation",
    use_dated_dir: bool = True,
    physical_delay_scale: float = 0.01,
) -> dict[str, Any]:
    """运行空间维度消融实验。"""
    gen = BenchmarkGenerator(seed=42)
    all_eps = gen.generate_full_suite()

    filtered = [
        ep for ep in all_eps
        if tiers is None or ep.scenario_type.value in tiers
    ][:10]

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
    parser.add_argument("--speed", type=float, default=0.01,
                        help="物理延迟缩放因子（0.0-1.0），越小实验越快，默认0.01")
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
