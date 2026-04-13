"""
AstroSASF · Experiments · Ablator
===================================
定向消融实验运行器 —— 验证 OoO-proposed 内部各机制对性能的贡献。

实验设计：
1. 基线对比（主实验）：
   - Sequential        ：严格顺序执行（理论下界）
   - Traditional DAG   ：传统 DAG 层并发
   - OoO-proposed      ：完整方案（基准）

2. 消融维度（内部机制）：
   - no_event_wakeup           ：禁用 OoO Scanner，改为轮询/阻塞等待
   - no_orthogonality_check    ：禁用资源正交性检查

实验原则：
- 以 OoO-proposed（完整方案）为基准
- 每个消融维度独立消融，保持其他机制不变
- 对比基线（Sequential / Traditional DAG）在同一实验中运行

Author: AstroSASF Team
Version: 10.0
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ── 动态项目根路径 ───────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_generator import BenchmarkGenerator, BenchmarkEpisode, ScenarioType
from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  消融维度定义（已实现）                                                    #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class AblationDim:
    """单个消融维度。"""
    name: str                       # 内部名称（如 "no_event_wakeup"）
    display_name: str               # 论文展示名称
    description: str                # 消融了什么的描述
    removes: list[str]              # 传给 ablated_dims 的字符串列表


# 当前代码中已实现的消融维度（与 core.py 中的 if "xxx" in ablated_dims 对应）
IMPLEMENTED_ABLATIONS = [
    AblationDim(
        name="no_event_wakeup",
        display_name="No Event-Wakeup",
        description="禁用 OoO Scanner，改为轮询 + asyncio.wait() 阻塞等待",
        removes=["event_wakeup"],
    ),
    AblationDim(
        name="no_orthogonality_check",
        display_name="No Orthogonality Check",
        description="禁用资源正交性检查，强制越级发射（即使资源存在交集）",
        removes=["orthogonality_check"],
    ),
]


# ────────────────────────────────────────────────────────────────────────────── #
#  Ablator Config                                                                #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class AblationExperiment:
    name: str
    display_name: str
    description: str
    ablated_dims: list[str]


@dataclass
class AblatorConfig:
    base_episodes: list[BenchmarkEpisode]
    ablation_experiments: list[AblationExperiment]
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/ablation"))
    use_dated_dir: bool = True
    physical_delay_scale: float = 0.1


# ────────────────────────────────────────────────────────────────────────────── #
#  Ablator                                                                      #
# ────────────────────────────────────────────────────────────────────────────── #

class Ablator:
    """定向消融实验运行器。"""

    def __init__(self, config: AblatorConfig) -> None:
        self.config = config
        self._results: list[dict[str, Any]] = []

    async def run(self) -> dict[str, Any]:
        """运行消融实验 + 基线对比。

        实验结构：
        1. 基线方法：Sequential, Traditional DAG（用于论文主对比表格）
        2. 消融变体：no_event_wakeup, no_orthogonality_check（验证内部机制贡献）
        3. 完整方案：full_proposed（基准）
        """
        episodes = self.config.base_episodes
        experiments = self.config.ablation_experiments

        output_dir = self._setup_output_dir()

        print(f"\n{'='*72}")
        print(f"  AstroSASF 消融实验 + 基线对比")
        print(f"  Episodes: {len(episodes)}")
        print(f"  消融变体: {len(experiments)}")
        print(f"  Speed: {self.config.physical_delay_scale}")
        print(f"  输出: {output_dir}")
        print(f"{'='*72}\n")

        all_results: list[dict[str, Any]] = []

        # ── 第一步：跑基线方法（用于论文主对比表格）────────────────────────────
        # Sequential 基线
        print(f"\n{'─'*72}")
        print(f"  ▶ Sequential（严格顺序执行，理论下界）")
        print(f"{'─'*72}")
        seq_suite = BenchmarkSuite(
            scheduler_mode=SchedulerMode.SEQUENTIAL,
            seed=self.config.seed,
            max_workers=3,
            verbose=False,
            physical_delay_scale=self.config.physical_delay_scale,
        )
        for ep in episodes:
            print(f"  [{ep.episode_id}]")
            result = await seq_suite.run_episode(ep)
            row = self._build_row("sequential", "Sequential", "", ep, result)
            all_results.append(row)
            self._append_csv_row(output_dir / "ablation_results.csv", row)

        # Traditional DAG 基线
        print(f"\n{'─'*72}")
        print(f"  ▶ Traditional DAG（层并发，乱序调度前的主流方案）")
        print(f"{'─'*72}")
        dag_suite = BenchmarkSuite(
            scheduler_mode=SchedulerMode.TRADITIONAL_DAG,
            seed=self.config.seed,
            max_workers=3,
            verbose=False,
            physical_delay_scale=self.config.physical_delay_scale,
        )
        for ep in episodes:
            print(f"  [{ep.episode_id}]")
            result = await dag_suite.run_episode(ep)
            row = self._build_row("traditional_dag", "Traditional DAG", "", ep, result)
            all_results.append(row)
            self._append_csv_row(output_dir / "ablation_results.csv", row)

        # ── 第二步：跑消融变体（验证内部机制贡献）───────────────────────────────
        total_runs = len(experiments) * len(episodes)
        run_idx = 0

        for exp in experiments:
            print(f"\n{'─'*72}")
            print(f"  ▶ {exp.display_name} ({exp.description})")
            print(f"{'─'*72}")

            suite = BenchmarkSuite(
                scheduler_mode=SchedulerMode.OOO_PROPOSED,
                seed=self.config.seed,
                max_workers=3,
                verbose=False,
                physical_delay_scale=self.config.physical_delay_scale,
                ablated_dims=set(exp.ablated_dims),
            )

            for ep in episodes:
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] {ep.episode_id}")
                result = await suite.run_episode(ep)
                row = self._build_row(exp.name, exp.display_name, ",".join(exp.ablated_dims), ep, result)
                all_results.append(row)
                self._append_csv_row(output_dir / "ablation_results.csv", row)

        # ── 第三步：跑完整方案基准 ─────────────────────────────────────────────
        print(f"\n{'─'*72}")
        print(f"  ▶ ★ Full Proposed（完整方案，不做消融）")
        print(f"{'─'*72}")

        suite = BenchmarkSuite(
            scheduler_mode=SchedulerMode.OOO_PROPOSED,
            seed=self.config.seed,
            max_workers=3,
            verbose=False,
            physical_delay_scale=self.config.physical_delay_scale,
            ablated_dims=set(),
        )

        for ep in episodes:
            print(f"  [{ep.episode_id}]")
            result = await suite.run_episode(ep)
            row = self._build_row("full_proposed", "★ Full Proposed", "", ep, result)
            all_results.append(row)
            self._append_csv_row(output_dir / "ablation_results.csv", row)

        self._results = all_results

        summary = self._compute_summary()
        self._save_summary(output_dir / "ablation_summary.json", summary)
        self._print_paper_table(summary)

        print(f"\n✅ 消融实验完成 → {output_dir}")
        return summary

    def _build_row(
        self,
        name: str,
        display: str,
        dims_str: str,
        ep: BenchmarkEpisode,
        result: Any,
    ) -> dict[str, Any]:
        row = {
            "ablation": name,
            "ablation_display": display,
            "ablated_dims": dims_str,
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
            "total_physical_ms": result.metrics.get("total_physical_ms", 0),
            "device_results_count": result.metrics.get("device_results_count", 0),
            "error": result.error or "",
        }
        # speedup_ratio 相对于 sequential 基线（后续汇总时填充）
        row["speedup_vs_seq"] = 0.0
        return row

    def _setup_output_dir(self) -> Path:
        base = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir
        if self.config.use_dated_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = base / f"ablation_{ts}"
        else:
            out = base
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _compute_summary(self) -> dict[str, Any]:
        """计算各方法/变体的平均指标，并计算 delta。

        实验结构：
        - 基线方法（sequential, traditional_dag）：不计算 delta，作为参照
        - 消融变体（no_event_wakeup 等）：相对于 full_proposed 计算 delta
        - 完整方案（full_proposed）：标记为基准

        speedup_vs_seq 计算：
        - sequential = 1.0（基准）
        - 其他 = sequential_makespan / method_makespan
        """
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
            # speedup_vs_seq 单独计算
            groups[key]["speedup_vs_seq"].append(row.get("speedup_vs_seq", 0.0))

        # 第一次汇总：计算各方法的平均 makespan
        raw_avg = {}
        for name, metrics in groups.items():
            raw_avg[name] = {m: (sum(v) / len(v) if v else 0) for m, v in metrics.items()}

        # 计算 speedup_vs_seq（基于平均 makespan）
        seq_makespan = raw_avg.get("sequential", {}).get("makespan_s", 0)
        for name in raw_avg:
            mk = raw_avg[name].get("makespan_s", 0)
            if seq_makespan > 0 and mk > 0:
                raw_avg[name]["speedup_vs_seq"] = round(seq_makespan / mk, 3)
            else:
                raw_avg[name]["speedup_vs_seq"] = 0.0

        # 构建 summary
        summary = {}
        for name, metrics in raw_avg.items():
            summary[name] = dict(metrics)
            summary[name]["_count"] = len(next(iter(groups[name].values())))

        # 计算相对于基准的 delta（仅消融变体相对于 full_proposed）
        baseline_metrics = summary.get("full_proposed", {})
        for name, metrics in summary.items():
            if name == "full_proposed":
                metrics["_is_baseline"] = True
                continue
            if name in ("sequential", "traditional_dag"):
                continue
            for m in metric_keys:
                base_val = baseline_metrics.get(m, 0)
                ablated_val = metrics.get(m, 0)
                if base_val != 0:
                    metrics[f"{m}_delta_pct"] = round((ablated_val - base_val) / base_val * 100, 2)
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

    def _print_paper_table(self, summary: dict) -> None:
        """打印论文级消融表格（Table 3）。

        表格结构：
        - Part 1: 基线方法对比（Sequential, Traditional DAG, OoO-proposed）
        - Part 2: 消融变体对比（验证内部机制贡献）
        """
        print(f"\n{'='*110}")
        print(f"  Table 3: 消融实验 + 基线对比 (speed={self.config.physical_delay_scale})")
        print(f"{'='*110}")

        # Part 1: 主对比表格
        print(f"\n{'─'*110}")
        print(f"  Part 1: 主对比（OoO-proposed vs 基线方法）")
        print(f"{'─'*110}")
        print(f"\n{'Variant':<22} {'Makespan':>10} {'SuccRate':>9} "
              f"{'Overlap':>9} {'Conf.Stall':>11} {'Promo#':>7} {'Speedup':>9}")
        print(f"{'-'*110}")

        baseline = summary.get("full_proposed", {})
        seq = summary.get("sequential", {})
        dag = summary.get("traditional_dag", {})

        # 显示顺序
        for name, display in [
            ("sequential", "Sequential"),
            ("traditional_dag", "Traditional DAG"),
            ("full_proposed", "★ Full Proposed"),
        ]:
            if name not in summary:
                continue
            m = summary[name]
            makespan = m.get("makespan_s", 0)
            succ = m.get("success_rate", 0)
            overlap = m.get("overlap_ratio", 0)
            stall = m.get("conflict_stall_time_ms", 0)
            ooo = m.get("ooo_promotion_count", 0)
            speedup = m.get("speedup_vs_seq", 0.0)

            is_base = m.get("_is_baseline", False)
            prefix = "★ " if is_base else "  "

            speedup_str = f"{speedup:>8.2f}x" if speedup > 0 else f"{'':>9}"
            print(
                f"{prefix}{display:<20} "
                f"{makespan:>10.3f}s "
                f"{succ:>8.1%} "
                f"{overlap:>8.1%} "
                f"{stall:>10.1f} "
                f"{ooo:>7.0f} "
                f"{speedup_str}"
            )

        # Part 2: 消融变体
        print(f"\n{'─'*110}")
        print(f"  Part 2: 消融变体（验证内部机制贡献，相对 ★ Full Proposed）")
        print(f"{'-'*110}")
        print(f"\n{'Variant':<22} {'Makespan':>10} {'ΔMakespan':>11} {'SuccRate':>9} "
              f"{'Overlap':>9} {'Conf.Stall':>11} {'Promo#':>7} {'Speedup':>9}")
        print(f"{'-'*110}")

        for name, expected_display in [
            ("no_event_wakeup", "No Event-Wakeup"),
            ("no_orthogonality_check", "No Orthogonality Check"),
        ]:
            if name not in summary:
                continue
            m = summary[name]
            makespan = m.get("makespan_s", 0)
            succ = m.get("success_rate", 0)
            overlap = m.get("overlap_ratio", 0)
            stall = m.get("conflict_stall_time_ms", 0)
            ooo = m.get("ooo_promotion_count", 0)
            speedup = m.get("speedup_vs_seq", 0.0)

            delta = m.get("makespan_s_delta_pct", 0)
            delta_str = f"{delta:>+10.1f}%"
            speedup_str = f"{speedup:>8.2f}x" if speedup > 0 else f"{'':>9}"

            print(
                f"  {name:<20} "
                f"{makespan:>10.3f}s "
                f"{delta_str} "
                f"{succ:>8.1%} "
                f"{overlap:>8.1%} "
                f"{stall:>10.1f} "
                f"{ooo:>7.0f} "
                f"{speedup_str}"
            )

        print(f"{'='*110}")
        print("\n★ = Full Proposed 基准行（完整方案）")
        print("ΔMakespan: 消融相对于基准的变化（正值=变慢/变差，负值=变快/变好）")
        print("Speedup: 相对于 Sequential 基线的加速比（越大越好）")
        print("ΔMakespan: 消融相对于基准的变化（正值=变慢/变差，负值=变快/变好）")


# ────────────────────────────────────────────────────────────────────────────── #
#  便捷入口                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

async def run_targeted_ablation(
    scenarios: list[str] | None = None,
    episodes_per_scenario: int = 3,
    output_dir: str = "results/ablation",
    physical_delay_scale: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """
    运行定向消融实验的便捷入口。

    消融维度（已实现）：
    - No Event-Wakeup：禁用 OoO Scanner，改为轮询
    - No Orthogonality Check：禁用资源正交性检查
    """
    from collections import defaultdict

    if scenarios is None:
        scenarios = ["heavy_conflict", "global_shared", "diamond_deep"]

    gen = BenchmarkGenerator(seed=seed)
    normal_eps = gen.generate_full_suite()
    extreme_eps = gen.generate_extreme_full_suite()

    by_scenario: dict[str, list] = defaultdict(list)
    for ep in normal_eps:
        by_scenario[ep.scenario_type.value].append(ep)
    for ep in extreme_eps:
        by_scenario[ep.scenario_type.value].append(ep)

    filtered: list[BenchmarkEpisode] = []
    for sc in scenarios:
        sc_eps = by_scenario.get(sc, [])
        if not sc_eps:
            print(f"[警告] 场景 '{sc}' 无 episode，跳过")
            continue
        import random
        rng = random.Random(seed)
        shuffled = sc_eps.copy()
        rng.shuffle(shuffled)
        filtered.extend(shuffled[:episodes_per_scenario])

    if not filtered:
        raise ValueError("没有找到有效的 episode。")

    from collections import Counter
    counts = Counter(ep.scenario_type.value for ep in filtered)
    print(f"[采样] 共 {len(filtered)} 条: {dict(counts)}")

    # 构建消融实验列表（只包含已实现的）
    experiments = [
        AblationExperiment(
            name=d.name,
            display_name=d.display_name,
            description=d.description,
            ablated_dims=d.removes,
        )
        for d in IMPLEMENTED_ABLATIONS
    ]

    config = AblatorConfig(
        base_episodes=filtered,
        ablation_experiments=experiments,
        seed=seed,
        output_dir=Path(output_dir),
        use_dated_dir=True,
        physical_delay_scale=physical_delay_scale,
    )

    ablator = Ablator(config)
    return await ablator.run()


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AstroSASF 定向消融实验")
    parser.add_argument("--scenarios", nargs="+",
                        default=["heavy_conflict", "alarm_recovery", "diamond_deep"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--speed", type=float, default=0.1)
    args = parser.parse_args()

    await run_targeted_ablation(
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes,
        output_dir=args.output_dir,
        physical_delay_scale=args.speed,
    )


if __name__ == "__main__":
    asyncio.run(main())
