"""
AstroSASF · Experiments · Component Comparison
===============================================
机制拆解对照实验 —— 分析 OoO-proposed 各机制模块的独立贡献。

实验目的（回答"哪个机制贡献最大"）：
- 与主实验（Main Comparison）不同：主实验对比外部基线，本实验拆解内部机制
- 与定向消融（Targeted Ablation）不同：消融是去掉 OoO-proposed 的某部分，
  而机制拆解是"以不同复杂度级别的系统与 OoO-proposed 对照"

四方法对照：
1. Async-only  ：仅有异步提交（同层并发），无 OoO Scanner，无事件驱动，无 ART 资源感知
                → 代表：裸并发层（Layer-level Concurrency）
2. Lock-only   ：仅有 DeviceRuntime 资源锁（顺序推进），无并发，无恢复
                → 代表：锁控制（Resource Locking）
3. Resume-only ：启用 orchestrator + 事件条件等待，禁用 OoO Scanner
                → 代表：事件恢复机制（Event-driven Recovery）
4. OoO-proposed：完整方案，全部机制启用
                → 作为锚点（Anchor）

关键区分（论文级说明）：
- Lock-only vs Resume-only：区分"被动等锁"和"事件唤醒"——两者都有锁，
  但 Resume-only 有 orchestrator 的 wait_for_condition 机制
- Resume-only vs Async-only：区分"有无 orchestrator 事件等待"——
  Async-only 完全不创建 orchestrator，Resume-only 有 orchestrator 但无 OoO Scanner
- Async-only vs Traditional DAG：两者代码相同（按层并发），但语义不同。
  在本文件中，Async-only 专用于机制拆解上下文。

Author: AstroSASF Team
Version: 9.0
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


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_generator import BenchmarkGenerator, BenchmarkEpisode
from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  Component Methods                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class ComponentMethod:
    """机制拆解中的单个方法。"""
    scheduler_mode: SchedulerMode
    display_name: str
    display_order: int  # 表格中显示顺序
    description: str     # 论文级说明


COMPONENT_METHODS: list[ComponentMethod] = [
    ComponentMethod(
        scheduler_mode=SchedulerMode.ASYNC_ONLY,
        display_name="Async-only",
        display_order=1,
        description="仅有异步提交（同层并发），无 OoO Scanner，无事件驱动，无资源感知",
    ),
    ComponentMethod(
        scheduler_mode=SchedulerMode.LOCK_ONLY,
        display_name="Lock-only",
        display_order=2,
        description="仅有 DeviceRuntime 资源锁（顺序推进），无并发，无恢复",
    ),
    ComponentMethod(
        scheduler_mode=SchedulerMode.RESUME_ONLY,
        display_name="Resume-only",
        display_order=3,
        description="启用 orchestrator + 事件条件等待，禁用 OoO Scanner（无乱序 promotion）",
    ),
    ComponentMethod(
        scheduler_mode=SchedulerMode.OOO_PROPOSED,
        display_name="OoO-proposed",
        display_order=4,
        description="完整方案，全部机制启用（锚点）",
    ),
]


# ────────────────────────────────────────────────────────────────────────────── #
#  Config                                                                       #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class ComponentConfig:
    episodes: list[BenchmarkEpisode]
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/component"))
    use_dated_dir: bool = True
    physical_delay_scale: float = 0.1


# ────────────────────────────────────────────────────────────────────────────── #
#  ComponentComparison Runner                                                    #
# ────────────────────────────────────────────────────────────────────────────── #

class ComponentComparison:
    """机制拆解对照实验运行器。"""

    def __init__(self, config: ComponentConfig) -> None:
        self.config = config
        self._results: list[dict[str, Any]] = []

    async def run(self) -> dict[str, Any]:
        """运行机制拆解对照实验。"""
        episodes = self.config.episodes

        output_dir = self._setup_output_dir()

        print(f"\n{'='*72}")
        print(f"  AstroSASF 机制拆解对照实验 (Component-wise Comparison)")
        print(f"  方法: {[m.display_name for m in COMPONENT_METHODS]}")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Speed: {self.config.physical_delay_scale}")
        print(f"  输出: {output_dir}")
        print(f"{'='*72}\n")

        total_runs = len(COMPONENT_METHODS) * len(episodes)
        run_idx = 0

        for method in sorted(COMPONENT_METHODS, key=lambda m: m.display_order):
            print(f"\n{'─'*72}")
            print(f"  ▶ {method.display_name} — {method.description}")
            print(f"{'─'*72}")

            suite = BenchmarkSuite(
                scheduler_mode=method.scheduler_mode,
                seed=self.config.seed,
                max_workers=3,
                verbose=True,
                physical_delay_scale=self.config.physical_delay_scale,
                ablated_dims=set(),
            )

            for ep in episodes:
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] {ep.episode_id}")
                result = await suite.run_episode(ep)

                row = {
                    "method": method.display_name,
                    "scheduler_mode": method.scheduler_mode.value,
                    "description": method.description,
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
                self._append_csv_row(output_dir / "component_results.csv", row)

        summary = self._compute_summary()
        self._save_summary(output_dir / "component_summary.json", summary)
        self._print_paper_table(summary)
        self._print_key_findings(summary)

        print(f"\n✅ 机制拆解完成 → {output_dir}")
        return summary

    def _setup_output_dir(self) -> Path:
        base = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir
        if self.config.use_dated_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = base / f"component_{ts}"
        else:
            out = base
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _compute_summary(self) -> dict[str, Any]:
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        metric_keys = [
            "makespan_s", "success_rate", "overlap_ratio",
            "conflict_stall_time_ms", "ooo_promotion_count",
            "cpu_busy_ratio", "avg_task_wait_time_ms",
            "alarm_response_latency_ms",
        ]

        for row in self._results:
            key = row["method"]
            for m in metric_keys:
                groups[key][m].append(row[m])

        summary = {}
        for name, metrics in groups.items():
            summary[name] = {m: (sum(v) / len(v) if v else 0) for m, v in metrics.items()}
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

    def _print_paper_table(self, summary: dict) -> None:
        """打印论文级机制拆解表格（Table 2）。"""
        all_scenarios = set()
        for m_data in summary.values():
            all_scenarios.update(m_data.keys())
        scenarios_sorted = sorted(all_scenarios)

        print(f"\n{'='*100}")
        print(f"  Table 2: 机制拆解对照结果 (Component-wise Comparison, speed={self.config.physical_delay_scale})")
        print(f"{'='*100}")
        print(f"\n{'Scenario':<22} {'Method':<16} {'Makespan':>9} {'SuccRate':>9} "
              f"{'Overlap':>9} {'Conf.Stall':>12} {'Promo#':>7} {'CPU%':>7}")
        print(f"{'-'*100}")

        for sc in scenarios_sorted:
            first = True
            count = 0
            for method in sorted(COMPONENT_METHODS, key=lambda m: m.display_order):
                m_name = method.display_name
                if m_name not in summary or sc not in summary[m_name]:
                    continue
                metrics = summary[m_name][sc]
                sc_label = sc if first else ""
                count += 1

                print(
                    f"{sc_label:<22} {m_name:<16} "
                    f"{metrics.get('makespan_s', 0):>9.3f}s "
                    f"{metrics.get('success_rate', 0):>8.1%} "
                    f"{metrics.get('overlap_ratio', 0):>8.1%} "
                    f"{metrics.get('conflict_stall_time_ms', 0):>11.1f} "
                    f"{metrics.get('ooo_promotion_count', 0):>7.0f} "
                    f"{metrics.get('cpu_busy_ratio', 0) * 100:>6.1f}%"
                )
                if count == 1:
                    first = False

        print(f"{'='*100}")

    def _print_key_findings(self, summary: dict) -> None:
        """打印关键发现摘要（用于论文讨论部分）。"""
        print(f"\n{'─'*72}")
        print(f"  关键发现摘要 (Key Findings)")
        print(f"{'─'*72}")

        # 计算各方法相对于 Async-only 的 makespan 变化
        async_ms = {}
        for m in summary:
            all_sc = summary[m]
            avg_ms = sum(s.get("makespan_s", 0) for s in all_sc.values()) / max(1, len(all_sc))
            async_ms[m] = avg_ms

        ooo_proposed = async_ms.get("OoO-proposed", 0)
        async_only = async_ms.get("Async-only", 0)
        lock_only = async_ms.get("Lock-only", 0)
        resume_only = async_ms.get("Resume-only", 0)

        print(f"\n  [平均 Makespan 对比]")
        for name, ms in sorted(async_ms.items(), key=lambda x: x[1]):
            if name != "Async-only" and async_only > 0:
                delta = (ms - async_only) / async_only * 100
                print(f"  {name:<20}: {ms:>8.3f}s  ({delta:>+7.1f}% vs Async-only)")
            else:
                print(f"  {name:<20}: {ms:>8.3f}s  (基准)")

        if async_only > 0 and ooo_proposed > 0:
            improvement = (async_only - ooo_proposed) / async_only * 100
            print(f"\n  [核心发现] OoO-proposed 相比 Async-only: Makespan 降低 {improvement:.1f}%")

        # OoO promotion 统计
        print(f"\n  [OoO Promotion 统计]")
        for m_name in ["Async-only", "Lock-only", "Resume-only", "OoO-proposed"]:
            if m_name in summary:
                promo = summary[m_name].get("_count", 0)
                total_promo = 0
                for sc, sc_data in summary[m_name].items():
                    if sc != "_count":
                        total_promo += sc_data.get("ooo_promotion_count", 0)
                avg_promo = total_promo / max(1, promo) if promo else 0
                print(f"  {m_name:<20}: 平均 OoO Promotion = {avg_promo:.1f}")

        print(f"{'─'*72}")


# ────────────────────────────────────────────────────────────────────────────── #
#  便捷入口                                                                  #
# ────────────────────────────────────────────────────────────────────────────── #

async def run_component_comparison(
    scenarios: list[str] | None = None,
    episodes_per_scenario: int = 3,
    output_dir: str = "results/component",
    physical_delay_scale: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """
    运行机制拆解对照实验的便捷入口。

    四方法：Async-only, Lock-only, Resume-only, OoO-proposed
    聚焦场景：heavy_conflict（最能体现锁竞争/事件恢复差异的场景）
    """
    from collections import defaultdict

    if scenarios is None:
        scenarios = ["heavy_conflict", "global_shared", "diamond_deep"]

    gen = BenchmarkGenerator(seed=seed)
    # 加载普通 suite 和 extreme suite
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

    config = ComponentConfig(
        episodes=filtered,
        seed=seed,
        output_dir=Path(output_dir),
        use_dated_dir=True,
        physical_delay_scale=physical_delay_scale,
    )

    runner = ComponentComparison(config)
    return await runner.run()


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AstroSASF 机制拆解对照实验")
    parser.add_argument("--scenarios", nargs="+",
                        default=["heavy_conflict", "alarm_recovery", "diamond_deep"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="results/component")
    parser.add_argument("--speed", type=float, default=0.1)
    args = parser.parse_args()

    await run_component_comparison(
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes,
        output_dir=args.output_dir,
        physical_delay_scale=args.speed,
    )


if __name__ == "__main__":
    asyncio.run(main())
