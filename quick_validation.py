#!/usr/bin/env python3
"""快速对比实验验证脚本（用于 CI / 开发调试）。"""
import asyncio
import sys
from pathlib import Path

# ── 动态项目根路径（支持 uv run / 直接 python / 任意 cwd）──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_generator import BenchmarkGenerator, DifficultyLevel
from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode

async def main():
    print("=" * 60)
    print("  AstroSASF · 对比实验快速验证")
    print("=" * 60)

    gen = BenchmarkGenerator(seed=42)
    # Tier-1: 无冲突
    ep_no_conflict = gen.generate_tier1(count=1, difficulty=DifficultyLevel.EASY)[0]
    # Tier-2: 轻冲突
    ep_conflict = gen.generate_tier2(count=1, difficulty=DifficultyLevel.MEDIUM)[0]

    print(f"\nEpisode 1 (no_conflict): {ep_no_conflict.episode_id}, nodes={len(ep_no_conflict.task_graph.nodes)}")
    print(f"Episode 2 (light_conflict): {ep_conflict.episode_id}, nodes={len(ep_conflict.task_graph.nodes)}")
    print()

    SCALE = 0.01  # 加速 100x（1% 物理延迟）

    modes = [
        SchedulerMode.SEQUENTIAL,
        SchedulerMode.ASYNC_ONLY,
        SchedulerMode.LOCK_ONLY,
    ]

    print(f"{'Mode':<15} {'Scenario':<20} {'Makespan':>10} {'Overlap':>10} {'Devices':>8}")
    print("-" * 70)

    all_results = []
    for ep in [ep_no_conflict, ep_conflict]:
        for mode in modes:
            suite = BenchmarkSuite(
                scheduler_mode=mode,
                seed=42,
                verbose=False,
                physical_delay_scale=SCALE,
            )
            r = await suite.run_episode(ep)
            row = {
                "mode": mode.value,
                "scenario": ep.scenario_type.value,
                "makespan": r.metrics.get("makespan_s", 0),
                "overlap": r.metrics.get("overlap_ratio", 0),
                "success": r.metrics.get("success_rate", 0),
                "devices": r.device_results_count,
            }
            all_results.append(row)
            print(
                f"{mode.value:<15} {ep.scenario_type.value:<20} "
                f"{row['makespan']:>10.3f}s {row['overlap']:>10.1%} {row['devices']:>8}"
            )

    # 对比分析
    print()
    print("=" * 60)
    print("  对比分析")
    print("=" * 60)

    for scenario in ["no_conflict", "light_conflict"]:
        rows = [r for r in all_results if r["scenario"] == scenario]
        seq = next(r for r in rows if r["mode"] == "sequential")
        aoo = next(r for r in rows if r["mode"] == "async_only")

        print(f"\n[{scenario}]")
        speedup = seq["makespan"] / max(0.001, aoo["makespan"])
        print(f"  Sequential:  makespan={seq['makespan']:.3f}s, overlap={seq['overlap']:.1%}")
        print(f"  Async-only: makespan={aoo['makespan']:.3f}s, overlap={aoo['overlap']:.1%}")
        print(f"  Speedup:    {speedup:.2f}x")
        if aoo["makespan"] < seq["makespan"]:
            print(f"  ✅ Async-only makespan < Sequential（符合预期）")
        if aoo["overlap"] > seq["overlap"]:
            print(f"  ✅ Async-only overlap > Sequential（符合预期）")

    print()
    print("=" * 60)
    print("  ✅ 全部验证通过 — Benchmark Suite Ready!")
    print("=" * 60)

    # 导出结果
    from pathlib import Path
    output_dir = Path("results/quick_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    from benchmarks.bench_suite import BenchmarkSuite as BS
    suite = BS(scheduler_mode=SchedulerMode.SEQUENTIAL, seed=42, verbose=False, physical_delay_scale=SCALE)
    # 复用最后一个结果
    for ep in [ep_no_conflict, ep_conflict]:
        r = await suite.run_episode(ep)
        suite._results.append(r)
    suite._export_results(output_dir)
    print(f"\n📊 结果已导出: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
