"""
AstroSASF · Experiments · Plotter
===============================
结果可视化脚本 —— 从 CSV/JSON 读取数据，生成论文级图表。

图表列表：
1. Makespan vs delay_scale         — 延迟缩放 vs Makespan
2. Overlap Ratio vs scheduler       — 不同调度器的 Overlap Ratio 对比
3. Waiting Time vs conflict_rate    — 冲突率 vs 平均等待时间
4. Resume Latency ablation          — Resume Latency 消融柱状图
5. Device Utilization by type       — 各设备利用率
6. Jitter vs concurrency            — 并发度 vs Jitter

依赖：matplotlib, numpy（pip install matplotlib numpy）
若未安装，plot_all() 会优雅降级并给出提示。

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ── 动态项目根路径（支持 uv run / 直接 python / 任意 cwd）──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    print("[Plotter] ⚠️  matplotlib 未安装，图表生成功能不可用（pip install matplotlib numpy）")

# ────────────────────────────────────────────────────────────────────────────── #
#  样式配置                                                                      #
# ────────────────────────────────────────────────────────────────────────────── #

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5.5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

BASELINE_COLORS = {
    "Sequential":   "#e74c3c",
    "Async-only":   "#f39c12",
    "OoO-lite":     "#3498db",
    "OoO-proposed": "#27ae60",
    "Lock-only":    "#9b59b6",
    "Resume-only":  "#1abc9c",
}

BASELINE_MARKERS = {
    "Sequential":   "o",
    "Async-only":   "s",
    "OoO-lite":     "^",
    "OoO-proposed": "D",
    "Lock-only":    "p",
    "Resume-only":  "h",
}


def _apply_style() -> None:
    for k, v in STYLE.items():
        plt.rcParams[k] = v


# ────────────────────────────────────────────────────────────────────────────── #
#  数据加载                                                                      #
# ────────────────────────────────────────────────────────────────────────────── #

def load_results(csv_path: Path) -> list[dict[str, Any]]:
    """从 CSV 加载实验结果。"""
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_summary(json_path: Path) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────────────── #
#  图表生成                                                                      #
# ────────────────────────────────────────────────────────────────────────────── #

def plot_makespan_vs_delay(
    rows: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Makespan vs physical_delay_scale。"""
    _apply_style()
    fig, ax = plt.subplots()

    delay_groups: dict[str, list[float]] = {}
    for row in rows:
        delay = float(row.get("delay_scale", 1.0))
        baseline = row.get("baseline", "unknown")
        makespan = float(row.get("makespan_s", 0))
        key = f"{baseline}_d{delay}"
        if key not in delay_groups:
            delay_groups[key] = []
        delay_groups[key].append(makespan)

    baselines_seen = set()
    for key, vals in sorted(delay_groups.items()):
        baseline = key.rsplit("_d", 1)[0]
        delay = float(key.rsplit("_d", 1)[1])
        baselines_seen.add(baseline)
        mean = np.mean(vals)
        label = baseline if baseline in BASELINE_COLORS else None
        color = BASELINE_COLORS.get(baseline, "#888888")
        marker = BASELINE_MARKERS.get(baseline, "o")
        ax.plot(delay, mean, marker=marker, color=color, markersize=8,
                label=label, linewidth=2)

    ax.set_xlabel("Physical Delay Scale (×)")
    ax.set_ylabel("Makespan (s)")
    ax.set_title("Makespan vs Physical Delay Scale")
    ax.legend(loc="best")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 图表已保存: {output_path}")
    plt.close()


def plot_overlap_ratio(
    rows: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Overlap Ratio vs Scheduler（柱状图）。"""
    _apply_style()
    fig, ax = plt.subplots()

    baseline_vals: dict[str, list[float]] = {}
    for row in rows:
        baseline = row.get("baseline", "unknown")
        overlap = float(row.get("overlap_ratio", 0))
        if baseline not in baseline_vals:
            baseline_vals[baseline] = []
        baseline_vals[baseline].append(overlap)

    baselines = []
    means = []
    stds = []
    colors = []
    for b in BASELINE_COLORS:
        if b in baseline_vals:
            baselines.append(b)
            means.append(np.mean(baseline_vals[b]))
            stds.append(np.std(baseline_vals[b]) if len(baseline_vals[b]) > 1 else 0)
            colors.append(BASELINE_COLORS[b])

    x = np.arange(len(baselines))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=4, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=15)
    ax.set_ylabel("Overlap Ratio")
    ax.set_title("Overlap Ratio by Scheduler")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 添加数值标签
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 图表已保存: {output_path}")
    plt.close()


def plot_waiting_time_vs_conflict(
    rows: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Average Task Waiting Time vs Conflict Rate。"""
    _apply_style()
    fig, ax = plt.subplots()

    for row in rows:
        baseline = row.get("baseline", "unknown")
        conflict_rate = float(row.get("conflict_rate", 0))
        wait_time = float(row.get("avg_task_wait_time_ms", 0))
        color = BASELINE_COLORS.get(baseline, "#888888")
        marker = BASELINE_MARKERS.get(baseline, "o")
        ax.scatter(conflict_rate, wait_time, color=color, marker=marker,
                   s=60, alpha=0.7, label=baseline)

    ax.set_xlabel("Resource Conflict Rate")
    ax.set_ylabel("Average Task Waiting Time (ms)")
    ax.set_title("Waiting Time vs Conflict Rate")
    ax.legend(loc="best")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 图表已保存: {output_path}")
    plt.close()


def plot_resume_latency_ablation(
    summary: dict[str, Any],
    output_path: Path | None = None,
) -> None:
    """Resume Latency 消融柱状图。"""
    _apply_style()
    fig, ax = plt.subplots()

    ablation_names = sorted(summary.keys())
    resume_latencies = [
        summary[name].get("resume_latency_ms", 0)
        for name in ablation_names
    ]
    success_rates = [
        summary[name].get("success_rate", 0)
        for name in ablation_names
    ]

    x = np.arange(len(ablation_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, resume_latencies, width, label="Resume Latency (ms)",
                   color="#3498db", alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, [s * 100 for s in success_rates], width,
                    label="Success Rate (%)", color="#27ae60", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_names, rotation=30, ha="right")
    ax.set_ylabel("Resume Latency (ms)")
    ax2.set_ylabel("Success Rate (%)")
    ax.set_title("Resume Latency Ablation Study")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 图表已保存: {output_path}")
    plt.close()


def plot_device_utilization(
    rows: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Device Utilization 按设备类型分组。"""
    _apply_style()
    fig, ax = plt.subplots()

    device_utils: dict[str, list[float]] = {}
    for row in rows:
        utils_str = row.get("device_utilization", "{}")
        import ast
        try:
            utils = ast.literal_eval(utils_str) if utils_str.startswith("{") else {}
        except Exception:
            utils = {}
        for dev, val in utils.items():
            dev_type = dev.rsplit("_", 1)[-1]
            if dev_type not in device_utils:
                device_utils[dev_type] = []
            device_utils[dev_type].append(float(val))

    dev_types = sorted(device_utils.keys())
    means = [np.mean(device_utils[d]) for d in dev_types]
    stds = [np.std(device_utils[d]) for d in dev_types]

    x = np.arange(len(dev_types))
    bars = ax.bar(x, means, yerr=stds, color="#8e44ad", alpha=0.8,
                  capsize=4, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dev_types, rotation=30, ha="right")
    ax.set_ylabel("Utilization")
    ax.set_title("Physical Device Utilization by Type")
    ax.set_ylim(0, 1.2)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 图表已保存: {output_path}")
    plt.close()


def plot_jitter_vs_concurrency(
    rows: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Jitter vs Concurrency。"""
    _apply_style()
    fig, ax = plt.subplots()

    for row in rows:
        baseline = row.get("baseline", "unknown")
        concurrency = int(row.get("concurrency_level", 1))
        jitter = float(row.get("jitter_ms", 0))
        color = BASELINE_COLORS.get(baseline, "#888888")
        marker = BASELINE_MARKERS.get(baseline, "o")
        ax.plot(concurrency, jitter, marker=marker, color=color,
                markersize=10, alpha=0.8)

    ax.set_xlabel("Concurrency Level")
    ax.set_ylabel("Jitter (ms)")
    ax.set_title("Scheduling Jitter vs Concurrency")
    ax.legend(loc="best")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 图表已保存: {output_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────── #
#  统一入口                                                                      #
# ────────────────────────────────────────────────────────────────────────────── #

def plot_all(input_dir: Path, output_dir: Path | None = None) -> None:
    """从 results 目录生成所有图表。"""
    if not _HAS_MATPLOTLIB:
        print("[Plotter] ⚠️  matplotlib 未安装，无法生成图表")
        return

    if output_dir is None:
        output_dir = input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / "all_results.csv"
    summary_path = input_dir / "summary.json"

    if not csv_path.exists():
        print(f"⚠️  未找到 {csv_path}，跳过图表生成")
        return

    print(f"\n📊 生成图表 → {output_dir}/")
    rows = load_results(csv_path)

    plot_makespan_vs_delay(rows, output_dir / "makespan_vs_delay.png")
    plot_overlap_ratio(rows, output_dir / "overlap_ratio.png")
    plot_waiting_time_vs_conflict(rows, output_dir / "waiting_time_vs_conflict.png")
    plot_jitter_vs_concurrency(rows, output_dir / "jitter_vs_concurrency.png")

    if summary_path.exists():
        summary = load_summary(summary_path)
        plot_resume_latency_ablation(summary, output_dir / "resume_latency_ablation.png")

    plot_device_utilization(rows, output_dir / "device_utilization.png")

    print(f"\n✅ 所有图表已保存至: {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AstroSASF 结果可视化")
    parser.add_argument("input_dir", help="实验结果目录（如 results/comparison）")
    parser.add_argument("--output-dir", default=None, help="图表输出目录")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else None
    plot_all(input_path, output_path)
