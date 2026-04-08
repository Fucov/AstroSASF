"""
AstroSASF · Scheduler · Metrics Collector
=========================================
Benchmark 指标统一采集器 —— 14 个核心指标全部打桩。

指标分组：
- 调度效率：Makespan / Scheduling Overhead / Jitter
- 资源利用：Physical Device Utilization / CPU Busy Ratio
- 恢复效率：Resume Latency / Overlap Ratio / Conflict Stall Time
            Prefix-Routing Hit Rate / Alarm Response Latency
- 安全/稳定性：Success Rate / Safety Rejection Rate / Fairness

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scheduler.device_runtime import ChaosEvent, DeviceResult


# ────────────────────────────────────────────────────────────────────────────── #
#  Task Lifecycle Record                                                       #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class TaskLifecycle:
    """单个任务的完整生命周期记录。"""
    task_id: str
    lab_id: str
    skill_name: str
    submitted_at: float = 0.0
    wait_started_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    failed_at: float | None = None
    wait_reason: str | None = None
    device_results: list[DeviceResult] = field(default_factory=list)
    ooo_promoted: bool = False
    checkpoint_saved: bool = False
    checkpoint_loaded: bool = False
    safety_rejected: bool = False
    priority: int = 2

    @property
    def wait_time_ms(self) -> float | None:
        if self.started_at is not None and self.wait_started_at is not None:
            return (self.started_at - self.wait_started_at) * 1000.0
        return None

    @property
    def total_time_ms(self) -> float | None:
        if self.completed_at is not None and self.submitted_at > 0:
            return (self.completed_at - self.submitted_at) * 1000.0
        if self.failed_at is not None and self.submitted_at > 0:
            return (self.failed_at - self.submitted_at) * 1000.0
        return None


# ────────────────────��───────────────────────────────────────────────────────── #
#  Metrics Collector                                                           #
# ────────────────────────────────────────────────────────────────────────────── #

class MetricsCollector:
    """14 个 benchmark 指标的统一采集器。"""

    def __init__(self, experiment_name: str = "default") -> None:
        self.experiment_name = experiment_name
        self._device_results: list[DeviceResult] = []
        self._task_lifecycles: dict[str, TaskLifecycle] = {}
        self._ooo_promotions: list[dict[str, Any]] = []
        self._alarm_events: list[dict[str, Any]] = []
        self._safety_rejections: list[dict[str, Any]] = []
        self._scheduling_rounds: list[float] = []
        self._experiment_start: float | None = None
        self._experiment_end: float | None = None
        self._total_compute_time_ms: float = 0.0
        self._prefix_cache: dict[str, Any] = {}
        self._prefix_hits: int = 0
        self._prefix_misses: int = 0
        self._alarm_triggers: dict[str, float] = {}  # alarm_id → triggered_at
        self._checkpoint_snapshots: dict[str, dict] = {}

    # ── 埋点 API ─────────────────────────────────────────────────────────────

    def experiment_start(self) -> None:
        self._experiment_start = time.monotonic()

    def experiment_end(self) -> None:
        self._experiment_end = time.monotonic()

    def record_device_result(self, result: DeviceResult) -> None:
        """【埋点 1-6 基础】每次设备调用后调用。"""
        self._device_results.append(result)

    def record_task_submit(
        self,
        task_id: str,
        lab_id: str,
        skill_name: str,
        priority: int = 2,
    ) -> None:
        lc = TaskLifecycle(
            task_id=task_id,
            lab_id=lab_id,
            skill_name=skill_name,
            priority=priority,
        )
        lc.submitted_at = time.monotonic()
        self._task_lifecycles[task_id] = lc

    def record_task_start(self, task_id: str) -> None:
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.started_at = time.monotonic()

    def record_task_complete(
        self,
        task_id: str,
        device_results: list[DeviceResult],
    ) -> None:
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.completed_at = time.monotonic()
            lc.device_results = list(device_results)

    def record_task_fail(self, task_id: str) -> None:
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.failed_at = time.monotonic()

    def record_task_wait(self, task_id: str, reason: str) -> None:
        lc = self._task_lifecycles.get(task_id)
        if lc and lc.wait_started_at is None:
            lc.wait_started_at = time.monotonic()
            lc.wait_reason = reason

    def record_task_resume(self, task_id: str) -> None:
        """【埋点 9】任务从等待中恢复（用于 Resume Latency 计算）。"""
        lc = self._task_lifecycles.get(task_id)
        if lc and lc.started_at is None:
            lc.started_at = time.monotonic()

    def record_ooo_promotion(
        self,
        task_id: str,
        reason: str,
        elapsed_since_submit_ms: float,
    ) -> None:
        """【埋点 7】记录 OoO 越级发射事件。"""
        self._ooo_promotions.append({
            "task_id": task_id,
            "reason": reason,
            "elapsed_since_submit_ms": elapsed_since_submit_ms,
            "timestamp": time.monotonic(),
        })
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.ooo_promoted = True

    def record_alarm_trigger(self, alarm_id: str, condition_expr: str) -> None:
        """【埋点 8 前置】记录报警触发时间。"""
        self._alarm_triggers[alarm_id] = time.monotonic()

    def record_alarm_response(
        self,
        alarm_id: str,
        condition_expr: str,
        first_action_task_id: str | None = None,
    ) -> None:
        """【埋点 8】Alarm Response Latency（first_action_task_id 启动时记录）。"""
        triggered_at = self._alarm_triggers.get(alarm_id)
        if triggered_at is None:
            triggered_at = time.monotonic()
        first_action_at = time.monotonic()
        response_latency_ms = (first_action_at - triggered_at) * 1000.0
        self._alarm_events.append({
            "alarm_id": alarm_id,
            "condition_expr": condition_expr,
            "triggered_at": triggered_at,
            "first_action_at": first_action_at,
            "response_latency_ms": response_latency_ms,
            "first_action_task_id": first_action_task_id,
        })

    def record_safety_rejection(
        self,
        task_id: str,
        device_id: str,
        reason: str,
    ) -> None:
        """【埋点 12】Safety Rejection。"""
        self._safety_rejections.append({
            "task_id": task_id,
            "device_id": device_id,
            "reason": reason,
            "rejected_at": time.monotonic(),
        })
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.safety_rejected = True

    def record_checkpoint_save(self, task_id: str, snapshot: dict | None = None) -> None:
        """【时间维度埋点】记录 checkpoint 保存。"""
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.checkpoint_saved = True
        if snapshot:
            self._checkpoint_snapshots[task_id] = snapshot

    def record_checkpoint_load(self, task_id: str) -> None:
        """【时间维度埋点】记录 checkpoint 加载。"""
        lc = self._task_lifecycles.get(task_id)
        if lc:
            lc.checkpoint_loaded = True

    def record_prefix_hit(self, key: str) -> None:
        self._prefix_hits += 1
        self._prefix_cache[key] = True

    def record_prefix_miss(self, key: str) -> None:
        self._prefix_misses += 1

    def record_scheduling_round_time(self, duration_ms: float) -> None:
        """【埋点 14】每轮调度耗时（用于 Jitter 计算）。"""
        self._scheduling_rounds.append(duration_ms)

    def add_compute_time(self, ms: float) -> None:
        """【埋点 5】CPU 计算时间累加。"""
        self._total_compute_time_ms += ms

    # ── 指标计算 ─────────────────────────────────────────────────────────────

    def finalize(self) -> dict[str, Any]:
        """计算所有 14 个指标。"""
        if self._experiment_start is None:
            return {}

        makespan = (self._experiment_end or self._experiment_start) - self._experiment_start
        total_tasks = len(self._task_lifecycles)
        completed_tasks = sum(
            1 for lc in self._task_lifecycles.values() if lc.completed_at is not None
        )
        failed_tasks = sum(
            1 for lc in self._task_lifecycles.values() if lc.failed_at is not None
        )

        # ── Metric 1: Makespan ────────────────────────────────────────────────
        makespan_s = makespan

        # ── Metric 2: Success Rate ────────────────────────────────────────────
        success_rate = completed_tasks / max(1, total_tasks)

        # ── Metric 3: Scheduling Overhead ─────────────────────────────────────
        avg_sched_round_ms = (
            sum(self._scheduling_rounds) / max(1, len(self._scheduling_rounds))
        )
        scheduling_overhead = avg_sched_round_ms / max(1, makespan * 1000.0)

        # ── Metric 4: Physical Device Utilization ─────────────────────────────
        device_busy_s: dict[str, float] = defaultdict(float)
        device_ids = set()
        for result in self._device_results:
            device_ids.add(result.device_id)
            device_busy_s[result.device_id] += result.total_latency_ms / 1000.0
        total_device_s = sum(device_busy_s.values())
        num_devices = max(1, len(device_ids))
        device_util = total_device_s / max(0.001, makespan * num_devices)

        # ── Metric 5: CPU Busy Ratio ───────────────────────────────────────────
        cpu_busy_ratio = self._total_compute_time_ms / max(1, makespan * 1000.0 * 3)

        # ── Metric 6: Average Task Waiting Time ─────────────────────────────────
        wait_times_ms: list[float] = []
        for lc in self._task_lifecycles.values():
            wt = lc.wait_time_ms
            if wt is not None:
                wait_times_ms.append(wt)
        avg_wait_time_ms = sum(wait_times_ms) / max(1, len(wait_times_ms))

        # ── Metric 7: Overlap Ratio ─────────────────────────────────────────────
        total_physical_ms = sum(r.total_latency_ms for r in self._device_results)
        if total_physical_ms > 0 and makespan > 0:
            overlap_ratio = 1.0 - (makespan * 1000.0 / total_physical_ms)
            overlap_ratio = max(0.0, min(1.0, overlap_ratio))
        else:
            overlap_ratio = 0.0

        # ── Metric 8: Conflict Stall Time ────────────────────────────────────────
        conflict_stall_ms = sum(
            r.total_latency_ms
            for r in self._device_results
            if r.wait_reason == "device_busy"
        )

        # ── Metric 9: Resume Latency ─────────────────────────────────────────────
        resume_latencies: list[float] = []
        for lc in self._task_lifecycles.values():
            if lc.checkpoint_loaded and lc.wait_time_ms is not None:
                resume_latencies.append(lc.wait_time_ms)
        avg_resume_latency_ms = sum(resume_latencies) / max(1, len(resume_latencies))

        # ── Metric 10: Prefix-Routing Hit Rate ───────────────────────────────────
        total_prefix = self._prefix_hits + self._prefix_misses
        prefix_hit_rate = self._prefix_hits / max(1, total_prefix)

        # ── Metric 11: Alarm Response Latency ────────────────────────────────────
        alarm_latencies = [e["response_latency_ms"] for e in self._alarm_events]
        avg_alarm_ms = sum(alarm_latencies) / max(1, len(alarm_latencies))

        # ── Metric 12: Safety Rejection Rate ────────────────────────────────────
        total_invocations = len(self._device_results)
        safety_rejections = len(self._safety_rejections)
        safety_rejection_rate = safety_rejections / max(1, total_invocations)

        # ── Metric 13: Fairness (Jain's Fairness Index) ──────────────────────────
        if wait_times_ms:
            n = len(wait_times_ms)
            sum_wt = sum(wait_times_ms)
            sum_sq = sum(w**2 for w in wait_times_ms)
            jains_fairness = (sum_wt**2) / (n * sum_sq) if sum_sq > 0 else 0.0
        else:
            jains_fairness = 0.0

        # ── Metric 14: Jitter ────────────────────────────────────────────────────
        if len(self._scheduling_rounds) > 1:
            mean_rt = sum(self._scheduling_rounds) / len(self._scheduling_rounds)
            variance = sum((r - mean_rt) ** 2 for r in self._scheduling_rounds) / len(
                self._scheduling_rounds
            )
            jitter_ms = variance ** 0.5
        else:
            jitter_ms = 0.0

        return {
            # ── 调度效率 ──────────────────────────────────────────────────────
            "makespan_s": round(makespan_s, 3),
            "scheduling_overhead": round(scheduling_overhead, 4),
            "jitter_ms": round(jitter_ms, 3),
            # ── 资源利用 ──────────────────────────────────────────────────────
            "physical_device_utilization": round(device_util, 4),
            "cpu_busy_ratio": round(cpu_busy_ratio, 4),
            # ── 恢复效率 ──────────────────────────────────────────────────────
            "overlap_ratio": round(overlap_ratio, 4),
            "conflict_stall_time_ms": round(conflict_stall_ms, 3),
            "resume_latency_ms": round(avg_resume_latency_ms, 3),
            "prefix_routing_hit_rate": round(prefix_hit_rate, 4),
            "alarm_response_latency_ms": round(avg_alarm_ms, 3),
            # ── 安全/稳定性 ───────────────────────────────────────────────────
            "success_rate": round(success_rate, 4),
            "safety_rejection_rate": round(safety_rejection_rate, 4),
            "jains_fairness": round(jains_fairness, 4),
            # ── 辅助信息 ──────────────────────────────────────────────────────
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "avg_task_wait_time_ms": round(avg_wait_time_ms, 3),
            "ooo_promotion_count": len(self._ooo_promotions),
            "alarm_count": len(self._alarm_events),
            "device_results_count": len(self._device_results),
            "prefix_hits": self._prefix_hits,
            "prefix_misses": self._prefix_misses,
        }

    # ── 导出 ─────────────────────────────────────────────────────────────────

    def export_csv(self, path: Path) -> None:
        """导出 metrics.csv（追加模式）。"""
        metrics = self.finalize()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(metrics.keys()) + "\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(",".join(str(v) for v in metrics.values()) + "\n")

    def export_jsonl(self, path: Path) -> None:
        """导出 events.jsonl（每行一个 DeviceResult）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for result in self._device_results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def export_timeline(self, path: Path) -> None:
        """导出 timeline.json（Gantt chart 数据）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        events = []
        for lc in self._task_lifecycles.values():
            events.append({
                "task_id": lc.task_id,
                "lab_id": lc.lab_id,
                "skill_name": lc.skill_name,
                "submitted_at": lc.submitted_at,
                "started_at": lc.started_at,
                "completed_at": lc.completed_at,
                "failed_at": lc.failed_at,
                "wait_reason": lc.wait_reason,
                "ooo_promoted": lc.ooo_promoted,
                "checkpoint_saved": lc.checkpoint_saved,
                "checkpoint_loaded": lc.checkpoint_loaded,
                "safety_rejected": lc.safety_rejected,
            })
        events.sort(key=lambda e: e["submitted_at"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

    def export_full_report(self, output_dir: Path) -> None:
        """导出完整报告。"""
        output_dir.mkdir(parents=True, exist_ok=True)

        self.export_csv(output_dir / "metrics.csv")
        self.export_jsonl(output_dir / "events.jsonl")
        self.export_timeline(output_dir / "timeline.json")

        report = {
            "experiment_name": self.experiment_name,
            "metrics": self.finalize(),
            "ooo_promotions": self._ooo_promotions,
            "alarm_events": self._alarm_events,
            "safety_rejections": self._safety_rejections,
            "device_results_summary": {
                "total": len(self._device_results),
                "by_device": {
                    dev: len([r for r in self._device_results if r.device_id == dev])
                    for dev in set(r.device_id for r in self._device_results)
                },
                "by_status": {
                    s: len([r for r in self._device_results if r.status == s])
                    for s in set(r.status for r in self._device_results)
                },
            },
            "scheduling_rounds": {
                "count": len(self._scheduling_rounds),
                "mean_ms": round(sum(self._scheduling_rounds) / max(1, len(self._scheduling_rounds)), 3),
            },
        }
        with open(output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📊 报告已导出至: {output_dir}/")
        for k, v in self.finalize().items():
            print(f"  {k}: {v}")

    @property
    def summary(self) -> dict[str, Any]:
        """快速摘要（不触发 finalize 的完整计算）。"""
        return {
            "experiment_name": self.experiment_name,
            "total_tasks": len(self._task_lifecycles),
            "device_invocations": len(self._device_results),
            "ooo_promotions": len(self._ooo_promotions),
            "alarm_events": len(self._alarm_events),
            "safety_rejections": len(self._safety_rejections),
            "scheduling_rounds": len(self._scheduling_rounds),
            "prefix_hit_rate": self._prefix_hits / max(1, self._prefix_hits + self._prefix_misses),
        }


__all__ = ["MetricsCollector", "TaskLifecycle"]
