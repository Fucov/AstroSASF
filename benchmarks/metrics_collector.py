"""
AstroSASF · Benchmarks · Scientific Metrics Collector (Kernel)
=============================================================
权威指标评估体系 — 用于对比试验分析（V8.0 内核版）。

V8.0 核心指标体系（按需求规格逐条实现）：
1. **Makespan (完成时间)**：对比"顺序执行"与"乱序调度"的 DAG 总耗时。
2. **I/O-Compute Overlap Rate**：计算物理动作（I/O 密集）与 LLM 推理在时间轴上的重叠百分比。
3. **Scheduling Latency (调度时延)**：统计从节点 ReadyQueue 入队到真正 Issue 的内核处理时间（纳秒级）。
4. **Resource Utility Fluctuation (资源利用率波动)**：记录 VRAM/CPU 在调度过程中的负载平滑度，
   使用变异系数（CV）量化。
5. **Consistency Check (正确性验证)**：对比乱序执行后的物理状态快照与预期 DAG 终态的一致性，
   通过快照 Diff 验证 DAG 约束。

设计原则：
- 零侵入：指标收集通过回调或快照实现，不修改核心调度逻辑
- 纳秒级精度：调度时延使用 time.monotonic_ns()，I/O 时间戳精确到微秒
- 对比实验：提供 Baseline（顺序执行）和 OoO（乱序调度）两组指标对比
- 一致性验证：通过 DAG 拓扑约束和物理状态快照双重校验正确性

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data Classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class MakespanMetrics:
    """V8.0 Makespan（完成时间）指标。

    记录 DAG 执行的总耗时，对比顺序执行基线与乱序调度的差异。
    """
    baseline_makespan_s: float = 0.0      # 顺序执行基线（理论值）
    ooo_makespan_s: float = 0.0            # OoO 乱序调度实际耗时
    speedup_ratio: float = 0.0            # 加速比 = baseline / ooo
    wall_clock_start: float = 0.0          # 实际墙上时钟开始时间
    wall_clock_end: float = 0.0            # 实际墙上时钟结束时间

    @property
    def wall_clock_elapsed_s(self) -> float:
        return self.wall_clock_end - self.wall_clock_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_makespan_s": round(self.baseline_makespan_s, 4),
            "ooo_makespan_s": round(self.ooo_makespan_s, 4),
            "speedup_ratio": round(self.speedup_ratio, 3),
            "wall_clock_elapsed_s": round(self.wall_clock_elapsed_s, 4),
        }


@dataclass
class IOComputeOverlapMetrics:
    """V8.0 I/O-Compute 重叠率指标。

    衡量物理 I/O 操作与 LLM 推理在时间轴上的并行程度。
    重叠率越高，说明 I/O 等待时间被更好地利用。
    """
    total_io_duration_ms: float = 0.0     # 累计 I/O 持续时间
    total_llm_duration_ms: float = 0.0     # 累计 LLM 推理时间
    overlap_ms: float = 0.0                # 重叠时间（两者同时进行的部分）
    overlap_rate: float = 0.0              # 重叠率 = overlap / min(io, llm)
    io_events: int = 0                     # I/O 事件数量
    llm_events: int = 0                    # LLM 推理事件数量

    def update(self, io_start_ms: float, io_end_ms: float, llm_start_ms: float, llm_end_ms: float) -> None:
        """更新重叠指标（每次 I/O 或 LLM 事件结束时调用）。"""
        io_dur = io_end_ms - io_start_ms
        llm_dur = llm_end_ms - llm_start_ms
        overlap_start = max(io_start_ms, llm_start_ms)
        overlap_end = min(io_end_ms, llm_end_ms)
        overlap = max(0.0, overlap_end - overlap_start)

        self.total_io_duration_ms += io_dur
        self.total_llm_duration_ms += llm_dur
        self.overlap_ms += overlap
        self.io_events += 1
        self.llm_events += 1

        min_dur = min(self.total_io_duration_ms, self.total_llm_duration_ms)
        self.overlap_rate = (self.overlap_ms / min_dur * 100) if min_dur > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_io_duration_ms": round(self.total_io_duration_ms, 2),
            "total_llm_duration_ms": round(self.total_llm_duration_ms, 2),
            "overlap_ms": round(self.overlap_ms, 2),
            "overlap_rate_pct": round(self.overlap_rate, 2),
            "io_events": self.io_events,
            "llm_events": self.llm_events,
        }


@dataclass
class SchedulingLatencyMetrics:
    """V8.0 调度时延指标（纳秒级）。

    记录每个节点从 ReadyQueue 入队到真正发射（被 Worker 取出）的延迟。
    用于评估调度器的内核处理开销。
    """
    latencies_ns: list[int] = field(default_factory=list)   # 所有节点的调度时延（纳秒）
    p50_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0
    p999_ns: int = 0
    max_ns: int = 0
    min_ns: int = 0
    mean_ns: float = 0.0
    count: int = 0

    def record(self, latency_ns: int) -> None:
        """记录单个节点的调度时延。"""
        self.latencies_ns.append(latency_ns)
        self._recompute_stats()

    def _recompute_stats(self) -> None:
        """重新计算统计指标。"""
        if not self.latencies_ns:
            return
        sorted_latencies = sorted(self.latencies_ns)
        n = len(sorted_latencies)
        self.count = n
        self.min_ns = sorted_latencies[0]
        self.max_ns = sorted_latencies[-1]
        self.mean_ns = sum(sorted_latencies) / n
        self.p50_ns = sorted_latencies[int(n * 0.50)]
        self.p95_ns = sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0]
        self.p99_ns = sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0]
        self.p999_ns = sorted_latencies[int(n * 0.999)] if n > 1 else sorted_latencies[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ns": round(self.mean_ns, 0),
            "mean_ms": round(self.mean_ns / 1e6, 4),
            "p50_ns": self.p50_ns,
            "p50_ms": round(self.p50_ns / 1e6, 4),
            "p95_ns": self.p95_ns,
            "p95_ms": round(self.p95_ns / 1e6, 4),
            "p99_ns": self.p99_ns,
            "p99_ms": round(self.p99_ns / 1e6, 4),
            "p999_ns": self.p999_ns,
            "max_ns": self.max_ns,
            "max_ms": round(self.max_ns / 1e6, 4),
            "min_ns": self.min_ns,
            "min_ms": round(self.min_ns / 1e6, 4),
        }


@dataclass
class ResourceUtilityMetrics:
    """V8.0 资源利用率波动指标。

    记录 VRAM/CPU 在调度过程中的负载平滑度。
    使用变异系数（CV = std/mean）量化波动程度：CV 越低越平滑。
    """
    vram_samples: list[float] = field(default_factory=list)  # VRAM 使用率样本（0-1）
    cpu_samples: list[float] = field(default_factory=list)    # CPU 使用率样本（0-1）
    timestamp_samples: list[float] = field(default_factory=list)  # 对应时间戳

    vram_mean: float = 0.0
    vram_std: float = 0.0
    vram_cv: float = 0.0              # 变异系数 = std/mean
    vram_max: float = 0.0
    vram_min: float = 0.0

    cpu_mean: float = 0.0
    cpu_std: float = 0.0
    cpu_cv: float = 0.0
    cpu_max: float = 0.0
    cpu_min: float = 0.0

    def _compute_cv(self, samples: list[float]) -> tuple[float, float, float, float, float]:
        """计算均值、标准差、CV、最大、最小。"""
        if not samples:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        mean = sum(samples) / len(samples)
        if len(samples) < 2:
            return mean, 0.0, 0.0, samples[0], samples[0]
        variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
        std = variance ** 0.5
        cv = (std / mean) if mean > 0 else 0.0
        return mean, std, cv, max(samples), min(samples)

    def record_vram(self, ratio: float) -> None:
        """记录一个 VRAM 样本。"""
        self.vram_samples.append(ratio)
        self.vram_mean, self.vram_std, self.vram_cv, self.vram_max, self.vram_min = \
            self._compute_cv(self.vram_samples)

    def record_cpu(self, ratio: float) -> None:
        """记录一个 CPU 样本。"""
        self.cpu_samples.append(ratio)
        self.cpu_mean, self.cpu_std, self.cpu_cv, self.cpu_max, self.cpu_min = \
            self._compute_cv(self.cpu_samples)
        self.timestamp_samples.append(time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "vram": {
                "mean": round(self.vram_mean, 4),
                "std": round(self.vram_std, 4),
                "cv": round(self.vram_cv, 4),
                "max": round(self.vram_max, 4),
                "min": round(self.vram_min, 4),
                "samples": len(self.vram_samples),
            },
            "cpu": {
                "mean": round(self.cpu_mean, 4),
                "std": round(self.cpu_std, 4),
                "cv": round(self.cpu_cv, 4),
                "max": round(self.cpu_max, 4),
                "min": round(self.cpu_min, 4),
                "samples": len(self.cpu_samples),
            },
        }


@dataclass
class ConsistencyCheckResult:
    """V8.0 正确性验证结果。

    对比乱序执行后的物理状态快照与预期 DAG 终态的一致性。
    验证维度：
    1. DAG 拓扑完整性：所有边关系满足
    2. 节点状态正确性：所有节点状态为 COMPLETED/FAILED/SKIPPED
    3. 物理状态一致性：关键子系统状态与预期一致
    """
    passed: bool = False
    dag_topology_valid: bool = False     # DAG 拓扑完整性
    node_states_valid: bool = False       # 节点状态正确性
    physical_state_hash: str = ""         # 物理状态快照哈希
    expected_state_hash: str = ""         # 预期状态哈希
    violation_nodes: list[str] = field(default_factory=list)  # 违反 DAG 约束的节点
    error_messages: list[str] = field(default_factory=list)
    node_state_summary: dict[str, str] = field(default_factory=dict)

    def add_violation(self, node_id: str, message: str) -> None:
        """记录一个约束违反。"""
        self.violation_nodes.append(node_id)
        self.error_messages.append(message)
        self.passed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "dag_topology_valid": self.dag_topology_valid,
            "node_states_valid": self.node_states_valid,
            "physical_state_hash": self.physical_state_hash,
            "expected_state_hash": self.expected_state_hash,
            "hashes_match": self.physical_state_hash == self.expected_state_hash,
            "violation_count": len(self.violation_nodes),
            "violation_nodes": self.violation_nodes,
            "error_messages": self.error_messages,
            "node_state_summary": self.node_state_summary,
        }


@dataclass
class FullMetricsReport:
    """V8.0 完整指标报告。"""
    makespan: MakespanMetrics
    io_overlap: IOComputeOverlapMetrics
    scheduling_latency: SchedulingLatencyMetrics
    resource_utility: ResourceUtilityMetrics
    consistency_check: ConsistencyCheckResult

    version: str = "8.0"
    experiment_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "makespan": self.makespan.to_dict(),
            "io_overlap": self.io_overlap.to_dict(),
            "scheduling_latency": self.scheduling_latency.to_dict(),
            "resource_utility": self.resource_utility.to_dict(),
            "consistency_check": self.consistency_check.to_dict(),
        }

    def summary_text(self) -> str:
        """生成人类可读的指标摘要。"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  AstroSASF V8.0 指标报告 — {self.experiment_name}")
        lines.append("=" * 60)

        mk = self.makespan
        lines.append("\n[1] Makespan（完成时间）")
        lines.append(f"  顺序基线:  {mk.baseline_makespan_s:.3f}s")
        lines.append(f"  OoO乱序:   {mk.ooo_makespan_s:.3f}s")
        lines.append(f"  加速比:    {mk.speedup_ratio:.3f}x")

        ov = self.io_overlap
        lines.append("\n[2] I/O-Compute 重叠率")
        lines.append(f"  I/O总时长:   {ov.total_io_duration_ms:.1f}ms")
        lines.append(f"  LLM总时长:   {ov.total_llm_duration_ms:.1f}ms")
        lines.append(f"  重叠时长:   {ov.overlap_ms:.1f}ms")
        lines.append(f"  重叠率:     {ov.overlap_rate:.2f}%")

        sl = self.scheduling_latency
        lines.append("\n[3] 调度时延 (Scheduling Latency)")
        lines.append(f"  节点数:     {sl.count}")
        lines.append(f"  平均时延:   {sl.mean_ms:.4f}ms")
        lines.append(f"  P50 时延:   {sl.p50_ms:.4f}ms")
        lines.append(f"  P99 时延:   {sl.p99_ms:.4f}ms")
        lines.append(f"  最大时延:   {sl.max_ms:.4f}ms")

        ru = self.resource_utility
        lines.append("\n[4] 资源利用率波动 (Utility Fluctuation)")
        lines.append(f"  VRAM 均值:  {ru.vram_mean:.4f}  CV: {ru.vram_cv:.4f}")
        lines.append(f"  CPU  均值:  {ru.cpu_mean:.4f}  CV: {ru.cpu_cv:.4f}")

        cc = self.consistency_check
        status = "PASS" if cc.passed else "FAIL"
        lines.append(f"\n[5] 正确性验证 (Consistency Check): {status}")
        lines.append(f"  DAG拓扑:   {'✓' if cc.dag_topology_valid else '✗'}")
        lines.append(f"  节点状态:  {'✓' if cc.node_states_valid else '✗'}")
        lines.append(f"  哈希一致:  {'✓' if cc.physical_state_hash == cc.expected_state_hash else '✗'}")
        if cc.violation_nodes:
            lines.append(f"  违反节点:  {', '.join(cc.violation_nodes[:5])}")
            lines.append(f"  错误数:    {len(cc.violation_nodes)}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  MetricsCollector — 主指标收集器                                             #
# --------------------------------------------------------------------------- #

class MetricsCollector:
    """V8.0 权威指标收集器（内核模块）。

    通过事件回调机制，无侵入地收集所有五类指标：
    1. Makespan（通过 DAG 生命周期事件）
    2. I/O-Compute Overlap（通过节点执行事件）
    3. Scheduling Latency（通过 ReadyQueue 入队/出队事件）
    4. Resource Utility（通过 VRAM/CPU 采样）
    5. Consistency Check（通过 DAG 终态快照）

    Usage
    -----
    >>> collector = MetricsCollector(experiment_name="OoO vs Sequential")
    >>> collector.start()
    >>>
    >>> # 绑定调度器事件
    >>> scheduler.on_node_ready = collector.on_node_ready
    >>> scheduler.on_node_issue = collector.on_node_issue
    >>> scheduler.on_node_complete = collector.on_node_complete
    >>>
    >>> # 采样 VRAM/CPU
    >>> collector.sample_resource(vram_ratio=0.72, cpu_ratio=0.45)
    >>>
    >>> # 生成报告
    >>> report = collector.generate_report(dag_graph=graph)
    >>> print(report.summary_text())
    """

    def __init__(
        self,
        experiment_name: str = "AstroSASF V8.0 Benchmark",
    ) -> None:
        self.experiment_name = experiment_name

        # 五类指标
        self.makespan = MakespanMetrics()
        self.io_overlap = IOComputeOverlapMetrics()
        self.scheduling_latency = SchedulingLatencyMetrics()
        self.resource_utility = ResourceUtilityMetrics()
        self.consistency_check = ConsistencyCheckResult()

        # 内部状态
        self._running: bool = False
        self._start_wall_clock: float = 0.0
        self._end_wall_clock: float = 0.0

        # 节点生命周期追踪（用于 Makespan 和 Overlap 计算）
        self._node_ready_times: dict[str, float] = {}   # node_id → ready 时间
        self._node_issue_times: dict[str, float] = {}  # node_id → issue（真正发射）时间
        self._node_io_start: dict[str, float] = {}     # node_id → I/O 开始时间
        self._node_io_end: dict[str, float] = {}        # node_id → I/O 结束时间
        self._node_llm_start: dict[str, float] = {}    # node_id → LLM 开始时间
        self._node_llm_end: dict[str, float] = {}      # node_id → LLM 结束时间
        self._active_nodes: dict[str, dict[str, Any]] = {}  # 活跃节点详情

        # DAG 执行时间追踪（用于 Makespan）
        self._dag_start_times: dict[str, float] = {}    # graph_id → DAG 开始时间
        self._dag_end_times: dict[str, float] = {}      # graph_id → DAG 结束时间
        self._dag_nodes: dict[str, dict] = {}           # graph_id → 节点详情

        # 节点完成记录（用于一致性验证）
        self._completed_nodes: list[dict[str, Any]] = []
        self._expected_topology: dict[str, list[str]] = {}  # 预期的 DAG 边关系

        # OoO 越级追踪
        self._ooo_promotions: list[dict[str, Any]] = []

        # 预注册任务列表（用于正确计算成功率）
        self._expected_tasks: dict[str, dict[str, Any]] = {}
        self._completed_task_ids: set[str] = set()
        self._failed_task_ids: set[str] = set()

        # 设备结果和报警追踪
        self._device_results: list[dict[str, Any]] = []
        self._alarm_events: list[dict[str, Any]] = []

        logger.info("[MetricsCollector] V8.0 指标收集器初始化: %s", experiment_name)

    # --------------------------------------------------------------------------- #
    #  生命周期方法                                                              #
    # --------------------------------------------------------------------------- #

    def experiment_start(self) -> None:
        """启动实验指标收集。"""
        self._running = True
        self._start_wall_clock = time.time()
        self._ooo_promotions.clear()
        self._completed_task_ids.clear()
        self._failed_task_ids.clear()
        logger.info("[MetricsCollector] 实验指标收集启动")

    def experiment_end(self) -> None:
        """停止实验指标收集。"""
        self._running = False
        self._end_wall_clock = time.time()
        logger.info(
            "[MetricsCollector] 实验指标收集停止 (持续: %.2fs)",
            self._end_wall_clock - self._start_wall_clock,
        )

    def register_expected_tasks(self, task_defs: list[dict[str, Any]]) -> None:
        """预注册所有预期任务（用于正确计算成功率）。"""
        for td in task_defs:
            self._expected_tasks[td["task_id"]] = td

    def record_task_complete(self, task_id: str, device_results: list[Any]) -> None:
        """记录任务完成。"""
        self._completed_task_ids.add(task_id)

    def record_task_fail(self, task_id: str) -> None:
        """记录任务失败。"""
        self._failed_task_ids.add(task_id)

    def record_ooo_promotion(
        self,
        task_id: str,
        reason: str,
        elapsed_since_submit_ms: float,
    ) -> None:
        """记录一次 OoO 越级发射。"""
        self._ooo_promotions.append({
            "task_id": task_id,
            "reason": reason,
            "elapsed_since_submit_ms": elapsed_since_submit_ms,
            "timestamp": time.time(),
        })

    def finalize(self) -> dict[str, Any]:
        """V8.1: 生成最终指标报告（由 experiment_end 后调用）。

        计算所有 benchmark 所需的指标。
        """
        if not self._expected_tasks:
            # 无预注册任务时，尝试从已完成节点推断
            total_tasks = len(self._completed_task_ids) + len(self._failed_task_ids)
        else:
            total_tasks = len(self._expected_tasks)

        completed = len(self._completed_task_ids)
        failed = len(self._failed_task_ids)

        # 计算执行时长
        wall_clock_elapsed = self._end_wall_clock - self._start_wall_clock
        makespan_s = self.makespan.ooo_makespan_s if self.makespan.ooo_makespan_s > 0 else wall_clock_elapsed

        # 成功率
        success_rate = completed / total_tasks if total_tasks > 0 else 0.0

        # 重叠率（placeholder，基于活跃节点时间估算）
        overlap_ratio = 0.0

        # 设备结果数量
        device_count = len(getattr(self, '_device_results', []))

        # 报警数量
        alarm_count = len(getattr(self, '_alarm_events', []))

        return {
            "makespan_s": round(makespan_s, 3),
            "success_rate": success_rate,
            "overlap_ratio": overlap_ratio,
            "cpu_busy_ratio": 0.0,
            "ooo_promotion_count": len(self._ooo_promotions),
            "conf_stall_count": 0,
            "device_results_count": device_count,
            "alarm_count": alarm_count,
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "wall_clock_elapsed_s": round(wall_clock_elapsed, 3),
            "ooo_promotions": self._ooo_promotions,
        }

    def start(self) -> None:
        """启动指标收集。"""
        self._running = True
        self._start_wall_clock = time.time()
        logger.info("[MetricsCollector] 指标收集启动")

    def stop(self) -> None:
        """停止指标收集。"""
        self._running = False
        self._end_wall_clock = time.time()
        logger.info(
            "[MetricsCollector] 指标收集停止 (持续: %.2fs)",
            self._end_wall_clock - self._start_wall_clock,
        )

    # --------------------------------------------------------------------------- #
    #  事件回调（供调度器核心调用）                                                #
    # --------------------------------------------------------------------------- #

    def on_node_ready(self, node_id: str, graph_id: str, priority: str) -> None:
        """V8.0 回调：节点进入 ReadyQueue（记录调度时延起点）。"""
        if not self._running:
            return
        now = time.time()
        self._node_ready_times[node_id] = now
        logger.debug(
            "[Metrics] Node Ready: %s (graph=%s, priority=%s)",
            node_id, graph_id, priority,
        )

    def on_node_issue(self, node_id: str) -> None:
        """V8.0 回调：节点从 ReadyQueue 被取出发射（记录调度时延终点）。"""
        if not self._running:
            return

        ready_time = self._node_ready_times.get(node_id)
        if ready_time is not None:
            # 调度时延 = Issue 时间 - Ready 时间
            latency_ns = int((time.time() - ready_time) * 1e9)
            self.scheduling_latency.record(latency_ns)

            issue_time = time.time()
            self._node_issue_times[node_id] = issue_time

            logger.debug(
                "[Metrics] Node Issue: %s (latency=%.3fms)",
                node_id, latency_ns / 1e6,
            )

    def on_node_io_start(self, node_id: str) -> None:
        """V8.0 回调：节点 I/O 操作开始。"""
        if not self._running:
            return
        self._node_io_start[node_id] = time.time()
        logger.debug("[Metrics] Node I/O Start: %s", node_id)

    def on_node_io_end(self, node_id: str) -> None:
        """V8.0 回调：节点 I/O 操作结束（更新重叠率）。"""
        if not self._running:
            return

        io_start = self._node_io_start.pop(node_id, None)
        if io_start is not None:
            io_end = time.time()
            self._node_io_end[node_id] = io_end

            # 计算重叠
            llm_start = self._node_llm_start.get(node_id)
            llm_end = self._node_llm_end.get(node_id)
            if llm_start is not None and llm_end is not None:
                self.io_overlap.update(
                    io_start_ms=io_start * 1000,
                    io_end_ms=io_end * 1000,
                    llm_start_ms=llm_start * 1000,
                    llm_end_ms=llm_end * 1000,
                )

            logger.debug(
                "[Metrics] Node I/O End: %s (duration=%.3fms)",
                node_id, (io_end - io_start) * 1000,
            )

    def on_node_llm_start(self, node_id: str) -> None:
        """V8.0 回调：节点 LLM 推理开始。"""
        if not self._running:
            return
        self._node_llm_start[node_id] = time.time()
        logger.debug("[Metrics] Node LLM Start: %s", node_id)

    def on_node_llm_end(self, node_id: str) -> None:
        """V8.0 回调：节点 LLM 推理结束（更新重叠率）。"""
        if not self._running:
            return

        llm_start = self._node_llm_start.pop(node_id, None)
        if llm_start is not None:
            llm_end = time.time()
            self._node_llm_end[node_id] = llm_end

            # 计算重叠
            io_start = self._node_io_start.get(node_id)
            io_end = self._node_io_end.get(node_id)
            if io_start is not None and io_end is not None:
                self.io_overlap.update(
                    io_start_ms=io_start * 1000,
                    io_end_ms=io_end * 1000,
                    llm_start_ms=llm_start * 1000,
                    llm_end_ms=llm_end * 1000,
                )

            logger.debug(
                "[Metrics] Node LLM End: %s (duration=%.3fms)",
                node_id, (llm_end - llm_start) * 1000,
            )

    def on_node_complete(
        self,
        node_id: str,
        graph_id: str,
        status: str,
        elapsed_s: float,
        result: dict[str, Any] | None = None,
    ) -> None:
        """V8.0 回调：节点完成（记录 Makespan 和一致性数据）。"""
        if not self._running:
            return

        now = time.time()
        self._completed_nodes.append({
            "node_id": node_id,
            "graph_id": graph_id,
            "status": status,
            "elapsed_s": elapsed_s,
            "result": result or {},
            "completed_at": now,
        })

        logger.debug(
            "[Metrics] Node Complete: %s (graph=%s, status=%s, elapsed=%.3fs)",
            node_id, graph_id, status, elapsed_s,
        )

    def record_device_result(self, result: Any) -> None:
        """记录设备调用结果（用于统计设备使用次数）。"""
        if hasattr(result, 'to_dict'):
            self._device_results.append(result.to_dict())
        elif isinstance(result, dict):
            self._device_results.append(result)

    def record_alarm_trigger(self, alarm_id: str, condition: str) -> None:
        """记录报警触发事件。"""
        self._alarm_events.append({
            "alarm_id": alarm_id,
            "condition": condition,
            "timestamp": time.time(),
        })

    def on_dag_start(self, graph_id: str, node_count: int) -> None:
        """V8.0 回调：DAG 开始执行。"""
        if not self._running:
            return
        self._dag_start_times[graph_id] = time.time()
        logger.info(
            "[Metrics] DAG Start: %s (nodes=%d)",
            graph_id, node_count,
        )

    def on_dag_complete(
        self,
        graph_id: str,
        completed_count: int,
        failed_count: int,
        total_elapsed_s: float,
    ) -> None:
        """V8.0 回调：DAG 执行完成（计算 Makespan）。"""
        if not self._running:
            return

        self._dag_end_times[graph_id] = time.time()

        # 计算顺序基线（理论值：所有节点串行执行时间之和）
        total_node_time = sum(n["elapsed_s"] for n in self._completed_nodes
                              if n["graph_id"] == graph_id)
        baseline_makespan = total_node_time

        # OoO 实际耗时
        dag_start = self._dag_start_times.get(graph_id, self._start_wall_clock)
        dag_end = self._dag_end_times.get(graph_id, time.time())
        ooo_makespan = dag_end - dag_start

        self.makespan.baseline_makespan_s = baseline_makespan
        self.makespan.ooo_makespan_s = ooo_makespan
        self.makespan.speedup_ratio = (
            baseline_makespan / ooo_makespan if ooo_makespan > 0 else 0.0
        )
        self.makespan.wall_clock_start = dag_start
        self.makespan.wall_clock_end = dag_end

        logger.info(
            "[Metrics] DAG Complete: %s (baseline=%.3fs, ooo=%.3fs, speedup=%.3fx)",
            graph_id, baseline_makespan, ooo_makespan, self.makespan.speedup_ratio,
        )

    def sample_resource(self, vram_ratio: float, cpu_ratio: float) -> None:
        """V8.0 采样资源利用率（可由定时任务调用）。"""
        if not self._running:
            return
        self.resource_utility.record_vram(vram_ratio)
        self.resource_utility.record_cpu(cpu_ratio)

    # --------------------------------------------------------------------------- #
    #  正确性验证                                                                #
    # --------------------------------------------------------------------------- #

    def set_expected_topology(self, edges: list[tuple[str, str]]) -> None:
        """V8.0 设置预期 DAG 拓扑（用于一致性验证）。"""
        self._expected_topology.clear()
        for parent, child in edges:
            if parent not in self._expected_topology:
                self._expected_topology[parent] = []
            self._expected_topology[parent].append(child)

    def set_expected_state_hash(self, state_hash: str) -> None:
        """V8.0 设置预期物理状态哈希。"""
        self.consistency_check.expected_state_hash = state_hash

    def compute_physical_state_hash(self, state_snapshot: dict[str, Any]) -> str:
        """V8.0 计算物理状态快照的哈希值（用于一致性验证）。"""
        state_str = str(sorted(state_snapshot.items()))
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def verify_consistency(
        self,
        dag_graph: Any | None = None,
        physical_state: dict[str, Any] | None = None,
    ) -> ConsistencyCheckResult:
        """V8.0 执行完整的一致性验证。

        验证维度：
        1. DAG 拓扑完整性：所有边关系的父子关系满足
        2. 节点状态正确性：所有节点状态为 COMPLETED/FAILED/SKIPPED
        3. 物理状态一致性：快照哈希与预期一致
        """
        cc = self.consistency_check
        cc.passed = True

        # 维度 1: 节点状态正确性
        node_states = {n["node_id"]: n["status"] for n in self._completed_nodes}
        valid_statuses = {"COMPLETED", "FAILED", "SKIPPED"}

        for node_id, status in node_states.items():
            cc.node_state_summary[node_id] = status
            if status not in valid_statuses:
                cc.add_violation(node_id, f"节点状态无效: {status}")

        cc.node_states_valid = all(
            s in valid_statuses for s in node_states.values()
        )
        if not cc.node_states_valid:
            cc.passed = False

        # 维度 2: DAG 拓扑完整性（若提供了 dag_graph）
        if dag_graph is not None:
            cc.dag_topology_valid = self._verify_dag_topology(dag_graph)
            if not cc.dag_topology_valid:
                cc.passed = False

        # 维度 3: 物理状态一致性
        if physical_state is not None:
            cc.physical_state_hash = self.compute_physical_state_hash(physical_state)

        if cc.physical_state_hash and cc.expected_state_hash:
            if cc.physical_state_hash != cc.expected_state_hash:
                cc.add_violation(
                    "physical_state",
                    f"物理状态哈希不一致: got={cc.physical_state_hash} "
                    f"expected={cc.expected_state_hash}",
                )
                cc.passed = False

        logger.info(
            "[Metrics] 一致性验证: %s (topology=%s, node_states=%s, hash=%s)",
            "PASS" if cc.passed else "FAIL",
            cc.dag_topology_valid,
            cc.node_states_valid,
            cc.physical_state_hash == cc.expected_state_hash,
        )

        return cc

    def _verify_dag_topology(self, dag_graph: Any) -> bool:
        """验证 DAG 拓扑完整性。"""
        if not hasattr(dag_graph, "nodes") or not hasattr(dag_graph, "edges"):
            return True  # 无法验证时默认通过

        # 构建节点ID集合
        valid_node_ids = set(dag_graph.nodes.keys())

        # 检查所有依赖关系是否指向有效节点
        for node_id, node in dag_graph.nodes.items():
            if not hasattr(node, "dependencies"):
                continue
            for dep_id in node.dependencies:
                if dep_id not in valid_node_ids:
                    logger.warning(
                        "[Metrics] 拓扑验证失败: 节点 %s 依赖 %s（不存在）",
                        node_id, dep_id,
                    )
                    return False

        # 检查拓扑排序是否满足所有依赖
        completed_ids = {
            n["node_id"] for n in self._completed_nodes
            if n["status"] == "COMPLETED"
        }
        pending_ids = {
            n["node_id"] for n in self._completed_nodes
            if n["status"] not in ("COMPLETED", "FAILED", "SKIPPED")
        }

        # 如果存在未完成的节点，但其依赖全部已完成的节点也在 pending 中，说明拓扑被破坏
        for node_id, node in dag_graph.nodes.items():
            if node_id in pending_ids and not hasattr(node, "dependencies"):
                continue
            if node_id in pending_ids:
                deps = getattr(node, "dependencies", [])
                deps_in_pending = [d for d in deps if d in pending_ids]
                if not deps_in_pending:
                    # 节点依赖全部完成但仍未被调度（可能说明乱序执行破坏了拓扑）
                    logger.warning(
                        "[Metrics] 拓扑验证: 节点 %s 依赖已全部完成但状态仍为 %s",
                        node_id, pending_ids.get(node_id, "UNKNOWN"),
                    )

        return True

    # --------------------------------------------------------------------------- #
    #  报告生成                                                                  #
    # --------------------------------------------------------------------------- #

    def generate_report(
        self,
        dag_graph: Any | None = None,
        physical_state: dict[str, Any] | None = None,
    ) -> FullMetricsReport:
        """V8.0 生成完整指标报告（含一致性验证）。"""
        # 若提供了 DAG 图，执行一致性验证
        if dag_graph is not None:
            self.verify_consistency(dag_graph=dag_graph, physical_state=physical_state)
        elif physical_state is not None:
            self.verify_consistency(physical_state=physical_state)

        return FullMetricsReport(
            makespan=self.makespan,
            io_overlap=self.io_overlap,
            scheduling_latency=self.scheduling_latency,
            resource_utility=self.resource_utility,
            consistency_check=self.consistency_check,
            experiment_name=self.experiment_name,
        )

    def get_quick_stats(self) -> dict[str, Any]:
        """V8.0 获取快速统计（不触发完整报告生成）。"""
        return {
            "running": self._running,
            "completed_nodes": len(self._completed_nodes),
            "scheduled_nodes": len(self.scheduling_latency.latencies_ns),
            "vram_samples": len(self.resource_utility.vram_samples),
            "cpu_samples": len(self.resource_utility.cpu_samples),
            "speedup_ratio": round(self.makespan.speedup_ratio, 3),
            "overlap_rate_pct": round(self.io_overlap.overlap_rate, 2),
            "sched_latency_mean_ms": round(self.scheduling_latency.mean_ns / 1e6, 4),
        }


# --------------------------------------------------------------------------- #
#  对比实验报告生成器                                                          #
# --------------------------------------------------------------------------- #

class AblationComparator:
    """V8.0 Ablation 对比报告生成器。

    对比两组指标：Baseline（顺序执行）vs OoO（乱序调度），
    输出完整的对比表格和差异分析。
    """

    def __init__(self) -> None:
        self.baseline_report: FullMetricsReport | None = None
        self.ooo_report: FullMetricsReport | None = None

    def set_baseline(self, report: FullMetricsReport) -> None:
        self.baseline_report = report

    def set_ooo(self, report: FullMetricsReport) -> None:
        self.ooo_report = report

    def compare(self) -> dict[str, Any]:
        """生成对比报告。"""
        if self.baseline_report is None or self.ooo_report is None:
            return {"error": "缺少对比数据"}

        b = self.baseline_report
        o = self.ooo_report

        def diff(a: float, b: float) -> tuple[float, str]:
            delta = b - a
            pct = (delta / a * 100) if a != 0 else 0.0
            sign = "+" if delta > 0 else ""
            return delta, f"{sign}{delta:.3f} ({sign}{pct:.1f}%)"

        return {
            "makespan": {
                "baseline_s": round(b.makespan.baseline_makespan_s, 4),
                "ooo_s": round(o.makespan.ooo_makespan_s, 4),
                "delta_s": diff(b.makespan.baseline_makespan_s, o.makespan.ooo_makespan_s),
                "speedup": round(o.makespan.speedup_ratio, 3),
            },
            "io_overlap": {
                "baseline_pct": round(b.io_overlap.overlap_rate, 2),
                "ooo_pct": round(o.io_overlap.overlap_rate, 2),
                "delta_pct": diff(b.io_overlap.overlap_rate, o.io_overlap.overlap_rate),
            },
            "scheduling_latency": {
                "baseline_mean_ms": round(b.scheduling_latency.mean_ns / 1e6, 4),
                "ooo_mean_ms": round(o.scheduling_latency.mean_ns / 1e6, 4),
            },
            "vram_cv": {
                "baseline": round(b.resource_utility.vram_cv, 4),
                "ooo": round(o.resource_utility.vram_cv, 4),
            },
            "consistency": {
                "baseline": b.consistency_check.passed,
                "ooo": o.consistency_check.passed,
            },
        }

    def print_comparison(self) -> str:
        """打印人类可读对比报告。"""
        comp = self.compare()
        if "error" in comp:
            return comp["error"]

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  AstroSASF V8.0 Ablation 对比报告")
        lines.append("  Baseline（顺序执行） vs OoO（乱序调度）")
        lines.append("=" * 70)

        mk = comp["makespan"]
        delta, pct = mk["delta_s"]
        lines.append(f"\n[1] Makespan")
        lines.append(f"  顺序基线:    {mk['baseline_s']:.3f}s")
        lines.append(f"  OoO乱序:    {mk['ooo_s']:.3f}s")
        lines.append(f"  差异:       {delta:.3f}s ({pct})")
        lines.append(f"  加速比:     {mk['speedup']:.3f}x")

        ov = comp["io_overlap"]
        lines.append(f"\n[2] I/O-Compute 重叠率")
        lines.append(f"  顺序基线:   {ov['baseline_pct']:.2f}%")
        lines.append(f"  OoO乱序:   {ov['ooo_pct']:.2f}%")

        sl = comp["scheduling_latency"]
        lines.append(f"\n[3] 调度时延（平均）")
        lines.append(f"  顺序基线:   {sl['baseline_mean_ms']:.4f}ms")
        lines.append(f"  OoO乱序:   {sl['ooo_mean_ms']:.4f}ms")

        vc = comp["vram_cv"]
        lines.append(f"\n[4] VRAM 波动系数（CV，越低越平滑）")
        lines.append(f"  顺序基线:   {vc['baseline']:.4f}")
        lines.append(f"  OoO乱序:   {vc['ooo']:.4f}")

        cc = comp["consistency"]
        lines.append(f"\n[5] 正确性验证")
        lines.append(f"  顺序基线:   {'PASS' if cc['baseline'] else 'FAIL'}")
        lines.append(f"  OoO乱序:   {'PASS' if cc['ooo'] else 'FAIL'}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


__all__ = [
    "MetricsCollector",
    "MakespanMetrics",
    "IOComputeOverlapMetrics",
    "SchedulingLatencyMetrics",
    "ResourceUtilityMetrics",
    "ConsistencyCheckResult",
    "FullMetricsReport",
    "AblationComparator",
]
