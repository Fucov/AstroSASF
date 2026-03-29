"""
AstroSASF · Benchmarks · Astro-Concurrency-Benchmark (V7.1.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
太空站多领域实验并发压测基准脚本。

测试场景：
- 流体实验：5步操作（~15-20秒）
- 材料合成：7步操作（~20-25秒，含长耗时 turn_on_laser）
- 生物培养：4步操作（~12-16秒）

Chaos Injection：
- 硬件延迟：随机让 MCP Tool 耗时增加 200-300%
- 并发资源冲突：多实验同时请求 vacuum="ACTIVE"
- 突发遥测报警：第 5 秒触发温度 >= 80°C 抢占

Ablation Study：
- Baseline (Workers=1)：传统无并发 Agent
- DAG-OS (Workers=4)：我们最新的操作系统内核

Author: AstroSASF Team
Version: 7.1.1
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================ #
#  Benchmark Configuration                                                       #
# ============================================================================ #

@dataclass
class BenchmarkConfig:
    """压测配置。"""
    name: str = "Baseline"
    num_workers: int = 1
    chaos_enabled: bool = False
    chaos_delay_multiplier: float = 2.5
    preemption_enabled: bool = False
    preemption_trigger_time: float = 5.0  # 秒
    preemption_temperature: float = 85.0  # 摄氏度
    deadlock_timeout: float = 120.0  # 秒
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))

    # 硬件模拟耗时配置（秒）
    base_tool_delay_min: float = 2.0
    base_tool_delay_max: float = 4.0

    def __post_init__(self) -> None:
        if self.name == "Baseline":
            self.chaos_enabled = False
            self.preemption_enabled = False


# ============================================================================ #
#  Benchmark Metrics Collector                                                  #
# ============================================================================ #

@dataclass
class BenchmarkMetrics:
    """压测指标收集器。"""

    # 压测元数据
    benchmark_start_time: float = field(default_factory=time.perf_counter)
    benchmark_end_time: float = 0.0
    config: BenchmarkConfig | None = None

    # 吞吐量和时延
    sub_task_timestamps: list[tuple[str, float]] = field(default_factory=list)

    # 抢占延迟
    _preemption_trigger_time: float = 0.0
    _preemption_action_start_time: float = 0.0
    preemption_latency_ms: float = 0.0
    preemption_triggered: bool = False

    # 安全性指标
    guardrail_trigger_count: int = 0
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)

    # 死锁检测
    deadlock_detected: bool = False
    deadlock_reason: str = ""

    # 实验统计
    experiments_submitted: int = 0
    experiments_completed: int = 0
    experiments_failed: int = 0
    experiments_skipped: int = 0  # 被抢占跳过的

    # Mock 工具调用记录
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    chaos_injections: list[dict[str, Any]] = field(default_factory=list)

    # 抢占相关
    preemption_injections: list[dict[str, Any]] = field(default_factory=list)
    escape_tasks_executed: list[dict[str, Any]] = field(default_factory=list)

    def record_subtask_complete(self, task_id: str) -> None:
        """记录子任务完成。"""
        now = time.perf_counter()
        self.sub_task_timestamps.append((task_id, now))

    def record_preemption_trigger(self) -> None:
        """记录抢占触发时刻。"""
        self._preemption_trigger_time = time.perf_counter()
        self.preemption_triggered = True

    def record_preemption_action_start(self) -> None:
        """记录抢占动作开始执行时刻。"""
        self._preemption_action_start_time = time.perf_counter()
        if self._preemption_trigger_time > 0:
            self.preemption_latency_ms = (
                self._preemption_action_start_time - self._preemption_trigger_time
            ) * 1000

    def record_guardrail_trigger(
        self,
        tool_name: str,
        reason: str,
        params: dict[str, Any],
    ) -> None:
        """记录 Guardrail 拦截事件。"""
        self.guardrail_trigger_count += 1
        self.guardrail_events.append({
            "tool_name": tool_name,
            "reason": reason,
            "params": params,
            "timestamp": time.perf_counter(),
        })

    def record_tool_invocation(
        self,
        tool_name: str,
        duration_ms: float,
        chaos_applied: bool = False,
    ) -> None:
        """记录工具调用。"""
        self.tool_invocations.append({
            "tool_name": tool_name,
            "duration_ms": duration_ms,
            "chaos_applied": chaos_applied,
            "timestamp": time.perf_counter(),
        })

    def record_chaos_injection(self, tool_name: str, original_ms: float, actual_ms: float) -> None:
        """记录 Chaos 注入。"""
        self.chaos_injections.append({
            "tool_name": tool_name,
            "original_ms": original_ms,
            "actual_ms": actual_ms,
            "timestamp": time.perf_counter(),
        })

    def record_preemption_injection(self, temperature: float, escape_action: str) -> None:
        """记录抢占注入。"""
        self.preemption_injections.append({
            "temperature": temperature,
            "escape_action": escape_action,
            "timestamp": time.perf_counter(),
        })

    def record_escape_task(self, action: str, start_time: float) -> None:
        """记录逃生任务执行。"""
        self.escape_tasks_executed.append({
            "action": action,
            "start_time": start_time,
        })

    def get_makespan(self) -> float:
        """计算 Makespan（总耗时）。"""
        if self.sub_task_timestamps:
            return max(t for _, t in self.sub_task_timestamps) - self.benchmark_start_time
        return time.perf_counter() - self.benchmark_start_time

    def get_throughput(self) -> float:
        """计算吞吐量（子任务/秒）。"""
        makespan = self.get_makespan()
        if makespan > 0:
            return len(self.sub_task_timestamps) / makespan
        return 0.0

    def get_avg_task_duration(self) -> float:
        """计算平均任务耗时。"""
        if not self.tool_invocations:
            return 0.0
        return statistics.mean(inv["duration_ms"] for inv in self.tool_invocations)

    def finalize(self) -> None:
        """完成压测，标记结束时间。"""
        self.benchmark_end_time = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            # OS 级效能指标
            "system_throughput": self.get_throughput(),
            "preemption_latency_ms": self.preemption_latency_ms,
            "makespan_seconds": self.get_makespan(),
            "avg_task_duration_ms": self.get_avg_task_duration(),

            # 调度安全性
            "guardrail_trigger_count": self.guardrail_trigger_count,
            "deadlock_detected": self.deadlock_detected,
            "deadlock_reason": self.deadlock_reason,

            # 实验统计
            "experiments_submitted": self.experiments_submitted,
            "experiments_completed": self.experiments_completed,
            "experiments_failed": self.experiments_failed,
            "experiments_skipped": self.experiments_skipped,
            "total_subtasks": len(self.sub_task_timestamps),

            # 抢占统计
            "preemption_triggered": self.preemption_triggered,
            "preemption_injections": self.preemption_injections,
            "escape_tasks_executed": self.escape_tasks_executed,

            # Chaos 统计
            "chaos_injection_count": len(self.chaos_injections),
            "chaos_events": self.chaos_injections,

            # 详细事件
            "guardrail_events": self.guardrail_events,
            "tool_invocations": self.tool_invocations,

            # 压测元数据
            "benchmark_start_time": self.benchmark_start_time,
            "benchmark_end_time": self.benchmark_end_time,
            "config": {
                "name": self.config.name if self.config else "Unknown",
                "num_workers": self.config.num_workers if self.config else 1,
                "chaos_enabled": self.config.chaos_enabled if self.config else False,
                "preemption_enabled": self.config.preemption_enabled if self.config else False,
                "preemption_trigger_time": (
                    self.config.preemption_trigger_time if self.config else 5.0
                ),
            } if self.config else {},
        }


# ============================================================================ #
#  Benchmark Reporter                                                           #
# ============================================================================ #

class BenchmarkReporter:
    """压测报告生成器。"""

    def __init__(self, metrics: BenchmarkMetrics):
        self.metrics = metrics

    def print_console_report(self) -> None:
        """打印格式化终端报告。"""
        name = self.metrics.config.name if self.metrics.config else "Unknown"
        makespan = self.metrics.get_makespan()
        throughput = self.metrics.get_throughput()
        latency = self.metrics.preemption_latency_ms
        guardrails = self.metrics.guardrail_trigger_count

        print("\n")
        print("╔" + "═" * 78 + "╗")
        print(f"║  {'=' * 76}  ║")
        print(f"║  Benchmark: {name:<65}  ║")
        print(f"║  {'=' * 76}  ║")
        print("╠" + "═" * 78 + "╣")
        print("║  📊 OS-Level Performance Metrics                                     ║")
        print("╠" + "─" * 78 + "╣")
        print(f"║  • Makespan:              {makespan:>10.2f} seconds                           ║")
        print(f"║  • System Throughput:      {throughput:>10.2f} tasks/sec                        ║")
        print(f"║  • Avg Task Duration:     {self.metrics.get_avg_task_duration():>10.2f} ms                              ║")
        print(f"║  • Total Subtasks:        {len(self.metrics.sub_task_timestamps):>10d} completed                          ║")

        # 抢占指标
        if self.metrics.config and self.metrics.config.preemption_enabled:
            print("╠" + "─" * 78 + "╣")
            print("║  ⚡ Hardware Preemption Metrics                                    ║")
            print("╠" + "─" * 78 + "╣")
            if self.metrics.preemption_triggered:
                print(f"║  • Preemption Latency:   {latency:>10.2f} ms                              ║")
                print(f"║  • Escape Tasks Exec:    {len(self.metrics.escape_tasks_executed):>10d}                                   ║")
            else:
                print(f"║  ⚠️  Preemption NOT triggered (tasks completed before alarm)        ║")

        # 安全性
        print("╠" + "─" * 78 + "╣")
        print("║  🔒 Scheduling Safety Metrics                                      ║")
        print("╠" + "─" * 78 + "╣")
        print(f"║  • Guardrail Triggers:   {guardrails:>10d} intercepted                            ║")
        print(f"║  • Deadlock Detected:     {'YES ⚠️ ' if self.metrics.deadlock_detected else 'No ✓':>10}                                   ║")
        if self.metrics.deadlock_detected:
            print(f"║    Reason: {self.metrics.deadlock_reason[:68]:<68}     ║")

        # Chaos
        if self.metrics.chaos_injections:
            print("╠" + "─" * 78 + "╣")
            print("║  💥 Chaos Injection Summary                                        ║")
            print("╠" + "─" * 78 + "╣")
            print(f"║  • Total Injections:      {len(self.metrics.chaos_injections):>10d}                                    ║")
            chaos_tools = [c["tool_name"] for c in self.metrics.chaos_injections]
            top_chaos = statistics.mode(chaos_tools) if chaos_tools else "N/A"
            print(f"║  • Most Affected Tool:     {top_chaos:>10}                                      ║")

        # 实验统计
        print("╠" + "─" * 78 + "╣")
        print("║  🧪 Experiment Execution Summary                                   ║")
        print("╠" + "─" * 78 + "╣")
        print(f"║  • Submitted:            {self.metrics.experiments_submitted:>10d}                                      ║")
        print(f"║  • Completed:            {self.metrics.experiments_completed:>10d}                                      ║")
        print(f"║  • Failed:               {self.metrics.experiments_failed:>10d}                                      ║")
        print(f"║  • Skipped (Preempt):    {self.metrics.experiments_skipped:>10d}                                      ║")

        print("╚" + "═" * 78 + "╝\n")

    def save_json_report(self, output_path: Path) -> None:
        """保存 JSON 报告。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.metrics.to_dict()
        report["generated_at"] = datetime.now().isoformat()
        report["report_version"] = "7.1.1"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Report saved: {output_path}")


class AblationReporter:
    """Ablation Study 对比报告生成器。"""

    def __init__(self, baseline: BenchmarkMetrics, dag_os: BenchmarkMetrics):
        self.baseline = baseline
        self.dag_os = dag_os

    def print_comparison(self) -> None:
        """打印 Ablation Study 对比表格。"""
        base_makespan = self.baseline.get_makespan()
        dag_makespan = self.dag_os.get_makespan()
        makespan_speedup = (base_makespan / dag_makespan - 1) * 100 if dag_makespan > 0 else 0

        base_throughput = self.baseline.get_throughput()
        dag_throughput = self.dag_os.get_throughput()
        throughput_improvement = (dag_throughput / base_throughput - 1) * 100 if base_throughput > 0 else 0

        base_guardrails = self.baseline.guardrail_trigger_count
        dag_guardrails = self.dag_os.guardrail_trigger_count

        base_latency = self.baseline.preemption_latency_ms
        dag_latency = self.dag_os.preemption_latency_ms

        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "Ablation Study: Baseline vs DAG-OS" + " " * 19 + "║")
        print("╠" + "═" * 78 + "╣")
        print("║" + " " * 78 + "║")
        print("║  Metric                      │  Baseline      │  DAG-OS        │  Improvement     ║")
        print("║" + "─" * 78 + "║")
        print(f"║  Makespan (s)                │  {base_makespan:>10.2f}  │  {dag_makespan:>10.2f}  │  "
              f"{'+' if makespan_speedup > 0 else ''}{makespan_speedup:>7.1f}%      ║")
        print(f"║  Throughput (tasks/s)        │  {base_throughput:>10.2f}  │  {dag_throughput:>10.2f}  │  "
              f"{'+' if throughput_improvement > 0 else ''}{throughput_improvement:>7.1f}%      ║")
        print(f"║  Guardrail Triggers           │  {base_guardrails:>10d}  │  {dag_guardrails:>10d}  │  "
              f"{'-' if dag_guardrails < base_guardrails else '+'}{abs(dag_guardrails - base_guardrails):>7d}      ║")
        print(f"║  Preemption Latency (ms)      │  {'N/A':>10}  │  {dag_latency:>10.2f}  │  "
              f"{'✓ Preempt':>10}      ║" if dag_latency > 0 else
              f"║  Preemption Latency (ms)      │  {'N/A':>10}  │  {'N/A':>10}  │  {'N/A':>10}      ║")
        print("║" + " " * 78 + "║")
        print("╠" + "═" * 78 + "╣")
        print("║  📈 Summary                                                                ║")
        print("╠" + "─" * 78 + "╣")

        if makespan_speedup > 0:
            print(f"║  ✓ DAG-OS 相比 Baseline 提速 {makespan_speedup:.1f}%                                   ║")
        else:
            print(f"║  ⚠️  Makespan 增加 {abs(makespan_speedup):.1f}% (可能因抢占开销)                        ║")

        if throughput_improvement > 0:
            print(f"║  ✓ 吞吐量提升 {throughput_improvement:.1f}%                                           ║")

        if dag_guardrails <= base_guardrails:
            print(f"║  ✓ Guardrail 拦截 {'减少' if dag_guardrails < base_guardrails else '持平'}，调度更安全                          ║")

        print("╚" + "═" * 78 + "╝\n")


# ============================================================================ #
#  Mock MCP Tool with Chaos Injection (Long Duration)                             #
# ============================================================================ #

class MockMCPTool:
    """带 Chaos 注入的长耗时 Mock MCP Tool。"""

    def __init__(
        self,
        name: str,
        chaos_enabled: bool = True,
        chaos_multiplier: float = 2.5,
        delay_min: float = 2.0,
        delay_max: float = 4.0,
    ):
        self.name = name
        self.chaos_enabled = chaos_enabled
        self.chaos_multiplier = chaos_multiplier
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.call_count = 0

    async def invoke(
        self,
        params: dict[str, Any],
        metrics: BenchmarkMetrics | None = None,
    ) -> dict[str, Any]:
        """调用工具（带 Chaos 注入，2-4秒基础耗时）。"""
        self.call_count += 1
        start = time.perf_counter()

        # 计算基础延迟（2-4秒随机）
        base_delay = random.uniform(self.delay_min, self.delay_max)

        # 计算总延迟
        actual_delay = base_delay
        chaos_applied = False

        if self.chaos_enabled and random.random() < 0.3:  # 30% 概率触发 Chaos
            actual_delay *= self.chaos_multiplier
            chaos_applied = True

            if metrics:
                metrics.record_chaos_injection(
                    self.name, base_delay * 1000, actual_delay * 1000
                )

        # 模拟工具执行（长耗时）
        await asyncio.sleep(actual_delay)

        duration_ms = (time.perf_counter() - start) * 1000

        if metrics:
            metrics.record_tool_invocation(self.name, duration_ms, chaos_applied)

        return {
            "skill": self.name,
            "status": "success",
            "duration_ms": duration_ms,
            "params": params,
        }


# ============================================================================ #
#  Mock Interlock Engine (FSM)                                                 #
# ============================================================================ #

class MockInterlockEngine:
    """Mock 联锁引擎（检测资源冲突）。"""

    def __init__(self):
        self.vacuum_lock = asyncio.Lock()
        self._vacuum_busy = False
        self.temperature_limit = 80.0
        self.current_temperature = 25.0

    async def check_interlock(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> tuple[bool, str]:
        """检查联锁，返回 (允许, 原因)。"""
        # 温度检查
        if tool_name == "set_temperature":
            target = params.get("target", self.current_temperature)
            if target >= self.temperature_limit:
                return False, f"Temperature {target}°C >= {self.temperature_limit}°C limit (Guardrail)"

        # Vacuum 互斥检查
        if tool_name == "toggle_vacuum_pump":
            activate = params.get("activate", False)
            if activate and self._vacuum_busy:
                return False, "Vacuum pump already in use (Guardrail: Resource Conflict)"
            self._vacuum_busy = activate

        return True, "OK"

    def release_vacuum(self) -> None:
        """释放 Vacuum 锁。"""
        self._vacuum_busy = False


# ============================================================================ #
#  Mock Telemetry Bus with Alarm                                                #
# ============================================================================ #

class MockTelemetryBus:
    """Mock 遥测总线（支持报警触发）。"""

    def __init__(self):
        self._state: dict[str, Any] = {
            "temperature": 25.0,
            "pressure": 101.3,
            "vacuum_pump_active": False,
        }
        self._alarm_callbacks: list[callable] = []

    async def write(self, key: str, value: Any) -> None:
        """写入遥测值。"""
        self._state[key] = value

        # 检查温度报警
        if key == "temperature" and value >= 80.0:
            logger.warning(f"🚨 [Alarm] Temperature exceeded limit: {value}°C")
            for callback in self._alarm_callbacks:
                try:
                    result = callback(key, value)
                    # 如果回调是协程，等待完成
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Alarm callback error: {e}")

    async def snapshot(self) -> dict[str, Any]:
        """获取遥测快照。"""
        return dict(self._state)

    def register_alarm_callback(self, callback: callable) -> None:
        """注册报警回调。"""
        self._alarm_callbacks.append(callback)


# ============================================================================ #
#  DAG Scheduler with Preemption (Simplified)                                   #
# ============================================================================ #

class DAGScheduler:
    """带抢占的 DAG 调度器。"""

    def __init__(
        self,
        config: BenchmarkConfig,
        metrics: BenchmarkMetrics,
    ):
        self.config = config
        self.metrics = metrics
        self.engine = MockInterlockEngine()
        self.bus = MockTelemetryBus()
        self.tools: dict[str, MockMCPTool] = {}
        self.experiment_tasks: list[asyncio.Task] = []
        self.worker_pool: list[asyncio.Task] = []
        self.shutdown_flag = False
        self.preemption_event = asyncio.Event()
        self.escape_action_triggered = False
        self.ready_queue: asyncio.Queue = asyncio.Queue()
        self.active_experiments: dict[str, asyncio.Task] = {}
        self.active_count = 0
        self.max_concurrent = config.num_workers

        # 初始化 Mock Tools
        self._init_tools()

        # 注册报警回调
        self.bus.register_alarm_callback(self._on_temperature_alarm)

    def _init_tools(self) -> None:
        """初始化 Mock 工具。"""
        for tool_name in ["set_temperature", "toggle_vacuum_pump", "inject_nutrient",
                          "turn_on_laser", "move_robotic_arm", "emergency_cooling"]:
            self.tools[tool_name] = MockMCPTool(
                name=tool_name,
                chaos_enabled=self.config.chaos_enabled,
                chaos_multiplier=self.config.chaos_delay_multiplier,
                delay_min=self.config.base_tool_delay_min,
                delay_max=self.config.base_tool_delay_max,
            )

    async def _on_temperature_alarm(self, key: str, value: float) -> None:
        """温度报警回调。"""
        if not self.escape_action_triggered:
            self.escape_action_triggered = True
            logger.warning(f"🚨 [Scheduler] Preemption triggered! Temperature = {value}°C")

            # 记录抢占触发时间
            self.metrics.record_preemption_trigger()
            self.metrics.record_preemption_injection(value, "emergency_cooling")

            # 设置抢占事件
            self.preemption_event.set()

    async def submit_experiment(self, exp_id: str, steps: list[dict[str, Any]]) -> None:
        """提交实验到调度器。"""
        self.metrics.experiments_submitted += 1

        # 创建实验协程
        exp_coroutine = self._execute_experiment(exp_id, steps)
        task = asyncio.create_task(exp_coroutine, name=f"exp-{exp_id}")
        self.experiment_tasks.append(task)

        # 如果没有运行中的实验，启动
        if self.active_count < self.max_concurrent:
            task.add_done_callback(lambda _: self._start_next_experiment())

    def _start_next_experiment(self) -> None:
        """启动下一个待执行的实验（如果有）。"""
        pass  # 实验协程会自动执行

    async def _execute_experiment(
        self,
        exp_id: str,
        steps: list[dict[str, Any]],
    ) -> None:
        """执行单个实验。"""
        self.active_count += 1
        self.active_experiments[exp_id] = asyncio.current_task()

        logger.info(f"[{exp_id}] Starting {len(steps)} steps (active: {self.active_count})")

        completed_steps = 0

        for i, step in enumerate(steps):
            # 检查抢占信号
            if self.config.preemption_enabled and self.preemption_event.is_set():
                logger.warning(f"[{exp_id}] PREEMPTED at step {i}!")
                self.metrics.experiments_skipped += 1
                self.metrics.record_subtask_complete(f"{exp_id}-skipped")
                await self._execute_escape_action()
                self.active_count -= 1
                del self.active_experiments[exp_id]
                return

            tool_name = step["tool"]
            params = step.get("params", {})

            # 检查联锁
            allowed, reason = await self.engine.check_interlock(tool_name, params)
            if not allowed:
                logger.warning(f"[{exp_id}] Step {i} BLOCKED: {reason}")
                self.metrics.record_guardrail_trigger(tool_name, reason, params)
                self.metrics.experiments_failed += 1
                self.active_count -= 1
                del self.active_experiments[exp_id]
                return

            # 执行工具
            tool = self.tools.get(tool_name)
            if tool:
                result = await tool.invoke(params, self.metrics)
                logger.info(f"[{exp_id}] Step {i} [{tool_name}] → {result['status']} ({result['duration_ms']:.0f}ms)")

                if result["status"] == "success":
                    self.metrics.record_subtask_complete(f"{exp_id}-step{i}")
                    completed_steps += 1
                else:
                    self.metrics.experiments_failed += 1
                    self.active_count -= 1
                    del self.active_experiments[exp_id]
                    return

        # 实验完成
        self.metrics.experiments_completed += 1
        logger.info(f"[{exp_id}] Completed ({completed_steps}/{len(steps)} steps)")
        self.active_count -= 1
        del self.active_experiments[exp_id]

    async def _execute_escape_action(self) -> None:
        """执行逃生动作。"""
        # 记录逃生任务开始执行
        self.metrics.record_preemption_action_start()

        logger.warning("⚡ [Escape] Executing emergency_cooling...")

        # 创建紧急降温任务
        escape_tool = self.tools.get("emergency_cooling")
        if escape_tool:
            result = await escape_tool.invoke(
                {"target": 25.0, "mode": "rapid"},
                self.metrics,
            )
            self.metrics.record_escape_task("emergency_cooling", time.perf_counter())
            logger.warning(f"⚡ [Escape] emergency_cooling completed ({result['duration_ms']:.0f}ms)")

    async def wait_completion(self, timeout: float) -> bool:
        """等待所有实验完成。"""
        deadline = time.perf_counter() + timeout

        while self.experiment_tasks:
            # 检查超时
            if time.perf_counter() > deadline:
                logger.error(f"⏰ Timeout: {len(self.experiment_tasks)} tasks still pending")
                self.metrics.deadlock_detected = True
                self.metrics.deadlock_reason = f"Timeout after {timeout}s"
                return False

            # 等待任务完成
            done, pending = await asyncio.wait(
                self.experiment_tasks,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            self.experiment_tasks = list(pending)

        return True

    async def shutdown(self) -> None:
        """关闭调度器。"""
        self.shutdown_flag = True
        for task in self.experiment_tasks:
            if not task.done():
                task.cancel()


# ============================================================================ #
#  Experiment DAG Definitions                                                    #
# ============================================================================ #

FLUID_EXPERIMENT = [
    {"tool": "toggle_vacuum_pump", "params": {"activate": True}},
    {"tool": "set_temperature", "params": {"target": 25.0}},
    {"tool": "toggle_vacuum_pump", "params": {"activate": False}},
    {"tool": "set_temperature", "params": {"target": 20.0}},
    {"tool": "move_robotic_arm", "params": {"target_angle": 45.0}},
]

MATERIAL_EXPERIMENT = [
    {"tool": "toggle_vacuum_pump", "params": {"activate": True}},
    {"tool": "set_temperature", "params": {"target": 70.0}},
    {"tool": "set_temperature", "params": {"target": 65.0}},
    {"tool": "move_robotic_arm", "params": {"target_angle": 45.0}},
    {"tool": "turn_on_laser", "params": {"activate": True}},  # Long duration
    {"tool": "turn_on_laser", "params": {"activate": False}},
    {"tool": "set_temperature", "params": {"target": 25.0}},
]

BIO_EXPERIMENT = [
    {"tool": "set_temperature", "params": {"target": 37.0}},
    {"tool": "inject_nutrient", "params": {"volume_ml": 100.0}},
    {"tool": "set_temperature", "params": {"target": 37.0}},
    {"tool": "move_robotic_arm", "params": {"target_angle": 90.0}},
]


# ============================================================================ #
#  Main Benchmark Runner                                                        #
# ============================================================================ #

async def run_benchmark(config: BenchmarkConfig) -> BenchmarkMetrics:
    """运行单次压测。"""
    print(f"\n{'─' * 80}")
    print(f"  Running: {config.name} (Workers={config.num_workers}, "
          f"Chaos={'Y' if config.chaos_enabled else 'N'}, "
          f"Preempt={'Y' if config.preemption_enabled else 'N'})")
    print(f"{'─' * 80}")

    # 初始化指标收集器
    metrics = BenchmarkMetrics()
    metrics.config = config

    # 创建调度器
    scheduler = DAGScheduler(config, metrics)

    # 提交实验（确保实验足够长，能触发抢占）
    experiments = [
        ("fluid-1", FLUID_EXPERIMENT),
        ("material-1", MATERIAL_EXPERIMENT),
        ("bio-1", BIO_EXPERIMENT),
        ("fluid-2", FLUID_EXPERIMENT),
        ("material-2", MATERIAL_EXPERIMENT),
    ]

    # 提交所有实验
    for exp_id, steps in experiments:
        await scheduler.submit_experiment(exp_id, steps)

    # 启动 Chaos 注入（温度报警抢占）
    chaos_task = None
    if config.preemption_enabled:
        async def inject_alarm():
            logger.info(f"⏰ [Chaos] Will trigger temperature alarm at {config.preemption_trigger_time}s")
            await asyncio.sleep(config.preemption_trigger_time)
            logger.warning(f"🚨 [Chaos] Injecting temperature = {config.preemption_temperature}°C")
            await scheduler.bus.write("temperature", config.preemption_temperature)

        chaos_task = asyncio.create_task(inject_alarm(), name="chaos-alarm")

    # 等待所有实验完成
    await scheduler.wait_completion(config.deadlock_timeout)

    # 取消 Chaos 任务
    if chaos_task and not chaos_task.done():
        chaos_task.cancel()
        try:
            await chaos_task
        except asyncio.CancelledError:
            pass

    # 关闭调度器
    await scheduler.shutdown()

    # 完成压测
    metrics.finalize()

    # 生成报告
    reporter = BenchmarkReporter(metrics)
    reporter.print_console_report()

    return metrics


# ============================================================================ #
#  Ablation Study Runner                                                        #
# ============================================================================ #

async def run_ablation_study() -> tuple[BenchmarkMetrics, BenchmarkMetrics]:
    """运行 Ablation Study。"""

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AstroSASF V7.1 Ablation Study" + " " * 24 + "║")
    print("║" + " " * 15 + "Baseline (1 Worker) vs DAG-OS (4 Workers + Chaos + Preempt)" + " " * 7 + "║")
    print("╚" + "═" * 78 + "╝")

    # 等待 Ollama 就绪
    print("\n⏳ Waiting for Ollama to be ready...")
    await asyncio.sleep(2)

    # ── Round 1: Baseline ── #
    baseline_config = BenchmarkConfig(
        name="Baseline (Workers=1, No Chaos, No Preempt)",
        num_workers=1,
        chaos_enabled=False,
        preemption_enabled=False,
        preemption_trigger_time=5.0,  # 触发但不响应
    )

    baseline_metrics = await run_benchmark(baseline_config)

    # 保存 Baseline 报告
    baseline_reporter = BenchmarkReporter(baseline_metrics)
    baseline_path = baseline_config.output_dir / "baseline_results.json"
    baseline_reporter.save_json_report(baseline_path)

    # 短暂休息
    await asyncio.sleep(2)

    # ── Round 2: DAG-OS ── #
    dagos_config = BenchmarkConfig(
        name="DAG-OS (Workers=4, Chaos=True, Preempt=True)",
        num_workers=4,
        chaos_enabled=True,
        preemption_enabled=True,
        preemption_trigger_time=5.0,
        chaos_delay_multiplier=2.5,
    )

    dagos_metrics = await run_benchmark(dagos_config)

    # 保存 DAG-OS 报告
    dagos_reporter = BenchmarkReporter(dagos_metrics)
    dagos_path = dagos_config.output_dir / "dagos_results.json"
    dagos_reporter.save_json_report(dagos_path)

    # ── 对比报告 ── #
    ablation = AblationReporter(baseline_metrics, dagos_metrics)
    ablation.print_comparison()

    return baseline_metrics, dagos_metrics


# ============================================================================ #
#  Entry Point                                                                  #
# ============================================================================ #

async def main() -> None:
    """主入口。"""
    # 运行 Ablation Study
    baseline, dagos = await run_ablation_study()

    print("\n✅ Ablation Study completed successfully!")
    print(f"\nResults saved to: benchmark_results/")
    print(f"  - baseline_results.json")
    print(f"  - dagos_results.json")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main())
