"""
AstroSASF · Benchmarks · Astro-Concurrency-Benchmark (V7.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
太空站多领域实验并发压测基准脚本。

测试场景：
- 流体实验：5步操作
- 材料合成：7步操作（含长耗时 turn_on_laser）
- 生物培养：4步操作

Chaos Injection：
- 硬件延迟：随机让 MCP Tool 耗时增加 300%
- 并发资源冲突：多实验同时请求 vacuum="ACTIVE"
- 突发遥测报警：第 5 秒触发温度 >= 80°C 抢占

Author: AstroSASF Team
Version: 7.1
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
    num_workers: int = 2
    chaos_enabled: bool = True
    chaos_delay_multiplier: float = 3.0
    preemption_trigger_time: float = 5.0  # 秒
    preemption_temperature: float = 85.0  # 摄氏度
    deadlock_timeout: float = 60.0  # 秒
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))


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
    completed_subtasks: list[float] = field(default_factory=list)  # (task_id, completion_time)
    sub_task_timestamps: list[tuple[str, float]] = field(default_factory=list)

    # 抢占延迟
    preemption_trigger_time: float = 0.0
    preemption_action_start_time: float = 0.0
    preemption_latency_ms: float = 0.0

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

    # Mock 工具调用记录
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    chaos_injections: list[dict[str, Any]] = field(default_factory=list)

    def record_subtask_complete(self, task_id: str) -> None:
        """记录子任务完成。"""
        now = time.perf_counter()
        self.sub_task_timestamps.append((task_id, now))
        self.completed_subtasks.append(now)

    def record_preemption_trigger(self) -> None:
        """记录抢占触发时刻。"""
        self.preemption_trigger_time = time.perf_counter()

    def record_preemption_action_start(self) -> None:
        """记录抢占动作开始执行时刻。"""
        self.preemption_action_start_time = time.perf_counter()
        if self.preemption_trigger_time > 0:
            self.preemption_latency_ms = (
                self.preemption_action_start_time - self.preemption_trigger_time
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

    def get_makespan(self) -> float:
        """计算 Makespan（总耗时）。"""
        if self.completed_subtasks:
            return max(self.completed_subtasks) - self.benchmark_start_time
        return time.perf_counter() - self.benchmark_start_time

    def get_throughput(self) -> float:
        """计算吞吐量（子任务/秒）。"""
        makespan = self.get_makespan()
        if makespan > 0:
            return len(self.completed_subtasks) / makespan
        return 0.0

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

            # 调度安全性
            "guardrail_trigger_count": self.guardrail_trigger_count,
            "deadlock_detected": self.deadlock_detected,
            "deadlock_reason": self.deadlock_reason,

            # 实验统计
            "experiments_submitted": self.experiments_submitted,
            "experiments_completed": self.experiments_completed,
            "experiments_failed": self.experiments_failed,
            "total_subtasks": len(self.completed_subtasks),

            # Chaos 统计
            "chaos_injection_count": len(self.chaos_injections),
            "chaos_events": self.chaos_injections,

            # 详细事件
            "guardrail_events": self.guardrail_events,
            "tool_invocations": self.tool_invocations,
            "sub_task_timestamps": [
                {"task_id": tid, "completion_time": ct}
                for tid, ct in self.sub_task_timestamps
            ],

            # 压测元数据
            "benchmark_start_time": self.benchmark_start_time,
            "benchmark_end_time": self.benchmark_end_time,
            "config": {
                "num_workers": self.config.num_workers if self.config else 2,
                "chaos_enabled": self.config.chaos_enabled if self.config else True,
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
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "AstroSASF V7.1 Concurrency Benchmark Report" + " " * 21 + "║")
        print("╠" + "═" * 78 + "╣")

        # 1. OS 级效能指标
        print("║  📊 OS-Level Performance Metrics                                     ║")
        print("╠" + "─" * 78 + "╣")
        print(f"║  • System Throughput:     {self.metrics.get_throughput():>8.2f} tasks/sec                         ║")
        print(f"║  • Preemption Latency:   {self.metrics.preemption_latency_ms:>8.2f} ms                              ║")
        print(f"║  • Makespan:              {self.metrics.get_makespan():>8.2f} seconds                            ║")
        print(f"║  • Total Subtasks:        {len(self.metrics.completed_subtasks):>8d} completed                          ║")

        # 2. 调度安全性
        print("╠" + "─" * 78 + "╣")
        print("║  🔒 Scheduling Safety Metrics                                      ║")
        print("╠" + "─" * 78 + "╣")
        print(f"║  • Guardrail Triggers:   {self.metrics.guardrail_trigger_count:>8d} intercepted                            ║")
        print(f"║  • Deadlock Detected:     {'YES ⚠️ ' if self.metrics.deadlock_detected else 'No ✓':>8}                                   ║")
        if self.metrics.deadlock_detected:
            print(f"║    Reason: {self.metrics.deadlock_reason[:68]:<68}     ║")

        # 3. Chaos Injection 统计
        if self.metrics.chaos_injections:
            print("╠" + "─" * 78 + "╣")
            print("║  💥 Chaos Injection Summary                                        ║")
            print("╠" + "─" * 78 + "╣")
            print(f"║  • Total Injections:      {len(self.metrics.chaos_injections):>8d}                                    ║")
            chaos_tools = [c["tool_name"] for c in self.metrics.chaos_injections]
            top_chaos = statistics.mode(chaos_tools) if chaos_tools else "N/A"
            print(f"║  • Most Affected Tool:     {top_chaos:>8}                                      ║")

        # 4. 实验执行统计
        print("╠" + "─" * 78 + "╣")
        print("║  🧪 Experiment Execution Summary                                   ║")
        print("╠" + "─" * 78 + "╣")
        print(f"║  • Experiments Submitted: {self.metrics.experiments_submitted:>8d}                                      ║")
        print(f"║  • Experiments Completed: {self.metrics.experiments_completed:>8d}                                      ║")
        print(f"║  • Experiments Failed:    {self.metrics.experiments_failed:>8d}                                      ║")

        # 5. 工具调用统计
        if self.metrics.tool_invocations:
            print("╠" + "─" * 78 + "╣")
            print("║  🔧 Tool Invocation Statistics                                    ║")
            print("╠" + "─" * 78 + "╣")
            tool_counts: dict[str, int] = {}
            tool_durations: dict[str, list[float]] = {}
            for inv in self.metrics.tool_invocations:
                name = inv["tool_name"]
                tool_counts[name] = tool_counts.get(name, 0) + 1
                tool_durations.setdefault(name, []).append(inv["duration_ms"])

            for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:5]:
                avg_dur = statistics.mean(tool_durations[tool_name])
                print(f"║  • {tool_name:<20} {count:>5} calls  avg: {avg_dur:>6.1f}ms               ║")

        print("╚" + "═" * 78 + "╝\n")

    def save_json_report(self, output_path: Path) -> None:
        """保存 JSON 报告。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.metrics.to_dict()
        report["generated_at"] = datetime.now().isoformat()
        report["report_version"] = "7.1"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Benchmark report saved to: {output_path}")


# ============================================================================ #
#  Mock MCP Tool with Chaos Injection                                           #
# ============================================================================ #

class MockMCPTool:
    """带 Chaos 注入的 Mock MCP Tool。"""

    def __init__(
        self,
        name: str,
        base_delay_ms: float,
        chaos_enabled: bool = True,
        chaos_multiplier: float = 3.0,
    ):
        self.name = name
        self.base_delay_ms = base_delay_ms
        self.chaos_enabled = chaos_enabled
        self.chaos_multiplier = chaos_multiplier
        self.call_count = 0

    async def invoke(
        self,
        params: dict[str, Any],
        metrics: BenchmarkMetrics | None = None,
    ) -> dict[str, Any]:
        """调用工具（带 Chaos 注入）。"""
        self.call_count += 1
        start = time.perf_counter()

        # 计算延迟
        delay_ms = self.base_delay_ms
        chaos_applied = False

        if self.chaos_enabled and random.random() < 0.3:  # 30% 概率触发 Chaos
            delay_ms *= self.chaos_multiplier
            chaos_applied = True

            if metrics:
                metrics.record_chaos_injection(
                    self.name, self.base_delay_ms, delay_ms
                )

        # 模拟工具执行
        await asyncio.sleep(delay_ms / 1000.0)

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
        self.arm_lock = asyncio.Lock()
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
                return False, f"Temperature {target}°C >= {self.temperature_limit}°C limit"

        # Vacuum 互斥检查
        if tool_name == "toggle_vacuum_pump":
            activate = params.get("activate", False)
            if activate:
                # 检查是否已被占用（简化模拟）
                if hasattr(self, "_vacuum_busy") and self._vacuum_busy:
                    return False, "Vacuum pump already in use by another experiment"
                self._vacuum_busy = True
            else:
                self._vacuum_busy = False

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
        self._alarms: dict[str, asyncio.Event] = {}
        self._alarm_callbacks: list[callable] = []

    async def write(self, key: str, value: Any) -> None:
        """写入遥测值。"""
        self._state[key] = value

        # 检查温度报警
        if key == "temperature" and value >= 80.0:
            for callback in self._alarm_callbacks:
                try:
                    callback(key, value)
                except Exception:
                    pass

    async def snapshot(self) -> dict[str, Any]:
        """获取遥测快照。"""
        return dict(self._state)

    def register_alarm_callback(self, callback: callable) -> None:
        """注册报警回调。"""
        self._alarm_callbacks.append(callback)


# ============================================================================ #
#  Mock DAG Scheduler (Simplified)                                             #
# ============================================================================ #

class MockDAGScheduler:
    """简化版 DAG 调度器（用于压测）。"""

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
        self.ready_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: list[asyncio.Task] = []
        self.shutdown_flag = False
        self.completed_count = 0
        self.failed_count = 0

        # 初始化 Mock Tools
        self._init_tools()

    def _init_tools(self) -> None:
        """初始化 Mock 工具。"""
        tool_configs = [
            ("set_temperature", 200),
            ("toggle_vacuum_pump", 150),
            ("inject_nutrient", 100),
            ("turn_on_laser", 3000),  # 长耗时模拟
            ("move_robotic_arm", 500),
        ]

        for name, delay in tool_configs:
            self.tools[name] = MockMCPTool(
                name=name,
                base_delay_ms=delay,
                chaos_enabled=self.config.chaos_enabled,
                chaos_multiplier=self.config.chaos_delay_multiplier,
            )

    async def submit_experiment(self, exp_id: str, steps: list[dict[str, Any]]) -> None:
        """提交实验到调度器。"""
        self.metrics.experiments_submitted += 1

        # 创建执行任务
        task = asyncio.create_task(
            self._execute_experiment(exp_id, steps),
            name=f"exp-{exp_id}",
        )
        self.running_tasks.append(task)

    async def _execute_experiment(
        self,
        exp_id: str,
        steps: list[dict[str, Any]],
    ) -> None:
        """执行单个实验。"""
        logger.info(f"[{exp_id}] 开始执行 {len(steps)} 步实验")

        for i, step in enumerate(steps):
            tool_name = step["tool"]
            params = step.get("params", {})

            # 检查联锁
            allowed, reason = await self.engine.check_interlock(tool_name, params)
            if not allowed:
                logger.warning(f"[{exp_id}] Step {i} 被 Guardrail 拦截: {reason}")
                self.metrics.record_guardrail_trigger(tool_name, reason, params)
                self.metrics.experiments_failed += 1
                return

            # 执行工具
            tool = self.tools.get(tool_name)
            if tool:
                result = await tool.invoke(params, self.metrics)
                logger.info(
                    f"[{exp_id}] Step {i} [{tool_name}] → {result['status']}"
                )

                if result["status"] == "success":
                    self.metrics.record_subtask_complete(f"{exp_id}-step{i}")
                else:
                    self.metrics.experiments_failed += 1
                    return

        # 实验完成
        self.metrics.experiments_completed += 1
        self.completed_count += 1
        logger.info(f"[{exp_id}] 实验完成")

    async def wait_completion(self, timeout: float) -> bool:
        """等待所有实验完成。"""
        deadline = time.perf_counter() + timeout

        while self.running_tasks:
            # 检查超时
            if time.perf_counter() > deadline:
                logger.error("⏰ 死锁检测：实验执行超时")
                self.metrics.deadlock_detected = True
                self.metrics.deadlock_reason = (
                    f"Timeout after {timeout}s with {len(self.running_tasks)} tasks pending"
                )
                return False

            # 等待任务完成
            done, pending = await asyncio.wait(
                self.running_tasks,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # 移除已完成的任务
            self.running_tasks = list(pending)

        return True

    async def shutdown(self) -> None:
        """关闭调度器。"""
        self.shutdown_flag = True
        for task in self.running_tasks:
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
    {"tool": "turn_on_laser", "params": {"activate": True}},  # 长耗时
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
#  Chaos Injection: Temperature Alarm Trigger                                   #
# ============================================================================ #

async def inject_temperature_alarm(
    bus: MockTelemetryBus,
    metrics: BenchmarkMetrics,
    trigger_time: float,
    target_temp: float,
) -> None:
    """在指定时刻注入温度报警（模拟硬件中断）。"""
    logger.info(f"💥 [Chaos] 等待 {trigger_time}s 后注入温度报警: {target_temp}°C")

    metrics.record_preemption_trigger()

    # 等待触发时间
    await asyncio.sleep(trigger_time)

    # 注入高温
    await bus.write("temperature", target_temp)
    logger.warning(f"🚨 [Chaos] 温度报警注入! temperature = {target_temp}°C")

    metrics.record_preemption_action_start()


# ============================================================================ #
#  Main Benchmark Runner                                                        #
# ============================================================================ #

async def run_benchmark(config: BenchmarkConfig) -> BenchmarkMetrics:
    """运行压测。"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "AstroSASF V7.1 Benchmark" + " " * 28 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  Workers: {config.num_workers}  |  Chaos: {'Enabled' if config.chaos_enabled else 'Disabled'}  |  "
          f"Delay×{config.chaos_delay_multiplier}  |  Preemption: {config.preemption_trigger_time}s   ║")
    print("╚" + "═" * 78 + "╝\n")

    # 初始化指标收集器
    metrics = BenchmarkMetrics()
    metrics.config = config

    # 创建调度器
    scheduler = MockDAGScheduler(config, metrics)

    # 注册温度报警回调
    async def temperature_alarm_callback(key: str, value: float) -> None:
        """温度报警回调。"""
        logger.warning(f"🚨 [Alarm] 温度报警触发: {key} = {value}°C")

    scheduler.bus.register_alarm_callback(temperature_alarm_callback)

    # 提交实验
    experiments = [
        ("fluid-exp-1", FLUID_EXPERIMENT),
        ("material-exp-1", MATERIAL_EXPERIMENT),
        ("bio-exp-1", BIO_EXPERIMENT),
        ("fluid-exp-2", FLUID_EXPERIMENT),
        ("material-exp-2", MATERIAL_EXPERIMENT),
    ]

    # 提交所有实验
    for exp_id, steps in experiments:
        await scheduler.submit_experiment(exp_id, steps)

    # 启动 Chaos 注入（温度报警）
    chaos_task = asyncio.create_task(
        inject_temperature_alarm(
            scheduler.bus,
            metrics,
            config.preemption_trigger_time,
            config.preemption_temperature,
        ),
        name="chaos-temperature-alarm",
    )

    # 等待所有实验完成
    success = await scheduler.wait_completion(config.deadlock_timeout)

    # 取消 Chaos 任务
    if not chaos_task.done():
        chaos_task.cancel()
        try:
            await chaos_task
        except asyncio.CancelledError:
            pass

    # 关闭调度器
    await scheduler.shutdown()

    # 完成压测
    metrics.finalize()

    return metrics


# ============================================================================ #
#  Entry Point                                                                  #
# ============================================================================ #

async def main() -> None:
    """主入口。"""
    # 配置
    config = BenchmarkConfig(
        num_workers=2,
        chaos_enabled=True,
        chaos_delay_multiplier=3.0,
        preemption_trigger_time=5.0,
        preemption_temperature=85.0,
        deadlock_timeout=60.0,
    )

    # 运行压测
    metrics = await run_benchmark(config)

    # 生成报告
    reporter = BenchmarkReporter(metrics)
    reporter.print_console_report()

    # 保存 JSON 报告
    output_path = config.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    reporter.save_json_report(output_path)

    print("\n✅ Benchmark completed successfully!")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main())
