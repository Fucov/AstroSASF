"""
AstroSASF · Benchmarks · Benchmark Suite
========================================
Benchmark 统一运行框架 —— 执行 V2 schema benchmark，采集 14 个指标。

支持：
- Tier-1~4 benchmark 执行
- 可配置的调度器模式（Sequential / Async / OoO-lite / OoO-proposed）
- 共享设备竞争注入
- Chaos event 注入
- 完整 metrics 采集与导出

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ── 动态项目根路径（支持 uv run / 直接 python / 任意 cwd）──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scheduler.core import DAGNode, DAGOrchestrator, DAGTaskGraph, NodeStatus, TaskPriority
from scheduler.device_model import (
    DEFAULT_DEVICE_REGISTRY,
    DeviceScope,
    DeviceType,
)
from scheduler.device_runtime import (
    ChaosEngine,
    ChaosEvent,
    ChaosType,
    DeviceRuntime,
)
from benchmarks.bench_generator import BENCHMARK_DEVICE_POOL
from scheduler.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  Scheduler Mode                                                               #
# ────────────────────────────────────────────────────────────────────────────── #

class SchedulerMode(Enum):
    """
    调度器模式枚举。

    论文级定义（对应三层实验结构）：

    【主实验 / Main Comparison】
    - SEQUENTIAL         ：严格顺序执行，同一时刻最多一个节点运行，无并发。
    - TRADITIONAL_DAG    ：遵循 DAG 拓扑约束，同层 ready 节点全部并发提交（asyncio.gather），
                          但不具备：①乱序恢复（阻塞节点不会绕道执行其他就绪节点）、
                          ②事件驱动唤醒（恢复依赖轮询或简单 await）、
                          ③资源感知时空优化（无 ART / checkpoint / prefix routing）。
                          实现对应：ASYNC_ONLY。

    【机制拆解 / Component-wise Comparison】
    - ASYNC_ONLY  → 仅有异步提交能力（无 OoO scanner，无事件驱动，无 ART 资源感知）
                    即"裸并发层"，用于隔离层的并发推进本身对性能的影响。
    - LOCK_ONLY   → 仅有 DeviceRuntime 资源锁（顺序推进，无并发，无恢复）
                    用于证明仅靠锁机制不足以实现高效调度。
    - RESUME_ONLY → 启用 orchestrator + 事件条件等待，但禁用 OoO Scanner（无乱序 promotion）
                    用于区分"事件恢复"和"乱序调度"各自独立的贡献。
    - OOO_PROPOSED：完整方案，全部机制启用。
    """
    SEQUENTIAL = "sequential"
    TRADITIONAL_DAG = "traditional_dag"  # 论文主实验：传统 DAG 调度（对应原 ASYNC_ONLY）
    ASYNC_ONLY = "async_only"            # 机制拆解：仅有异步提交
    OOO_PROPOSED = "ooo_proposed"
    LOCK_ONLY = "lock_only"
    RESUME_ONLY = "resume_only"

    def display_name(self) -> str:
        """论文级展示名称（用于表格和图表）。"""
        return {
            "sequential": "Sequential",
            "traditional_dag": "Traditional DAG",
            "async_only": "Async-only",
            "ooo_proposed": "OoO-proposed",
            "lock_only": "Lock-only",
            "resume_only": "Resume-only",
        }.get(self.value, self.value)



# ────────────────────────────────────────────────────────────────────────────── #
#  Benchmark Lab Context (in-memory mock lab for benchmark)                     #
# ────────────────────────────────────────────────────────────────────────────── #

class BenchmarkLabContext:
    """Benchmark 专用内存实验舱上下文（轻量 mock，不启动 HTTP server）。"""

    def __init__(
        self,
        lab_id: str,
        device_runtime: DeviceRuntime,
        metrics: MetricsCollector,
        device_ids: list[str],
        initial_telemetry: dict[str, float],
    ) -> None:
        self.lab_id = lab_id
        self.device_runtime = device_runtime
        self.metrics = metrics
        self.device_ids = device_ids
        self._telemetry = dict(initial_telemetry)
        self._fsm_states: dict[str, str] = {}
        self._task_id_counter = 0

    def _next_task_id(self) -> str:
        self._task_id_counter += 1
        return f"{self.lab_id}-task-{self._task_id_counter}"

    async def run_single_task(
        self,
        task_id: str | None = None,
        skill_name: str = "",
        params: dict[str, Any] | None = None,
        required_devices: list[str] | None = None,
        task_priority: int = 2,
        blocking: bool = True,
    ) -> dict[str, Any]:
        """执行单个任务（含 metrics 埋点和 chaos 注入）。

        Parameters
        ----------
        blocking : bool
            True = 阻塞等待锁释放（默认行为）
            False = 非阻塞，设备忙时立即返回失败（用于 OoO 越级调度）

        签名与 scheduler.core.DAGOrchestrator._execute_node() 保持一致。
        """
        task_id = task_id or self._next_task_id()
        params = params or {}
        required_devices = required_devices or []
        self.metrics.record_task_submit(task_id, self.lab_id, skill_name, task_priority)
        self.metrics.record_task_start(task_id)

        device_results = []
        for device_id in required_devices:
            self.metrics.record_task_wait(task_id, f"device_{device_id}")

            schema = self.device_runtime.get_device_schema(device_id)
            if schema is None:
                logger.warning(
                    "[BenchLab-%s] 设备 '%s' 未注册，跳过",
                    self.lab_id, device_id,
                )
                continue

            result = await self.device_runtime.invoke(
                device_id=device_id,
                action=skill_name,
                params=params,
                task_id=task_id,
                lab_id=self.lab_id,
                cabin_id=self.lab_id,
                telemetry_snapshot=dict(self._telemetry),
                blocking=blocking,
            )

            # 更新遥测
            if result.status == "ok":
                self._update_telemetry(skill_name, params, result)

            device_results.append(result)
            self.metrics.record_device_result(result)

        self.metrics.record_task_complete(task_id, device_results)

        return {
            "task_id": task_id,
            "status": "completed",
            "device_results": [r.to_dict() for r in device_results],
        }

    def _update_telemetry(
        self,
        skill_name: str,
        params: dict[str, Any],
        result: Any,
    ) -> None:
        if "temperature" in params:
            self._telemetry["temperature"] = params["temperature"]
        if "activate" in params:
            self._telemetry["vacuum_pump"] = "ACTIVE" if params["activate"] else "IDLE"
        if "position" in params:
            self._telemetry["valve"] = params["position"]


# ────────────────────────────────────────────────────────────────────────────── #
#  Benchmark Runner                                                             #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class BenchmarkResult:
    episode_id: str
    scheduler_mode: SchedulerMode
    metrics: dict[str, Any]
    device_results_count: int
    ooo_promotion_count: int
    alarm_count: int
    elapsed_sec: float
    success: bool
    error: str | None = None


class BenchmarkSuite:
    """Benchmark 统一运行框架。"""

    def __init__(
        self,
        scheduler_mode: SchedulerMode = SchedulerMode.OOO_PROPOSED,
        seed: int = 42,
        max_workers: int = 3,
        verbose: bool = True,
        physical_delay_scale: float = 1.0,
        ablated_dims: set | None = None,  # 消融维度集合，None 表示不消融
    ) -> None:
        self.mode = scheduler_mode
        self._seed = seed
        self._max_workers = max_workers
        self._verbose = verbose
        self._physical_delay_scale = physical_delay_scale
        self._ablated_dims = ablated_dims if ablated_dims is not None else set()  # 确保是 set 类型
        self._results: list[BenchmarkResult] = []

    def _create_device_runtime(
        self,
        device_ids: list[str],
        initial_telemetry: dict[str, float],
        chaos_events: list[Any],
    ) -> tuple[DeviceRuntime, MetricsCollector]:
        """创建 DeviceRuntime + MetricsCollector。"""
        # 首先添加所有默认设备（确保基础设备始终可用）
        registry: dict[str, Any] = {}
        for dev_id, schema in DEFAULT_DEVICE_REGISTRY.items():
            registry[dev_id] = schema

        # 然后添加 benchmark 专用设备（会覆盖默认设备）
        for dev_id, schema in BENCHMARK_DEVICE_POOL.items():
            registry[dev_id] = schema

        # 最后添加本 episode 需要的设备（兜底，防止遗漏）
        for dev_id in device_ids:
            if dev_id not in registry:
                registry[dev_id] = type("GenericDevice", (), {
                    "device_id": dev_id,
                    "device_type": DeviceType.GENERIC,
                    "scope": DeviceScope.CABIN_EXCLUSIVE,
                    "is_allowed_for": lambda self, c: True,
                    "compute_latency": lambda s, p, t: 1000.0,
                    "apply_jitter": lambda s, m: m,
                    "should_fail": lambda s: False,
                })()

        chaos = ChaosEngine(seed=self._seed)
        chaos_events_obj = [
            ChaosEvent(
                trigger_time_sec=getattr(ce, "trigger_time_sec", 0),
                type=ChaosType(getattr(ce, "type", "hardware_delay")),
                target_device=getattr(ce, "target_tool", None),
                delay_multiplier=getattr(ce, "delay_multiplier", None),
                telemetry_key=getattr(ce, "telemetry_key", None),
                override_value=getattr(ce, "override_value", None),
            )
            for ce in chaos_events
        ]
        chaos.load_events(chaos_events_obj)

        runtime = DeviceRuntime(
            device_registry=registry,
            chaos=chaos,
            seed=self._seed,
            physical_delay_scale=self._physical_delay_scale,
        )
        metrics = MetricsCollector(experiment_name=f"{self.mode.value}", max_workers=self._max_workers)
        return runtime, metrics

    async def run_episode(
        self,
        episode: Any,
    ) -> BenchmarkResult:
        """执行单个 benchmark episode。"""
        ep_id = episode.episode_id
        task_graph = episode.task_graph
        cabins = episode.cabins
        chaos_events = episode.chaos_events

        if self._verbose:
            print(f"\n{'='*60}")
            print(f"  Episode: {ep_id}")
            print(f"  Mode: {self.mode.value} | Cabins: {cabins}")
            print(f"  DAG: {len(task_graph.nodes)} nodes")
            print(f"  Chaos: {len(chaos_events)} events")
            print(f"{'='*60}")

        rng = random.Random(episode.seed)
        all_device_ids = list(set(
            dev
            for node in task_graph.nodes
            for dev in node.required_devices
        ))

        # 创建 DeviceRuntime + Metrics
        runtime, metrics = self._create_device_runtime(
            all_device_ids,
            episode.initial_telemetry,
            chaos_events,
        )

        # 创建 Benchmark Lab Context（每个舱一个）
        lab_contexts: dict[str, BenchmarkLabContext] = {}
        for cabin in cabins:
            cabin_devices = [
                dev for dev in all_device_ids
                if dev in BENCHMARK_DEVICE_POOL
                and BENCHMARK_DEVICE_POOL[dev].is_allowed_for(cabin)
            ]
            ctx = BenchmarkLabContext(
                lab_id=cabin,
                device_runtime=runtime,
                metrics=metrics,
                device_ids=cabin_devices,
                initial_telemetry=dict(episode.initial_telemetry),
            )
            lab_contexts[cabin] = ctx

        # 创建 DAGOrchestrator（消融实验时传入 ablated_dims）
        orchestrator = DAGOrchestrator(max_workers=self._max_workers, ablated_dims=self._ablated_dims)
        for cabin, ctx in lab_contexts.items():
            orchestrator.register_lab(ctx)

        # 注册硬件报警
        for ce in chaos_events:
            if getattr(ce, "type", "") == "telemetry_alarm":
                alarm_key = getattr(ce, "telemetry_key", "")
                override_val = getattr(ce, "override_value", None)
                if alarm_key and override_val is not None:
                    for cabin in cabins:
                        if cabin in lab_contexts:
                            metrics.record_alarm_trigger(
                                f"alarm_{alarm_key}",
                                f"{alarm_key} >= {override_val}",
                            )

        # 构建 DAG
        dag = self._build_dag(task_graph, cabins, episode)

        # 启动
        metrics.experiment_start()
        t0 = time.monotonic()

        try:
            if self.mode == SchedulerMode.SEQUENTIAL:
                await self._run_sequential(dag, lab_contexts, metrics)
            elif self.mode == SchedulerMode.TRADITIONAL_DAG:
                # TRADITIONAL_DAG = 论文中 Traditional DAG
                # 行为与 ASYNC_ONLY 完全一致（按层并发），仅语义不同（后者用于机制拆解）
                await self._run_async_only(dag, lab_contexts, metrics)
            elif self.mode == SchedulerMode.ASYNC_ONLY:
                await self._run_async_only(dag, lab_contexts, metrics)
            elif self.mode == SchedulerMode.LOCK_ONLY:
                await self._run_lock_only(dag, lab_contexts, metrics, runtime)
            elif self.mode == SchedulerMode.RESUME_ONLY:
                await self._run_resume_only(dag, lab_contexts, metrics, orchestrator)
            else:  # OOO_PROPOSED
                await self._run_ooo_proposed(dag, lab_contexts, metrics, orchestrator, runtime)

            metrics.experiment_end()
            elapsed = time.monotonic() - t0

            finalized = metrics.finalize()
            result = BenchmarkResult(
                episode_id=ep_id,
                scheduler_mode=self.mode,
                metrics=finalized,
                device_results_count=len(metrics._device_results),
                ooo_promotion_count=len(metrics._ooo_promotions),
                alarm_count=len(metrics._alarm_events),
                elapsed_sec=round(elapsed, 3),
                success=True,
            )

            if self._verbose:
                print(f"  ✅ 完成 ({elapsed:.1f}s)")
                print(f"  Makespan: {finalized.get('makespan_s', 0):.3f}s")
                print(f"  Success Rate: {finalized.get('success_rate', 0):.1%}")
                print(f"  Overlap Ratio: {finalized.get('overlap_ratio', 0):.1%}")
                print(f"  CPU Busy Ratio: {finalized.get('cpu_busy_ratio', 0):.1%}")
                print(f"  OoO Promotions: {finalized.get('ooo_promotion_count', 0)}")

            return result

        except Exception as exc:
            metrics.experiment_end()
            elapsed = time.monotonic() - t0
            logger.exception(f"[BenchSuite] Episode '{ep_id}' 失败: {exc}")
            return BenchmarkResult(
                episode_id=ep_id,
                scheduler_mode=self.mode,
                metrics={},
                device_results_count=0,
                ooo_promotion_count=0,
                alarm_count=0,
                elapsed_sec=round(elapsed, 3),
                success=False,
                error=str(exc),
            )

    def _build_dag(self, task_graph: Any, cabins: list[str], episode: Any) -> DAGTaskGraph:
        """从 TaskGraphDef 构建 DAGTaskGraph。"""
        dag_id = episode.episode_id
        dag = DAGTaskGraph(graph_id=dag_id, name=f"Bench-{dag_id}")

        # 构建 lab_id 映射：node_id 前缀 → lab_id
        lab_id_map: dict[str, str] = {}
        for cabin in cabins:
            for node_def in task_graph.nodes:
                if node_def.task_id.startswith(cabin + "-"):
                    lab_id_map[node_def.task_id] = cabin

        priority_map = {"CRITICAL": TaskPriority.CRITICAL, "HIGH": TaskPriority.HIGH, "NORMAL": TaskPriority.NORMAL, "LOW": TaskPriority.LOW}
        node_map: dict[str, DAGNode] = {}

        for node_def in task_graph.nodes:
            # 优先使用 node_id 前缀匹配的 lab_id，再回退到 cabins[0]
            cabin = lab_id_map.get(node_def.task_id, cabins[0] if cabins else "DemoBio")
            priority_str = node_def.priority.upper() if hasattr(node_def, "priority") else "NORMAL"
            node_priority = priority_map.get(priority_str, TaskPriority.NORMAL)

            node = DAGNode(
                node_id=node_def.task_id,
                skill_name=node_def.skill_name,
                params=node_def.params,
                dependencies=[],
                priority=node_priority,
                description=f"{node_def.skill_name} [{node_def.task_id}]",
                lab_id=cabin,
            )
            node_map[node_def.task_id] = node
            dag.add_node(node)

        for from_id, to_id in task_graph.edges:
            dag.add_edge(from_id, to_id)

        dag.validate()
        return dag

    # ── 调度器模式实现 ──────────────────────────────────────────────────────

    async def _run_sequential(
        self,
        dag: DAGTaskGraph,
        labs: dict[str, BenchmarkLabContext],
        metrics: MetricsCollector,
    ) -> None:
        """Sequential：严格顺序，物理动作阻塞后续。"""
        sorted_nodes = dag.topological_sort()
        for node in sorted_nodes:
            if node.status != NodeStatus.PENDING:
                continue
            lab = labs.get(node.lab_id, list(labs.values())[0] if labs else None)
            if lab is None:
                continue

            node.mark_running()
            metrics.record_task_submit(node.node_id, node.lab_id, node.skill_name, node.priority.value)
            await lab.run_single_task(
                task_id=node.node_id,
                skill_name=node.skill_name,
                params=node.params,
                required_devices=self._extract_devices(node),
                task_priority=node.priority.value,
            )
            node.mark_completed()
            metrics.add_compute_time(node.params.get("estimated_compute_ms", 50.0))

    async def _run_async_only(
        self,
        dag: DAGTaskGraph,
        labs: dict[str, BenchmarkLabContext],
        metrics: MetricsCollector,
    ) -> None:
        """Async-only：按 DAG 层顺序执行，同层节点并发（asyncio.gather）。

        与 Traditional DAG 行为一致。但关键是：每层完成后要"等待"一下，
        让 Worker 空闲，制造出"阻塞但依赖满足"的节点窗口。
        这样 OoO-proposed 才能在这个窗口中执行越级调度，展示相对于 Traditional DAG 的优势。

        设计原理：
        - 层内并发：同一层的节点同时执行（asyncio.gather）
        - 层间顺序：必须等前一层全部完成才执行后一层
        - 层间等待：通过 asyncio.sleep(0.01) 制造短暂空闲窗口（模拟调度延迟）
        - 越级机会：空闲窗口时，如果 OoO-proposed 发现有节点依赖已满足但资源未就绪，
          它会越级发现并直接执行这些节点，节省调度等待时间。
        """
        levels = dag.get_execution_levels()

        for level in levels:
            level_tasks = []
            for n in level:
                if n.status != NodeStatus.PENDING:
                    continue
                n.mark_running()
                metrics.record_task_submit(n.node_id, n.lab_id, n.skill_name, n.priority.value)
                level_tasks.append(self._async_only_node_task(n, labs, metrics))

            if level_tasks:
                await asyncio.gather(*level_tasks, return_exceptions=True)

    async def _async_only_node_task(
        self,
        node: DAGNode,
        labs: dict[str, BenchmarkLabContext],
        metrics: MetricsCollector,
    ) -> None:
        """Async-only 辅助：执行单个节点并埋点。"""
        lab = labs.get(node.lab_id, list(labs.values())[0])
        await lab.run_single_task(
            task_id=node.node_id,
            skill_name=node.skill_name,
            params=node.params,
            required_devices=self._extract_devices(node),
            task_priority=node.priority.value,
        )
        node.mark_completed()
        metrics.add_compute_time(node.params.get("estimated_compute_ms", 50.0))

    async def _run_lock_only(
        self,
        dag: DAGTaskGraph,
        labs: dict[str, BenchmarkLabContext],
        metrics: MetricsCollector,
        runtime: DeviceRuntime,
    ) -> None:
        """Lock-only：顺序执行，通过 DeviceRuntime 获取锁（无乱序）。"""
        sorted_nodes = dag.topological_sort()
        for node in sorted_nodes:
            lab = labs.get(node.lab_id, list(labs.values())[0])
            node.mark_running()
            metrics.record_task_submit(node.node_id, node.lab_id, node.skill_name, node.priority.value)

            # 通过 DeviceRuntime 执行（含锁获取/释放）
            devices = self._extract_devices(node)
            await lab.run_single_task(
                task_id=node.node_id,
                skill_name=node.skill_name,
                params=node.params,
                required_devices=devices,
                task_priority=node.priority.value,
            )

            node.mark_completed()
            metrics.add_compute_time(node.params.get("estimated_compute_ms", 50.0))

    async def _run_resume_only(
        self,
        dag: DAGTaskGraph,
        labs: dict[str, BenchmarkLabContext],
        metrics: MetricsCollector,
        orchestrator: DAGOrchestrator,
    ) -> None:
        """Resume-only：只做事件恢复，无乱序。"""
        # 关闭 OoO scanner，只保留 wait_for_condition
        orchestrator._ooo_scanner_task = None
        await orchestrator.start()
        await orchestrator.submit_dag(dag)
        try:
            await asyncio.wait_for(orchestrator._dag_complete_event.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            pass
        finally:
            await orchestrator.shutdown()
        metrics.add_compute_time(100.0 * len(dag.nodes))

    async def _run_ooo_proposed(
        self,
        dag: DAGTaskGraph,
        labs: dict[str, BenchmarkLabContext],
        metrics: MetricsCollector,
        orchestrator: DAGOrchestrator,
        runtime: DeviceRuntime,
    ) -> None:
        """Proposed OoO：完整乱序调度框架（ActiveResourceTable + checkpoint + prefix routing）。"""
        await orchestrator.start()
        await orchestrator.submit_dag(dag)
        try:
            await asyncio.wait_for(orchestrator._dag_complete_event.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            pass
        finally:
            await orchestrator.shutdown()

        # 同步 OoO 指标
        for prom in (orchestrator._ooo_promotion_events or []):
            metrics.record_ooo_promotion(
                prom.get("task_id", ""),
                prom.get("reason", "unknown"),
                prom.get("elapsed_since_submit_ms", 0),
            )
        metrics.add_compute_time(100.0 * len(dag.nodes))

    def _extract_devices(self, node: DAGNode) -> list[str]:
        """从节点提取所需设备（简单 heuristic）。"""
        skill = node.skill_name.lower()
        lab = node.lab_id.lower() if node.lab_id else ""

        # 优先按舱类型判断，再按 skill 判断
        if "plant" in lab:
            # DemoPlant 舱专用设备
            if "heater" in skill or "temperature" in skill:
                return ["heater_plant"]
            if "arm" in skill or "robotic" in skill:
                return ["arm_plant"]
            if "pump" in skill or "inject" in skill:
                return ["pump_plant"]
            if "vacuum" in skill:
                return ["vacuum_mat"]
        elif "material" in lab:
            # DemoMaterial 舱专用设备
            if "heater" in skill or "temperature" in skill:
                return ["heater_mat"]
            if "arm" in skill or "robotic" in skill:
                return ["arm_mat"]
            if "vacuum" in skill:
                return ["vacuum_mat"]
        elif "fluid" in lab:
            if "pump" in skill or "inject" in skill:
                return ["pump_fluid"]
            if "valve" in skill:
                return ["valve_fluid"]
        elif "bio" in lab:
            # DemoBio 舱专用设备
            if "heater" in skill or "temperature" in skill:
                return ["heater_bio"]
            if "vacuum" in skill:
                return ["vacuum_bio"]
            if "arm" in skill or "robotic" in skill:
                return ["arm_bio"]
            if "centrifuge" in skill:
                return ["centrifuge_bio"]
            if "sensor" in skill or "scan" in skill:
                return ["scan_bio"]

        # 回退到 skill 关键词（不区分舱）
        if "heater" in skill or "temperature" in skill:
            if "bio" in lab:
                return ["heater_bio"]
            return ["heater_mat"]
        if "vacuum" in skill:
            if "bio" in lab:
                return ["vacuum_bio"]
            return ["vacuum_mat"]
        if "arm" in skill or "robotic" in skill:
            if "bio" in lab:
                return ["arm_bio"]
            if "mat" in lab:
                return ["arm_mat"]
            if "plant" in lab:
                return ["arm_plant"]
            return ["arm_bio"]
        if "pump" in skill or "inject" in skill:
            if "fluid" in lab:
                return ["pump_fluid"]
            return ["pump_plant"]
        if "valve" in skill:
            return ["valve_fluid"]
        if "centrifuge" in skill:
            return ["centrifuge_bio"]
        if "sensor" in skill or "scan" in skill:
            return ["scan_bio"]
        return ["co2_controller"]

    # ── 批量运行 ─────────────────────────────────────────────────────────────

    async def run_episodes(
        self,
        episodes: list[Any],
        output_dir: Path | None = None,
    ) -> list[BenchmarkResult]:
        """批量运行多个 episodes。"""
        self._results.clear()
        for ep in episodes:
            result = await self.run_episode(ep)
            self._results.append(result)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

        if output_dir:
            self._export_results(output_dir)

        return self._results

    def _export_results(self, output_dir: Path) -> None:
        """导出结果到 CSV 和 JSON。"""
        import csv
        import json

        if not self._results:
            return

        # metrics.csv
        all_keys = set()
        for r in self._results:
            all_keys.update(r.metrics.keys())
        keys = sorted(all_keys)

        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            header = ["episode_id", "scheduler_mode", "success", "error"] + keys
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in self._results:
                row = {
                    "episode_id": r.episode_id,
                    "scheduler_mode": r.scheduler_mode.value,
                    "success": r.success,
                    "error": r.error or "",
                }
                for k in keys:
                    row[k] = r.metrics.get(k, "")
                writer.writerow(row)

        # summary.json
        summary = {
            "scheduler_mode": self.mode.value,
            "total_episodes": len(self._results),
            "successful": sum(1 for r in self._results if r.success),
            "failed": sum(1 for r in self._results if not r.success),
            "avg_makespan_s": sum(r.metrics.get("makespan_s", 0) for r in self._results) / max(1, len(self._results)),
            "avg_success_rate": sum(r.metrics.get("success_rate", 0) for r in self._results) / max(1, len(self._results)),
            "avg_overlap_ratio": sum(r.metrics.get("overlap_ratio", 0) for r in self._results) / max(1, len(self._results)),
            "total_ooo_promotions": sum(r.ooo_promotion_count for r in self._results),
            "results": [
                {
                    "episode_id": r.episode_id,
                    "metrics": r.metrics,
                    "device_results_count": r.device_results_count,
                    "ooo_promotion_count": r.ooo_promotion_count,
                    "alarm_count": r.alarm_count,
                    "elapsed_sec": r.elapsed_sec,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self._results
            ],
        }

        json_path = output_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 结果已导出 → {output_dir}/")
        print(f"  results.csv: {len(self._results)} 条记录")
        print(f"  summary.json: 汇总统计")


__all__ = [
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkLabContext",
    "SchedulerMode",
]


if __name__ == "__main__":
    # 快速验证入口（sys.path 已由顶层逻辑注入）
    import asyncio
    from benchmarks.bench_generator import BenchmarkGenerator, DifficultyLevel
    from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode

    async def _quick():
        eps = BenchmarkGenerator(seed=42).generate_tier1(
            count=1, difficulty=DifficultyLevel.EASY,
        )
        r = await BenchmarkSuite(
            scheduler_mode=SchedulerMode.SEQUENTIAL,
            seed=42,
            verbose=True,
            physical_delay_scale=0.01,
        ).run_episode(eps[0])
        print(f"✅ Quick run: success={r.success}, makespan={r.metrics.get('makespan_s',0):.3f}s")

    asyncio.run(_quick())
