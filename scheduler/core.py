"""
AstroSASF · Scheduler · DAG Orchestrator (Kernel)
================================================
理论/实践双轨 DAG 调度器。

核心机制：
- 理论智能体（Planner）：调用 LLM 将自然语言解析为 DAG 任务图
- 实践智能体（Worker）：从 ReadyQueue 获取节点，执行 MCP Tool
- DAG 依赖状态机：PENDING → READY → RUNNING → COMPLETED/FAILED
- 双队列机制：ReadyQueue（就绪）| BlockedQueue（阻塞）
- 结算时触发依赖解除，递归检查下游节点
- V7.2 新增：硬件级抢占 + 动态优先级 Aging 机制

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from scheduler.models import (
    DEFAULT_AGING_FACTOR,
    DEFAULT_REBALANCE_INTERVAL,
    DAGExecutionResult,
    DAGNode,
    DAGTaskGraph,
    HardwareInterruptTask,
    NodeStatus,
    TaskPriority,
)

logger = logging.getLogger(__name__)


@dataclass
class DAGOrchestrator:
    """理论/实践双轨 DAG 调度器 (V7.2)。

    V7.2 新增功能：
    - 硬件级抢占：接收 TelemetryBus 报警，直接注入逃生任务
    - 动态优先级 Aging：防止低优任务饿死
    - LLM Task Cancel：硬件中断时强行中止 LLM 推理

    调度流程：
    1. **提交阶段**：理论智能体生成 DAG 后，无依赖节点入 ReadyQueue，
       其余入 BlockedQueue。
    2. **执行阶段**：实践智能体（Worker）从 ReadyQueue 获取节点执行。
    3. **结算阶段**：节点完成后，触发依赖解除，检查 BlockedQueue，
       将依赖已满足的节点移入 ReadyQueue。
    4. **硬件抢占**：TelemetryBus 报警触发时，立即注入 CRITICAL 任务，
       并 Cancel 正在执行的 LLM 推理。
    """

    max_workers: int = 3

    # ── 实验柜注册 ── #
    _labs: dict[str, Any] = field(default_factory=dict, init=False)
    _tool_registrar: Any = field(default=None, init=False)
    _macro_registrar: Any = field(default=None, init=False)

    # ── DAG 调度核心 ── #
    _ready_queue: asyncio.PriorityQueue = field(
        default_factory=asyncio.PriorityQueue, init=False,
    )
    _blocked_queue: list[DAGNode] = field(default_factory=list, init=False)
    _running_nodes: dict[str, DAGNode] = field(default_factory=dict, init=False)
    _completed_nodes: list[DAGNode] = field(default_factory=list, init=False)
    _failed_nodes: list[DAGNode] = field(default_factory=list, init=False)

    # ── DAG 图管理 ── #
    _active_graphs: dict[str, DAGTaskGraph] = field(default_factory=dict, init=False)
    _completed_graphs: list[DAGTaskGraph] = field(default_factory=list, init=False)

    # ── Worker 管理 ── #
    _workers: list[asyncio.Task] = field(default_factory=list, init=False)
    _shutdown_flag: bool = field(default=False, init=False)

    # ── 并发控制 ── #
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _node_counter: int = field(default=0, init=False)

    # ── V7.2: 硬件级抢占 ── #
    _llm_tasks: dict[str, asyncio.Task] = field(default_factory=dict, init=False)
    _interrupt_queue: asyncio.Queue[HardwareInterruptTask] = field(
        default_factory=asyncio.Queue, init=False,
    )
    _hardware_interrupt_event: asyncio.Event = field(
        default_factory=asyncio.Event, init=False,
    )
    _aging_rebalance_task: asyncio.Task | None = field(default=None, init=False)
    _aging_factor: float = field(default=DEFAULT_AGING_FACTOR, init=False)
    _rebalance_interval: float = field(default=DEFAULT_REBALANCE_INTERVAL, init=False)
    _dag_complete_event: asyncio.Event = field(default=None, init=False)

    # --------------------------------------------------------------------------- #
    #  实验柜注册                                                                #
    # --------------------------------------------------------------------------- #

    def register_lab(self, env: Any) -> None:
        """注册实验柜环境（供调度器分发任务）。"""
        self._labs[env.lab_id] = env
        logger.info("[DAG调度器] 注册实验柜: %s", env.lab_id)

    def register_lab_hardware_alarm(
        self,
        lab_id: str,
        alarm_id: str,
        condition_expr: str,
        interrupt_action_skill: str,
        interrupt_action_params: dict[str, Any] | None = None,
        severity: TaskPriority = TaskPriority.CRITICAL,
    ) -> None:
        """为指定实验柜注册硬件报警。"""
        env = self._labs.get(lab_id)
        if env is None:
            raise ValueError(f"实验柜 '{lab_id}' 未注册")

        bus = getattr(env, "_bus", None)
        if bus is None:
            raise ValueError(f"实验柜 '{lab_id}' 未绑定 TelemetryBus")

        def _interrupt_callback(interrupt: HardwareInterruptTask) -> None:
            asyncio.create_task(
                self._handle_hardware_interrupt(interrupt),
                name=f"hw-int-{interrupt.interrupt_id}",
            )

        bus.register_alarm(
            alarm_id=alarm_id,
            condition_expr=condition_expr,
            interrupt_action_skill=interrupt_action_skill,
            interrupt_action_params=interrupt_action_params,
            severity=severity,
        )
        bus.register_alarm_callback(_interrupt_callback)
        bus.start_alarm_monitor()

        logger.info(
            "[DAG调度器] 注册硬件报警: lab=%s, alarm=%s, condition='%s' → %s",
            lab_id, alarm_id, condition_expr, interrupt_action_skill,
        )

    # --------------------------------------------------------------------------- #
    #  V7.2: 硬件级抢占 (Hardware Preemption)                                       #
    # --------------------------------------------------------------------------- #

    async def _handle_hardware_interrupt(
        self,
        interrupt: HardwareInterruptTask,
    ) -> None:
        """处理硬件中断 —— 注入逃生任务并 Cancel LLM 推理。"""
        logger.warning(
            "🚨 [硬件中断] 触发! ID=%s, 描述=%s, 逃生动作=%s",
            interrupt.interrupt_id, interrupt.description, interrupt.action_skill,
        )

        await self._cancel_all_llm_tasks(reason=interrupt.description)

        suspended_nodes: list[DAGNode] = []
        async with self._lock:
            for node_id, node in self._running_nodes.items():
                node.mark_skipped()
                suspended_nodes.append(node)
                logger.info(
                    "⏸️  [硬件中断] 挂起任务: %s (原状态: %s)",
                    node_id, node.status.name,
                )

        escape_node = interrupt.to_dag_node(priority=TaskPriority.CRITICAL)
        escape_node.lab_id = interrupt.lab_id

        logger.info(
            "⚡ [硬件中断] 注入逃生任务: %s → %s (CRITICAL)",
            escape_node.node_id, escape_node.skill_name,
        )

        await self._enqueue_ready_node(escape_node)
        self._hardware_interrupt_event.set()

    async def _cancel_all_llm_tasks(self, reason: str) -> None:
        """Cancel 所有正在进行的 LLM 推理任务。"""
        if not self._llm_tasks:
            logger.debug("[硬件中断] 当前无正在进行的 LLM 任务")
            return

        logger.warning(
            "🛑 [硬件中断] 正在 Cancel %d 个 LLM 推理任务... 原因: %s",
            len(self._llm_tasks), reason,
        )

        cancelled = []
        for task_id, task in list(self._llm_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled.append(task_id)
                logger.info("🛑 [硬件中断] 已发送 Cancel 信号: %s", task_id)

        if cancelled:
            done, pending = await asyncio.wait(
                self._llm_tasks.values(),
                timeout=2.0,
            )
            for task_id in cancelled:
                task = self._llm_tasks.pop(task_id, None)
                if task is not None and task.cancelled():
                    logger.info("✅ [硬件中断] LLM 任务已取消: %s", task_id)
                elif task is not None and not task.done():
                    logger.warning("⚠️  LLM 任务未能及时取消: %s", task_id)

    def register_llm_task(self, task_id: str, task: asyncio.Task) -> None:
        """注册一个正在进行的 LLM 推理任务。"""
        self._llm_tasks[task_id] = task
        logger.debug("[LLM Task 注册] %s", task_id)

    def unregister_llm_task(self, task_id: str) -> None:
        """取消注册 LLM 推理任务。"""
        self._llm_tasks.pop(task_id, None)

    # --------------------------------------------------------------------------- #
    #  V7.2: 动态优先级 Aging 机制                                               #
    # --------------------------------------------------------------------------- #

    async def _aging_rebalance_loop(self) -> None:
        """队列重平衡协程 —— 周期性提升长时间等待任务的优先级。"""
        logger.info(
            "[Aging] 启动队列重平衡协程 (间隔: %.1fs, 因子: %.3f)",
            self._rebalance_interval, self._aging_factor,
        )

        while not self._shutdown_flag:
            try:
                await asyncio.sleep(self._rebalance_interval)
            except asyncio.CancelledError:
                break

            if self._shutdown_flag:
                break

            await self._rebalance_ready_queue()

        logger.info("[Aging] 队列重平衡协程已退出")

    async def _rebalance_ready_queue(self) -> None:
        """重平衡就绪队列 —— 提升长时间等待任务的优先级。"""
        if self._ready_queue.empty():
            return

        items_to_requeue: list[tuple[tuple[float, float], DAGNode | None]] = []

        while not self._ready_queue.empty():
            try:
                item = await asyncio.wait_for(
                    self._ready_queue.get(),
                    timeout=0.1,
                )
                items_to_requeue.append(item)
            except asyncio.TimeoutError:
                break

        if not items_to_requeue:
            return

        rebalanced = False
        for score, node in items_to_requeue:
            if node is None:
                await self._ready_queue.put((score, node))
                continue

            new_score = node.get_ready_score()

            if new_score != score:
                logger.debug(
                    "[Aging] 提升任务 '%s' 优先级: %.3f → %.3f",
                    node.node_id, score[0], new_score[0],
                )
                rebalanced = True

            await self._ready_queue.put((new_score, node))

        if rebalanced:
            logger.info("[Aging] 队列重平衡完成 (处理 %d 个任务)", len(items_to_requeue))

    def set_aging_params(self, factor: float, interval: float) -> None:
        """设置 Aging 参数。"""
        self._aging_factor = factor
        self._rebalance_interval = interval
        logger.info(
            "[Aging] 参数更新: factor=%.3f, interval=%.1f",
            self._aging_factor, self._rebalance_interval,
        )

    # --------------------------------------------------------------------------- #
    #  DAG 提交 (提交阶段)                                                        #
    # --------------------------------------------------------------------------- #

    async def submit_dag(self, dag_graph: DAGTaskGraph) -> str:
        """提交 DAG 图到调度器。"""
        if not dag_graph._validated:
            dag_graph.validate()

        async with self._lock:
            self._active_graphs[dag_graph.graph_id] = dag_graph

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  📥 DAG 调度器: 提交任务图 '%s'                       ║", dag_graph.name)
        logger.info("║  图 ID: %-44s ║", dag_graph.graph_id)
        logger.info("║  节点数: %-44d ║", len(dag_graph.nodes))
        logger.info("╚" + "═" * 60 + "╝")

        await self._classify_and_enqueue_nodes(dag_graph)
        return dag_graph.graph_id

    async def _classify_and_enqueue_nodes(self, dag_graph: DAGTaskGraph) -> None:
        """将 DAG 节点分类并入队。"""
        ready_count = 0
        blocked_count = 0

        for node in dag_graph.nodes.values():
            if self._can_node_run(node, dag_graph):
                await self._enqueue_ready_node(node)
                ready_count += 1
            else:
                async with self._lock:
                    self._blocked_queue.append(node)
                blocked_count += 1

        logger.info(
            "[DAG调度器] 节点分类完成: %d 就绪, %d 阻塞",
            ready_count, blocked_count,
        )

    def _can_node_run(self, node: DAGNode, dag_graph: DAGTaskGraph) -> bool:
        """检查节点是否可以运行（所有依赖都已完成）。"""
        if not node.dependencies:
            return True
        return all(
            dag_graph.nodes[dep_id].status == NodeStatus.COMPLETED
            for dep_id in node.dependencies
            if dep_id in dag_graph.nodes
        )

    async def _enqueue_ready_node(self, node: DAGNode) -> None:
        """将节点加入就绪队列。"""
        node.status = NodeStatus.READY
        priority_score = node.get_ready_score()
        await self._ready_queue.put((priority_score, node))
        logger.debug(
            "[DAG调度器] 节点 '%s' 入 ReadyQueue (优先级: %s)",
            node.node_id, node.priority.name,
        )

    # --------------------------------------------------------------------------- #
    #  DAG 结算 (结算阶段)                                                        #
    # --------------------------------------------------------------------------- #

    async def _settle_completed_node(
        self,
        completed_node: DAGNode,
        dag_graph: DAGTaskGraph,
    ) -> None:
        """结算完成的节点，解除下游依赖。"""
        logger.info(
            "[DAG调度器] 结算节点 '%s' (状态: %s)",
            completed_node.node_id, completed_node.status.name,
        )

        async with self._lock:
            still_blocked: list[DAGNode] = []

            for blocked_node in self._blocked_queue:
                if blocked_node.graph_id != dag_graph.graph_id:
                    still_blocked.append(blocked_node)
                    continue

                deps_satisfied = all(
                    dag_graph.nodes[dep_id].status == NodeStatus.COMPLETED
                    for dep_id in blocked_node.dependencies
                    if dep_id in dag_graph.nodes
                )

                if deps_satisfied:
                    await self._enqueue_ready_node(blocked_node)
                    logger.info(
                        "[DAG调度器] 依赖解除: '%s' → '%s' 已就绪",
                        completed_node.node_id, blocked_node.node_id,
                    )
                else:
                    still_blocked.append(blocked_node)

            self._blocked_queue = still_blocked

        await self._check_dag_completion(dag_graph)

    async def _check_dag_completion(self, dag_graph: DAGTaskGraph) -> None:
        """检查 DAG 是否完全执行完毕。"""
        if dag_graph.is_complete():
            async with self._lock:
                if dag_graph.graph_id in self._active_graphs:
                    del self._active_graphs[dag_graph.graph_id]
                    self._completed_graphs.append(dag_graph)

            logger.info("")
            logger.info("╔" + "═" * 60 + "╗")
            logger.info("║  🎉 DAG '%s' 执行完毕!                               ║", dag_graph.name)
            logger.info("║  完成: %d | 失败: %d | 跳过: %d                      ║",
                        dag_graph.get_completed_count(),
                        dag_graph.get_failed_count(),
                        sum(1 for n in dag_graph.nodes.values() if n.status == NodeStatus.SKIPPED))
            logger.info("╚" + "═" * 60 + "╝")

            self._dag_complete_event.set()

    # --------------------------------------------------------------------------- #
    #  Worker 生命周期 (执行阶段)                                                 #
    # --------------------------------------------------------------------------- #

    async def start(self) -> None:
        """启动 Worker 协程池和 Aging 重平衡协程。"""
        self._shutdown_flag = False
        self._dag_complete_event = asyncio.Event()

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🚀 DAG 调度内核启动 (V7.2 Dual-Track + HW Preemption)  ║")
        logger.info("║  Workers: %-3d | ReadyQueue | BlockedQueue               ║", self.max_workers)
        logger.info("║  理论智能体: DAG Planner | 实践智能体: DAG Workers    ║")
        logger.info("║  Aging 重平衡: %.1fs 间隔 | 因子: %.3f                     ║",
                    self._rebalance_interval, self._aging_factor)
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("")

        self._aging_rebalance_task = asyncio.create_task(
            self._aging_rebalance_loop(),
            name="aging-rebalance",
        )

        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"dag-worker-{i}",
            )
            self._workers.append(worker)

    async def shutdown(self, timeout: float = 30.0) -> list[DAGExecutionResult]:
        """优雅关闭调度器，返回所有 DAG 执行结果。"""
        self._shutdown_flag = True

        if self._aging_rebalance_task is not None:
            self._aging_rebalance_task.cancel()
            try:
                await asyncio.wait_for(self._aging_rebalance_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._aging_rebalance_task.cancel()
            finally:
                self._aging_rebalance_task = None

        await self._cancel_all_llm_tasks(reason="调度器关闭")

        for _ in self._workers:
            await self._ready_queue.put((float('inf'), None))

        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout)

        self._workers.clear()

        for lab_id, env in self._labs.items():
            bus = getattr(env, "_bus", None)
            if bus is not None and hasattr(bus, "stop_alarm_monitor"):
                try:
                    await bus.stop_alarm_monitor()
                except Exception:
                    pass

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🛑 DAG 调度内核已关闭                                    ║")
        logger.info("║  完成 DAG 图: %-3d                                        ║",
                    len(self._completed_graphs))
        logger.info("╚" + "═" * 60 + "╝")

        return [self._build_dag_result(g) for g in self._completed_graphs]

    async def _worker_loop(self, worker_id: int) -> None:
        """实践智能体 Worker 协程 — 从 ReadyQueue 取节点执行。"""
        logger.info("[Worker-%d] 实践智能体就绪", worker_id)

        while not self._shutdown_flag:
            try:
                if self._hardware_interrupt_event.is_set():
                    self._hardware_interrupt_event.clear()
                    logger.info("[Worker-%d] 检测到硬件中断信号", worker_id)

                priority_score, node = await asyncio.wait_for(
                    self._ready_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if node is None:
                break

            try:
                await self._execute_node(worker_id, node)
            except Exception as exc:
                logger.exception(
                    "[Worker-%d] 节点 [%s] 执行时发生未捕获异常: %s",
                    worker_id, node.node_id, exc,
                )
                try:
                    node.mark_failed(f"Worker异常: {exc}")
                    if node.graph_id and node.graph_id in self._active_graphs:
                        dag_graph = self._active_graphs[node.graph_id]
                        await self._settle_completed_node(node, dag_graph)
                except Exception:
                    pass

            self._ready_queue.task_done()

        logger.info("[Worker-%d] 实践智能体关闭", worker_id)

    async def _execute_node(self, worker_id: int, node: DAGNode) -> None:
        """执行单个 DAG 节点。"""
        dag_graph = self._active_graphs.get(node.graph_id)
        if dag_graph is None:
            logger.warning(
                "[Worker-%d] 节点 '%s' 找不到所属 DAG 图，已跳过",
                worker_id, node.node_id,
            )
            return

        node.mark_running()
        async with self._lock:
            self._running_nodes[node.node_id] = node

        logger.info(
            "⚡ [Worker-%d] 实践智能体执行: [%s] %s → %s (实验柜: %s)",
            worker_id, node.node_id, node.skill_name,
            node.priority.name, node.lab_id,
        )

        env = self._labs.get(node.lab_id)
        if env is None:
            node.mark_failed(f"实验柜 '{node.lab_id}' 不存在")
            async with self._lock:
                self._running_nodes.pop(node.node_id, None)
                self._failed_nodes.append(node)
            await self._settle_completed_node(node, dag_graph)
            return

        try:
            result = await env.run_single_task(
                task_description=f"[{node.node_id}] {node.skill_name}",
                suspend_event=None,
            )
            node.mark_completed(result)
            async with self._lock:
                self._running_nodes.pop(node.node_id, None)
                self._completed_nodes.append(node)
        except Exception as exc:
            logger.exception("[Worker-%d] 节点执行异常: [%s]", worker_id, node.node_id)
            node.mark_failed(str(exc))
            async with self._lock:
                self._running_nodes.pop(node.node_id, None)
                self._failed_nodes.append(node)

        elapsed = node.elapsed_time or 0
        logger.info(
            "✅ [Worker-%d] 节点完成: [%s] %s → %s (耗时: %.1fs)",
            worker_id, node.node_id, node.skill_name,
            node.status.name, elapsed,
        )

        await self._settle_completed_node(node, dag_graph)

    # --------------------------------------------------------------------------- #
    #  便捷方法                                                                  #
    # --------------------------------------------------------------------------- #

    async def run_dag(
        self,
        dag_graph: DAGTaskGraph,
        planner_llm_calls: int = 0,
        planner_time_ms: float = 0.0,
    ) -> DAGExecutionResult:
        """一站式执行 DAG 图：启动 → 提交 → 等待完成 → 关闭。"""
        await self.start()
        await self.submit_dag(dag_graph)

        try:
            await asyncio.wait_for(
                self._dag_complete_event.wait(),
                timeout=3600.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[DAG调度器] DAG '%s' 执行超时", dag_graph.name)

        results = await self.shutdown()
        result = results[0] if results else self._build_dag_result(dag_graph)

        result.planner_llm_calls = planner_llm_calls
        result.planner_time_ms = planner_time_ms
        result.worker_llm_calls = 0

        logger.info(
            "[DAG调度器] V7.2 LLM 埋点统计: graph_id=%s, "
            "planner_llm_calls=%d, planner_time_ms=%.2f, worker_llm_calls=%d",
            result.graph_id, result.planner_llm_calls, result.planner_time_ms, result.worker_llm_calls,
        )

        return result

    def _build_dag_result(self, dag_graph: DAGTaskGraph) -> DAGExecutionResult:
        """构建 DAG 执行结果。"""
        total_time = (
            max((n.end_time or 0) for n in dag_graph.nodes.values()) -
            dag_graph.created_at
        )

        return DAGExecutionResult(
            graph_id=dag_graph.graph_id,
            status="completed" if not dag_graph.has_failures()
                   else ("partial" if dag_graph.get_completed_count() > 0 else "failed"),
            total_nodes=len(dag_graph.nodes),
            completed_nodes=dag_graph.get_completed_count(),
            failed_nodes=dag_graph.get_failed_count(),
            total_time=total_time,
            execution_levels=len(dag_graph.get_execution_levels()),
            node_results=[n.to_dict() for n in dag_graph.nodes.values()],
        )

    @property
    def lab_ids(self) -> list[str]:
        return list(self._labs.keys())

    @property
    def ready_count(self) -> int:
        return self._ready_queue.qsize()

    @property
    def blocked_count(self) -> int:
        return len(self._blocked_queue)

    @property
    def running_count(self) -> int:
        return len(self._running_nodes)

    @property
    def status_summary(self) -> dict[str, Any]:
        """获取调度器状态摘要。"""
        return {
            "ready_queue": self.ready_count,
            "blocked_queue": self.blocked_count,
            "running_nodes": self.running_count,
            "completed_nodes": len(self._completed_nodes),
            "failed_nodes": len(self._failed_nodes),
            "active_graphs": len(self._active_graphs),
            "completed_graphs": len(self._completed_graphs),
            "llm_tasks_in_progress": len(self._llm_tasks),
            "interrupt_queue_size": self._interrupt_queue.qsize(),
            "aging_factor": self._aging_factor,
            "rebalance_interval": self._rebalance_interval,
        }


__all__ = [
    "DAGOrchestrator",
    "DAGNode",
    "DAGTaskGraph",
    "DAGExecutionResult",
    "NodeStatus",
    "TaskPriority",
    "HardwareInterruptTask",
    "HardwareAlarm",
    "compute_dynamic_priority",
    "DAG_PLANNER_PROMPT_TEMPLATE",
    "DAG_VALIDATOR_PROMPT",
]
