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
- V7.4 新增：OoO 乱序越级执行 + 事件驱动零开销 I/O 挂起

Author: AstroSASF Team
Version: 7.4
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


# --------------------------------------------------------------------------- #
#  ActiveResourceTable — 资源锁感知的 OoO 调度基础                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ResourceLock:
    """单个硬件资源的锁记录。"""
    resource_name: str
    owner_node_id: str | None  # 当前持有者（None = 空闲）
    locked_at: float            # 锁定时间戳


class ActiveResourceTable:
    """运行时资源占用表（V7.4 新增）。

    维护一张全局的硬件资源占用表，支持：
    - 查询单个资源是否被占用
    - 查询某节点是否持有一个或多个资源
    - 在 OoO 调度前做准入检查（防止资源争用）
    - 在节点完成时自动释放资源

    锁顺序协议（Dead-lock Prevention）：
    - 当需要同时锁定多个资源时，必须按 resource_name 字母序依次加锁
    - 这消除了"循环等待"类型的死锁

    Usage
    -----
    >>> table = ActiveResourceTable()
    >>> table.acquire("camera", "node-A")
    True
    >>> table.is_free("camera")
    False
    >>> table.acquire("camera", "node-B")  # 已被 node-A 占用 → 失败
    False
    >>> table.release("node-A")  # node-A 完成，自动释放
    {'camera'}
    """

    def __init__(self) -> None:
        self._locks: dict[str, ResourceLock] = {}
        self._node_resources: dict[str, set[str]] = {}  # node_id → set of resource names
        self._lock_obj: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(
        self,
        resource_name: str,
        node_id: str,
    ) -> bool:
        """原子性地申请持有某资源。

        Returns
        -------
        bool
            True = 成功获取；False = 已被其他节点占用。
        """
        async with self._lock_obj:
            existing = self._locks.get(resource_name)
            if existing is not None and existing.owner_node_id is not None:
                if existing.owner_node_id != node_id:
                    return False  # 已被其他节点占用

            self._locks[resource_name] = ResourceLock(
                resource_name=resource_name,
                owner_node_id=node_id,
                locked_at=time.monotonic(),
            )
            if node_id not in self._node_resources:
                self._node_resources[node_id] = set()
            self._node_resources[node_id].add(resource_name)

            logger.debug(
                "[ResourceTable] 资源占用: node=%s, resource=%s",
                node_id, resource_name,
            )
            return True

    async def release(self, node_id: str) -> set[str]:
        """节点完成后释放其持有的所有资源（原子操作）。

        Returns
        -------
        set[str]
            被释放的资源名集合。
        """
        async with self._lock_obj:
            freed: set[str] = set()
            resources = self._node_resources.pop(node_id, set())
            for res_name in resources:
                lock = self._locks.get(res_name)
                if lock is not None and lock.owner_node_id == node_id:
                    lock.owner_node_id = None
                    freed.add(res_name)
                    logger.debug(
                        "[ResourceTable] 资源释放: node=%s, resource=%s",
                        node_id, res_name,
                    )
            return freed

    def is_free(self, resource_name: str) -> bool:
        """查询某资源是否空闲（无需加锁，只读快照）。"""
        lock = self._locks.get(resource_name)
        return lock is None or lock.owner_node_id is None

    def get_holder(self, resource_name: str) -> str | None:
        """查询某资源的当前持有者。"""
        lock = self._locks.get(resource_name)
        return lock.owner_node_id if lock else None

    def get_node_resources(self, node_id: str) -> frozenset[str]:
        """查询某节点当前持有的资源集合。"""
        return frozenset(self._node_resources.get(node_id, set()))

    def get_all_held_resources(self) -> dict[str, str | None]:
        """获取所有资源的占用快照（resource_name → owner_node_id）。"""
        return {
            res: lock.owner_node_id
            for res, lock in self._locks.items()
        }


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

    # ── V7.4: OoO 乱序越级执行 ── #
    _resource_table: ActiveResourceTable = field(
        default_factory=ActiveResourceTable, init=False,
    )
    _ooo_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    # 每次 Worker 从 ReadyQueue 取不到节点时（阻塞），调度器尝试 OoO 推进
    _ooo_lookahead_triggered: int = field(default=0, init=False)
    # OoO 成功越级执行的节点数
    _ooo_execution_count: int = field(default=0, init=False)
    # I/O 与计算重叠的累计毫秒数
    _io_compute_overlap_ms: float = field(default=0.0, init=False)
    # 每当一个 Worker 在 I/O 等待时记录开始时间；I/O 结束时累积到 _io_compute_overlap_ms
    _active_io_windows: dict[str, float] = field(default_factory=dict, init=False)
    # 空闲 Worker 槽位阈值（当 RunningNodes 数量 < max_workers 时，触发 Look-ahead）
    _min_idle_worker_threshold: int = 1
    # OoO 最大等待窗口（毫秒），超时后强制回退
    _ooo_max_wait_ms: float = 2000.0

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
            loop = asyncio.get_running_loop()
            loop.create_task(
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
    #  V7.4: OoO 乱序越级执行 (Out-of-Order Look-ahead Scheduler)                   #
    # --------------------------------------------------------------------------- #
    # 防死锁策略：
    # 1. 锁顺序协议：_resource_table 内部在同时锁多个资源时按字母序依次加锁
    # 2. OoO 三重准入门：(a) 逻辑依赖全部 COMPLETED (b) 资源未被占用 (c) 联锁不拦截
    # 3. 资源预约原子性：检查与预约在同一 _ooo_lock 下完成，不分拆
    # 4. 推进保证：每次 OoO 推进后立即更新 _resource_table，不留幽灵锁
    # 5. 降级兜底：超时 _ooo_max_wait_ms 后强制回退，正常结算
    # --------------------------------------------------------------------------- #

    async def _check_ooo_promotion(
        self,
        node: DAGNode,
        dag_graph: DAGTaskGraph,
    ) -> tuple[bool, str]:
        """OoO 越级提取的三重准入门检查。

        Returns
        -------
        tuple[bool, str]
            (can_promote, reason_if_not)
        """
        # Gate 1: 逻辑依赖已全部完成（已在 BlockedQueue 中说明）
        all_deps_done = all(
            dag_graph.nodes[dep_id].status == NodeStatus.COMPLETED
            for dep_id in node.dependencies
            if dep_id in dag_graph.nodes
        )
        if not all_deps_done:
            return False, "仍有逻辑依赖未完成"

        # Gate 2: 物理资源未被锁定（查询 ActiveResourceTable）
        required_resources = self._extract_required_resources(node)
        for res in required_resources:
            if not self._resource_table.is_free(res):
                holder = self._resource_table.get_holder(res)
                return False, f"资源 '{res}' 已被 '{holder}' 占用"

        # Gate 3: 联锁引擎不拦截（可选，如果接入了 interlock_engine）
        interlock_allowed = await self._check_interlock(node)
        if not interlock_allowed:
            return False, "联锁引擎拦截"

        return True, "OK"

    def _extract_required_resources(self, node: DAGNode) -> list[str]:
        """从节点参数中提取所需的硬件资源列表。

        策略：从 skill_name 和 params 中提取资源标识符。
        可通过 _tool_registrar 扩展（暂不强制依赖）。
        """
        resources: list[str] = []
        # 从 skill_name 中提取资源前缀（如 "camera_*.py" → "camera"）
        if node.skill_name:
            parts = node.skill_name.lower().split("_")
            if parts:
                resources.append(parts[0])
        # 从 lab_id 推断（同一实验柜硬件互斥）
        if node.lab_id:
            resources.append(f"lab:{node.lab_id}")
        return resources

    async def _check_interlock(self, node: DAGNode) -> bool:
        """查询联锁引擎，判断节点是否可以执行（可扩展接入 labs/interlock_engine）。"""
        # 默认允许；如果接入了 interlock_engine，可以在这里调用其 API
        # labs/interlock_engine.InterlockEngine.check(node) → bool
        # 此处保持默认 True，避免强制依赖 labs 模块
        return True

    async def _look_ahead_and_promote(self) -> int:
        """主动遍历 BlockedQueue，执行资源锁感知的越级推进。

        触发时机：Worker Pool 有空闲槽位但 ReadyQueue 为空时。

        算法：
        1. 遍历所有 BlockedQueue 中的节点
        2. 对每个节点执行三重准入门检查（依赖 / 资源 / 联锁）
        3. 若检查通过：在 _ooo_lock 下将节点预约资源并移入 ReadyQueue
        4. 记录 OoO 执行次数

        Returns
        -------
        int
            本次调用成功越级推进的节点数。

        防死锁核心：
        - 资源预约与 ReadyQueue 入队在同一把 _ooo_lock 下完成
        - 不存在"检查通过但预约失败"的竞态窗口
        """
        promoted = 0

        async with self._ooo_lock:
            # 获取 BlockedQueue 快照（遍历时不上锁，允许并发 I/O 回调正常结算）
            blocked_snapshot = list(self._blocked_queue)
            for node in blocked_snapshot:
                if node.status != NodeStatus.PENDING:
                    continue

                dag_graph = self._active_graphs.get(node.graph_id)
                if dag_graph is None:
                    continue

                can_promote, reason = await self._check_ooo_promotion(node, dag_graph)
                if not can_promote:
                    logger.debug(
                        "[OoO] 节点 '%s' 越级检查未通过: %s",
                        node.node_id, reason,
                    )
                    continue

                # 原子性资源预约（在 _ooo_lock 下）
                required_resources = self._extract_required_resources(node)
                reservation_ok = True
                for res in sorted(required_resources):  # 字母序 → 锁顺序协议
                    acquired = await self._resource_table.acquire(res, node.node_id)
                    if not acquired:
                        reservation_ok = False
                        # 回滚已预约的资源（保持原子性）
                        await self._resource_table.release(node.node_id)
                        logger.warning(
                            "[OoO] 资源预约失败，回滚: node=%s, resource=%s",
                            node.node_id, res,
                        )
                        break

                if not reservation_ok:
                    continue

                # 从 BlockedQueue 移除（在 _ooo_lock 下）
                self._blocked_queue = [n for n in self._blocked_queue if n.node_id != node.node_id]

                # 注入 ReadyQueue（立即可被任何空闲 Worker 抢到）
                node.status = NodeStatus.READY
                priority_score = node.get_ready_score()
                await self._ready_queue.put((priority_score, node))

                self._ooo_execution_count += 1
                promoted += 1
                self._ooo_lookahead_triggered += 1

                logger.info(
                    "🔀 [OoO] 越级推进成功: 节点 '%s' (skill=%s, reason=%s) "
                    "→ ReadyQueue | OoO累计: %d",
                    node.node_id, node.skill_name, reason, self._ooo_execution_count,
                )

        return promoted

    def _start_io_overlap_tracking(self, node_id: str) -> None:
        """记录某节点开始 I/O 等待的时间戳（用于 I/O-计算重叠度统计）。"""
        self._active_io_windows[node_id] = time.monotonic()
        logger.debug(
            "[IoOverlap] I/O 等待开始: node=%s (活跃 I/O 窗口: %d)",
            node_id, len(self._active_io_windows),
        )

    def _end_io_overlap_tracking(self, node_id: str) -> None:
        """节点 I/O 等待结束，累加重叠时长到统计指标。"""
        start = self._active_io_windows.pop(node_id, None)
        if start is not None:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._io_compute_overlap_ms += elapsed_ms
            logger.debug(
                "[IoOverlap] I/O 窗口结束: node=%s, duration=%.1fms (累计: %.1fms)",
                node_id, elapsed_ms, self._io_compute_overlap_ms,
            )

    # --------------------------------------------------------------------------- #
    #  Worker 生命周期 (执行阶段)                                                 #
    # --------------------------------------------------------------------------- #

    async def start(self) -> None:
        """启动 Worker 协程池和 Aging 重平衡协程。"""
        self._shutdown_flag = False
        self._dag_complete_event = asyncio.Event()

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🚀 DAG 调度内核启动 (V7.4 OoO + Event-Driven I/O)     ║")
        logger.info("║  Workers: %-3d | ReadyQueue | BlockedQueue               ║", self.max_workers)
        logger.info("║  理论智能体: DAG Planner | 实践智能体: DAG Workers    ║")
        logger.info("║  Aging 重平衡: %.1fs 间隔 | 因子: %.3f                     ║",
                    self._rebalance_interval, self._aging_factor)
        logger.info("║  OoO 越级执行: 启用 | 资源锁感知: 启用                  ║")
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("")

        self._aging_rebalance_task = asyncio.get_running_loop().create_task(
            self._aging_rebalance_loop(),
            name="aging-rebalance",
        )

        for i in range(self.max_workers):
            worker = asyncio.get_running_loop().create_task(
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
        """实践智能体 Worker 协程 — 从 ReadyQueue 取节点执行。

        V7.4 增强：
        - ReadyQueue 为空时，触发 _look_ahead_and_promote() 尝试 OoO 越级提取
        - 当 RunningNodes 数量 < max_workers 时，说明有 Worker 在等待 I/O
          （或完全挂起），此时 OoO 推进可以提升资源利用率
        """
        logger.info("[Worker-%d] 实践智能体就绪", worker_id)

        while not self._shutdown_flag:
            try:
                if self._hardware_interrupt_event.is_set():
                    self._hardware_interrupt_event.clear()
                    logger.info("[Worker-%d] 检测到硬件中断信号", worker_id)

                # V7.4: 动态超时 — 有空闲槽位时缩短等待，允许更快触发 OoO
                async with self._lock:
                    running_now = len(self._running_nodes)
                idle_slots = self.max_workers - running_now
                queue_timeout = 0.2 if idle_slots > 0 else 1.0

                try:
                    priority_score, node = await asyncio.wait_for(
                        self._ready_queue.get(),
                        timeout=queue_timeout,
                    )
                except asyncio.TimeoutError:
                    # V7.4: ReadyQueue 为空，尝试 OoO 越级推进
                    if idle_slots > 0 and self._blocked_queue:
                        promoted = await self._look_ahead_and_promote()
                        if promoted > 0:
                            logger.info(
                                "[Worker-%d] OoO 推进了 %d 个节点，触发 Worker 重新竞争",
                                worker_id, promoted,
                            )
                    continue

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
            # V7.4: 节点完成，自动释放其持有的所有硬件资源
            await self._resource_table.release(node.node_id)
            self._end_io_overlap_tracking(node.node_id)
        except Exception as exc:
            logger.exception("[Worker-%d] 节点执行异常: [%s]", worker_id, node.node_id)
            node.mark_failed(str(exc))
            async with self._lock:
                self._running_nodes.pop(node.node_id, None)
                self._failed_nodes.append(node)
            # V7.4: 节点失败也必须释放资源，防止幽灵锁
            await self._resource_table.release(node.node_id)
            self._end_io_overlap_tracking(node.node_id)

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
        """构建 DAG 执行结果（含 V7.4 OoO/Overlap 指标）。"""
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
            # V7.4 新增指标
            planner_llm_calls=0,
            planner_time_ms=0.0,
            worker_llm_calls=0,
            ooo_execution_count=self._ooo_execution_count,
            io_compute_overlap_ms=self._io_compute_overlap_ms,
            ooo_lookahead_triggered=self._ooo_lookahead_triggered,
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
        """获取调度器状态摘要（含 V7.4 OoO 指标）。"""
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
            # V7.4 OoO 指标
            "ooo_execution_count": self._ooo_execution_count,
            "ooo_lookahead_triggered": self._ooo_lookahead_triggered,
            "io_compute_overlap_ms": self._io_compute_overlap_ms,
            "active_io_windows": len(self._active_io_windows),
            "resource_locks": self._resource_table.get_all_held_resources(),
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
    "ActiveResourceTable",
    "ResourceLock",
    "compute_dynamic_priority",
    "DAG_PLANNER_PROMPT_TEMPLATE",
    "DAG_VALIDATOR_PROMPT",
]
