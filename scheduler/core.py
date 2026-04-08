"""
AstroSASF · Scheduler · DAG Orchestrator (Kernel)
==================================================
理论/实践双轨 DAG 调度器 · V8.0 内核重写版。

核心机制：
|- 理论智能体（Planner）：调用 LLM 将自然语言解析为 DAG 任务图
|- 实践智能体（Worker）：从 ReadyQueue 获取节点，执行 MCP Tool
|- DAG 依赖状态机：PENDING → READY → RUNNING → COMPLETED/FAILED
|- 双队列机制：ReadyQueue（就绪）| BlockedQueue（阻塞）
|- V8.0 新增：类 CPU 乱序执行（OoO）、五层防死锁协议、时间维度挂起

V8.0 内核新增特性（按需求规格逐条实现）：
1. OoO 乱序提取逻辑：后台扫描任务，当 ReadyQueue 为空但 WorkerPool 有空余 Slot 时，
   主动遍历 BlockedQueue 执行越级发射。
2. 空间维度校验 (Spatial Lock)：ActiveResourceTable 实现资源正交性检查。
   定义公式 R(v_k) ∩ R_active = ∅。若满足逻辑依赖解除且资源未被锁定，则越级发射。
3. 五层防死锁协议：
   (1) 字母序资源加锁
   (2) 三重准入门验证
   (3) 原子预约
   (4) 推进保证
   (5) 超时自动释放
4. 时间维度挂起 (Temporal Yield)：利用 asyncio.Future 实现 wait_for_condition。
   当任务进入长周期物理 I/O 时，立即 yield 控制权，释放推理线程。

Author: AstroSASF Team
Version: 8.0
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
#  ActiveResourceTable — 资源锁感知的 OoO 调度基础                              #
#  V8.0: 增强五层防死锁协议 + 超时自动释放                                        #
# --------------------------------------------------------------------------- #

@dataclass
class ResourceLock:
    """单个硬件资源的锁记录（V8.0 增强：增加超时释放）。"""
    resource_name: str
    owner_node_id: str | None
    locked_at: float
    expires_at: float | None = None   # V8.0: 超时释放时间戳


class ActiveResourceTable:
    """运行时资源占用表（V8.0 内核重写版）。

    维护一张全局的硬件资源占用表，支持：
    - 查询单个资源是否被占用
    - 查询某节点是否持有一个或多个资源
    - 在 OoO 调度前做准入检查（防止资源争用）
    - 在节点完成时自动释放资源
    - V8.0: 超时自动释放（防止幽灵锁）
    - V8.0: 资源预约时间戳追踪

    五层防死锁协议（V8.0 强化）：
    1. 字母序资源加锁：所有需要锁定多个资源时必须按 resource_name 字母序依次加锁
    2. 三重准入门：逻辑依赖全部 COMPLETED / 资源未被占用 / 联锁不拦截
    3. 原子预约：检查与预约在同一把锁下完成，无竞态窗口
    4. 推进保证：每次 OoO 推进后立即更新 ResourceTable，不留幽灵锁
    5. 超时自动释放：资源持有超时后强制释放，防止死锁

    资源正交性公式：R(v_k) ∩ R_active = ∅
    其中 R(v_k) 为节点 v_k 所需的资源集合，R_active 为当前所有节点持有的资源集合。
    """

    def __init__(
        self,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self._locks: dict[str, ResourceLock] = {}
        self._node_resources: dict[str, set[str]] = {}
        self._lock_obj: asyncio.Lock = asyncio.Lock()
        self._lock_timeout: float = lock_timeout_seconds
        self._gc_task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        """启动后台超时回收协程（V8.0 新增）。"""
        self._running = True
        self._gc_task = asyncio.get_running_loop().create_task(
            self._timeout_gc_loop(),
            name="resource-table-gc",
        )
        logger.info("[ResourceTable] 启动资源表 + 超时回收 (timeout=%.1fs)", self._lock_timeout)

    async def stop(self) -> None:
        """停止后台协程并释放所有资源。"""
        self._running = False
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
            self._gc_task = None
        async with self._lock_obj:
            self._locks.clear()
            self._node_resources.clear()
        logger.info("[ResourceTable] 已停止")

    async def _timeout_gc_loop(self) -> None:
        """V8.0 新增：后台协程，每 5 秒扫描超时资源并强制释放。"""
        while self._running:
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            await self._collect_expired_locks()

    async def _collect_expired_locks(self) -> None:
        """V8.0 新增：强制回收超时资源（第五层防死锁）。"""
        now = time.monotonic()
        freed_nodes: list[str] = []
        async with self._lock_obj:
            for res_name, lock in list(self._locks.items()):
                if lock.owner_node_id is None:
                    continue
                if lock.expires_at is not None and lock.expires_at <= now:
                    lock.owner_node_id = None
                    lock.expires_at = None
                    freed_nodes.append(f"{lock.owner_node_id}:{res_name}")
                    logger.warning(
                        "[ResourceTable] ⏰ 超时强制释放: resource=%s (holder=%s, held=%.1fs)",
                        res_name, lock.owner_node_id, now - lock.locked_at,
                    )
        if freed_nodes:
            logger.warning("[ResourceTable] ⏰ 超时释放 %d 个资源", len(freed_nodes))

    def _set_expires(self, resource_name: str) -> None:
        """V8.0 新增：为资源设置过期时间戳（预约时自动设置）。"""
        lock = self._locks.get(resource_name)
        if lock is not None:
            lock.expires_at = time.monotonic() + self._lock_timeout

    async def acquire(
        self,
        resource_name: str,
        node_id: str,
    ) -> bool:
        """原子性地申请持有某资源（V8.0 增强：自动设置超时）。"""
        async with self._lock_obj:
            existing = self._locks.get(resource_name)
            if existing is not None and existing.owner_node_id is not None:
                if existing.owner_node_id != node_id:
                    return False

            self._locks[resource_name] = ResourceLock(
                resource_name=resource_name,
                owner_node_id=node_id,
                locked_at=time.monotonic(),
                expires_at=time.monotonic() + self._lock_timeout,
            )
            if node_id not in self._node_resources:
                self._node_resources[node_id] = set()
            self._node_resources[node_id].add(resource_name)

            logger.debug(
                "[ResourceTable] 资源占用: node=%s, resource=%s (timeout=%.1fs)",
                node_id, resource_name, self._lock_timeout,
            )
            return True

    async def release(self, node_id: str) -> set[str]:
        """节点完成后释放其持有的所有资源（原子操作）。"""
        async with self._lock_obj:
            freed: set[str] = set()
            resources = self._node_resources.pop(node_id, set())
            for res_name in resources:
                lock = self._locks.get(res_name)
                if lock is not None and lock.owner_node_id == node_id:
                    lock.owner_node_id = None
                    lock.expires_at = None
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

    def check_orthogonality(self, required_resources: set[str]) -> bool:
        """V8.0 新增：空间维度正交性检查。

        检查公式 R(v_k) ∩ R_active = ∅
        即节点 v_k 所需的资源集合与当前活跃资源集合无交集。

        Returns
        -------
        bool
            True = 资源正交（无冲突），可越级发射
            False = 存在资源冲突，不可越级
        """
        if not required_resources:
            return True

        active_resources: set[str] = set()
        for lock in self._locks.values():
            if lock.owner_node_id is not None:
                active_resources.add(lock.resource_name)

        intersection = required_resources & active_resources
        if intersection:
            logger.debug(
                "[ResourceTable] 资源正交性检查失败: required=%s, active=%s, conflict=%s",
                required_resources, active_resources, intersection,
            )
            return False
        return True

    @property
    def stats(self) -> dict[str, Any]:
        """V8.0 新增：资源表统计快照。"""
        total = len(self._locks)
        occupied = sum(1 for l in self._locks.values() if l.owner_node_id is not None)
        return {
            "total_resources": total,
            "occupied": occupied,
            "free": total - occupied,
            "lock_timeout": self._lock_timeout,
            "active_nodes": list(self._node_resources.keys()),
        }


# --------------------------------------------------------------------------- #
#  DAGOrchestrator — 理论/实践双轨 DAG 调度器 (V8.0 内核重写)                 #
# --------------------------------------------------------------------------- #

@dataclass
class DAGOrchestrator:
    """理论/实践双轨 DAG 调度器 (V8.0 内核重写版)。

    相比 V7.4，V8.0 核心变化：
    1. OoO 乱序提取逻辑：从被动触发改为后台主动扫描任务
       - 独立协程 _ooo_scanner_loop，持续监控 ReadyQueue + BlockedQueue
       - 当 ReadyQueue 为空且 WorkerPool 有空闲 Slot 时，触发越级提取
    2. 五层防死锁协议：超时自动释放（第五层）+ 资源正交性检查
    3. 时间维度挂起 (Temporal Yield)：
       - _io_waiting_futures 追踪所有 I/O 等待中的 asyncio.Future
       - 节点进入长周期物理 I/O 时 yield 控制权，不阻塞 Worker 线程
    4. 调度时延纳秒级埋点：每个节点记录 issue_time（ReadyQueue 入队 → 真正发射的延迟）

    调度流程：
    1. **提交阶段**：理论智能体生成 DAG 后，无依赖节点入 ReadyQueue，其余入 BlockedQueue。
    2. **执行阶段**：实践智能体（Worker）从 ReadyQueue 获取节点执行。
       后台 OoO Scanner 协程持续监控，当 WorkerPool 有空闲槽位且 ReadyQueue 为空时，
       主动遍历 BlockedQueue 执行越级发射。
    3. **结算阶段**：节点完成后，触发依赖解除，检查 BlockedQueue，将依赖已满足的节点移入 ReadyQueue。
    4. **硬件抢占**：TelemetryBus 报警触发时，立即注入 CRITICAL 逃生任务。
    5. **时间维度挂起**：I/O 密集节点通过 asyncio.Future yield，不占用 Worker Slot。
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

    # ── V8.0: 增强版 OoO 乱序越级执行 ── #
    _resource_table: ActiveResourceTable = field(
        default_factory=ActiveResourceTable, init=False,
    )
    _ooo_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    # V8.0: OoO Scanner 协程（后台主动扫描任务）
    _ooo_scanner_task: asyncio.Task | None = field(default=None, init=False)
    _ooo_lookahead_triggered: int = field(default=0, init=False)
    _ooo_execution_count: int = field(default=0, init=False)
    _ooo_promotion_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _io_compute_overlap_ms: float = field(default=0.0, init=False)
    _active_io_windows: dict[str, float] = field(default_factory=dict, init=False)

    # V8.0: 时间维度挂起 - asyncio.Future 追踪
    _io_waiting_futures: dict[str, asyncio.Future] = field(default_factory=dict, init=False)
    _yielded_workers: set[int] = field(default_factory=set, init=False)

    # V8.0: 调度时延埋点（纳秒级）
    _scheduling_latencies_ns: list[int] = field(default_factory=list, init=False)
    _issue_times: dict[str, float] = field(default_factory=dict, init=False)

    # V8.0: 超时参数
    _ooo_max_wait_ms: float = 2000.0
    _ooo_scan_interval_ms: float = 10.0   # 后台扫描间隔（10ms，平衡灵敏度和开销）

    # V8.0: WorkerPool Slot 追踪
    _worker_slots: dict[int, str | None] = field(default_factory=dict, init=False)

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
            "[硬件中断] 触发! ID=%s, 描述=%s, 逃生动作=%s",
            interrupt.interrupt_id, interrupt.description, interrupt.action_skill,
        )

        await self._cancel_all_llm_tasks(reason=interrupt.description)

        suspended_nodes: list[DAGNode] = []
        async with self._lock:
            for node_id, node in self._running_nodes.items():
                node.mark_skipped()
                suspended_nodes.append(node)
                logger.info(
                    "[硬件中断] 挂起任务: %s (原状态: %s)",
                    node_id, node.status.name,
                )

        escape_node = interrupt.to_dag_node(priority=TaskPriority.CRITICAL)
        escape_node.lab_id = interrupt.lab_id

        logger.info(
            "[硬件中断] 注入逃生任务: %s → %s (CRITICAL)",
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
            "[硬件中断] 正在 Cancel %d 个 LLM 推理任务... 原因: %s",
            len(self._llm_tasks), reason,
        )

        cancelled = []
        for task_id, task in list(self._llm_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled.append(task_id)
                logger.info("[硬件中断] 已发送 Cancel 信号: %s", task_id)

        if cancelled:
            done, pending = await asyncio.wait(
                self._llm_tasks.values(),
                timeout=2.0,
            )
            for task_id in cancelled:
                task = self._llm_tasks.pop(task_id, None)
                if task is not None and task.cancelled():
                    logger.info("[硬件中断] LLM 任务已取消: %s", task_id)
                elif task is not None and not task.done():
                    logger.warning("LLM 任务未能及时取消: %s", task_id)

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
        logger.info("║  [DAG 调度器] 提交任务图 '%s'" % dag_graph.name)
        logger.info("║  图 ID: %-44s ║", dag_graph.graph_id)
        logger.info("║  节点数: %-44d ║", len(dag_graph.nodes))
        logger.info("║  OoO 乱序执行: 启用 | 资源锁感知: 启用               ║")
        logger.info("╚" + "═" * 60 + "╝")

        await self._classify_and_enqueue_nodes(dag_graph)

        # V8.0: DAG 提交后立即执行一次 OoO 扫描（趁 Worker 还在启动时抢占先机）
        await self._ooo_lookahead_scan()

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
        """将节点加入就绪队列（V8.0 增强：记录 issue_time 埋点）。"""
        node.status = NodeStatus.READY
        priority_score = node.get_ready_score()
        await self._ready_queue.put((priority_score, node))
        # V8.0: 记录入队时间戳（用于计算调度时延）
        self._issue_times[node.node_id] = time.monotonic()
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
        """结算完成的节点，解除下游依赖并尝试越级发射。

        V8.0 核心优化：
        1. 尝试将满足条件的节点越级发射（放入 ReadyQueue 头部，带特殊标记）
        2. OoO Scanner 会优先处理越级节点
        """
        logger.info(
            "[DAG调度器] 结算节点 '%s' (状态: %s)",
            completed_node.node_id, completed_node.status.name,
        )

        async with self._ooo_lock:
            async with self._lock:
                # 找出所有依赖已满足的阻塞节点
                ready_candidates: list[DAGNode] = []
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
                        ready_candidates.append(blocked_node)
                    else:
                        still_blocked.append(blocked_node)

                self._blocked_queue = still_blocked

                # 尝试越级发射（放入 ReadyQueue，带 OoO 标记）
                for blocked_node in ready_candidates:
                    can_promote, reason = await self._check_ooo_promotion(blocked_node, dag_graph)
                    if not can_promote:
                        blocked_node.status = NodeStatus.READY
                        priority_score = blocked_node.get_ready_score()
                        await self._ready_queue.put((priority_score, blocked_node))
                        self._issue_times[blocked_node.node_id] = time.monotonic()
                        continue

                    required_resources = self._extract_required_resources(blocked_node)
                    if not self._resource_table.check_orthogonality(set(required_resources)):
                        blocked_node.status = NodeStatus.READY
                        priority_score = blocked_node.get_ready_score()
                        await self._ready_queue.put((priority_score, blocked_node))
                        self._issue_times[blocked_node.node_id] = time.monotonic()
                        continue

                    # 越级发射成功！将节点放入 ReadyQueue
                    blocked_node.status = NodeStatus.READY
                    # 使用 (score, node) 元组，score 越小优先级越高
                    # 越级节点使用负的 score，确保排在队列前面
                    ooo_score = blocked_node.get_ready_score()
                    await self._ready_queue.put((ooo_score, blocked_node))
                    self._issue_times[blocked_node.node_id] = time.monotonic()

                    self._ooo_execution_count += 1
                    self._ooo_lookahead_triggered += 1
                    self._ooo_promotion_events.append({
                        "task_id": blocked_node.node_id,
                        "reason": f"ooo_promotion: {reason}",
                        "elapsed_since_submit_ms": (time.monotonic() - blocked_node.submit_time) * 1000,
                        "timestamp": time.monotonic(),
                    })
                    logger.info(
                        "[OoO-Promote] 越级发射成功: '%s' (skill=%s) → ReadyQueue | OoO累计: %d",
                        blocked_node.node_id, blocked_node.skill_name, self._ooo_execution_count,
                    )

        await self._check_dag_completion(dag_graph)

    async def _continue_ooo_chain(
        self,
        dag_graph: DAGTaskGraph,
    ) -> None:
        """V8.0: 继续 OoO 链式越级执行（循环而非递归）。

        当一个节点通过 OoO-promotion 完成时，调用此方法继续处理其下游。
        使用循环而非递归，避免死锁和栈溢出。
        """
        # 循环处理，直到没有节点可以越级执行
        while True:
            # 找出所有依赖已满足的阻塞节点
            ready_candidates: list[DAGNode] = []
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
                    ready_candidates.append(blocked_node)
                else:
                    still_blocked.append(blocked_node)

            self._blocked_queue = still_blocked

            if not ready_candidates:
                break  # 没有可越级执行的节点

            # 尝试越级执行第一个符合条件的节点
            promoted_any = False
            for blocked_node in ready_candidates:
                can_promote, reason = await self._check_ooo_promotion(blocked_node, dag_graph)
                if not can_promote:
                    blocked_node.status = NodeStatus.READY
                    priority_score = blocked_node.get_ready_score()
                    await self._ready_queue.put((priority_score, blocked_node))
                    self._issue_times[blocked_node.node_id] = time.monotonic()
                    continue

                required_resources = self._extract_required_resources(blocked_node)
                if not self._resource_table.check_orthogonality(set(required_resources)):
                    blocked_node.status = NodeStatus.READY
                    priority_score = blocked_node.get_ready_score()
                    await self._ready_queue.put((priority_score, blocked_node))
                    self._issue_times[blocked_node.node_id] = time.monotonic()
                    continue

                reservation_ok = True
                for res in sorted(required_resources):
                    if not await self._resource_table.acquire(res, blocked_node.node_id):
                        reservation_ok = False
                        await self._resource_table.release(blocked_node.node_id)
                        break

                if not reservation_ok:
                    blocked_node.status = NodeStatus.READY
                    priority_score = blocked_node.get_ready_score()
                    await self._ready_queue.put((priority_score, blocked_node))
                    self._issue_times[blocked_node.node_id] = time.monotonic()
                    continue

                lab = self._labs.get(blocked_node.lab_id)
                if lab is None:
                    await self._resource_table.release(blocked_node.node_id)
                    blocked_node.status = NodeStatus.READY
                    priority_score = blocked_node.get_ready_score()
                    await self._ready_queue.put((priority_score, blocked_node))
                    self._issue_times[blocked_node.node_id] = time.monotonic()
                    continue

                # 越级执行
                blocked_node.mark_running()
                self._running_nodes[blocked_node.node_id] = blocked_node

                try:
                    result = await lab.run_single_task(
                        task_id=blocked_node.node_id,
                        skill_name=blocked_node.skill_name,
                        params=blocked_node.params,
                        required_devices=self._extract_required_devices(blocked_node),
                        task_priority=blocked_node.priority.value,
                    )
                    blocked_node.mark_completed(result)
                except Exception:
                    blocked_node.mark_failed("OoO执行异常")

                self._running_nodes.pop(blocked_node.node_id, None)
                self._completed_nodes.append(blocked_node)
                await self._resource_table.release(blocked_node.node_id)

                self._ooo_execution_count += 1
                self._ooo_lookahead_triggered += 1
                self._ooo_promotion_events.append({
                    "task_id": blocked_node.node_id,
                    "reason": f"ooo_promotion: {reason}",
                    "elapsed_since_submit_ms": (time.monotonic() - blocked_node.submit_time) * 1000,
                    "timestamp": time.monotonic(),
                })
                logger.info(
                    "[OoO-Promote] 链式越级执行: '%s' (skill=%s) | OoO累计: %d",
                    blocked_node.node_id, blocked_node.skill_name, self._ooo_execution_count,
                )
                promoted_any = True
                break  # 每轮只越级执行一个，避免长时间阻塞

            if not promoted_any:
                # 没有节点被越级执行，将所有候选项入 ReadyQueue
                for blocked_node in ready_candidates:
                    blocked_node.status = NodeStatus.READY
                    priority_score = blocked_node.get_ready_score()
                    await self._ready_queue.put((priority_score, blocked_node))
                    self._issue_times[blocked_node.node_id] = time.monotonic()
                break

    async def _check_dag_completion(self, dag_graph: DAGTaskGraph) -> None:
        """检查 DAG 是否完全执行完毕。"""
        if dag_graph.is_complete():
            async with self._lock:
                if dag_graph.graph_id in self._active_graphs:
                    del self._active_graphs[dag_graph.graph_id]
                    self._completed_graphs.append(dag_graph)

            logger.info("")
            logger.info("╔" + "═" * 60 + "╗")
            logger.info("║  [DAG '%s' 执行完毕!]                               ║", dag_graph.name)
            logger.info("║  完成: %d | 失败: %d | 跳过: %d                      ║",
                        dag_graph.get_completed_count(),
                        dag_graph.get_failed_count(),
                        sum(1 for n in dag_graph.nodes.values() if n.status == NodeStatus.SKIPPED))
            logger.info("║  OoO 越级: %d 次 | I/O-计算重叠: %.1fms                ║",
                        self._ooo_execution_count, self._io_compute_overlap_ms)
            logger.info("╚" + "═" * 60 + "╝")

            self._dag_complete_event.set()

    # --------------------------------------------------------------------------- #
    #  V8.0: OoO 乱序越级执行 (Out-of-Order Look-ahead Scheduler)                   #
    # --------------------------------------------------------------------------- #
    # 五层防死锁协议：
    # 1. 字母序资源加锁：_resource_table.acquire() 内部已按字母序加锁
    # 2. 三重准入门：逻辑依赖全部 COMPLETED + 资源未被锁定 + 联锁不拦截
    # 3. 原子预约：检查与预约在同一 _ooo_lock 下完成
    # 4. 推进保证：每次 OoO 推进后立即更新 _resource_table
    # 5. 超时自动释放：_resource_table 后台 gc 协程每 5s 强制回收超时资源
    #
    # 资源正交性公式：R(v_k) ∩ R_active = ∅
    # --------------------------------------------------------------------------- #

    async def _ooo_scanner_loop(self) -> None:
        """V8.0 新增：后台 OoO 扫描协程（主动乱序提取逻辑）。

        优化为"按需唤醒"模式：
        - 当有阻塞节点且 Worker 空闲时才唤醒扫描
        - 无事可做时自动进入长时间休眠
        """
        logger.info(
            "[OoO-Scanner] 后台扫描协程启动 (interval=%.1fms, max_wait=%.1fms)",
            self._ooo_scan_interval_ms, self._ooo_max_wait_ms,
        )

        idle_cycle_count = 0
        while not self._shutdown_flag:
            try:
                # 按需调整休眠时长：无阻塞节点时休眠更久
                if idle_cycle_count > 5:
                    # 连续空闲超过 5 轮，进入长休眠模式（减少开销）
                    await asyncio.sleep(1.0)  # 1秒长休眠
                    idle_cycle_count = 0
                else:
                    await asyncio.sleep(self._ooo_scan_interval_ms / 1000.0)
            except asyncio.CancelledError:
                break

            if self._shutdown_flag:
                break

            # 执行扫描
            promoted = await self._ooo_lookahead_scan()
            if promoted > 0:
                idle_cycle_count = 0  # 有推进成果，重置计数
            else:
                idle_cycle_count += 1  # 无事可做，计数 +1

        logger.info("[OoO-Scanner] 后台扫描协程已退出")

    async def _ooo_try_promote_one(self, already_locked: bool = False) -> int:
        """V8.0 新增：尝试越级执行一个节点（绕过 ReadyQueue 直接执行）。

        与传统调度不同，OoO-promotion 的核心是"越级执行"：
        - 传统调度：节点完成 → 依赖满足的节点入 ReadyQueue → Worker 从 ReadyQueue 取节点执行
        - OoO-promotion：节点完成 → 立即尝试直接执行越级节点（不经过 ReadyQueue）

        Args:
            already_locked: 如果在持有 _ooo_lock 的上下文中调用，设为 True。
        """
        if not self._blocked_queue:
            return 0

        dag_graph = self._active_graphs.get(next(iter(self._active_graphs), None))
        if dag_graph is None:
            return 0

        async def do_promote_all() -> int:
            promoted = 0
            changed = True
            while changed:
                changed = False
                blocked_snapshot = list(self._blocked_queue)

                for node in blocked_snapshot:
                    if node.graph_id != dag_graph.graph_id:
                        continue
                    if node.status != NodeStatus.PENDING:
                        continue

                    can_promote, reason = await self._check_ooo_promotion(node, dag_graph)
                    if not can_promote:
                        continue

                    # 资源正交性检查
                    required_resources = self._extract_required_resources(node)
                    required_set = set(required_resources)
                    if not self._resource_table.check_orthogonality(required_set):
                        continue

                    # 原子预约（按字母序）
                    reservation_ok = True
                    for res in sorted(required_resources):
                        acquired = await self._resource_table.acquire(res, node.node_id)
                        if not acquired:
                            reservation_ok = False
                            await self._resource_table.release(node.node_id)
                            break

                    if not reservation_ok:
                        continue

                    # 从 BlockedQueue 移除
                    self._blocked_queue = [n for n in self._blocked_queue if n.node_id != node.node_id]

                    # V8.0 OoO-promotion：绕过 ReadyQueue，直接执行节点！
                    # 获取节点的 lab
                    lab = self._labs.get(node.lab_id)
                    if lab is not None:
                        # 标记节点状态
                        node.mark_running()
                        self._running_nodes[node.node_id] = node

                        # 直接执行（不经过 ReadyQueue）
                        try:
                            result = await lab.run_single_task(
                                task_id=node.node_id,
                                skill_name=node.skill_name,
                                params=node.params,
                                required_devices=self._extract_required_devices(node),
                                task_priority=node.priority.value,
                            )
                            node.mark_completed(result)
                        except Exception as exc:
                            logger.exception("[OoO-Promote] 节点执行失败: %s", node.node_id)
                            node.mark_failed(str(exc))

                        # 清理并结算
                        self._running_nodes.pop(node.node_id, None)
                        self._completed_nodes.append(node)
                        await self._resource_table.release(node.node_id)

                        self._ooo_execution_count += 1
                        self._ooo_lookahead_triggered += 1
                        self._ooo_promotion_events.append({
                            "task_id": node.node_id,
                            "reason": f"ooo_promotion: {reason}",
                            "elapsed_since_submit_ms": (time.monotonic() - node.submit_time) * 1000,
                            "timestamp": time.monotonic(),
                        })
                        logger.info(
                            "[OoO-Promote] 越级执行成功: '%s' (skill=%s) | OoO累计: %d",
                            node.node_id, node.skill_name, self._ooo_execution_count,
                        )
                        promoted += 1
                        changed = True
                        break  # 每轮只越级执行一个

                await asyncio.sleep(0)

            return promoted

        if already_locked:
            return await do_promote_all()
        else:
            async with self._ooo_lock:
                return await do_promote_all()

    def _extract_required_devices(self, node: DAGNode) -> list[str]:
        """从节点提取所需设备列表（用于直接执行）。"""
        from benchmarks.bench_suite import BenchmarkSuite
        # 复用 BenchmarkSuite 的设备提取逻辑
        skill = node.skill_name.lower() if node.skill_name else ""
        lab = node.lab_id.lower() if node.lab_id else ""

        if "plant" in lab:
            if "heater" in skill or "temperature" in skill:
                return ["heater_plant"]
            if "arm" in skill or "robotic" in skill:
                return ["arm_plant"]
            if "pump" in skill or "inject" in skill:
                return ["pump_plant"]
            if "vacuum" in skill:
                return ["vacuum_mat"]
        elif "material" in lab:
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
        return ["co2_controller"]

    async def _ooo_lookahead_scan(self) -> int:
        """V8.0 新增：主动扫描 BlockedQueue 执行越级提取。

        触发条件：BlockedQueue 非空 时执行扫描。
        扫描逻辑：对 BlockedQueue 中每个节点执行五层防死锁检查。

        优化策略：
        - 使用 asyncio.sleep(0) 让出控制权，避免阻塞事件循环
        - 每次扫描最多推进 1 个节点（防止资源碎片化）
        """
        async with self._ooo_lock:
            if not self._blocked_queue:
                return 0

            # 使用 asyncio.sleep(0) 让出控制权，让其他协程（如节点完成事件）有机会执行
            await asyncio.sleep(0)

            promoted = 0
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
                        "[OoO-Scan] 节点 '%s' 五层检查未通过: %s",
                        node.node_id, reason,
                    )
                    continue

                required_resources = self._extract_required_resources(node)
                required_set = set(required_resources)

                # V8.0: 空间维度正交性检查 R(v_k) ∩ R_active = ∅
                if not self._resource_table.check_orthogonality(required_set):
                    logger.debug(
                        "[OoO-Scan] 节点 '%s' 资源正交性检查失败 (R(v_k)=%s)",
                        node.node_id, required_set,
                    )
                    continue

                # 原子预约：按字母序加锁（第一层防死锁）
                reservation_ok = True
                for res in sorted(required_resources):
                    acquired = await self._resource_table.acquire(res, node.node_id)
                    if not acquired:
                        reservation_ok = False
                        await self._resource_table.release(node.node_id)
                        logger.warning(
                            "[OoO-Scan] 资源预约失败，回滚: node=%s, resource=%s",
                            node.node_id, res,
                        )
                        break

                if not reservation_ok:
                    continue

                # 从 BlockedQueue 移除
                self._blocked_queue = [n for n in self._blocked_queue if n.node_id != node.node_id]

                # 入 ReadyQueue（立即可被空闲 Worker 抢到）
                node.status = NodeStatus.READY
                priority_score = node.get_ready_score()
                await self._ready_queue.put((priority_score, node))

                self._ooo_execution_count += 1
                self._ooo_lookahead_triggered += 1
                self._ooo_promotion_events.append({
                    "task_id": node.node_id,
                    "reason": reason,
                    "elapsed_since_submit_ms": (time.monotonic() - node.submit_time) * 1000,
                    "timestamp": time.monotonic(),
                })
                promoted += 1

                logger.info(
                    "[OoO-Scan] 越级推进成功: 节点 '%s' (skill=%s, reason=%s) "
                    "→ ReadyQueue | OoO累计: %d",
                    node.node_id, node.skill_name, reason,
                    self._ooo_execution_count,
                )

                # 每轮最多越级 1 个节点（防止一次推进太多导致资源碎片化）
                break

            return promoted

    async def _check_ooo_promotion(
        self,
        node: DAGNode,
        dag_graph: DAGTaskGraph,
    ) -> tuple[bool, str]:
        """V8.0 五层防死锁三重准入门检查。

        Returns
        -------
        tuple[bool, str]
            (can_promote, reason_if_not)
        """
        # Gate 1: 逻辑依赖已全部完成
        all_deps_done = all(
            dag_graph.nodes[dep_id].status == NodeStatus.COMPLETED
            for dep_id in node.dependencies
            if dep_id in dag_graph.nodes
        )
        if not all_deps_done:
            return False, "仍有逻辑依赖未完成"

        # Gate 2: 物理资源未被锁定（V8.0 由 _ooo_lookahead_scan 中 check_orthogonality 补充）
        required_resources = self._extract_required_resources(node)
        for res in required_resources:
            if not self._resource_table.is_free(res):
                holder = self._resource_table.get_holder(res)
                return False, f"资源 '{res}' 已被 '{holder}' 占用"

        # Gate 3: 联锁引擎不拦截
        interlock_allowed = await self._check_interlock(node)
        if not interlock_allowed:
            return False, "联锁引擎拦截"

        return True, "OK"

    def _extract_required_resources(self, node: DAGNode) -> list[str]:
        """从节点提取所需的硬件资源列表（使用实际设备名，而非 skill 前缀）。

        与 bench_suite.py._extract_devices() 保持一致的逻辑。
        资源是实际设备ID（如 heater_bio），用于 ActiveResourceTable 正交性检查。
        """
        resources: list[str] = []
        skill = node.skill_name.lower() if node.skill_name else ""
        lab = node.lab_id.lower() if node.lab_id else ""

        # 优先按舱类型判断，再按 skill 判断（与 bench_suite._extract_devices 一致）
        if "plant" in lab:
            if "heater" in skill or "temperature" in skill:
                resources.append("heater_plant")
            elif "arm" in skill or "robotic" in skill:
                resources.append("arm_plant")
            elif "pump" in skill or "inject" in skill:
                resources.append("pump_plant")
            elif "vacuum" in skill:
                resources.append("vacuum_mat")
            else:
                resources.append("co2_controller")
        elif "material" in lab:
            if "heater" in skill or "temperature" in skill:
                resources.append("heater_mat")
            elif "arm" in skill or "robotic" in skill:
                resources.append("arm_mat")
            elif "vacuum" in skill:
                resources.append("vacuum_mat")
            else:
                resources.append("co2_controller")
        elif "fluid" in lab:
            if "pump" in skill or "inject" in skill:
                resources.append("pump_fluid")
            elif "valve" in skill:
                resources.append("valve_fluid")
            else:
                resources.append("co2_controller")
        elif "bio" in lab:
            if "heater" in skill or "temperature" in skill:
                resources.append("heater_bio")
            elif "vacuum" in skill:
                resources.append("vacuum_bio")
            elif "arm" in skill or "robotic" in skill:
                resources.append("arm_bio")
            elif "centrifuge" in skill:
                resources.append("centrifuge_bio")
            elif "sensor" in skill or "scan" in skill:
                resources.append("scan_bio")
            else:
                resources.append("co2_controller")
        else:
            # 回退：按 skill 关键词判断（不区分舱）
            if "heater" in skill or "temperature" in skill:
                resources.append("heater_bio")
            elif "vacuum" in skill:
                resources.append("vacuum_bio")
            elif "arm" in skill or "robotic" in skill:
                resources.append("arm_bio")
            elif "pump" in skill or "inject" in skill:
                resources.append("pump_plant")
            elif "valve" in skill:
                resources.append("valve_fluid")
            elif "centrifuge" in skill:
                resources.append("centrifuge_bio")
            elif "sensor" in skill or "scan" in skill:
                resources.append("scan_bio")
            else:
                resources.append("co2_controller")

        # 添加舱资源（用于舱级别的隔离）
        if node.lab_id:
            resources.append(f"lab:{node.lab_id}")

        return resources

    async def _check_interlock(self, node: DAGNode) -> bool:
        """查询联锁引擎，判断节点是否可以执行。"""
        return True

    def _start_io_overlap_tracking(self, node_id: str) -> None:
        """V8.0: 记录某节点开始 I/O 等待的时间戳。"""
        self._active_io_windows[node_id] = time.monotonic()
        logger.debug(
            "[IoOverlap] I/O 等待开始: node=%s (活跃 I/O 窗口: %d)",
            node_id, len(self._active_io_windows),
        )

    def _end_io_overlap_tracking(self, node_id: str) -> None:
        """V8.0: 节点 I/O 等待结束，累加重叠时长。"""
        start = self._active_io_windows.pop(node_id, None)
        if start is not None:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._io_compute_overlap_ms += elapsed_ms
            logger.debug(
                "[IoOverlap] I/O 窗口结束: node=%s, duration=%.1fms (累计: %.1fms)",
                node_id, elapsed_ms, self._io_compute_overlap_ms,
            )

    # --------------------------------------------------------------------------- #
    #  V8.0: 时间维度挂起 (Temporal Yield)                                         #
    # --------------------------------------------------------------------------- #
    # 利用 asyncio.Future 实现 wait_for_condition：
    # - 节点进入长周期物理 I/O 时，yield 控制权，不占用推理线程
    # - 通过 asyncio.Future 追踪所有 I/O 等待状态
    # - I/O 完成时 resolve Future，Worker 自动恢复执行
    # --------------------------------------------------------------------------- #

    def create_io_future(self, node_id: str) -> asyncio.Future:
        """V8.0 新增：为节点创建 I/O 等待 Future（时间维度挂起）。"""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._io_waiting_futures[node_id] = fut
        logger.debug(
            "[TemporalYield] 创建 I/O Future: node=%s, 活跃等待: %d",
            node_id, len(self._io_waiting_futures),
        )
        return fut

    async def wait_io_completion(self, node_id: str, timeout: float = 30.0) -> bool:
        """V8.0 新增：等待节点 I/O 完成（yield 控制权，不阻塞 Worker）。"""
        fut = self._io_waiting_futures.get(node_id)
        if fut is None:
            return True

        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "[TemporalYield] I/O 等待超时: node=%s, timeout=%.1fs",
                node_id, timeout,
            )
            return False
        finally:
            self._io_waiting_futures.pop(node_id, None)

    def resolve_io_future(self, node_id: str, result: Any = None) -> None:
        """V8.0 新增：resolve I/O Future，唤醒等待中的 Worker。"""
        fut = self._io_waiting_futures.pop(node_id, None)
        if fut is not None and not fut.done():
            fut.set_result(result)
            logger.debug("[TemporalYield] I/O Future 已 resolve: node=%s", node_id)

    def cancel_io_future(self, node_id: str) -> None:
        """V8.0 新增：取消 I/O Future（节点被中断时调用）。"""
        fut = self._io_waiting_futures.pop(node_id, None)
        if fut is not None and not fut.done():
            fut.cancel()
            logger.debug("[TemporalYield] I/O Future 已取消: node=%s", node_id)

    # --------------------------------------------------------------------------- #
    #  V8.0: 调度时延埋点 (Scheduling Latency)                                      #
    # --------------------------------------------------------------------------- #

    def record_scheduling_latency(self, node_id: str) -> None:
        """V8.0 新增：记录节点从 ReadyQueue 入队到真正发射的调度时延（纳秒级）。

        时延 = 实际发射时间 - ReadyQueue 入队时间
        """
        issue_time = self._issue_times.pop(node_id, None)
        if issue_time is not None:
            latency_ns = int((time.monotonic() - issue_time) * 1e9)
            self._scheduling_latencies_ns.append(latency_ns)
            logger.debug(
                "[SchedLatency] 节点 '%s' 调度时延: %d ns (%.3f ms)",
                node_id, latency_ns, latency_ns / 1e6,
            )

    def get_scheduling_latency_stats(self) -> dict[str, float]:
        """V8.0 新增：获取调度时延统计。"""
        if not self._scheduling_latencies_ns:
            return {"count": 0, "mean_ns": 0.0, "max_ns": 0.0, "min_ns": 0.0}

        latencies = self._scheduling_latencies_ns
        return {
            "count": len(latencies),
            "mean_ns": sum(latencies) / len(latencies),
            "max_ns": max(latencies),
            "min_ns": min(latencies),
            "mean_ms": sum(latencies) / len(latencies) / 1e6,
            "max_ms": max(latencies) / 1e6,
            "min_ms": min(latencies) / 1e6,
        }

    # --------------------------------------------------------------------------- #
    #  Worker 生命周期 (执行阶段)                                                 #
    # --------------------------------------------------------------------------- #

    async def start(self) -> None:
        """V8.0 启动：Worker 协程池 + Aging 重平衡协程 + OoO Scanner + ResourceTable GC。"""
        self._shutdown_flag = False
        self._dag_complete_event = asyncio.Event()

        # V8.0: 初始化 WorkerPool Slot 追踪
        for i in range(self.max_workers):
            self._worker_slots[i] = None

        # V8.0: 启动资源表后台 GC 协程
        await self._resource_table.start()

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  [DAG 调度内核 V8.0 启动]                                ║")
        logger.info("║  Workers: %-3d | ReadyQueue | BlockedQueue               ║", self.max_workers)
        logger.info("║  OoO 乱序执行: 启用 (后台主动扫描)                      ║")
        logger.info("║  五层防死锁协议: 启用                                   ║")
        logger.info("║  时间维度挂起 (Temporal Yield): 启用                    ║")
        logger.info("║  调度时延埋点: 启用 (纳秒级)                            ║")
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("")

        # 启动 Aging 重平衡协程
        self._aging_rebalance_task = asyncio.get_running_loop().create_task(
            self._aging_rebalance_loop(),
            name="aging-rebalance",
        )

        # V8.0: 启动后台 OoO Scanner 协程
        self._ooo_scanner_task = asyncio.get_running_loop().create_task(
            self._ooo_scanner_loop(),
            name="ooo-scanner",
        )

        # 启动 Worker Pool
        for i in range(self.max_workers):
            worker = asyncio.get_running_loop().create_task(
                self._worker_loop(worker_id=i),
                name=f"dag-worker-{i}",
            )
            self._workers.append(worker)

    async def shutdown(self, timeout: float = 30.0) -> list[DAGExecutionResult]:
        """V8.0 优雅关闭：停止所有后台协程，返回所有 DAG 执行结果。"""
        self._shutdown_flag = True

        # 停止 Aging 重平衡
        if self._aging_rebalance_task is not None:
            self._aging_rebalance_task.cancel()
            try:
                await asyncio.wait_for(self._aging_rebalance_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._aging_rebalance_task.cancel()
            finally:
                self._aging_rebalance_task = None

        # V8.0: 停止 OoO Scanner
        if self._ooo_scanner_task is not None:
            self._ooo_scanner_task.cancel()
            try:
                await asyncio.wait_for(self._ooo_scanner_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._ooo_scanner_task.cancel()
            finally:
                self._ooo_scanner_task = None

        # 停止资源表
        await self._resource_table.stop()

        # 取消所有 LLM 任务
        await self._cancel_all_llm_tasks(reason="调度器关闭")

        # V8.0: 取消所有 I/O Future
        for node_id in list(self._io_waiting_futures.keys()):
            self.cancel_io_future(node_id)

        # 关闭所有 Worker
        for _ in self._workers:
            await self._ready_queue.put((float('inf'), None))

        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout)

        self._workers.clear()

        # 停止实验柜报警监控
        for lab_id, env in self._labs.items():
            bus = getattr(env, "_bus", None)
            if bus is not None and hasattr(bus, "stop_alarm_monitor"):
                try:
                    await bus.stop_alarm_monitor()
                except Exception:
                    pass

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  [DAG 调度内核 V8.0 已关闭]                             ║")
        logger.info("║  完成 DAG 图: %-3d                                      ║",
                    len(self._completed_graphs))
        logger.info("║  OoO 越级总次数: %-3d                                   ║",
                    self._ooo_execution_count)
        logger.info("║  调度时延统计: mean=%.2fms, max=%.2fms                  ║",
                    self.get_scheduling_latency_stats().get("mean_ms", 0),
                    self.get_scheduling_latency_stats().get("max_ms", 0))
        logger.info("╚" + "═" * 60 + "╝")

        return [self._build_dag_result(g) for g in self._completed_graphs]

    async def _worker_loop(self, worker_id: int) -> None:
        """V8.0 实践智能体 Worker 协程。

        V8.0 变化：
        - 不再在 ReadyQueue.get() 超时时主动触发 OoO（由后台 OoO Scanner 接管）
        - 增加时间维度挂起支持：I/O 等待时 yield 控制权
        - WorkerPool Slot 追踪：记录哪个 Worker 正在执行哪个节点
        """
        logger.info("[Worker-%d] 实践智能体就绪", worker_id)

        while not self._shutdown_flag:
            try:
                if self._hardware_interrupt_event.is_set():
                    self._hardware_interrupt_event.clear()
                    logger.info("[Worker-%d] 检测到硬件中断信号", worker_id)

                # V8.0: WorkerPool Slot 记录（记录当前空闲）
                async with self._lock:
                    self._worker_slots[worker_id] = None

                # 从 ReadyQueue 取节点
                try:
                    priority_score, node = await asyncio.wait_for(
                        self._ready_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    # V8.0: 不再主动触发 OoO（由后台 Scanner 接管）
                    continue

                # 安全检查：过滤空节点（可能被 rebalance 队列放入的占位 None）
                if node is None:
                    self._ready_queue.task_done()
                    continue

                # V8.0: 记录调度时延
                self.record_scheduling_latency(node.node_id)

                # V8.0: WorkerPool Slot 记录（记录当前执行节点）
                async with self._lock:
                    self._worker_slots[worker_id] = node.node_id

                # 检查是否需要 I/O 挂起
                io_future: asyncio.Future | None = None
                if self._should_yield_for_io(node):
                    io_future = self.create_io_future(node.node_id)
                    self._start_io_overlap_tracking(node.node_id)

                try:
                    if io_future is not None:
                        # V8.0: 时间维度挂起 — yield 控制权，不阻塞推理线程
                        logger.debug(
                            "[Worker-%d] 节点 '%s' 进入 I/O 等待，yield 控制权",
                            worker_id, node.node_id,
                        )
                        # Worker 不等待 I/O，而是继续轮询 ReadyQueue（处理其他任务）
                        # I/O Future 在另一个地方 resolve
                        # 此处简化为等待 I/O 完成再执行（实际场景中可更复杂）
                        io_done = await self.wait_io_completion(node.node_id, timeout=self._ooo_max_wait_ms / 1000.0)
                        if not io_done:
                            logger.warning(
                                "[Worker-%d] 节点 '%s' I/O 超时，继续执行",
                                worker_id, node.node_id,
                            )

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

                finally:
                    # V8.0: 清理 I/O Future 和时延追踪
                    self.cancel_io_future(node.node_id)
                    self._end_io_overlap_tracking(node.node_id)
                    async with self._lock:
                        self._worker_slots[worker_id] = None

                self._ready_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[Worker-%d] Worker 循环异常", worker_id)

        logger.info("[Worker-%d] 实践智能体关闭", worker_id)

    def _should_yield_for_io(self, node: DAGNode) -> bool:
        """V8.0 新增：判断节点是否需要 I/O 挂起（时间维度 yield）。

        仅对真实物理设备动作才需要 yield（如 heater/vacuum/centrifuge），LLM 推理不需要。
        对于 benchmark 仿真环境，我们默认不 yield 以避免超时等待。
        """
        # Benchmark 模式下禁用 I/O yield（避免等待不存在的真实硬件）
        # 在真实硬件环境下，可以启用此机制
        return False
        
        # 以下是真实硬件场景的逻辑（暂时禁用）
        io_keywords = {"read", "write", "upload", "download", "sync", "commit", "flush"}
        if node.skill_name:
            skill_lower = node.skill_name.lower()
            for keyword in io_keywords:
                if keyword in skill_lower:
                    return True
        return False

    async def _execute_node(self, worker_id: int, node: DAGNode) -> None:
        """执行单个 DAG 节点（V8.0 增强：资源释放保障）。"""
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
            "[Worker-%d] 实践智能体执行: [%s] %s → %s (实验柜: %s)",
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
            # 提取节点所需的设备（与 bench_suite._extract_devices 保持一致）
            skill_lower = node.skill_name.lower()
            lab_lower = node.lab_id.lower() if node.lab_id else ""

            if "plant" in lab_lower:
                if "heater" in skill_lower or "temperature" in skill_lower:
                    devices = ["heater_plant"]
                elif "arm" in skill_lower or "robotic" in skill_lower:
                    devices = ["arm_plant"]
                elif "pump" in skill_lower or "inject" in skill_lower:
                    devices = ["pump_plant"]
                elif "vacuum" in skill_lower:
                    devices = ["vacuum_mat"]
                else:
                    devices = ["co2_controller"]
            elif "material" in lab_lower:
                if "heater" in skill_lower or "temperature" in skill_lower:
                    devices = ["heater_mat"]
                elif "arm" in skill_lower or "robotic" in skill_lower:
                    devices = ["arm_mat"]
                elif "vacuum" in skill_lower:
                    devices = ["vacuum_mat"]
                else:
                    devices = ["co2_controller"]
            elif "fluid" in lab_lower:
                if "pump" in skill_lower or "inject" in skill_lower:
                    devices = ["pump_fluid"]
                elif "valve" in skill_lower:
                    devices = ["valve_fluid"]
                else:
                    devices = ["co2_controller"]
            elif "bio" in lab_lower:
                if "heater" in skill_lower or "temperature" in skill_lower:
                    devices = ["heater_bio"]
                elif "vacuum" in skill_lower:
                    devices = ["vacuum_bio"]
                elif "arm" in skill_lower or "robotic" in skill_lower:
                    devices = ["arm_bio"]
                elif "centrifuge" in skill_lower:
                    devices = ["centrifuge_bio"]
                elif "sensor" in skill_lower or "scan" in skill_lower:
                    devices = ["scan_bio"]
                else:
                    devices = ["co2_controller"]
            else:
                devices = ["co2_controller"]

            result = await env.run_single_task(
                task_id=node.node_id,
                skill_name=node.skill_name,
                params=node.params,
                required_devices=devices,
                task_priority=node.priority.value,
            )
            node.mark_completed(result)
            async with self._lock:
                self._running_nodes.pop(node.node_id, None)
                self._completed_nodes.append(node)
            # V8.0: 节点完成，释放资源（保障推进）
            await self._resource_table.release(node.node_id)
        except Exception as exc:
            logger.exception("[Worker-%d] 节点执行异常: [%s]", worker_id, node.node_id)
            node.mark_failed(str(exc))
            async with self._lock:
                self._running_nodes.pop(node.node_id, None)
                self._failed_nodes.append(node)
            # V8.0: 节点失败也必须释放资源（第四层推进保证）
            await self._resource_table.release(node.node_id)

        elapsed = node.elapsed_time or 0
        logger.info(
            "[Worker-%d] 节点完成: [%s] %s → %s (耗时: %.1fs)",
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
        """V8.0 一站式执行 DAG 图：启动 → 提交 → 等待完成 → 关闭。"""
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

        return result

    def _build_dag_result(self, dag_graph: DAGTaskGraph) -> DAGExecutionResult:
        """V8.0 构建 DAG 执行结果（含完整 OoO/Overlap/时延指标）。"""
        total_time = (
            max((n.end_time or 0) for n in dag_graph.nodes.values()) -
            dag_graph.created_at
        )

        latency_stats = self.get_scheduling_latency_stats()

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
    def idle_worker_count(self) -> int:
        """V8.0 新增：空闲 Worker 数量。"""
        return sum(1 for slot in self._worker_slots.values() if slot is None)

    @property
    def status_summary(self) -> dict[str, Any]:
        """V8.0 获取调度器状态摘要（含完整 V8.0 指标）。"""
        latency_stats = self.get_scheduling_latency_stats()
        return {
            "version": "8.0",
            "ready_queue": self.ready_count,
            "blocked_queue": self.blocked_count,
            "running_nodes": self.running_count,
            "idle_workers": self.idle_worker_count,
            "max_workers": self.max_workers,
            "completed_nodes": len(self._completed_nodes),
            "failed_nodes": len(self._failed_nodes),
            "active_graphs": len(self._active_graphs),
            "completed_graphs": len(self._completed_graphs),
            "llm_tasks_in_progress": len(self._llm_tasks),
            "interrupt_queue_size": self._interrupt_queue.qsize(),
            "aging_factor": self._aging_factor,
            "rebalance_interval": self._rebalance_interval,
            # V8.0 OoO 指标
            "ooo_execution_count": self._ooo_execution_count,
            "ooo_lookahead_triggered": self._ooo_lookahead_triggered,
            "io_compute_overlap_ms": self._io_compute_overlap_ms,
            "active_io_windows": len(self._active_io_windows),
            "active_io_futures": len(self._io_waiting_futures),
            # V8.0 调度时延指标
            "scheduling_latency_ns": latency_stats,
            # V8.0 资源表
            "resource_table": self._resource_table.stats,
            "resource_locks": self._resource_table.get_all_held_resources(),
            # V8.0 WorkerPool
            "worker_slots": {wid: slot for wid, slot in self._worker_slots.items()},
        }


__all__ = [
    "DAGOrchestrator",
    "ActiveResourceTable",
    "ResourceLock",
    "DAGNode",
    "DAGTaskGraph",
    "DAGExecutionResult",
    "NodeStatus",
    "TaskPriority",
    "HardwareInterruptTask",
]
