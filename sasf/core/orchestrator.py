"""
AstroSASF · Core · DAGOrchestrator (V7.0 — Dual-Track DAG Scheduling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
理论/实践双轨调度器。

核心机制：
- 理论智能体（Planner）：调用 LLM 将自然语言解析为 DAG 任务图
- 实践智能体（Worker）：从 ReadyQueue 获取节点，执行 MCP Tool
- DAG 依赖状态机：PENDING → READY → RUNNING → COMPLETED/FAILED
- 双队列机制：ReadyQueue（就绪）| BlockedQueue（阻塞）
- 结算时触发依赖解除，递归检查下游节点

Author: AstroSASF Team
Version: 7.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sasf.core.config_loader import SASFConfig
from sasf.core.environment import LaboratoryEnvironment
from sasf.core.models import (
    DAGExecutionResult,
    DAGNode,
    DAGTaskGraph,
    NodeStatus,
    TaskPriority,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  DAGOrchestrator (V7.0)                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class DAGOrchestrator:
    """理论/实践双轨 DAG 调度器 (V7.0)。

    调度流程：
    1. **提交阶段**：理论智能体生成 DAG 后，无依赖节点入 ReadyQueue，
       其余入 BlockedQueue。
    2. **执行阶段**：实践智能体（Worker）从 ReadyQueue 获取节点执行。
    3. **结算阶段**：节点完成后，触发依赖解除，检查 BlockedQueue，
       将依赖已满足的节点移入 ReadyQueue。

    Parameters
    ----------
    config : SASFConfig
        全局配置
    max_workers : int | None
        并发 Worker 数（默认取 config.orchestrator.max_concurrent_labs）
    """

    config: SASFConfig
    max_workers: int | None = None

    # ── 实验柜注册 ── #
    _labs: dict[str, LaboratoryEnvironment] = field(default_factory=dict, init=False)
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

    def __post_init__(self) -> None:
        if self.max_workers is None:
            self.max_workers = self.config.orchestrator.max_concurrent_labs

    # --------------------------------------------------------------------------- #
    #  实验柜注册                                                                #
    # --------------------------------------------------------------------------- #

    def register_lab(self, env: LaboratoryEnvironment) -> None:
        """注册实验柜环境（供调度器分发任务）。"""
        self._labs[env.lab_id] = env
        logger.info("[DAG调度器] 注册实验柜: %s", env.lab_id)

    def spawn_laboratory(
        self,
        lab_id: str,
        engine: Any,
        tool_registrar: Any = None,
        macro_registrar: Any = None,
        initial_telemetry: dict[str, Any] | None = None,
    ) -> LaboratoryEnvironment:
        """创建并注册实验柜。"""
        env = LaboratoryEnvironment(
            lab_id=lab_id,
            config=self.config,
            engine=engine,
            tool_registrar=tool_registrar,
            macro_registrar=macro_registrar,
            initial_telemetry=initial_telemetry or {},
        )
        self.register_lab(env)
        return env

    # --------------------------------------------------------------------------- #
    #  DAG 提交 (提交阶段)                                                        #
    # --------------------------------------------------------------------------- #

    async def submit_dag(self, dag_graph: DAGTaskGraph) -> str:
        """提交 DAG 图到调度器。

        Parameters
        ----------
        dag_graph : DAGTaskGraph
            理论智能体生成的 DAG 任务图

        Returns
        -------
        str
            DAG 图 ID

        Raises
        ------
        ValueError
            如果 DAG 验证失败（循环依赖等）
        """
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

        # 分类节点：无依赖 → ReadyQueue，有依赖 → BlockedQueue
        await self._classify_and_enqueue_nodes(dag_graph)

        return dag_graph.graph_id

    async def _classify_and_enqueue_nodes(self, dag_graph: DAGTaskGraph) -> None:
        """将 DAG 节点分类并入队。

        - 无依赖（或依赖已完成）的节点 → ReadyQueue
        - 有未完成依赖的节点 → BlockedQueue
        """
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
        """结算完成的节点，解除下游依赖。

        触发事件：检查 BlockedQueue，将依赖已被满足的节点移入 ReadyQueue。
        """
        logger.info(
            "[DAG调度器] 结算节点 '%s' (状态: %s)",
            completed_node.node_id, completed_node.status.name,
        )

        async with self._lock:
            # 遍历阻塞队列，寻找依赖被满足的节点
            still_blocked: list[DAGNode] = []

            for blocked_node in self._blocked_queue:
                if blocked_node.graph_id != dag_graph.graph_id:
                    still_blocked.append(blocked_node)
                    continue

                # 检查所有依赖是否都已完成
                deps_satisfied = all(
                    dag_graph.nodes[dep_id].status == NodeStatus.COMPLETED
                    for dep_id in blocked_node.dependencies
                    if dep_id in dag_graph.nodes
                )

                if deps_satisfied:
                    # 依赖已满足，移入就绪队列
                    await self._enqueue_ready_node(blocked_node)
                    logger.info(
                        "[DAG调度器] 依赖解除: '%s' → '%s' 已就绪",
                        completed_node.node_id, blocked_node.node_id,
                    )
                else:
                    still_blocked.append(blocked_node)

            self._blocked_queue = still_blocked

        # 检查 DAG 是否完成
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

            # 唤醒等待者
            self._dag_complete_event.set()

    # --------------------------------------------------------------------------- #
    #  Worker 生命周期 (执行阶段)                                                 #
    # --------------------------------------------------------------------------- #

    async def start(self) -> None:
        """启动 Worker 协程池。"""
        self._shutdown_flag = False
        self._dag_complete_event = asyncio.Event()

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🚀 DAG 调度内核启动 (V7.0 Dual-Track Scheduling)        ║")
        logger.info("║  Workers: %-3d | ReadyQueue | BlockedQueue               ║", self.max_workers)
        logger.info("║  理论智能体: DAG Planner | 实践智能体: DAG Workers        ║")
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("")

        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"dag-worker-{i}",
            )
            self._workers.append(worker)

    async def shutdown(self, timeout: float = 30.0) -> list[DAGExecutionResult]:
        """优雅关闭调度器，返回所有 DAG 执行结果。"""
        self._shutdown_flag = True

        # 向每个 Worker 发送毒丸
        for _ in self._workers:
            await self._ready_queue.put((float('inf'), None))  # 毒丸

        # 等待 Workers 退出
        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout)

        self._workers.clear()

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
                # 从就绪队列获取节点
                priority_score, node = await asyncio.wait_for(
                    self._ready_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # 调度器关闭时，优雅退出
                break

            # 处理毒丸
            if node is None:
                break

            # ── 执行节点（带异常捕获，防止 Worker 崩溃）── #
            try:
                await self._execute_node(worker_id, node)
            except Exception as exc:
                logger.exception(
                    "[Worker-%d] 节点 [%s] 执行时发生未捕获异常: %s",
                    worker_id, node.node_id, exc,
                )
                # 尝试将节点标记为失败
                try:
                    node.mark_failed(f"Worker异常: {exc}")
                    if node.graph_id and node.graph_id in self._active_graphs:
                        dag_graph = self._active_graphs[node.graph_id]
                        await self._settle_completed_node(node, dag_graph)
                except Exception:
                    pass

            # 标记队列完成
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

        # 标记为运行中
        node.mark_running()
        async with self._lock:
            self._running_nodes[node.node_id] = node

        logger.info("")
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
                suspend_event=None,  # DAG 模式不使用挂起
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

        # ── 结算阶段：触发依赖解除 ── #
        await self._settle_completed_node(node, dag_graph)

    # --------------------------------------------------------------------------- #
    #  便捷方法                                                                  #
    # --------------------------------------------------------------------------- #

    async def run_dag(self, dag_graph: DAGTaskGraph) -> DAGExecutionResult:
        """一站式执行 DAG 图：启动 → 提交 → 等待完成 → 关闭。"""
        await self.start()
        await self.submit_dag(dag_graph)

        # 等待 DAG 完成
        try:
            await asyncio.wait_for(
                self._dag_complete_event.wait(),
                timeout=self.config.orchestrator.dag_execution_timeout or 3600,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[DAG调度器] DAG '%s' 执行超时",
                dag_graph.name,
            )

        results = await self.shutdown()
        return results[0] if results else self._build_dag_result(dag_graph)

    def _build_dag_result(self, dag_graph: DAGTaskGraph) -> DAGExecutionResult:
        """构建 DAG 执行结果。"""
        total_time = (
            max(n.end_time or 0 for n in dag_graph.nodes.values()) -
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
        }

    def get_node_status(self, node_id: str) -> str | None:
        """查询节点状态。"""
        for graph in self._active_graphs.values():
            if node_id in graph.nodes:
                return graph.nodes[node_id].status.name
        for node in self._completed_nodes:
            if node.node_id == node_id:
                return node.status.name
        for node in self._failed_nodes:
            if node.node_id == node_id:
                return node.status.name
        return None

    def get_graph_status(self, graph_id: str) -> dict[str, Any] | None:
        """查询 DAG 图状态。"""
        if graph_id in self._active_graphs:
            return self._active_graphs[graph_id].get_execution_summary()
        for graph in self._completed_graphs:
            if graph.graph_id == graph_id:
                return graph.get_execution_summary()
        return None


__all__ = [
    "DAGOrchestrator",
    "DAGNode",
    "DAGTaskGraph",
    "DAGExecutionResult",
    "NodeStatus",
    "TaskPriority",
]
