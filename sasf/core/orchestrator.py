"""
AstroSASF · Core · Orchestrator (V5.1 — Priority Scheduling Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于优先级队列的抢占式任务调度器。

核心机制：
- ``TaskPriority`` 四级优先级（CRITICAL/HIGH/NORMAL/LOW）
- ``asyncio.PriorityQueue`` 全局调度池
- Worker 协程池从队列取任务分发执行
- CRITICAL 任务入队时自动挂起低优先级运行中任务
- 挂起通过 ``asyncio.Event`` 实现：每步执行前 ``await event.wait()``
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from sasf.core.config_loader import SASFConfig
from sasf.core.environment import LaboratoryEnvironment

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  TaskPriority                                                                #
# --------------------------------------------------------------------------- #

class TaskPriority(IntEnum):
    """任务优先级（数值越小越优先）。"""
    CRITICAL = 0   # 紧急异常响应
    HIGH = 1       # 核心科学任务
    NORMAL = 2     # 常规任务
    LOW = 3        # 清理/待机


# --------------------------------------------------------------------------- #
#  ScheduledTask                                                               #
# --------------------------------------------------------------------------- #

@dataclass(order=False)
class ScheduledTask:
    """调度任务描述符。"""
    priority: TaskPriority
    task_id: str
    lab_id: str
    description: str
    submit_time: float = field(default_factory=time.monotonic)

    # 运行时状态
    status: str = "pending"           # pending / running / suspended / completed / error
    result: dict[str, Any] | None = field(default=None, repr=False)
    _suspend_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _future: asyncio.Future | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._suspend_event.set()     # 默认放行

    # PriorityQueue 需要可比较
    def __lt__(self, other: ScheduledTask) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.submit_time < other.submit_time

    def suspend(self) -> None:
        """挂起任务（清除 Event → 下一个 await 点阻塞）。"""
        if self.status == "running":
            self._suspend_event.clear()
            self.status = "suspended"
            logger.warning(
                "⏸️  [调度器] 任务挂起: [%s] %s (优先级: %s)",
                self.task_id[:8], self.description, self.priority.name,
            )

    def resume(self) -> None:
        """恢复任务（设置 Event → 阻塞的 await 放行）。"""
        if self.status == "suspended":
            self._suspend_event.set()
            self.status = "running"
            logger.info(
                "▶️  [调度器] 任务恢复: [%s] %s (优先级: %s)",
                self.task_id[:8], self.description, self.priority.name,
            )

    async def wait_if_suspended(self) -> None:
        """挂起检查点 — Environment 每步执行前调用。"""
        await self._suspend_event.wait()


# --------------------------------------------------------------------------- #
#  Orchestrator (V5.1)                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class Orchestrator:
    """优先级抢占式任务调度器 (V5.1)。

    Parameters
    ----------
    config : SASFConfig
    max_workers : int
        并发 Worker 数（默认取 ``config.orchestrator.max_concurrent_labs``）。
    """

    config: SASFConfig
    max_workers: int | None = None

    _labs: dict[str, LaboratoryEnvironment] = field(default_factory=dict, init=False)
    _queue: asyncio.PriorityQueue = field(default_factory=asyncio.PriorityQueue, init=False)
    _running_tasks: dict[str, ScheduledTask] = field(default_factory=dict, init=False)
    _completed_tasks: list[ScheduledTask] = field(default_factory=list, init=False)
    _workers: list[asyncio.Task] = field(default_factory=list, init=False)
    _shutdown_flag: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.max_workers is None:
            self.max_workers = self.config.orchestrator.max_concurrent_labs

    # ---- 实验柜注册 -------------------------------------------------------- #

    def register_lab(self, env: LaboratoryEnvironment) -> None:
        """注册实验柜环境（供调度器分发任务）。"""
        self._labs[env.lab_id] = env
        logger.info("[调度器] 注册实验柜: %s", env.lab_id)

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

    # ---- 任务提交 ---------------------------------------------------------- #

    async def submit_task(
        self,
        lab_id: str,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: str | None = None,
    ) -> ScheduledTask:
        """向调度队列提交任务。

        CRITICAL 任务入队时自动触发抢占。
        """
        if lab_id not in self._labs:
            raise ValueError(f"[调度器] 未注册的实验柜: '{lab_id}'")

        task = ScheduledTask(
            priority=priority,
            task_id=task_id or uuid.uuid4().hex[:12],
            lab_id=lab_id,
            description=description,
        )

        logger.info(
            "📥 [调度器] 提交任务: [%s] %s → %s (优先级: %s)",
            task.task_id[:8], description, lab_id, priority.name,
        )

        # ── 抢占逻辑：CRITICAL 任务挂起所有低优先级运行中任务 ── #
        if priority == TaskPriority.CRITICAL:
            self._preempt_lower_priority(priority)

        await self._queue.put(task)
        return task

    def _preempt_lower_priority(self, incoming_priority: TaskPriority) -> None:
        """挂起所有优先级低于 incoming_priority 的运行中任务。"""
        suspended_count = 0
        for tid, task in self._running_tasks.items():
            if task.priority > incoming_priority and task.status == "running":
                task.suspend()
                suspended_count += 1

        if suspended_count > 0:
            logger.warning(
                "🚨 [调度器] 抢占! %d 个低优先级任务已挂起 (因 %s 任务入队)",
                suspended_count, incoming_priority.name,
            )

    def _resume_suspended_tasks(self, completed_priority: TaskPriority) -> None:
        """CRITICAL 任务完成后，恢复被其挂起的任务。"""
        resumed_count = 0
        for tid, task in self._running_tasks.items():
            if task.status == "suspended":
                task.resume()
                resumed_count += 1

        if resumed_count > 0:
            logger.info(
                "✅ [调度器] %s 任务完成，%d 个挂起任务已恢复",
                completed_priority.name, resumed_count,
            )

    # ---- Worker 生命周期 --------------------------------------------------- #

    async def start(self) -> None:
        """启动 Worker 协程池。"""
        self._shutdown_flag = False
        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🚀 调度内核启动 (V5.1 Priority Scheduling Kernel)       ║")
        logger.info("║  Workers: %-3d | 队列: PriorityQueue                      ║", self.max_workers)
        logger.info("║  优先级: CRITICAL > HIGH > NORMAL > LOW                   ║")
        logger.info("╚" + "═" * 60 + "╝")
        logger.info("")

        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"scheduler-worker-{i}",
            )
            self._workers.append(worker)

    async def shutdown(self, timeout: float = 30.0) -> list[ScheduledTask]:
        """优雅关闭调度器，返回所有已完成任务。"""
        self._shutdown_flag = True

        # 向每个 Worker 发送毒丸
        for _ in self._workers:
            await self._queue.put(None)  # type: ignore

        # 等待 Workers 退出
        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout)

        self._workers.clear()

        logger.info("")
        logger.info("╔" + "═" * 60 + "╗")
        logger.info("║  🛑 调度内核已关闭                                        ║")
        logger.info("║  已完成任务: %-3d                                          ║", len(self._completed_tasks))
        logger.info("╚" + "═" * 60 + "╝")

        return list(self._completed_tasks)

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker 协程 — 从队列取任务执行。"""
        logger.info("[Worker-%d] 就绪", worker_id)

        while not self._shutdown_flag:
            item = await self._queue.get()
            if item is None:
                break  # 毒丸 → 退出

            task: ScheduledTask = item
            task.status = "running"
            self._running_tasks[task.task_id] = task

            logger.info("")
            logger.info(
                "⚡ [Worker-%d] 开始执行: [%s] %s (优先级: %s, 实验柜: %s)",
                worker_id, task.task_id[:8], task.description,
                task.priority.name, task.lab_id,
            )

            env = self._labs.get(task.lab_id)
            if env is None:
                task.status = "error"
                task.result = {"status": "error", "detail": f"实验柜 '{task.lab_id}' 不存在"}
            else:
                try:
                    result = await env.run_single_task(
                        task_description=task.description,
                        suspend_event=task._suspend_event,
                    )
                    task.result = result
                    task.status = "completed"
                except Exception as exc:
                    logger.exception(
                        "❌ [Worker-%d] 任务异常: [%s] %s",
                        worker_id, task.task_id[:8], exc,
                    )
                    task.status = "error"
                    task.result = {"status": "error", "detail": str(exc)}

            # ── 任务完成后处理 ── #
            self._running_tasks.pop(task.task_id, None)
            self._completed_tasks.append(task)

            elapsed = time.monotonic() - task.submit_time
            logger.info(
                "✅ [Worker-%d] 任务完成: [%s] %s → %s (耗时: %.1fs)",
                worker_id, task.task_id[:8], task.description,
                task.status, elapsed,
            )

            # CRITICAL 完成后恢复被挂起的任务
            if task.priority == TaskPriority.CRITICAL:
                self._resume_suspended_tasks(task.priority)

            self._queue.task_done()

    # ---- 便捷方法 ---------------------------------------------------------- #

    async def run_all(self) -> list[dict[str, Any]]:
        """兼容 V5 接口：启动调度器 → 等待队列清空 → 关闭。"""
        await self.start()
        await self._queue.join()
        completed = await self.shutdown()
        return [t.result or {} for t in completed]

    @property
    def lab_ids(self) -> list[str]:
        return list(self._labs.keys())

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def running_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": t.task_id,
                "priority": t.priority.name,
                "lab_id": t.lab_id,
                "description": t.description,
                "status": t.status,
            }
            for t in self._running_tasks.values()
        ]
