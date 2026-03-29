"""
AstroSASF · Core · Models (V7.0 — Dual-Track DAG Scheduling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
理论/实践双轨调度机制的核心数据结构。

核心概念：
- DAGNode: DAG 图中的单个节点，代表一个可执行的任务单元
- DAGTaskGraph: 完整实验的 DAG 图结构
- 状态机: PENDING → READY → RUNNING → COMPLETED/FAILED

Author: AstroSASF Team
Version: 7.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  NodeStatus & Priority                                                     #
# --------------------------------------------------------------------------- #

class NodeStatus(Enum):
    """DAG 节点状态机。"""
    PENDING = auto()    # 等待依赖满足
    READY = auto()      # 已就绪，可执行
    RUNNING = auto()    # 正在执行
    COMPLETED = auto()  # 执行成功
    FAILED = auto()     # 执行失败
    SKIPPED = auto()    # 被跳过（如前置条件不满足）


class TaskPriority(IntEnum):
    """任务优先级（数值越小越优先）。保留 V5.1 接口兼容性。"""
    CRITICAL = 0   # 紧急异常响应
    HIGH = 1       # 核心科学任务
    NORMAL = 2     # 常规任务
    LOW = 3        # 清理/待机


# Aging 机制配置
DEFAULT_AGING_FACTOR: float = 0.1  # 每秒提升的优先级分数（数值越小优先级越高）
DEFAULT_REBALANCE_INTERVAL: float = 5.0  # 队列重平衡周期（秒）
DEFAULT_MAX_AGING_BOOST: float = 10.0  # 单个任务最大 Aging 提升分数


def compute_dynamic_priority(
    base_priority: TaskPriority,
    submit_time: float,
    aging_factor: float = DEFAULT_AGING_FACTOR,
    max_boost: float = DEFAULT_MAX_AGING_BOOST,
) -> float:
    """计算动态优先级（含 Aging 机制）。

    算法：Dynamic_Score = Base_Score - min(Aging_Boost, Max_Boost)
    其中 Aging_Boost = (Current_Time - Submit_Time) * Aging_Factor

    等待越久的任务，Boost 越大，Dynamic_Score 越小，优先级越高。

    Parameters
    ----------
    base_priority : TaskPriority
        静态基础优先级
    submit_time : float
        任务入队时间（time.monotonic）
    aging_factor : float
        Aging 提升速率（默认 0.1，即每秒提升 0.1 分）
    max_boost : float
        最大 Aging 提升上限（默认 10.0）

    Returns
    -------
    float
        动态优先级分数（越小越优先）
    """
    elapsed = time.monotonic() - submit_time
    aging_boost = min(elapsed * aging_factor, max_boost)
    return float(base_priority.value) - aging_boost


# --------------------------------------------------------------------------- #
#  DAGNode                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class DAGNode:
    """DAG 图中的单个节点，代表一个可执行的任务单元。

    Attributes
    ----------
    node_id : str
        节点唯一标识符，通常由 LLM 生成（如 "step1", "prepare_001"）
    skill_name : str
        调用的 MCP Tool 名称（白名单中的工具）
    params : dict[str, Any]
        工具调用参数
    dependencies : list[str]
        依赖的节点 ID 列表。只有当所有依赖节点都 COMPLETED 时，
        此节点才会被移动到 READY 队列。
    status : NodeStatus
        当前节点状态
    priority : TaskPriority
        节点优先级（用于 READY 队列排序）
    description : str
        节点描述（可选，供日志和调试使用）
    submit_time : float
        节点提交到调度器的时间戳
    start_time : float | None
        节点开始执行的时间戳
    end_time : float | None
        节点完成执行的时间戳
    result : dict[str, Any] | None
        工具执行结果
    error_msg : str | None
        错误信息（如果执行失败）
    lab_id : str
        所属实验柜 ID
    graph_id : str | None
        所属 DAG 图 ID（由 add_node 自动绑定）

    Example
    -------
    >>> node = DAGNode(
    ...     node_id="step1",
    ...     skill_name="heater_set_temperature",
    ...     params={"temperature": 37.0, "duration": 60},
    ...     dependencies=[],
    ...     priority=TaskPriority.NORMAL,
    ...     lab_id="bio_lab_01"
    ... )
    """
    node_id: str
    skill_name: str = ""
    graph_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    description: str = ""
    submit_time: float = field(default_factory=time.monotonic)
    start_time: float | None = None
    end_time: float | None = None
    result: dict[str, Any] | None = None
    error_msg: str | None = None
    lab_id: str = ""

    # ── Aging 机制 ── #
    _aging_factor: float = field(default=DEFAULT_AGING_FACTOR, init=False, repr=False)
    _max_aging_boost: float = field(default=DEFAULT_MAX_AGING_BOOST, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = uuid.uuid4().hex[:12]
        if not self.lab_id:
            self.lab_id = "default"

    @property
    def elapsed_time(self) -> float | None:
        """计算节点执行耗时（秒）。"""
        if self.start_time is None:
            return None
        end = self.end_time or time.monotonic()
        return end - self.start_time

    @property
    def dynamic_priority(self) -> float:
        """计算动态优先级（含 Aging）。

        公式：Dynamic_Score = Base_Score - min(Aging_Boost, Max_Boost)
        等待越久的任务，优先级越高。
        """
        return compute_dynamic_priority(
            base_priority=self.priority,
            submit_time=self.submit_time,
            aging_factor=self._aging_factor,
            max_boost=self._max_aging_boost,
        )

    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点（无下游依赖）。"""
        return len(self.dependencies) == 0

    def mark_running(self) -> None:
        """标记为运行中。"""
        self.status = NodeStatus.RUNNING
        self.start_time = time.monotonic()

    def mark_completed(self, result: dict[str, Any] | None = None) -> None:
        """标记为完成。"""
        self.status = NodeStatus.COMPLETED
        self.end_time = time.monotonic()
        if result is not None:
            self.result = result

    def mark_failed(self, error_msg: str) -> None:
        """标记为失败。"""
        self.status = NodeStatus.FAILED
        self.end_time = time.monotonic()
        self.error_msg = error_msg

    def mark_skipped(self) -> None:
        """标记为跳过。"""
        self.status = NodeStatus.SKIPPED
        self.end_time = time.monotonic()

    def get_ready_score(self) -> tuple[float, float]:
        """计算就绪队列优先级得分（使用动态优先级）。

        Returns
        -------
        tuple[float, float]
            (动态优先级分数, 提交时间) 用于 asyncio.PriorityQueue 排序
            动态优先级越低越优先，相同时按提交时间排序
        """
        return (self.dynamic_priority, self.submit_time)

    def set_aging_params(self, factor: float, max_boost: float) -> None:
        """设置 Aging 参数。"""
        self._aging_factor = factor
        self._max_aging_boost = max_boost

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "node_id": self.node_id,
            "skill_name": self.skill_name,
            "params": self.params,
            "dependencies": self.dependencies,
            "status": self.status.name,
            "priority": self.priority.name,
            "description": self.description,
            "submit_time": self.submit_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": self.elapsed_time,
            "result": self.result,
            "error_msg": self.error_msg,
            "lab_id": self.lab_id,
        }


# --------------------------------------------------------------------------- #
#  DAGTaskGraph                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class DAGTaskGraph:
    """完整实验的 DAG 图结构。

    支持拓扑排序、循环依赖检测、依赖链分析等图操作。

    Attributes
    ----------
    graph_id : str
        图的唯一标识符
    name : str
        图的名称（通常是实验名称）
    nodes : dict[str, DAGNode]
        节点 ID → DAGNode 的映射
    created_at : float
        图创建时间戳

    Example
    -------
    >>> graph = DAGTaskGraph(graph_id="exp_001", name="细胞培养实验")
    >>> graph.add_node(DAGNode(node_id="step1", skill_name="incubator_init", ...))
    >>> graph.add_node(DAGNode(node_id="step2", skill_name="heater_set", depends_on=["step1"]))
    >>> graph.validate()  # 检测循环依赖
    True
    """

    graph_id: str
    name: str = ""
    nodes: dict[str, DAGNode] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    _in_degree: dict[str, int] = field(default_factory=dict, init=False)
    _out_degree: dict[str, list[str]] = field(default_factory=dict, init=False)
    _reverse_adj: dict[str, list[str]] = field(default_factory=dict, init=False)
    _validated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"DAG-{self.graph_id[:8]}"

    def add_node(self, node: DAGNode) -> None:
        """添加节点到图中。

        Parameters
        ----------
        node : DAGNode
            要添加的节点

        Raises
        ------
        ValueError
            如果节点 ID 已存在
        """
        if node.node_id in self.nodes:
            raise ValueError(f"节点 ID '{node.node_id}' 已存在于图中")

        node.graph_id = self.graph_id
        self.nodes[node.node_id] = node
        self._validated = False  # 需要重新验证

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """添加边（依赖关系）。

        Parameters
        ----------
        from_node_id : str
            被依赖的节点 ID（前置节点）
        to_node_id : str
            依赖该节点的节点 ID（后继节点）

        Raises
        ------
        KeyError
            如果节点 ID 不存在
        """
        if from_node_id not in self.nodes:
            raise KeyError(f"源节点 '{from_node_id}' 不存在于图中")
        if to_node_id not in self.nodes:
            raise KeyError(f"目标节点 '{to_node_id}' 不存在于图中")

        # to_node 依赖于 from_node，所以 from_node 完成才能执行 to_node
        self.nodes[to_node_id].dependencies.append(from_node_id)
        self._validated = False

    def remove_node(self, node_id: str) -> None:
        """从图中移除节点。

        Parameters
        ----------
        node_id : str
            要移除的节点 ID

        Raises
        ------
        KeyError
            如果节点不存在
        """
        if node_id not in self.nodes:
            raise KeyError(f"节点 '{node_id}' 不存在于图中")

        # 从其他节点的依赖列表中移除
        for node in self.nodes.values():
            if node_id in node.dependencies:
                node.dependencies.remove(node_id)

        del self.nodes[node_id]
        self._validated = False

    def validate(self) -> bool:
        """验证图的合法性。

        检查：
        1. 所有依赖引用的节点是否存在
        2. 图中是否存在循环依赖

        Returns
        -------
        bool
            True 表示合法，False 表示存在错误

        Raises
        ------
        ValueError
            如果检测到非法依赖或循环依赖
        """
        # 检查所有依赖引用是否有效
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(
                        f"节点 '{node_id}' 引用了不存在的依赖节点 '{dep_id}'"
                    )

        # Kahn 算法检测循环依赖
        in_degree = {nid: 0 for nid in self.nodes}
        adj_list: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                # dep_id 完成才能执行 node_id
                adj_list[dep_id].append(node_id)
                in_degree[node_id] += 1

        # 拓扑排序
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        count = 0

        while queue:
            node_id = queue.popleft()
            count += 1
            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        self._validated = True

        if count != len(self.nodes):
            cycle_nodes = [nid for nid in self.nodes if in_degree[nid] > 0]
            raise ValueError(
                f"检测到循环依赖！循环节点: {cycle_nodes}"
            )

        # 缓存计算结果
        self._in_degree = in_degree
        self._out_degree = adj_list
        self._build_reverse_adjacency()

        return True

    def _build_reverse_adjacency(self) -> None:
        """构建反向邻接表（从子节点到父节点）。"""
        self._reverse_adj = {nid: [] for nid in self.nodes}
        for parent_id, children in self._out_degree.items():
            for child_id in children:
                self._reverse_adj[child_id].append(parent_id)

    def get_ready_nodes(self) -> list[DAGNode]:
        """获取所有就绪节点（无依赖或依赖已完成的节点）。

        Returns
        -------
        list[DAGNode]
            就绪节点列表，按优先级排序
        """
        if not self._validated:
            self.validate()

        ready = []
        for node_id, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue

            # 检查所有依赖是否已完成
            all_deps_done = all(
                self.nodes[dep_id].status == NodeStatus.COMPLETED
                for dep_id in node.dependencies
            )

            if all_deps_done:
                node.status = NodeStatus.READY
                ready.append(node)

        # 按优先级排序
        ready.sort(key=lambda n: n.get_ready_score())
        return ready

    def get_blocked_count(self) -> int:
        """获取阻塞节点数量。"""
        return sum(
            1 for node in self.nodes.values()
            if node.status == NodeStatus.PENDING
        )

    def get_completed_count(self) -> int:
        """获取已完成节点数量。"""
        return sum(
            1 for node in self.nodes.values()
            if node.status == NodeStatus.COMPLETED
        )

    def get_failed_count(self) -> int:
        """获取失败节点数量。"""
        return sum(
            1 for node in self.nodes.values()
            if node.status == NodeStatus.FAILED
        )

    def topological_sort(self) -> list[DAGNode]:
        """返回拓扑排序后的节点列表。

        Returns
        -------
        list[DAGNode]
            按拓扑顺序排列的节点列表

        Raises
        ------
        ValueError
            如果图中存在循环依赖
        """
        if not self._validated:
            self.validate()

        result = []
        visited = set()
        temp_mark = set()

        def visit(node_id: str) -> None:
            if node_id in temp_mark:
                raise ValueError(f"循环依赖检测: {node_id}")
            if node_id in visited:
                return

            temp_mark.add(node_id)
            for child_id in self._out_degree.get(node_id, []):
                visit(child_id)
            temp_mark.remove(node_id)
            visited.add(node_id)
            result.append(self.nodes[node_id])

        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def get_execution_levels(self) -> list[list[DAGNode]]:
        """获取执行层级（同一层可并行执行）。

        Returns
        -------
        list[list[DAGNode]]
            按层级分组的节点列表。例如 [[A], [B, C], [D]] 表示
            A 先执行，然后 B 和 C 可并行执行，最后 D 执行。
        """
        if not self._validated:
            self.validate()

        levels: list[list[DAGNode]] = []
        remaining = set(self.nodes.keys())

        while remaining:
            # 找出所有入度为 0 的节点（没有未完成的依赖）
            current_level = []
            for node_id in remaining:
                deps = set(self.nodes[node_id].dependencies)
                ready_deps = deps & (set(self.nodes.keys()) - remaining)
                if deps <= ready_deps:  # 所有依赖都已完成
                    current_level.append(self.nodes[node_id])

            if not current_level:
                raise ValueError("无法继续分层，可能存在循环依赖")

            levels.append(current_level)
            remaining -= {n.node_id for n in current_level}

        return levels

    def is_complete(self) -> bool:
        """检查图是否完全执行完毕（所有节点都完成或失败）。"""
        return all(
            node.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)
            for node in self.nodes.values()
        )

    def has_failures(self) -> bool:
        """检查是否有节点失败。"""
        return any(node.status == NodeStatus.FAILED for node in self.nodes.values())

    def get_execution_summary(self) -> dict[str, Any]:
        """获取执行摘要。"""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "total_nodes": len(self.nodes),
            "completed": self.get_completed_count(),
            "failed": self.get_failed_count(),
            "blocked": self.get_blocked_count(),
            "ready": len([n for n in self.nodes.values() if n.status == NodeStatus.READY]),
            "running": len([n for n in self.nodes.values() if n.status == NodeStatus.RUNNING]),
            "is_complete": self.is_complete(),
            "has_failures": self.has_failures(),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "created_at": self.created_at,
            "execution_summary": self.get_execution_summary(),
        }

    @classmethod
    def from_llm_output(
        cls,
        llm_output: list[dict[str, Any]],
        graph_id: str | None = None,
        lab_id: str = "",
        default_priority: TaskPriority = TaskPriority.NORMAL,
    ) -> DAGTaskGraph:
        """从 LLM 输出构建 DAG 图。

        Parameters
        ----------
        llm_output : list[dict[str, Any]]
            LLM 输出的任务列表，格式：
            [
                {"id": "step1", "skill": "tool_name", "params": {...}, "depends_on": []},
                {"id": "step2", "skill": "tool_name", "params": {...}, "depends_on": ["step1"]},
                ...
            ]
        graph_id : str | None
            图 ID，默认自动生成
        lab_id : str
            实验柜 ID
        default_priority : TaskPriority
            默认优先级

        Returns
        -------
        DAGTaskGraph
            构建好的 DAG 图

        Raises
        ------
        ValueError
            如果 LLM 输出格式不正确

        Example
        -------
        >>> llm_output = [
        ...     {"id": "step1", "skill": "incubator_init", "params": {}, "depends_on": []},
        ...     {"id": "step2", "skill": "heater_set", "params": {"temp": 37}, "depends_on": ["step1"]},
        ... ]
        >>> graph = DAGTaskGraph.from_llm_output(llm_output, lab_id="bio_lab_01")
        >>> graph.validate()
        True
        """
        if not llm_output:
            raise ValueError("LLM 输出为空")

        if not isinstance(llm_output, list):
            raise ValueError(f"LLM 输出必须是列表，实际类型: {type(llm_output)}")

        graph_id = graph_id or uuid.uuid4().hex[:12]

        graph = cls(graph_id=graph_id, name=f"DAG-Graph-{graph_id}")

        # 第一遍：创建所有节点
        node_ids = set()
        for item in llm_output:
            if not isinstance(item, dict):
                raise ValueError(f"任务项必须是字典，实际类型: {type(item)}")

            node_id = item.get("id")
            if not node_id:
                node_id = f"node_{uuid.uuid4().hex[:8]}"
                logger.warning("LLM 输出缺少 'id' 字段，自动生成: %s", node_id)

            if node_id in node_ids:
                raise ValueError(f"发现重复的节点 ID: {node_id}")

            node_ids.add(node_id)

            node = DAGNode(
                node_id=node_id,
                skill_name=item.get("skill", ""),
                params=item.get("params", {}),
                dependencies=list(item.get("depends_on", [])),
                priority=default_priority,
                description=item.get("description", ""),
                lab_id=lab_id,
            )
            graph.add_node(node)

        # 验证所有依赖引用存在
        for node in graph.nodes.values():
            for dep_id in node.dependencies:
                if dep_id not in node_ids:
                    raise ValueError(
                        f"节点 '{node.node_id}' 引用了不存在的依赖: '{dep_id}'. "
                        f"可用节点: {list(node_ids)}"
                    )

        # 验证无循环依赖
        graph.validate()

        return graph


# --------------------------------------------------------------------------- #
#  DAGExecutionResult                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class DAGExecutionResult:
    """DAG 执行结果摘要。"""
    graph_id: str
    status: str  # "completed", "failed", "partial"
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    total_time: float
    execution_levels: int
    node_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "status": self.status,
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "total_time": self.total_time,
            "execution_levels": self.execution_levels,
            "node_results": self.node_results,
        }


# --------------------------------------------------------------------------- #
#  HardwareInterruptTask (硬实时逃生任务)                                       #
# --------------------------------------------------------------------------- #

@dataclass
class HardwareInterruptTask:
    """硬件级抢占任务 —— 来自 TelemetryBus 的紧急逃生指令。

    当 TelemetryBus 检测到危险条件时，会绕过 LLM 规划层，
    直接向调度器注入此类型的最高优先级任务。

    Attributes
    ----------
    interrupt_id : str
        中断唯一标识符
    description : str
        人类可读的中断描述（如 "温度超限：82℃ > 80℃"）
    action_skill : str
        逃生动作对应的 MCP Tool 名称
    action_params : dict[str, Any]
        逃生动作参数
    lab_id : str
        目标实验柜 ID
    source_condition : str
        触发的原始条件表达式
    timestamp : float
        中断触发时间戳
    """
    interrupt_id: str
    description: str
    action_skill: str
    action_params: dict[str, Any] = field(default_factory=dict)
    lab_id: str = "default"
    source_condition: str = ""
    timestamp: float = field(default_factory=time.monotonic)

    def to_dag_node(self, priority: TaskPriority = TaskPriority.CRITICAL) -> DAGNode:
        """转换为 DAGNode 用于调度执行。"""
        return DAGNode(
            node_id=f"INT-{self.interrupt_id}",
            skill_name=self.action_skill,
            params=self.action_params,
            dependencies=[],
            priority=priority,
            description=f"[HARDWARE INTERRUPT] {self.description}",
            lab_id=self.lab_id,
        )


@dataclass
class HardwareAlarm:
    """硬件报警注册条目。"""
    alarm_id: str
    condition_expr: str                    # 布尔表达式，如 "temperature >= 80"
    interrupt_action_skill: str             # 触发时执行的 MCP Tool
    interrupt_action_params: dict[str, Any] # 触发时执行的参数
    severity: TaskPriority = TaskPriority.CRITICAL
    enabled: bool = True
    trigger_count: int = field(default=0, init=False)
    last_trigger_time: float | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return hash(self.alarm_id)


# --------------------------------------------------------------------------- #
#  LLM Prompt Templates (for DAG Generation)                                  #
# --------------------------------------------------------------------------- #

DAG_PLANNER_PROMPT_TEMPLATE = '''你是太空实验柜的**理论智能体 (Planner)**，负责将用户的自然语言任务解析为**有向无环图 (DAG)** 结构。

## 核心任务
将用户的任务指令拆解为多个 MCP Tool 原子调用，并明确标注它们之间的**前后依赖关系**。

## ⚠️ 严格规则 (违反将导致系统崩溃)

1. **只允许使用以下 MCP Tools** —— 这是完整的白名单，不存在其他工具：
{tools_whitelist}

2. **绝对禁止**创造不在上述列表中的工具名！
   - ❌ 禁止输出 SOP/Skill 的名称作为 tool_name
   - ❌ 禁止编造任何不在白名单中的工具
   - ✅ 每个步骤的 "skill" 字段必须精确匹配白名单中的名称

3. **依赖关系规则**：
   - `depends_on` 为空数组 `[]` 表示该步骤无前置依赖，可以立即执行
   - `depends_on` 中列出的节点 ID 必须在**之前执行完成**才能执行当前步骤
   - 多个 `depends_on` 表示**AND 关系**（所有依赖都完成后才执行）

4. **如果用户提到某个 SOP**，你必须：
   - 阅读下方 SOP 文档理解操作流程
   - 将 SOP 中的每一步翻译为白名单中的底层 MCP Tool 调用
   - 标注正确的依赖关系

## 可用 MCP Tools 详细说明
{tools_description}

## SOP 上下文 (如有)
{skills_context}

## 输出格式 (严格 JSON，不允许任何其他文字)

```json
[
  {{
    "id": "唯一标识符（如 step1, prepare_001）",
    "skill": "白名单中的工具名称",
    "params": {{工具调用参数}},
    "depends_on": ["前置节点ID列表"],
    "description": "可选的步骤描述"
  }},
  ...
]
```

### 重要约束
- `id` 字段在整个列表中必须**唯一**
- `depends_on` 中引用的 ID 必须在数组中出现
- **禁止创建循环依赖**（A 依赖 B，B 依赖 A）
- 如果步骤之间没有依赖，设置为 `depends_on: []`

## 用户指令
{task}

## 请严格按上述格式输出 JSON：'''

DAG_VALIDATOR_PROMPT = '''你是 DAG 验证智能体。请检查以下任务计划是否存在问题。

## 任务计划
{plan_json}

## 可用工具白名单
{tools_whitelist}

## 检查项目
1. 所有 `skill` 是否都在白名单中？
2. 所有 `depends_on` 引用是否存在？
3. 是否存在循环依赖？
4. `id` 是否唯一？
5. `params` 是否符合工具的 JSON Schema？

请返回检查结果：
- 如果全部通过：返回 `{{"valid": true, "issues": []}}`
- 如果有问题：返回 `{{"valid": false, "issues": ["问题1", "问题2", ...]}}`'''
