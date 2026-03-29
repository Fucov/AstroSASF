# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · Edge-RAG + **硬件级抢占** + **动态优先级 Aging** + 正交联锁 + Guard + Macro + **DAG 双轨调度**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Fucov/AstroSASF)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Fucov/AstroSASF)
---

## 项目概述

AstroSASF 是面向空间站科学实验柜的**多智能体协作调度框架**。核心矛盾：大模型推理的 _"概率性/高延迟"_ 与物理硬件控制的 _"确定性/硬实时"_ 之间的冲突。

### V7.1 核心设计

> **理论/实践双轨调度** + **DAG 依赖图** + **LLM 语义路由** + **硬件级抢占** + **动态优先级 Aging** + **正交联锁** + **Guard** + **Macro**。

| 概念 | 层级 | 本质 | 管理者 |
|------|------|------|--------|
| **理论智能体 (Planner)** | 认知层 | LLM 将自然语言解析为 DAG 任务图 | `graph_builder.py` |
| **实践智能体 (Worker)** | 执行层 | 从 ReadyQueue 取节点，执行 MCP Tool | `orchestrator.py` |
| **DAGNode** | 核心层 | 带依赖关系的可执行任务单元 | `models.py` |
| **MCP Tools** | 中间件层 | 底层原子操作接口 + **Guard 声明式安全守卫** | `middleware/mcp_registry.py` |
| **TelemetryBus** | 物理层 | 遥测数据存储 + **硬件报警监控** | `physics/telemetry_bus.py` |
| **Hardware Interrupt** | 核心层 | 越过 LLM 层，直接注入 CRITICAL 逃生任务 | `orchestrator.py` |

---

## 核心特性

| 能力 | 模块 | 描述 |
|------|------|------|
| **DAG 双轨调度** | `orchestrator.py` | 理论智能体生成 DAG → 实践智能体执行 |
| **依赖状态机** | `models.py` | PENDING → READY → RUNNING → COMPLETED/FAILED |
| **Ready/Blocked 队列** | `orchestrator.py` | 无依赖入 ReadyQueue，有依赖入 BlockedQueue |
| **结算依赖解除** | `orchestrator.py` | 节点完成后递归检查下游，移入 ReadyQueue |
| **循环依赖检测** | `models.py` | Kahn 算法入度检测，拒绝非法 DAG |
| **语义路由** | `graph_builder.py` | LLM router_node 意图分析，动态选择 SOP |
| **多领域知识库** | `skills_catalog/` | 流体实验 / 生物培养 / 材料合成 |
| **优先级调度** | `orchestrator.py` | PriorityQueue + Worker 池 + CRITICAL 抢占 |
| **Guard 装饰器** | `mcp_registry.py` | `@mcp_tool(forbid_states=..., telemetry_rules=...)` |
| **Macro 绑定** | `mcp_registry.py` | `bind_macro("heat_50", "set_temperature", {"target": 50})` |
| **正交联锁引擎** | `interlock_engine.py` | 子系统独立状态 + `ast` 安全求值 |
| **HITL** | 应用层注入 | `graph.compile(checkpointer=MemorySaver())` |
| **硬件级抢占 (V7.1)** | `telemetry_bus.py` | TelemetryBus 监测危险条件，注入 CRITICAL 逃生任务 |
| **LLM Task Cancel (V7.1)** | `orchestrator.py` | 硬件中断时强行 Cancel 正在进行的 LLM 推理 |
| **动态优先级 Aging (V7.1)** | `models.py` | 防止低优任务饿死，队列周期性重平衡 |

---

## V7.1 硬件级抢占机制

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│  TelemetryBus (V5.1) — 1553B 总线模拟                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  register_alarm(condition="temperature >= 80",                    ││
│  │                 action="emergency_cooling")                      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                          │
│                              ▼ 报警触发回调                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Orchestrator._handle_hardware_interrupt()                        ││
│  │  1. Cancel 所有 LLM Task (asyncio.Task.cancel())                ││
│  │  2. 挂起当前运行中的 Worker 任务 (mark_skipped)                   ││
│  │  3. 注入 CRITICAL 逃生任务到 ReadyQueue                         ││
│  │  4. 唤醒 Worker 重新选择                                         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 动态优先级 Aging 算法

```python
# 公式：Dynamic_Score = Base_Score - min(Aging_Boost, Max_Boost)
# 其中 Aging_Boost = (Current_Time - Submit_Time) * Aging_Factor

def compute_dynamic_priority(base_priority, submit_time, aging_factor=0.1, max_boost=10.0):
    elapsed = time.monotonic() - submit_time
    aging_boost = min(elapsed * aging_factor, max_boost)
    return float(base_priority.value) - aging_boost
```

- **等待越久** → Aging_Boost 越大 → Dynamic_Score 越小 → **优先级越高**
- 防止低优先级任务长期处于就绪队列导致"饿死"
- 后台协程每 5 秒对 PriorityQueue 进行重排序

### 使用示例

```python
# 注册硬件报警
orchestrator.register_lab_hardware_alarm(
    lab_id="Lab-Alpha",
    alarm_id="temp_overheat",
    condition_expr="temperature >= 80",
    interrupt_action_skill="emergency_cooling",
    interrupt_action_params={"mode": "rapid", "target": 25.0},
    severity=TaskPriority.CRITICAL,
)

# 报警触发时，系统自动：
# 1. Cancel 所有 LLM 推理
# 2. 挂起当前任务
# 3. 注入 emergency_cooling 逃生任务（CRITICAL 优先级）
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  Core Layer — DAGOrchestrator (V7.1 HW Preemption Scheduler)        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  理论智能体 (Planner)                                            │ │
│  │  LLM → 自然语言 → DAG 任务图 (带依赖关系)                         │ │
│  │  ↑ LLM Task 可被硬件中断 Cancel                                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  实践智能体 (Workers)                                            │ │
│  │  ReadyQueue ← [节点A, 节点B] → 执行 → 结算 → 依赖解除            │ │
│  │                          ↑                                      │ │
│  │                    BlockedQueue                                  │ │
│  │                    [节点C(依赖A), 节点D(依赖A,B)]                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  硬件抢占层 (V7.1)                                               │ │
│  │  TelemetryBus → 报警监控协程 → 回调 → Orchestrator              │ │
│  │  HardwareInterruptTask → CRITICAL 逃生任务注入                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ LaboratoryEnvironment (suspend_event checkpoint)                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  (config.yaml 驱动 · Headless / HITL 可选)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Cognition Layer                                                    │
│  ┌──────────────────┐  ┌───────────────────────────────────────┐   │
│  │ SkillLoader (V5)  │  │ LangGraph StateGraph + LLM             │   │
│  │ Macro-aware SOP   │  │ dag_planner_node (V7.0 DAG Generator) │   │
│  └──────────────────┘  └───────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  Middleware Layer ★ 核心资产                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ MCPToolRegistry ← @mcp_tool(Guard) + Macro                     │ │
│  │ SpaceMCPCodec(自动握手) · SpaceWire · Gateway                   │ │
│  │ A2ARouter (Pub/Sub)                                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  Physics Layer — InterlockEngine + TelemetryBus (V5.1)              │
│  (正交子系统状态 · 跨系统联锁规则 · ast 安全求值 · 硬件报警监控)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```text
AstroSASF/
├── config.yaml                     # LLM + 中间件配置
├── fsm_rules.yaml                  # 联锁规则（正交子系统 + 安全约束）
├── sasf/                           # 【核心框架包 — 零业务词汇】
│   ├── core/
│   │   ├── config_loader.py        # YAML 解析 + LLM 工厂
│   │   ├── models.py              # ★ V7.0 DAG 数据结构
│   │   ├── orchestrator.py        # ★ V7.0 DAG 双轨调度器
│   │   └── environment.py          # 5 步装配器 (Headless)
│   ├── middleware/
│   │   ├── mcp_registry.py         # @mcp_tool(Guard) + bind_macro()
│   │   ├── a2a_protocol.py         # A2A Pub/Sub 消息路由
│   │   ├── codec.py               # 动态字典 Space-MCP 编解码
│   │   ├── gateway.py              # 协议转换网关
│   │   └── virtual_bus.py         # SpaceWire 总线模拟
│   ├── cognition/
│   │   ├── state.py                # LangGraph TypedDict (V7.0)
│   │   ├── graph_builder.py        # ★ V7.0 DAG Planner
│   │   └── skill_loader.py         # SKILL.md SOP + Macro 感知
│   └── physics/
│       ├── interlock_engine.py     # 正交联锁引擎 (替代 FSM)
│       └── telemetry_bus.py        # 遥测数据总线
├── skills_catalog/
│   ├── fluid_experiment/
│   │   └── SKILL.md
│   ├── bio_culture/
│   │   └── SKILL.md
│   └── material_synthesis/
│       └── SKILL.md
└── examples/
    └── space_station_demo.py       # V7.0 全链路演示
```

---

## 快速开始

```bash
ollama serve && ollama pull qwen2.5:7b
pip install -r requirements.txt
python examples/space_station_demo.py
```

---

## V7.0 核心机制：理论/实践双轨调度

### 概述

V7.0 引入**双轨智能体概念**，将任务规划与任务执行解耦：

| 轨道 | 智能体 | 职责 | 位置 |
|------|--------|------|------|
| **理论轨道** | Planner | LLM 生成 DAG 任务图 | `graph_builder.py` |
| **实践轨道** | Worker | 从队列取节点，执行 MCP Tool | `orchestrator.py` |

### DAG 状态机

```
PENDING ──┬── 依赖未满足 ──→ 保持在 BlockedQueue
          │
          └── 依赖已满足 ──→ READY ──→ 入 ReadyQueue
                                              │
READY ─────────────────────────────────────────┤
    │                                         │
    ▼                                         ▼
RUNNING ←── Worker 取节点执行            COMPLETED
    │                                         │
    ├─── 执行成功 ──→ mark_completed() ──────┘
    │                     │
    │                     └──→ _settle_completed_node()
    │                              │
    └─── 执行失败 ──→ mark_failed() ──→ SKIPPED (下游节点)
```

### 三阶段调度流程

#### 1. 提交阶段 (Submit Phase)

```
理论智能体 (Planner) 解析任务 → 生成 DAG 任务图
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ DAGTaskGraph.validate() │
                              │  循环依赖检测 (Kahn)     │
                              └─────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
           depends_on == []                         depends_on != []
                    │                                         │
                    ▼                                         ▼
           入 ReadyQueue                            入 BlockedQueue
```

#### 2. 执行阶段 (Execution Phase)

```
Worker 协程池
     │
     ├── Worker-0: 从 ReadyQueue 取节点 → 执行 → mark_completed()
     ├── Worker-1: 从 ReadyQueue 取节点 → 执行 → mark_completed()
     └── Worker-N: ...
```

#### 3. 结算阶段 (Settlement Phase)

```
节点 A 完成 → _settle_completed_node(A)
                    │
                    ▼
           遍历 BlockedQueue
                    │
                    ├── 节点 B (依赖 [A]) → A 已完成 → 移入 ReadyQueue
                    ├── 节点 C (依赖 [A, D]) → D 未完成 → 保持 Blocked
                    └── 节点 D (依赖 []) → 无依赖 → 移入 ReadyQueue
```

### 核心数据结构

#### DAGNode

```python
@dataclass
class DAGNode:
    node_id: str              # 节点唯一标识
    skill_name: str          # MCP Tool 名称
    params: dict             # 工具参数
    dependencies: list[str]   # 依赖节点 ID 列表
    status: NodeStatus       # PENDING/READY/RUNNING/COMPLETED/FAILED
    priority: TaskPriority    # CRITICAL/HIGH/NORMAL/LOW
    graph_id: str            # 所属 DAG 图 ID
    lab_id: str              # 实验柜 ID
```

#### DAGTaskGraph

```python
@dataclass
class DAGTaskGraph:
    graph_id: str
    name: str
    nodes: dict[str, DAGNode]  # 节点映射

    def validate(self) -> bool:
        """Kahn 算法检测循环依赖"""

    def get_ready_nodes(self) -> list[DAGNode]:
        """获取所有就绪节点（依赖已满足）"""

    def topological_sort(self) -> list[DAGNode]:
        """拓扑排序"""

    def get_execution_levels(self) -> list[list[DAGNode]]:
        """获取执行层级（同一层可并行）"""
```

### LLM DAG 生成 Prompt

```python
DAG_PLANNER_PROMPT_TEMPLATE = '''你是太空实验柜的**理论智能体 (Planner)**，
负责将用户的自然语言任务解析为**有向无环图 (DAG)** 结构。

## 核心任务
将用户的任务指令拆解为多个 MCP Tool 原子调用，
并明确标注它们之间的**前后依赖关系**。

## 输出格式
[
  {
    "id": "唯一标识符",
    "skill": "白名单中的工具名称",
    "params": {工具调用参数},
    "depends_on": ["前置节点ID列表"],
    "description": "步骤描述"
  },
  ...
]

## 重要约束
- `depends_on` 为空表示无前置依赖
- 多个依赖表示 AND 关系（全部完成后才执行）
- **禁止创建循环依赖**
'''
```

### 并发锁机制

```python
async def _settle_completed_node(self, completed_node, dag_graph):
    """结算完成的节点，解除下游依赖"""
    async with self._lock:  # ★ 关键：并发安全
        still_blocked = []
        for blocked_node in self._blocked_queue:
            deps_satisfied = all(
                dag_graph.nodes[dep_id].status == NodeStatus.COMPLETED
                for dep_id in blocked_node.dependencies
            )
            if deps_satisfied:
                await self._enqueue_ready_node(blocked_node)
            else:
                still_blocked.append(blocked_node)
        self._blocked_queue = still_blocked
```

### 循环依赖检测

```python
def validate(self) -> bool:
    """Kahn 算法检测循环依赖"""
    in_degree = {nid: 0 for nid in self.nodes}
    adj_list = {nid: [] for nid in self.nodes}

    for node_id, node in self.nodes.items():
        for dep_id in node.dependencies:
            adj_list[dep_id].append(node_id)
            in_degree[node_id] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    count = 0

    while queue:
        node_id = queue.popleft()
        count += 1
        for neighbor in adj_list[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if count != len(self.nodes):
        cycle_nodes = [nid for nid in self.nodes if in_degree[nid] > 0]
        raise ValueError(f"检测到循环依赖！循环节点: {cycle_nodes}")
```

### DAG 执行示例

```python
from sasf.core.orchestrator import DAGOrchestrator
from sasf.core.models import DAGTaskGraph, DAGNode, TaskPriority

# 初始化 DAG 调度器
orchestrator = DAGOrchestrator(config=config, max_workers=4)
await orchestrator.start()

# 理论智能体生成 DAG
dag_plan = [
    {"id": "init", "skill": "incubator_init", "params": {}, "depends_on": []},
    {"id": "heat", "skill": "set_temperature", "params": {"target": 37}, "depends_on": ["init"]},
    {"id": "observe", "skill": "camera_capture", "params": {}, "depends_on": ["init"]},
    {"id": "analyze", "skill": "ml_analyze", "params": {}, "depends_on": ["heat", "observe"]},
]

# 构建 DAG 图
dag_graph = DAGTaskGraph.from_llm_output(dag_plan, lab_id="bio_lab_01")

# 提交执行
await orchestrator.submit_dag(dag_graph)

# 等待完成
result = await orchestrator.run_dag(dag_graph)
print(f"DAG 状态: {result.status}")
print(f"完成节点: {result.completed_nodes}/{result.total_nodes}")
```

---

## V6.2 核心机制（向后兼容）

### LLM 语义路由 (`cognition/graph_builder.py`)

**问题**：BM25 等传统 NLP 算法无法处理中文语义，边缘场景下评分全为 0。

**方案**：用 LLM 本身做语义路由 — 新增 `router_node` 节点，轻量 Prompt 选择最匹配的 SOP。

```
LangGraph V6.2 节点流:
  router_node → planner_node → operator_node ⇄ execute_node → END
```

#### Router 工作流

```
1. router_node 收到任务 "开始进行太空生物细胞培养"
2. 构造 Router Prompt：列出所有 SOP name + description
3. LLM 输出: "bio_culture" (仅一个词)
4. 存入 State: selected_skill = "bio_culture"
5. planner_node 读取 → 向 catalog 获取完整 SOP → 注入 Prompt
```

#### 异常容错

- JSON 提取失败 → `plan=[]`, `error_msg=LLM原文`, `final_result.status="failed"`
- LLM 拒绝执行 → 优雅结束，不抛异常
- 四层 JSON 防护：代码块剥离 → 直接解析 → 正则提取 → ast.literal_eval

---

## V5.1 优先级抢占式调度（向后兼容）

```python
# V5.1 API 仍然可用
from sasf.core.orchestrator import Orchestrator, TaskPriority

scheduler = Orchestrator(config=config, max_workers=2)
env = scheduler.spawn_laboratory(lab_id="Lab-01", engine=engine, ...)

await scheduler.start()
await scheduler.submit_task("Lab-01", "常规实验", TaskPriority.NORMAL)
await scheduler.submit_task("Lab-01", "紧急复位", TaskPriority.CRITICAL)
await scheduler.shutdown()
```

---

## 一、正交联锁引擎 (`physics/interlock_engine.py`)

**问题**：单体 FSM 状态数 = 子系统1状态数 × 子系统2 × ... → 状态爆炸。

**方案**：每个子系统独立维护状态，跨系统安全由联锁规则保证。

```yaml
# fsm_rules.yaml
subsystems:
  thermal: [IDLE, HEATING, COOLING]
  vacuum:  [IDLE, ACTIVE]
  arm:     [IDLE, MOVING]

interlocks:
  - condition: "vacuum == 'ACTIVE' and arm == 'MOVING'"
    message: "真空激活时禁止移动机械臂"
  - condition: "temperature >= 80"
    message: "温度过高，安全停机"
```

---

## 二、Guard 装饰器 (`middleware/mcp_registry.py`)

```python
@registry.mcp_tool(
    require_states={"thermal": "IDLE"},    # 前提：thermal 必须 IDLE
    forbid_states={"vacuum": "ACTIVE"},    # 禁止：vacuum 不能 ACTIVE
    telemetry_rules=["temperature < 80"],   # 遥测：温度必须 < 80℃
)
async def set_temperature(ctx: MCPToolContext, target: float) -> dict:
    """设置舱内温度目标值（℃）"""
    ...
```

---

## 三、Macro 参数预绑定

```python
registry.bind_macro("heat_to_50", "set_temperature", {"target": 50.0},
                    description="快速加热到 50℃")
```

- Macro 注册为独立 Tool，对 LLM **零参数或少参数**调用
- Codec 词表**自动包含** Macro 名
- SkillLoader 在 Prompt 中**自动引导** LLM 优先调用 Macro

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ (async/await, TypedDict, ast, dataclasses) |
| LLM | Ollama (Qwen2.5) / DeepSeek / 阿里云百炼 |
| 工作流 | LangGraph (StateGraph + MemorySaver) |
| 配置 | PyYAML |
| 安全求值 | ast.parse + 白名单节点遍历 |
| 二进制编码 | struct (标准库) |
| 并发 | asyncio (标准库) |
| DAG 调度 | asyncio.PriorityQueue + asyncio.Lock |

---

## 版本历史

| 版本 | 日期 | 变化 |
|------|------|------|
| V7.0 | 2026-03 | **双轨调度**：DAG 依赖图 + 理论/实践智能体分离 |
| V6.2 | 2026-01 | **语义路由**：LLM router_node 替代 BM25 |
| V5.1 | 2025-10 | **优先级调度**：PriorityQueue + 抢占机制 |
| V5.0 | 2025-08 | **正交联锁**：FSM → InterlockEngine |

---

## License

MIT
