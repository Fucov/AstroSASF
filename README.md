# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · Edge-RAG + 抢占调度 + 正交联锁 + Guard + Macro

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目概述

AstroSASF 是面向空间站科学实验柜的**多智能体协作调度框架**。核心矛盾：大模型推理的 _"概率性/高延迟"_ 与物理硬件控制的 _"确定性/硬实时"_ 之间的冲突。

### V6.0 核心设计

> **Edge-RAG 动态检索** + **优先级抢占调度** + **正交联锁** + **Guard** + **Macro**。

| 概念 | 层级 | 本质 | 管理者 |
|------|------|------|--------|
| **MCP Tools** | 中间件层 | 底层原子操作接口 + **Guard 声明式安全守卫** | `middleware/mcp_registry.py` |
| **Macro** | 中间件层 | 参数预绑定的快捷 Tool（类似 `functools.partial`） | `mcp_registry.bind_macro()` |
| **OpenAI Skills** | 认知层 | 标准操作程序 SOP + **Macro 感知上下文** | `cognition/skill_loader.py` |
| **InterlockEngine** | 物理层 | 正交子系统状态 + 跨系统联锁规则引擎 | `physics/interlock_engine.py` |

---

## 核心特性

| 能力 | 模块 | 描述 |
|------|------|------|
| **Edge-RAG** | `skill_loader.py` | BM25-lite 零依赖检索，动态匹配最相关 SOP |
| **多领域知识库** | `skills_catalog/` | 流体实验 / 生物培养 / 材料合成 |
| **优先级调度** | `orchestrator.py` | PriorityQueue + Worker 池 + CRITICAL 抢占 |
| **Guard 装饰器** | `mcp_registry.py` | `@mcp_tool(forbid_states=..., telemetry_rules=...)` |
| **Macro 绑定** | `mcp_registry.py` | `bind_macro("heat_50", "set_temperature", {"target": 50})` |
| **正交联锁引擎** | `interlock_engine.py` | 子系统独立状态 + `ast` 安全求值 |
| **动态字典压缩** | `codec.py` | 自动握手含 Macro 名 |
| **Macro 感知 SOP** | `skill_loader.py` | 自动引导 LLM 优先调用 Macro |
| **HITL** | 应用层注入 | `graph.compile(checkpointer=MemorySaver())` |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Core Layer — Orchestrator (V5.1 Priority Scheduler)        │
│  PriorityQueue │ Worker Pool │ Preemption (asyncio.Event)   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ LaboratoryEnvironment (suspend_event checkpoint)     │    │
│  └──────────────────────────────────────────────────────┘    │
│  (config.yaml 驱动 · Headless / HITL 可选)                   │
├─────────────────────────────────────────────────────────────┤
│  Cognition Layer                                            │
│  ┌──────────────────┐  ┌──────────────────────────────┐     │
│  │ SkillLoader (V5)  │  │ LangGraph StateGraph + LLM   │     │
│  │ Macro-aware SOP   │  │ (Ollama/DeepSeek/百炼)       │     │
│  └──────────────────┘  └──────────────────────────────┘     │
├──┬──────────────────────────────────────────────────────────┤
│  │  Middleware Layer ★ 核心资产                              │
│  │  ┌────────────────────────────────────────────────┐      │
│  │  │ MCPToolRegistry ← @mcp_tool(Guard) + Macro    │      │
│  │  │ SpaceMCPCodec(自动握手) · SpaceWire · Gateway  │      │
│  │  │ A2ARouter (Pub/Sub)                            │      │
│  │  └────────────────────────────────────────────────┘      │
├──┴──────────────────────────────────────────────────────────┤
│  Physics Layer — InterlockEngine + TelemetryBus             │
│  (正交子系统状态 · 跨系统联锁规则 · ast 安全求值)             │
└─────────────────────────────────────────────────────────────┘
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
│   │   ├── orchestrator.py         # 多实验柜编排器
│   │   └── environment.py          # 5 步装配器 (Headless)
│   ├── middleware/
│   │   ├── mcp_registry.py         # @mcp_tool(Guard) + bind_macro()
│   │   ├── a2a_protocol.py         # A2A Pub/Sub 消息路由
│   │   ├── codec.py                # 动态字典 Space-MCP 编解码
│   │   ├── gateway.py              # 协议转换网关
│   │   └── virtual_bus.py          # SpaceWire 总线模拟
│   ├── cognition/
│   │   ├── state.py                # LangGraph TypedDict
│   │   ├── graph_builder.py        # StateGraph + 严格 Planner
│   │   └── skill_loader.py         # SKILL.md SOP + Macro 感知
│   └── physics/
│       ├── interlock_engine.py     # 正交联锁引擎 (替代 FSM)
│       └── telemetry_bus.py        # 遥测数据总线
├── skills_catalog/
│   └── fluid_experiment/
│       └── SKILL.md
└── examples/
    └── space_station_demo.py       # V5 全链路演示
```

---

## 快速开始

```bash
ollama serve && ollama pull qwen2.5:7b
pip install -r requirements.txt
python examples/space_station_demo.py
```

## V6.0 核心机制

### 〇、Edge-RAG 轻量级边缘检索增强 (`cognition/skill_loader.py`)

**问题**：太空站边缘节点算力受限，无法运行向量数据库或 Embedding 模型，但多领域知识库持续增长，全量注入 Prompt 会浪费 Token。

**方案**：纯 Python 标准库 BM25-lite（`collections.Counter` + `math.log`），零第三方依赖。

```python
# planner_node 内部：每个任务动态检索最相关的 SOP
retrieved = catalog.retrieve_relevant_skills(task, top_k=1)
# → [{"name": "bio_culture", "score": 0.85, "context": "..."}]
```

#### BM25 公式
```
IDF(t)  = ln((N - df + 0.5) / (df + 0.5) + 1)
Score   = Σ IDF(t) × tf(t) × (k1+1) / (tf(t) + k1 × (1 - b + b × dl/avgdl))
```

#### 多领域知识库
| 领域 | SKILL.md | 关键工具 |
|------|----------|----------|
| 流体实验 | `fluid_experiment` | set_temperature, vacuum, arm |
| 生物培养 | `bio_culture` | set_temperature(37℃), inject_nutrient |
| 材料合成 | `material_synthesis` | vacuum, turn_on_laser, set_temperature |

---

### 〇、优先级抢占式调度内核 (`core/orchestrator.py`)

**问题**：`asyncio.gather` 仅支持同级并发，无法区分任务优先级，更无法在紧急异常时抢占资源。

**方案**：`PriorityQueue` + Worker 协程池 + `asyncio.Event` 抢占/挂起。

```
优先级枚举:
  CRITICAL = 0   (紧急异常响应 — 最高)
  HIGH     = 1   (核心科学任务)
  NORMAL   = 2   (常规任务)
  LOW      = 3   (清理/待机)
```

#### 抢占序列

```
时刻 T=0    提交 NORMAL 任务 → Worker-0 取出 → 开始执行
               ┌──────────────────────────────────┐
时刻 T=2    │ 🚨 CRITICAL 任务入队               │
               │ → Orchestrator._preempt()        │
               │ → NORMAL 任务 event.clear() 挂起  │
               └──────────────────────────────────┘
            Worker-1 取出 CRITICAL → 执行紧急安全复位

时刻 T=T₁   CRITICAL 完成 → _resume_suspended_tasks()
            → NORMAL 任务 event.set() 恢复 → 继续执行
```

#### API

```python
scheduler = Orchestrator(config=config, max_workers=2)
env = scheduler.spawn_laboratory(lab_id="Lab-01", engine=engine, ...)

await scheduler.start()
await scheduler.submit_task("Lab-01", "常规实验", TaskPriority.NORMAL)
await scheduler.submit_task("Lab-01", "紧急复位", TaskPriority.CRITICAL)
await scheduler.shutdown()
```

---

### 一、正交联锁引擎 (`physics/interlock_engine.py`)

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

联锁表达式通过 `ast.parse()` → 白名单 AST 节点验证 → 安全求值，**不使用 `eval()`**。

---

### 二、Guard 装饰器 (`middleware/mcp_registry.py`)

```python
@registry.mcp_tool(
    require_states={"thermal": "IDLE"},    # 前提：thermal 必须 IDLE
    forbid_states={"vacuum": "ACTIVE"},    # 禁止：vacuum 不能 ACTIVE
    telemetry_rules=["temperature < 80"],  # 遥测：温度必须 < 80℃
)
async def set_temperature(ctx: MCPToolContext, target: float) -> dict:
    """设置舱内温度目标值（℃）"""
    ...
```

`invoke()` 自动在执行前检查所有 Guard 条件，不满足则抛出 `SecurityGuardrailException`。

---

### 三、Macro 参数预绑定

```python
registry.bind_macro("heat_to_50", "set_temperature", {"target": 50.0},
                    description="快速加热到 50℃")
```

- Macro 注册为独立 Tool，对 LLM **零参数或少参数**调用
- Codec 词表**自动包含** Macro 名
- SkillLoader 在 Prompt 中**自动引导** LLM 优先调用 Macro

---

### 四、Codec 词表自动握手

```
1. MCPToolRegistry 注册 Tools + bind_macro()
2. registry.all_vocabulary() 返回完整词汇表（含 Macro 名）
3. SpaceMCPCodec 按字母序自动分配 Token ID

   0x01 ← 'activate'
   0x02 ← 'arm_home'          ← Macro!
   0x03 ← 'arm_to_dock'       ← Macro!
   ...
```

**零代码维护**：新增 Tool 或 Macro → 词表自动更新。

---

### 五、Macro 感知 SOP (`cognition/skill_loader.py`)

SkillLoader 接收 Registry 引用，在 `get_all_skills_context()` 中自动追加：

```
## 🔗 可用宏指令 (Macro)
- `heat_to_50` → set_temperature({"target": 50.0}) — 快速加热到 50℃
- `arm_to_observation` → move_robotic_arm({"target_angle": 45.0}) — 观测位
```

LLM 可直接调用 `heat_to_50` 而非 `set_temperature(target=50)`。

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

---

## License

MIT
