# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · MCP Tools + OpenAI Skills 解耦架构

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Fucov/AstroSASF)

---

## 项目概述

AstroSASF 是面向空间站科学实验柜的**多智能体协作调度框架**。核心矛盾：大模型推理的 _"概率性/高延迟"_ 与物理硬件控制的 _"确定性/硬实时"_ 之间的冲突。

### V4.2 核心设计

> **MCP Tools ≠ OpenAI Skills**。这是两种完全不同的抽象层次。

| 概念 | 层级 | 本质 | 管理者 |
|------|------|------|--------|
| **MCP Tools** | 中间件层 | 底层原子操作接口（`set_temperature`, `move_arm`） | `middleware/mcp_registry.py` |
| **OpenAI Skills** | 认知层 | 标准操作程序 SOP（描述**如何**组合 Tools） | `cognition/skill_loader.py` |

---

## 核心特性

| 能力 | 模块 | 描述 |
|------|------|------|
| **自动反射 Schema** | `mcp_registry.py` | `@mcp_tool` 装饰器自动从 Type Hints 生成 JSON Schema |
| **动态字典压缩** | `codec.py` | 启动期从 Registry 协商词汇表，动态分配 Token ID |
| **SKILL.md 知识** | `skill_loader.py` | 解析 YAML Frontmatter + Markdown SOP 注入 LLM Prompt |
| **A2A Protocol** | `a2a_protocol.py` | 标准消息信封 + 路由器审计日志 |
| **协议网关** | `gateway.py` | 纯透传：Registry 查找 → 压缩 → 执行 → 回传 |
| **HITL** | `environment.py` | Human-in-the-Loop 每步中断审批 |
| **LangGraph** | `graph_builder.py` | StateGraph 循环 + MCP Tools/Skills 双上下文 |
| **FSM 护栏** | `shadow_fsm.py` | 确定性状态机，不可绕过的安全拦截 |
| **崩溃安全** | `orchestrator.py` | 环境崩溃时抢救已产生的统计数据 |

---

## 系统架构

### 四层解耦 + MCP/Skills 分离

```
┌─────────────────────────────────────────────────────────────┐
│  Core Layer — Orchestrator / LaboratoryEnvironment          │
│  (N 个隔离环境 · config.yaml 驱动 · HITL 中断循环)           │
├─────────────────────────────────────────────────────────────┤
│  Cognition Layer                                            │
│  ┌──────────────┐  ┌────────────────────────────────────┐   │
│  │ SkillLoader   │  │ LangGraph StateGraph + LLM         │   │
│  │ (SKILL.md SOP)│  │ (可替换: Ollama/DeepSeek/百炼)     │   │
│  └──────────────┘  └────────────────────────────────────┘   │
├──┬──────────────────────────────────────────────────────────┤
│  │  Middleware Layer ★ 核心资产                              │
│  │  ┌─────────────────────────────────────────────────┐     │
│  │  │ MCPToolRegistry ← @mcp_tool      A2ARouter     │     │
│  │  │ SpaceMCPCodec(动态字典) · SpaceWire · Gateway   │     │
│  │  └─────────────────────────────────────────────────┘     │
├──┴──────────────────────────────────────────────────────────┤
│  Physics Layer — ShadowFSM + TelemetryBus                   │
│  (确定性安全护栏 · @mcp_tool 声明式注册)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```text
AstroSASF/
├── config.yaml                             # 统一配置
├── requirements.txt / pyproject.toml
├── README.md
│
├── sasf/                                   # 【核心框架包】
│   ├── core/
│   │   ├── config_loader.py                # YAML 解析 + LLM 工厂
│   │   ├── orchestrator.py                 # 多实验柜编排器 (崩溃安全)
│   │   └── environment.py                  # 5 步装配器 (HITL)
│   ├── middleware/                          # ★ 核心中间件
│   │   ├── mcp_registry.py                 # @mcp_tool 自动反射 Schema
│   │   ├── a2a_protocol.py                 # A2A 消息标准 + 路由器
│   │   ├── codec.py                        # 动态字典 Space-MCP 编解码
│   │   ├── gateway.py                      # 协议转换网关 (纯透传)
│   │   └── virtual_bus.py                  # SpaceWire 总线模拟
│   ├── cognition/
│   │   ├── state.py                        # LangGraph TypedDict 状态
│   │   ├── graph_builder.py                # StateGraph (Tools + Skills 双上下文)
│   │   └── skill_loader.py                 # SKILL.md SOP 加载器
│   └── physics/
│       ├── shadow_fsm.py                   # FSM + @mcp_tool 注册
│       └── telemetry_bus.py                # 遥测数据总线
│
├── skills_catalog/                         # 【OpenAI Skills 知识套件】
│   └── fluid_experiment/
│       └── SKILL.md                        # 微重力流体实验 SOP
│
└── examples/
    └── space_station_demo.py               # 全链路演示
```

---

## 快速开始

```bash
ollama serve && ollama pull qwen2.5:7b
pip install -r requirements.txt
python examples/space_station_demo.py
```

---

## 技术原理与工作链路

### 一、@mcp_tool 自动反射 Schema (`middleware/mcp_registry.py`)

#### 问题

V4.1 的 `register()` 需要手动传入 `param_schema` 字典，增加维护负担且容易与实际函数签名不同步。

#### 原理

**Python Introspection** — 装饰器通过 `inspect.signature()` + `typing.get_type_hints()` 自动反射目标函数的形参名、类型注解、默认值，动态生成 **OpenAI Function Calling 兼容** 的 JSON Schema：

```python
# 物理层开发者只写自然的 Python 函数，零 Schema 代码：
@registry.mcp_tool
async def set_temperature(ctx: MCPToolContext, target: float) -> dict:
    """设置舱内温度目标值（℃）"""
    ...
```

自动生成的 Schema：
```json
{
  "type": "function",
  "function": {
    "name": "set_temperature",
    "description": "设置舱内温度目标值（℃）",
    "parameters": {
      "type": "object",
      "properties": {"target": {"type": "number"}},
      "required": ["target"]
    }
  }
}
```

#### 类型映射

| Python Type Hint | JSON Schema Type |
|-----------------|------------------|
| `float` | `"number"` |
| `int` | `"integer"` |
| `str` | `"string"` |
| `bool` | `"boolean"` |

#### Handler 签名变化

```
V4.1:  handler(ctx: SkillContext, params: dict) → dict     # 手动解构参数
V4.2:  handler(ctx: MCPToolContext, target: float) → dict   # 关键字参数直接传入
```

---

### 二、动态字典压缩 (`middleware/codec.py`)

#### 问题

V4.1 使用硬编码的 `_STR_TO_TOKEN` 字典。每新增一个 MCP Tool 都必须手动维护映射表，违背了"新增 Tool 只需写一个函数"的设计目标。

#### 原理

**启动期协商机制** — 系统启动时的装配流程：

```
1. MCPToolRegistry 注册所有 @mcp_tool
2. registry.all_vocabulary() 返回排序后的完整词汇表
3. SpaceMCPCodec.__post_init__() 按词汇表序自动分配 Token ID

   词汇表: ['activate', 'detail', 'error', 'fsm_state', 'move_robotic_arm',
            'set_temperature', 'skill', 'status', 'success',
            'target', 'target_angle', 'toggle_vacuum_pump']

   动态分配:
   0x01 ← 'activate'
   0x02 ← 'detail'
   0x03 ← 'error'
   0x04 ← 'fsm_state'
   0x05 ← 'move_robotic_arm'
   0x06 ← 'set_temperature'
   0x07 ← 'skill'
   0x08 ← 'status'
   0x09 ← 'success'
   0x0A ← 'target'
   0x0B ← 'target_angle'
   0x0C ← 'toggle_vacuum_pump'
```

**零代码维护**：新增 MCP Tool → `all_vocabulary()` 自动包含新词 → Codec 自动分配 Token。

#### 帧结构

```
┌────────┬──────────┬──────────┬────────────┬─────────────────┐
│ Magic  │ Tool     │ N Params │ Param Key  │ Type + Value    │ ...
│ 0xA5   │ Token ID │ uint8    │ Token/0xFF │ (1+N bytes)     │
│ 1 byte │ 1 byte   │ 1 byte   │ 1 byte     │ variable        │
└────────┴──────────┴──────────┴────────────┴─────────────────┘
```

#### 效果

```
JSON:  {"skill": "set_temperature", "params": {"target": 50.0}}  = 56 字节
Binary: a5 06 01 0a 01 42 48 00 00                               =  9 字节
压缩率: 83.9%
```

---

### 三、OpenAI Skills 标准操作程序 (`cognition/skill_loader.py`)

#### 问题

LLM 擅长理解自然语言但缺乏领域知识。单纯给出 MCP Tool 列表，LLM 不知道**复合实验的最佳操作顺序**。

#### 原理

按 **OpenAI/Skills 标准**，每个 Skill 是一个包含 `SKILL.md` 的目录：

```yaml
# skills_catalog/fluid_experiment/SKILL.md
---
name: fluid_experiment
description: 微重力流体实验标准操作程序
---

# 微重力流体实验 SOP

## Workflow
### Step 1: 环境准备
1. 调用 `set_temperature` 设温到 25℃
2. 等待稳定...
```

`OpenAISkillCatalog` 在系统启动时：
1. 递归扫描 `skills_catalog/` 目录下的 `SKILL.md`
2. 解析 YAML Frontmatter（name, description）
3. 提取 Markdown 正文（SOP Workflow）
4. 通过 `get_all_skills_context()` 格式化为 Prompt 片段

#### Planner Prompt 结构

```
你是太空实验柜的规划智能体 (Planner)。

## 可用 MCP Tools (底层原子操作)        ← 自动反射的 Schema
- set_temperature: ...
- move_robotic_arm: ...

## 已加载的 OpenAI Skills (SOP)         ← SKILL.md 知识
### fluid_experiment
1. 先设温 25℃...
2. 再移臂到 90°...

用户指令: 请做微重力流体实验
```

---

### 四、A2A 通信协议 (`middleware/a2a_protocol.py`)

所有 Agent 间通信封装为 `A2AMessage` 标准信封：

```python
A2AMessage(sender, receiver, intent, payload, timestamp, sequence)
```

6 种意图类型：

| Intent | 方向 | 含义 |
|--------|------|------|
| `TASK_REQUEST` | System → Planner | 新任务提交 |
| `PLAN_GENERATED` | Planner → Operator | 规划完成 |
| `SKILL_INVOCATION` | Operator → Gateway | MCP Tool 调用 |
| `SKILL_RESULT` | Gateway → Operator | 执行结果 |
| `ERROR_CORRECTION` | Operator → LLM | 错误修正 |
| `EXECUTION_COMPLETE` | Operator → System | 全部完成 |

`A2ARouter` 统一分配递增序列号、记录完整日志、提供审计能力。

---

### 五、MCP Tool 全链路执行 (Gateway)

```
LangGraph execute_node
  │
  └──▶ gateway.invoke_tool("set_temperature", {"target": 50.0})
          │
          ├── 1. A2ARouter.route(SKILL_INVOCATION)
          ├── 2. Codec.encode() → binary (动态字典)
          ├── 3. VirtualSpaceWire.transmit()
          ├── 4. Codec.decode() → 还原 JSON
          ├── 5. MCPToolRegistry.invoke() → handler(ctx, target=50.0)
          │       ├── FSM.validate_and_transition()
          │       └── TelemetryBus.write()
          ├── 6. Codec.encode_response()
          ├── 7. VirtualSpaceWire.transmit() (上行)
          ├── 8. Codec.decode_response()
          └── 9. A2ARouter.route(SKILL_RESULT)
```

---

### 六、LangGraph 状态图 + HITL

```
START ──▶ planner_node ──▶ operator_node ──┬──▶ ⏸️ HITL ──▶ execute_node ──┐
                                           │                               │
                                           │◀──────── (循环) ──────────────┘
                                           └── done ──▶ END
```

- `interrupt_before=["execute_node"]` + `MemorySaver` 检查点
- HITL 输入: `y` 批准 / `n` 中止 / `JSON` 修正参数

---

### 七、FSM 安全护栏 (`physics/shadow_fsm.py`)

确定性有限状态机，**不可被 LLM 或上层逻辑绕过**。

约束示例：
- 温度 ≥ 80℃ 时禁止加热 (`START_HEATING`)
- 气压 < 50kPa 时禁止移动机械臂 (`MOVE_ROBOTIC_ARM`)

违规时抛出 `SecurityGuardrailException`，由 Gateway 捕获并返回 error。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ (async/await, TypedDict, inspect) |
| LLM | Ollama (Qwen2.5) / DeepSeek / 阿里云百炼 |
| 工作流 | LangGraph (StateGraph + MemorySaver) |
| LLM 接口 | langchain-ollama / langchain-openai |
| 配置 | PyYAML |
| 二进制编码 | struct (标准库) |
| 并发 | asyncio (标准库) |

---

## License

MIT
