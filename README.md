# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · Middleware-First Architecture

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Fucov/AstroSASF)

---

## 项目概述

AstroSASF 是面向空间站科学实验柜的**多智能体协作调度框架**。核心矛盾：大模型推理的 _"概率性/高延迟"_ 与物理硬件控制的 _"确定性/硬实时"_ 之间的冲突。

**设计哲学 — Middleware-First（中间件优先）：**

> 中间件是系统的**核心资产**，LangGraph 和 LLM 只是可替换的应用层工具。
> 网关不写任何业务代码，物理设备主动向中间件注册能力。

---

## 核心特性

| 能力 | 模块 | 描述 |
|------|------|------|
| **SkillRegistry** | `middleware/skill_registry.py` | 实例级技能注册中心，物理设备主动注册 |
| **A2A Protocol** | `middleware/a2a_protocol.py` | 消息信封 `A2AMessage` + 路由器审计 |
| **Space-MCP 压缩** | `middleware/codec.py` | JSON → 单字节 Token 二进制帧，压缩率 > 85% |
| **SpaceWire 模拟** | `middleware/virtual_bus.py` | 200Kbps 带宽受限总线模拟 |
| **协议网关** | `middleware/gateway.py` | 纯透传：Registry 查找 → 执行 → A2A 记录 |
| **HITL** | `core/environment.py` | Human-in-the-Loop 每步中断审批 |
| **LangGraph** | `cognition/graph_builder.py` | StateGraph 循环工作流 + LLM 错误修正 |
| **配置驱动** | `config.yaml` | LLM/带宽/并发统一配置 |
| **FSM 护栏** | `physics/shadow_fsm.py` | 确定性状态机，零妥协安全拦截 |

---

## 系统架构

### 四层解耦 + Middleware-First

```
┌─────────────────────────────────────────────────────────────┐
│  Core Layer — Orchestrator / LaboratoryEnvironment          │
│  (N 个隔离环境 · config.yaml 驱动 · HITL 中断循环)           │
├─────────────────────────────────────────────────────────────┤
│  Cognition Layer — LangGraph StateGraph + LLM               │
│  (可替换: Ollama / DeepSeek / 阿里云百炼)                     │
├──┬──────────────────────────────────────────────────────────┤
│  │  Middleware Layer ★ 核心资产                              │
│  │  ┌─────────────────────────────────────────────────┐     │
│  │  │ SkillRegistry ← 物理层注册       A2ARouter      │     │
│  │  │ SpaceMCPCodec · VirtualSpaceWire · Gateway      │     │
│  │  └─────────────────────────────────────────────────┘     │
├──┴──────────────────────────────────────────────────────────┤
│  Physics Layer — ShadowFSM + TelemetryBus                   │
│  (确定性安全护栏 · 主动注册 Skills)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```text
AstroSASF/
├── config.yaml                             # 统一配置文件 (LLM/带宽/并发)
├── requirements.txt                        # Python 依赖
├── pyproject.toml                          # PEP 621 项目元数据
├── README.md
│
├── sasf/                                   # 【核心框架包】
│   ├── __init__.py
│   ├── core/
│   │   ├── config_loader.py                # YAML 解析 + LLM 工厂
│   │   ├── orchestrator.py                 # 多实验柜编排器 (崩溃安全)
│   │   └── environment.py                  # 单实验柜装配器 (HITL 中断循环)
│   ├── middleware/                          # ★ 核心中间件
│   │   ├── skill_registry.py               # 技能注册中心
│   │   ├── a2a_protocol.py                 # A2A 消息标准 + 路由器
│   │   ├── codec.py                        # Space-MCP 二进制编解码
│   │   ├── gateway.py                      # 协议转换网关 (纯透传)
│   │   └── virtual_bus.py                  # SpaceWire 总线模拟
│   ├── cognition/
│   │   ├── state.py                        # LangGraph TypedDict 状态
│   │   └── graph_builder.py                # StateGraph 构建器 (HITL)
│   └── physics/
│       ├── shadow_fsm.py                   # FSM + register_default_skills()
│       └── telemetry_bus.py                # 遥测数据总线
│
└── examples/                               # 【演示用例】
    └── space_station_demo.py               # 全链路演示 (含 HITL 交互)
```

---

## 快速开始

### 前置条件

1. **Python 3.10+**
2. **Ollama**（本地推理）或 **DeepSeek/阿里云百炼** API Key

```bash
# 本地 Ollama
ollama serve && ollama pull qwen2.5:7b
```

### 安装与运行

```bash
pip install -r requirements.txt
python examples/space_station_demo.py
```

### HITL 交互

运行后，LLM 生成计划后，**每个 Skill 执行前会暂停**：

```
──────────────────────────────────────────────────────────
🛡️ HITL | 即将执行: set_temperature({"target": 50.0})
  [y/回车] 批准  |  [n] 中止  |  [JSON] 修正参数
──────────────────────────────────────────────────────────
>>>
```

- 输入 `y` 或回车 → 批准执行
- 输入 `n` → 中止整个任务
- 输入 JSON → 修正参数后执行，例如 `{"skill": "set_temperature", "params": {"target": 30.0}}`

### 切换远端 API

编辑 `config.yaml`：

```yaml
llm:
  provider: "openai_compatible"
  base_url: "https://api.deepseek.com/v1"
  api_key: "sk-xxxxxx"
  model_name: "deepseek-chat"
```

---

## 技术原理与工作链路

### 一、Space-MCP 静态字典压缩 (`middleware/codec.py`)

#### 问题

航天总线（SpaceWire ~200Mbps，MIL-STD-1553B 仅 1Mbps）带宽极为有限。MCP 工具调用的 JSON 文本中充斥高频重复的键名（`"set_temperature"`, `"target"`, `"status"`）和冗余的引号、花括号等结构字符，直接传输将导致严重的网络拥塞。

#### 原理

**静态字典映射** — 在编译期预置一张「高频字符串 → 单字节 Token ID」映射表：

```python
# 字典摘录 (完整见 _STR_TO_TOKEN)
"set_temperature"  → 0x01    # 15 字节 → 1 字节
"move_robotic_arm" → 0x02    # 16 字节 → 1 字节
"target"           → 0x10    #  6 字节 → 1 字节
"status"           → 0x21    #  6 字节 → 1 字节
"success"          → 0x30    #  7 字节 → 1 字节
```

**数值紧凑编码** — 使用 `struct.pack` 替代 ASCII 文本表示：

| 类型 | 文本形式 | 二进制编码 | 节省 |
|------|----------|-----------|------|
| `float 50.0` | `"50.0"` (4 字节) | `struct >f` (4 字节) | 0 |
| `bool true` | `"true"` (4 字节) | `0x01` (1 字节) | 75% |
| 字符串键 | `"set_temperature"` (17 字节含引号) | `0x01` (1 字节) | 94% |

#### 帧结构

```
┌────────┬──────────┬──────────┬─────────────┬────────────────────┐
│ Magic  │ Skill    │ N Params │ Param Key 1 │ Type + Value 1     │ ...
│ 0xA5   │ Token ID │ uint8    │ Token ID    │ (1+N bytes)        │
│ 1 byte │ 1 byte   │ 1 byte   │ 1 byte      │ variable           │
└────────┴──────────┴──────────┴─────────────┴────────────────────┘
```

#### 效果

一次典型的 `set_temperature(target=50.0)` 调用：

```
JSON 原文:  {"skill": "set_temperature", "params": {"target": 50.0}}
            = 56 字节

Space-MCP:  a5 01 01 10 01 42 48 00 00
            = 9 字节

压缩率:     83.9%
```

---

### 二、MCP Skill 注册与执行链路 (`middleware/skill_registry.py`)

#### 问题

V1-V3 的 Gateway 硬编码了 `set_temperature` 等 Skill 函数，同时承担"协议转换"和"业务执行"双重职责，违反了单一职责原则。当需要新增或修改 Skill 时，必须修改网关代码本身。

#### 原理

**控制反转 (IoC)** — 引入 `SkillRegistry` 作为中间件暴露的标准注册端口。Skill 的定义权交还给物理设备层。

#### 注册流程

```
1. LaboratoryEnvironment.__post_init__()
   │
   ├── 创建 SkillRegistry(lab_id)
   │
   ├── 创建 ShadowFSM(lab_id) + TelemetryBus(lab_id)
   │
   └── 调用 register_default_skills(registry, fsm, bus)
       │
       ├── registry.register("set_temperature",  handler, description, schema)
       ├── registry.register("move_robotic_arm",  handler, description, schema)
       └── registry.register("toggle_vacuum_pump", handler, description, schema)
```

每个 handler 是一个闭包函数，签名为 `(ctx: SkillContext, params: dict) → dict`。
`SkillContext` 封装了 `fsm`, `bus`, `lab_id`，使 handler 无需依赖 Gateway 类型。

#### 调用执行链路

```
Gateway.invoke_skill("set_temperature", {"target": 50.0})
  │
  ├── 1. A2ARouter.route(SKILL_INVOCATION)          # 记录调用消息
  ├── 2. Codec.encode(request) → bytearray          # JSON → 二进制
  ├── 3. VirtualSpaceWire.transmit(binary)           # 模拟物理传输
  ├── 4. Codec.decode(wire_data) → dict              # 还原 JSON
  ├── 5. SkillRegistry.invoke(name, params, context) # 查表 → 调用 handler
  │       └── handler(ctx, params)
  │           ├── ctx.fsm.validate_and_transition()  # FSM 校验
  │           └── ctx.bus.write("temperature", 50.0)  # 遥测更新
  ├── 6. Codec.encode_response(result) → bytearray   # 响应压缩
  ├── 7. VirtualSpaceWire.transmit(resp_binary)       # 回传
  ├── 8. Codec.decode_response(wire_data) → dict      # 还原
  └── 9. A2ARouter.route(SKILL_RESULT)                # 记录结果消息
```

网关从头到尾**不包含任何业务逻辑** — 纯粹的协议透传代理。

---

### 三、A2A 通信协议 (`middleware/a2a_protocol.py`)

#### 问题

V3 引入 LangGraph 后，Agent 间的通信被框架内部的状态传递吞没，无法独立观测、审计和追溯。作为中间件项目，A2A 的存在感必须在日志中清晰可见。

#### 原理

**标准消息信封** — 所有 Agent 间通信必须封装为 `A2AMessage`：

```python
@dataclass(frozen=True)
class A2AMessage:
    sender: str         # "Lab-Alpha::Planner"
    receiver: str       # "Lab-Alpha::Operator"
    intent: A2AIntent   # PLAN_GENERATED / SKILL_INVOCATION / ...
    payload: Any        # 消息正文
    timestamp: float    # UNIX 时间戳
    sequence: int       # 递增序列号（Router 分配）
```

**意图类型** — 明确定义 6 种 A2A 意图：

| Intent | 方向 | 含义 |
|--------|------|------|
| `TASK_REQUEST` | System → Planner | 新任务提交 |
| `PLAN_GENERATED` | Planner → Operator | 规划完成 |
| `SKILL_INVOCATION` | Operator → Gateway | 技能调用请求 |
| `SKILL_RESULT` | Gateway → Operator | 技能执行结果 |
| `ERROR_CORRECTION` | Operator → LLM | 错误修正请求 |
| `EXECUTION_COMPLETE` | Operator → System | 全部完成 |

#### 路由流

```
[System] ──TASK_REQUEST──▶ [Planner]
[Planner] ──PLAN_GENERATED──▶ [Operator]
  ┌───────────────────────────────────────┐
  │ [Operator] ──SKILL_INVOCATION──▶ [GW] │ ← 循环
  │ [GW] ──SKILL_RESULT──▶ [Operator]     │
  │ (如失败) [Operator] ──ERROR_CORRECTION──▶ [LLM]
  └───────────────────────────────────────┘
[Operator] ──EXECUTION_COMPLETE──▶ [System]
```

`A2ARouter` 统一分配递增 `sequence` 编号，记录完整日志，提供审计能力。

---

### 四、LangGraph 状态图 + HITL (`cognition/graph_builder.py`)

#### 原理

使用 LangGraph `StateGraph` 构建带循环的工作流：

```
START ──▶ planner_node ──▶ operator_node ──┬──▶ ⏸️ HITL ──▶ execute_node ──┐
                                           │                               │
                                           │◀──────── (循环) ──────────────┘
                                           └── done ──▶ END
```

- **planner_node**: 调用 LLM 将自然语言拆解为 JSON 步骤
- **operator_node**: 四路条件判断（FSM 修正 / 成功推进 / 提取步骤 / 完成）
- **execute_node**: 通过 SpaceMCPGateway 执行 Skill
- **HITL 中断**: `interrupt_before=["execute_node"]` + `MemorySaver` 检查点

#### HITL 循环

```python
compiled = graph.compile(
    checkpointer=MemorySaver(),        # 检查点后端
    interrupt_before=["execute_node"],  # 中断点
)

# 执行循环
state = await graph.ainvoke(initial_state, config)
while graph.get_state(config).next:     # 还有未执行的节点
    human_input = input(">>> y/n/JSON: ")
    if human_input == "n":
        break
    if is_json(human_input):
        graph.update_state(config, {"current_step": corrected})  # 修正参数
    state = await graph.ainvoke(None, config)  # 继续执行
```

#### 错误修正机制

当 FSM 拦截指令时，`operator_node` 自动：
1. 调用 LLM 生成修正方案
2. 如果 LLM 建议 `"skip"` → 跳过该步骤
3. 每步最多重试 `_MAX_RETRIES_PER_STEP = 2` 次
4. 超过重试次数 → 记录错误日志并继续下一步

---

### 五、FSM 安全护栏 (`physics/shadow_fsm.py`)

#### 原理

独立于 LLM 推理链路的**确定性有限状态机**。所有硬件指令必须通过 FSM 校验才能执行。

#### 状态转移表

```
IDLE ──START_HEATING──▶ HEATING ──STOP_HEATING──▶ IDLE
IDLE ──START_COOLING──▶ COOLING ──STOP_COOLING──▶ IDLE
IDLE ──MOVE_ARM──▶ MOVING ──STOP_ARM──▶ IDLE
IDLE ──ACTIVATE_VACUUM──▶ VACUUM_ACTIVE ──DEACTIVATE──▶ IDLE
*    ──EMERGENCY_STOP──▶ EMERGENCY ──RESET──▶ IDLE
```

#### 物理安全约束

```python
_SAFETY_CONSTRAINTS = {
    Action.START_HEATING:    [("temperature", ">=", 80.0)],   # ≥80℃ 禁止加热
    Action.MOVE_ROBOTIC_ARM: [("pressure",    "<",  50.0)],   # <50kPa 禁止移臂
}
```

违规时抛出 `SecurityGuardrailException`，**不可被任何上层逻辑绕过**。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ (async/await, TypedDict) |
| LLM | Ollama (Qwen2.5) / DeepSeek / 阿里云百炼 |
| 工作流 | LangGraph (StateGraph + MemorySaver) |
| LLM 接口 | langchain-ollama / langchain-openai |
| 配置 | PyYAML |
| 二进制编码 | struct (标准库) |
| 并发 | asyncio (标准库) |

---

## License

MIT
