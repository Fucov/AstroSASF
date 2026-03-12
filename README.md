# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目概述

AstroSASF 是一个面向空间站科学实验柜的**多智能体协作调度框架**，解决大模型推理的 _"概率性 / 高延迟"_ 与物理硬件控制的 _"确定性 / 硬实时"_ 之间的核心矛盾。

**设计哲学：** 将 LLM 的不确定性严格限制在认知层（Cognition），通过 Space-MCP 协议压缩中间件（Middleware）与 FSM 安全护栏（Physics）两层确定性屏障，确保物理硬件绝不接收到非法指令。

---

## 核心特性

| 能力 | 版本 | 描述 |
|------|------|------|
| **Human-in-the-loop** | V3.1 | MemorySaver + interrupt_before 实现科学家审核关卡 |
| **LangGraph 状态图** | V3 | 基于 StateGraph 的循环工作流：Planner → Operator → Execute |
| **Ollama LLM 接入** | V3 | ChatOllama (Qwen2.5) 驱动的任务规划与错误修正 |
| **Space-MCP 压缩** | V2 | 静态字典编解码器，JSON → 二进制帧压缩率 > 85% |
| **虚拟 SpaceWire** | V2 | 200Kbps 带宽受限总线模拟，按字节计算传输延迟 |
| **FSM 安全护栏** | V1 | 影子设备有限状态机，拦截 LLM 幻觉指令，零妥协 |
| **多系统并发** | V1 | N 套实验柜环境完全隔离，asyncio.gather 并发运行 |
| **A2A 通信** | V1 | 基于 asyncio.Queue 的进程内 Pub/Sub 消息路由 |
| **MCP Skills** | V1 | 将硬件操作抽象为标准 Model Context Protocol 工具接口 |

---

## 系统架构

### 四层解耦架构

```
┌─────────────────────────────────────────────────────────────┐
│  Core Layer — Orchestrator / LaboratoryEnvironment          │
│  (编排 N 个隔离环境，asyncio.gather 并发)                     │
├─────────────────────────────────────────────────────────────┤
│  Cognition Layer — LangGraph StateGraph + ChatOllama        │
│  (LLM 规划 → 状态流转 → 错误修正循环)                         │
├─────────────────────────────────────────────────────────────┤
│  Middleware Layer — Codec + VirtualSpaceWire + Gateway       │
│  (JSON ↔ Binary 压缩 · 低带宽总线模拟 · 协议转换)             │
├─────────────────────────────────────────────────────────────┤
│  Physics Layer — ShadowFSM + TelemetryBus                   │
│  (确定性状态机 · 遥测数据总线 · 安全护栏)                      │
└─────────────────────────────────────────────────────────────┘
```

### LangGraph 状态图拓扑 (V3.1 认知层 + HITL)

```
                    ┌───────────────┐
    START ────────▶ │ planner_node  │ (LLM 生成 JSON 计划)
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
              ┌───▶ │ operator_node │ (判断: 提取步骤 / 修正错误 / 完成)
              │     └───────┬───────┘
              │             │
              │    ┌────────┴────────┐
              │    │ has_step   done │
              │    ▼                 ▼
              │  ┌──────────┐    ┌──────┐
              └──│ execute  │    │ END  │
                 │ _node    │    └──────┘
                 │ ⏸ HITL   │
                 └──────────┘
           (interrupt_before → 人工审核 → 调用 SpaceMCPGateway)
```

### Space-MCP 协议转换链路 (V2 中间件层)

```
LangGraph ──JSON──▶ Gateway ──encode──▶ SpaceWire ──▶ FSM
                                              │
LangGraph ◀──JSON── Gateway ◀──decode── SpaceWire ◀──┘
```

---

## 目录结构

```text
AstroSASF/
├── main.py                         # 入口点：V3.1 LangGraph + HITL + Space-MCP 全链路演示
├── requirements.txt                # Python 依赖声明
├── pyproject.toml                  # PEP 621 项目元数据
├── README.md
│
├── core/                           # 核心编排层
│   ├── orchestrator.py             # 多实验柜编排器 (asyncio.gather)
│   └── environment.py              # 单实验柜运行时 (HITL 循环 + LangGraph + Middleware + Physics)
│
├── cognition/                      # 认知层 (V3 活跃模块)
│   ├── state.py                    # [V3] LangGraph TypedDict 状态定义
│   ├── graph_builder.py            # [V3] StateGraph 构建：planner/operator/execute 节点
│   ├── agent_nexus.py              # [V1 遗留] A2A Pub/Sub 通信总线 ⚠️
│   ├── base_agent.py               # [V1 遗留] 智能体基类 (Mock LLM) ⚠️
│   └── specialized_agents.py       # [V2 遗留] Planner + Operator 智能体 ⚠️
│
├── middleware/                     # 中间件层
│   ├── codec.py                    # [V2] 静态字典二进制编解码器
│   ├── gateway.py                  # [V2] Space-MCP 协议转换网关
│   ├── virtual_bus.py              # [V2] SpaceWire 总线模拟
│   └── mcp_gateway.py             # [V1 遗留] 原始 MCP 网关 ⚠️
│
└── physics/                        # 物理硬件层
    ├── shadow_fsm.py               # [V1] 影子设备有限状态机
    └── telemetry_bus.py            # [V1] 遥测数据总线
```

### ⚠️ 遗留模块说明

以下模块属于 V1/V2 迭代遗留，**当前版本 (V3.1) 不再导入或使用**，保留原因如下：

| 模块 | 原始版本 | 保留原因 |
|------|----------|----------|
| `cognition/agent_nexus.py` | V1 | 架构演进参考；未来如需恢复低依赖模式可复用 |
| `cognition/base_agent.py` | V1 | Mock LLM 实现可在无 Ollama 环境下用于单元测试 |
| `cognition/specialized_agents.py` | V2 | 展示 V1→V3 的架构演进脉络，用于毕设论文对比分析 |
| `middleware/mcp_gateway.py` | V1 | 不含 Space-MCP 压缩的直连版本；可用于压缩效果对比基准测试 |

> **注意：** 这些文件不会对 V3.1 的运行产生任何影响。如需清理，可安全删除，不会影响框架功能。

---

## 快速开始

### 前置条件

1. **Python 3.10+**
2. **Ollama** — 本地运行 Qwen2.5 模型

```bash
# 安装 Ollama（如果尚未安装）
# macOS: brew install ollama
# 或访问 https://ollama.com/download

# 启动 Ollama 并拉取模型
ollama serve
ollama pull qwen2.5:7b
```

### 安装与运行

```bash
# 克隆项目
git clone <repo-url>
cd AstroSASF

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

### 预期输出

程序将展示：
1. **LLM 规划阶段** — Qwen2.5 将自然语言任务拆解为 Skill 调用序列
2. **Human-in-the-loop 审核** — 每步 Skill 执行前暂停，等待科学家确认（y/n/JSON 覆盖）
3. **Space-MCP 压缩** — 每个 Skill 调用的 JSON → Binary 压缩率和 hex dump
4. **SpaceWire 传输** — 按字节计算的物理总线传输延迟
5. **FSM 状态迁移** — 设备状态机的确定性校验日志
6. **错误修正循环** — 当 FSM 拦截非法指令时，LLM 尝试生成修正方案
7. **统计汇总** — 编解码器 / 总线 / 任务执行的综合统计

---

## 技术原理

### 1. 静态字典压缩 (`middleware/codec.py`)

航天总线带宽极低（SpaceWire 约 200Mbps，1553B 仅 1Mbps），MCP 工具调用的 JSON 文本充斥大量重复的高频键名（如 `"set_temperature"`, `"target"`, `"status"`）。

**方案：** 构建预置静态字典，将高频字符串映射为单字节 Token ID：

```python
"set_temperature"  → 0x01   # 15 字节 → 1 字节
"move_robotic_arm" → 0x02   # 16 字节 → 1 字节
"target"           → 0x10   #  6 字节 → 1 字节
```

数值参数使用 `struct.pack` 紧凑编码（`float32` = 4 字节，`bool` = 1 字节），最终将 ~85 字节的 JSON 压缩至 ~6 字节的二进制帧。

### 2. 影子设备 FSM (`physics/shadow_fsm.py`)

FSM 是系统的**绝对安全底线**，独立于 LLM 推理链路：

- 仅包含确定性的状态转移表（`dict[tuple[State, Action], State]`）
- 物理安全约束（如温度 ≥ 80℃ 时禁止加热）
- 违规时抛出 `SecurityGuardrailException`，不可被静默忽略

### 3. LangGraph 状态图 (`cognition/graph_builder.py`)

使用 LangGraph `StateGraph` 构建带循环的工作流：

- **planner_node**: 调用 ChatOllama 将自然语言拆解为 JSON 步骤计划
- **operator_node**: 条件判断（Step 提取 / FSM 错误修正 / 流程结束）
- **execute_node**: 通过 SpaceMCPGateway 执行 Skill

错误修正机制：当 FSM 拦截指令时，operator_node 调用 LLM 生成修正方案，最多重试 2 次后跳过。

### 4. Human-in-the-loop 审核 (`core/environment.py`)

使用 LangGraph 的 `MemorySaver` + `interrupt_before=["execute_node"]` 机制：

1. **图编译时** — 注入 `MemorySaver` 作为 checkpointer，指定 `execute_node` 为中断点
2. **运行时** — 图在 `operator_node` 提取出待执行步骤后、进入 `execute_node` 前**暂停**
3. **控制台交互** — 在终端打印即将执行的 Skill 名称和参数，等待科学家输入：
   - `y` — 确认执行
   - `n` — 终止整个任务
   - `{"target": 40.0}` — 用新 JSON 覆盖当前步骤参数（通过 `graph.update_state`）
4. **恢复执行** — 调用 `graph.ainvoke(None, config)` 从中断点恢复

每个任务使用独立的 `thread_id`（如 `Lab-Alpha::task_0`），确保 MemorySaver 中的检查点互不干扰。

### 5. 多系统隔离

每个 `LaboratoryEnvironment` 实例拥有独立的：
- `TelemetryBus` + `ShadowFSM`（物理层）
- `SpaceMCPCodec` + `VirtualSpaceWire` + `SpaceMCPGateway`（中间件层）
- `ChatOllama` + 编译后的 `CompiledGraph` + `MemorySaver`（认知层）

通过 `Orchestrator.run_all()` + `asyncio.gather` 实现真正的零干扰并发。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ (async/await, TypedDict) |
| LLM | Ollama / Qwen2.5:7b |
| 工作流引擎 | LangGraph (StateGraph + MemorySaver) |
| LLM 接口 | langchain-ollama (ChatOllama) |
| 人工审核 | LangGraph interrupt_before / update_state |
| 二进制编码 | struct (标准库) |
| 并发 | asyncio (标准库) |

---

## License

MIT
