# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · **C/S 服务化架构** · **DAG 双轨调度** · **硬件级抢占** · **正交联锁安全**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-blue.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目概述

AstroSASF 是专为**空间站科学实验柜**设计的智能体调度框架，解决大语言模型（LLM）推理的**概率性/高延迟**与物理硬件控制的**确定性/硬实时**之间的根本矛盾。

**核心架构**：采用 C/S 分层设计，将任务**规划**（LLM 推理）与任务**执行**（确定性调度）彻底解耦。

**快速开始**：

```bash
# 终端 1 - 启动 LLM 服务
ollama serve && ollama pull qwen2.5:7b

# 终端 1 - 启动服务端
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# 终端 2 - 运行演示
python demo_integration.py
```

---

## 项目目录结构

```
AstroSASF/
│
├── config.yaml                     # LLM 服务 + 中间件全局配置
├── fsm_rules.yaml                  # 正交子系统状态 + 联锁安全规则
├── requirements.txt                # Python 依赖清单
├── README.md                       # 项目文档
│
├── server.py                       # ★ FastAPI HTTP 服务端入口
├── demo_integration.py             # 客户端集成演示脚本
│
├── sasf/                           # 【核心框架包 — 零业务词汇】
│   ├── __init__.py
│   │
│   ├── core/                       # ===== 核心调度层 =====
│   │   ├── config_loader.py        # YAML 配置解析 + LLM 工厂模式
│   │   ├── models.py               # ★ DAG 数据结构（DAGNode / DAGTaskGraph）
│   │   ├── orchestrator.py         # ★ DAG 双轨调度器 + 硬件抢占
│   │   └── environment.py          # 实验室环境装配器（Headless）
│   │
│   ├── cognition/                  # ===== 认知规划层 =====
│   │   ├── state.py                # LangGraph TypedDict 状态定义
│   │   ├── graph_builder.py        # ★ DAG Planner（LLM → DAG 转换）
│   │   └── skill_loader.py         # 技能库加载器（Macro 感知 SOP）
│   │
│   ├── middleware/                  # ===== 中间件层 =====
│   │   ├── mcp_registry.py         # @mcp_tool 装饰器 + Guard + Macro 绑定
│   │   ├── a2a_protocol.py         # Agent-to-Agent 协议路由（Pub/Sub）
│   │   ├── codec.py                # 动态字典 Space-MCP 编解码
│   │   ├── gateway.py              # 协议转换网关
│   │   └── virtual_bus.py          # SpaceWire 总线模拟
│   │
│   └── physics/                    # ===== 物理模拟层 =====
│       ├── interlock_engine.py     # 正交联锁引擎（FSM 替代方案）
│       └── telemetry_bus.py         # 遥测数据总线（1553B 协议模拟）
│
├── labs_catalog/                   # 【实验舱目录 — 业务词汇区】
│   ├── shared/
│   │   └── shared_tools.py         # 跨舱公共工具
│   ├── Lab-Bio/
│   │   ├── lab_config.yaml          # 生物舱 FSM 配置
│   │   └── custom_tools.py         # 生物培养专属工具
│   └── Lab-Fluid/
│       ├── lab_config.yaml          # 流体舱 FSM 配置
│       └── custom_tools.py         # 流体实验专属工具
│
├── skills_catalog/                  # 【领域技能库】
│   ├── fluid_experiment/            # 流体实验 SOP
│   ├── bio_culture/                # 生物培养 SOP
│   ├── material_synthesis/         # 材料合成 SOP
│   ├── emergency_fire_response/    # 火情应急响应 SOP
│   └── plant_growth_monitor/       # 植物生长监测 SOP
│
├── datasets/                        # 【评测数据集】
│   └── astro_bench.jsonl           # 50 条标准化测试数据
│
├── tools/                           # 【工具脚本】
│   └── generate_dataset.py         # 数据集生成器
│
├── benchmarks/                      # 【基准测试】
│   ├── run_dataset_eval.py         # 评测运行器
│   └── dataset_report.json          # 评测报告输出
│
└── examples/                        # 【示例代码】
    └── space_station_demo.py       # 全链路演示脚本
```

### 目录分层设计哲学

| 层级 | 目录 | 职责 | 特点 |
|------|------|------|------|
| **核心框架层** | `sasf/` | DAG 调度、LLM 路由、联锁安全 | 零业务词汇，完全通用 |
| **业务适配层** | `labs_catalog/` | 实验舱工具、FSM 配置 | 业务词汇，可配置 |
| **领域知识层** | `skills_catalog/` | SOP 流程、Prompt 模板 | 领域专业，可扩展 |

---

## 核心功能

### 1. DAG 双轨调度（V7.0）

**理论轨道**（Planner）：LLM 将自然语言解析为 DAG 任务图，仅需 1 次 LLM 调用。

**实践轨道**（Workers）：从 ReadyQueue 抢占节点执行，**零 LLM 调用**。

```
自然语言指令
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  理论智能体 (Planner)                                        │
│  LangGraph → LLM → DAG 任务图（一次 LLM 调用）                 │
│  输出: [{"id":"A","skill":"set_temp",...}, {"id":"B",...}]  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  实践智能体 (Workers)                                         │
│  ReadyQueue ← [节点A, 节点B] → 并发执行 → 结算 → 依赖解除    │
│                           ↑                                  │
│                     BlockedQueue                             │
│                     [节点C(依赖A)]                           │
└─────────────────────────────────────────────────────────────┘
```

### 2. 硬件级抢占机制（V7.1）

TelemetryBus 监测危险条件 → 注入 CRITICAL 逃生任务 → 强制 Cancel LLM 推理。

### 3. 正交联锁安全

子系统独立状态 + 联锁规则 YAML 配置 + AST 白名单安全求值。

### 4. Guard 声明式安全守卫

```python
@registry.mcp_tool(
    require_states={"thermal": "IDLE"},     # 前提条件
    forbid_states={"vacuum": "ACTIVE"},    # 禁止条件
    telemetry_rules=["temperature < 80"],  # 遥测约束
)
async def set_temperature(ctx, target: float):
    ...
```

### 5. Macro 参数预绑定

```python
registry.bind_macro("heat_50", "set_temperature", {"target": 50.0})
# LLM 调用 "heat_50" 即可，无需传递参数
```

### 6. 动态优先级 Aging

等待越久的任务优先级动态提升，防止低优先级任务饿死。

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AstroSASF 系统架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     客户端层 (Client)                       │ │
│  │   航天员指令 → 自然语言解析 → HTTP 请求 → 结果展示         │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     服务端层 (Server)                       │ │
│  │                                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │              认知层 (Cognition Layer)                │  │ │
│  │  │    Planner → LangGraph StateGraph → DAG Generator   │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │                           │                                │ │
│  │                           ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │              核心层 (Core Layer)                     │  │ │
│  │  │   DAG Orchestrator → PriorityQueue → Worker Pool     │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │                           │                                │ │
│  │                           ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │              中间件层 (Middleware Layer)              │  │ │
│  │  │      MCP Registry → Dynamic Codec → A2A Router       │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │                           │                                │ │
│  │                           ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │              物理层 (Physics Layer)                   │  │ │
│  │  │    Interlock Engine → TelemetryBus → 1553B Bus       │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### API 端点

```
GET   /api/v1/labs                     # 列出所有实验舱
GET   /api/v1/labs/{lab_id}/meta       # 获取舱体元数据
POST  /api/v1/labs/{lab_id}/execute    # 执行工具调用
POST  /api/v1/llm/chat                 # LLM 推理接口
```

---

## 基准测试成果

| 指标 | Baseline | DAG-OS | 提升 |
|------|----------|--------|------|
| **Avg Makespan** | 18.42s | 7.31s | **2.52×** |
| **LLM Calls (总计)** | 174 | 50 | **↓71.3%** |
| **Hard Survival Rate** | 0.0% | 100.0% | **+100%** |
| **Avg Preemption** | N/A | <1ms | — |

> **极限场景**：Baseline 在 Hard 难度下 15/15 全部失败（报警触发 Guardrail），DAG-OS 15/15 全部成功。

---

## 技术栈

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| **语言** | Python 3.10+ | async/await、TypedDict、dataclasses |
| **LLM** | Ollama (Qwen2.5) / DeepSeek / 阿里云百炼 | 推理引擎 |
| **工作流** | LangGraph (StateGraph + MemorySaver) | DAG 生成与状态管理 |
| **服务** | FastAPI | HTTP REST API |
| **配置** | PyYAML | 规则外部化配置 |
| **安全求值** | `ast.parse` + 白名单节点遍历 | 联锁表达式安全执行 |
| **并发** | `asyncio` (标准库) | 异步任务调度 |
| **调度** | `asyncio.PriorityQueue` + `asyncio.Lock` | DAG 任务排队与并发控制 |
| **总线模拟** | 1553B 协议 | 遥测数据实时传输 |

---

## 版本历史

| 版本 | 日期 | 核心变化 |
|------|------|----------|
| **V7.1** | 2026-03 | 数据集评测 + LLM 算力埋点 + 物理模拟层 |
| **V7.0** | 2026-03 | DAG 双轨调度 + 理论/实践智能体分离 |
| **V6.2** | 2026-01 | LLM 语义路由替代 BM25 |
| **V5.1** | 2025-10 | 优先级抢占式调度 |
| **V5.0** | 2025-08 | 正交联锁引擎替代单体 FSM |

---

## License

MIT
