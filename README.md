# AstroSASF V7.2 — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · **内核化架构** · **DAG 双轨调度** · **硬件级抢占** · **多节点 LLM 网关**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-blue.svg)](https://fastapi.tiangolo.com/)
[![asyncio](https://img.shields.io/badge/asyncio-async%2Fawait-purple.svg)](https://docs.python.org/3/library/asyncio.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目概述

AstroSASF V7.2 是专为**空间站科学实验柜**设计的智能体调度框架，解决大语言模型（LLM）推理的**概率性/高延迟**与物理硬件控制的**确定性/硬实时**之间的根本矛盾。

**V7.2 核心变化：内核化 + 分布式网关**
- **内核提取**：调度逻辑、VRAM 管理、LLM 网关下沉为独立内核模块
- **扁平化**：减少目录嵌套深度，`catalog` 层被拆分到 `infra/` 和 `labs/`
- **多实例网关**：`config.yaml` 配置驱动多后端路由，支持 Ollama / SGLang / vLLM 混合部署
- **生命周期解耦**：AstroSASF 不管理 LLM 进程，只连接已运行的远端推理服务
- **演示驱动**：`demo/demo_mission.py` 提供完整的三智能体协作演示

**快速开始：**

```bash
# 启动 AstroSASF 服务（推荐）
uv run uvicorn interface.server:app --reload --host 0.0.0.0 --port 8000

# 运行三智能体协作演示
uv run python demo/demo_mission.py

# 单任务演示
uv run python demo/demo_mission.py --lab DemoBio --mission "将培养舱温度设置为37°C"
```

> **LLM 服务（可选）**：如果未启动 LLM 服务，Planner Agent 会自动降级为关键词匹配模式（demo-safe）。如需真实 LLM 推理，请确保 `config.yaml` 中配置的后端地址可达（Ollama / SGLang / vLLM）。


---

## 项目目录结构

```
AstroSASF/
│
├── config.yaml                     # ★ Gateway 多后端 + 中间件全局配置
├── pyproject.toml                 # Python 项目配置
│
├── infra/                        # ===== 内核基础设施层 =====
│   ├── __init__.py
│   ├── llm/                      # LLM 配置与实例池
│   │   ├── config_loader.py     # YAML 配置解析（多后端 GatewayConfig）
│   │   └── instance_pool.py     # LLM 实例池（健康检查、VRAM 监控）
│   ├── routing/                  # 请求路由与熔断
│   │   ├── prefix_balancer.py   # 前缀感知负载均衡（支持 SGLang RadixAttention）
│   │   └── vram_breaker.py      # VRAM 感知熔断器（优先级准入控制）
│   └── gateway/                   # LLM 网关
│       ├── proxy.py              # 分布式 LLM 网关反向代理
│       └── api.py                # LLM 网关 FastAPI 集成
│
├── scheduler/                    # ===== 内核调度层（零业务词汇）=====
│   ├── __init__.py
│   ├── models.py                 # DAG 数据结构（DAGNode / DAGTaskGraph）
│   ├── core.py                   # DAG 双轨调度器（优先级队列 + 硬件抢占）
│   ├── a2a_protocol.py          # Agent-to-Agent 通信协议（Pub/Sub）
│   ├── virtual_bus.py            # SpaceWire 总线模拟（低带宽约束）
│   └── telemetry.py              # 遥测总线 + 硬件告警监控
│
├── interface/                    # ===== 北向接口层 =====
│   ├── __init__.py
│   ├── gateway.py               # Space-MCP 协议转换网关
│   ├── server.py                # ★ FastAPI HTTP 服务端入口
│   └── facade.py                # API 统一门面（聚合 Gateway/Loader/Scheduler）
│
├── labs/                        # ===== 物理模拟与 MCP 注册 =====
│   ├── __init__.py
│   ├── mcp_registry.py         # @mcp_tool 装饰器 + Guard + Macro 绑定
│   ├── interlock_engine.py     # 正交联锁引擎（YAML 规则 + AST 安全求值）
│   ├── mcp_codec.py            # 动态字典 Space-MCP 编解码器
│   └── lab_loader.py           # 实验舱目录扫描与动态加载
│
├── demo/                        # ===== 演示资产 =====
│   ├── __init__.py
│   ├── demo_mission.py         # ★ V7.2 三智能体协作演示入口
│   ├── assets/
│   │   ├── labs/               # 演示用实验舱配置
│   │   │   ├── DemoBio/        # 生物实验舱
│   │   │   └── DemoFluid/      # 流体实验舱
│   │   └── skills/             # 演示用技能 SOP
│   │       ├── bio_culture/
│   │       ├── fluid_experiment/
│   │       ├── material_synthesis/
│   │       ├── plant_growth_monitor/
│   │       └── emergency_fire_response/
│   └── agents/                  # 协作智能体实现
│       ├── __init__.py
│       ├── base_agent.py        # Agent 基类（HTTP 客户端 + 消息总线）
│       ├── qa_agent.py          # Q&A Agent（意图解析）
│       ├── planner_agent.py     # Planner Agent（LLM 生成 DAG 步骤）
│       └── executor_agent.py    # Executor Agent（按序执行 MCP Tools）
│
│
├── datasets/                     # 评测数据集
└── benchmarks/                  # 基准测试
```

### 目录分层设计哲学

| 层级 | 目录 | 职责 | 特点 |
|------|------|------|------|
| **内核基础设施** | `infra/` | LLM 管理、路由、负载均衡、熔断 | 零业务词汇，完全通用 |
| **内核调度** | `scheduler/` | DAG 调度、A2A 协议、遥测、虚拟总线 | 零业务词汇，完全通用 |
| **北向接口** | `interface/` | FastAPI 服务、协议网关、门面聚合 | 零业务词汇 |
| **物理适配** | `labs/` | MCP 注册、联锁引擎、编解码、实验舱加载 | 允许业务词汇（物理层） |
| **演示资产** | `demo/` | 三智能体协作演示 | 完整场景示例 |
| 演示配置 | `demo/assets/labs/` | 演示实验舱（DemoBio / DemoFluid） | 业务词汇，可扩展 |
| **旧代码** | `sasf/` | V7.1 遗留代码 | ⚠️ 已废弃 |

---

## 核心功能

### 1. DAG 双轨调度

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

### 2. 前缀感知 LLM 网关（V7.2 新增）

支持 SGLang RadixAttention 的 KV-Cache 复用：

```python
# 前缀哈希路由：相同 SOP/系统提示的请求 → 同一实例
# 减少重复 prompt 的 GPU 计算，显著降低 TTFT
prefix_hash = sha256(system_prompt + task_desc)
# → 路由到历史健康实例，或 fallback 到最少连接实例
```

### 3. VRAM 感知熔断器（V7.2 新增）

优先级敏感的准入控制，保护 LLM 实例不过载：

```
CRITICAL 任务 → 任何实例有空间即可执行
HIGH 任务     → VRAM < 80% 才准入
NORMAL 任务   → VRAM < 70% 才准入
LOW 任务      → VRAM < 60% 才准入
→ 全部满载 → 排队等待
```

### 4. 硬件级抢占机制

TelemetryBus 监测危险条件 → 注入 CRITICAL 逃生任务 → 强制 Cancel LLM 推理。

### 5. 正交联锁安全

子系统独立状态 + 联锁规则 YAML 配置 + AST 白名单安全求值。

### 6. Guard 声明式安全守卫

```python
@registry.mcp_tool(
    require_states={"heater": "IDLE"},      # 前提条件
    forbid_states={"vacuum": "ACTIVE"},     # 禁止条件
    telemetry_rules=["temperature < 80"],  # 遥测约束
)
async def control_heater(ctx, temperature: float, duration: int):
    ...
```

### 7. Macro 参数预绑定

```python
registry.bind_macro("cell_culture_temp", "control_heater",
                    {"temperature": 37.0, "duration": 3600})
# LLM 调用 "cell_culture_temp" 即可，无需传递参数
```

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AstroSASF V7.2 系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    客户端层 (Client)                                    │  │
│  │   航天员 / Agent → HTTP 请求 → REST API → 结果展示                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              北向接口层 (interface/)                                    │  │
│  │    FastAPI Server ← Facade ← Agent (Q&A / Planner / Executor)          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              分布式 LLM 网关 (infra/gateway/)                          │  │
│  │                                                                         │  │
│  │  PrefixAwareLoadBalancer ──→ VRAMWatermarkBreaker ──→ LLMInstancePool │  │
│  │       (前缀哈希路由)              (优先级准入)           (多节点管理)       │  │
│  │                                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │  Backend 1   │  │  Backend 2   │  │  Backend N   │  ← config.yaml   │  │
│  │  │  Ollama      │  │  SGLang     │  │  vLLM       │                 │  │
│  │  │  localhost   │  │  192.168.x  │  │  192.168.x  │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              内核调度层 (scheduler/)                                    │  │
│  │   DAG Orchestrator → PriorityQueue → Worker Pool                       │  │
│  │   A2A Protocol │ VirtualBus │ Telemetry + Alarm                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              物理适配层 (labs/)                                         │  │
│  │   MCPToolRegistry │ InterlockEngine │ SpaceMCPCodec                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              物理模拟层 (实验舱)                                          │  │
│  │   DemoBio │ DemoFluid │ ... (位于 demo/assets/labs/)                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### config.yaml 示例（多后端配置）

```yaml
gateway:
  vram_high_watermark: 0.85      # 高水位（限流）
  vram_critical_watermark: 0.92  # 熔断线

  backends:
    - url: "http://localhost:11434"        # 本地 Ollama（fallback）
      provider: "ollama"
      model_name: "qwen2.5:7b"
      weight: 1
      tags: ["local", "fallback"]

    - url: "http://192.168.1.10:8000"     # SGLang GPU 节点
      provider: "sglang"
      model_name: "qwen2.5-7b-instruct"
      weight: 3
      tags: ["v100", "node-1"]
```

### API 端点

```
GET   /health                        # 健康检查
GET   /api/v1/labs                   # 列出所有实验舱
GET   /api/v1/labs/{lab_id}/meta     # 获取舱体元数据（含 MCP Tools）
POST  /api/v1/labs/{lab_id}/execute  # 执行工具调用
POST  /api/v1/llm/chat               # LLM 推理接口（经 GatewayProxy）
```

---

## 三智能体协作流程

```
┌────────────────────────────────────────────────────────────────────┐
│                      demo_mission.py                                │
│                                                                    │
│  用户请求                                                          │
│      │                                                            │
│      ▼                                                            │
│  [Q&A Agent]  ──── plan_request ──→  [Planner Agent]             │
│      ↑                                      │                      │
│      │           execution_complete          ▼                      │
│      └──────  Execution Report  ←────  [Executor Agent]            │
│                                                                    │
│  所有 Agent 通过 HTTP (httpx) 调用 interface/server.py               │
│  Executor Agent 按序执行 MCP Tools，汇报结果给 Q&A Agent              │
└────────────────────────────────────────────────────────────────────┘
```

**运行方式：**

```bash
# 默认并发演示（两个实验舱同时执行）
python demo/demo_mission.py

# 指定任务（单任务模式）
python demo/demo_mission.py \
    --lab DemoBio \
    --mission "将培养舱温度设置为37°C，然后注入50ml营养液"

# 指定自定义实验舱目录
python demo/demo_mission.py --catalog ./my_labs --skills ./my_skills
```

---

## 技术栈

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| **语言** | Python 3.10+ | async/await、TypedDict、dataclasses |
| **LLM** | Ollama / DeepSeek / 阿里云百炼 / vLLM / SGLang | 推理引擎 |
| **工作流** | LangGraph (StateGraph) | DAG 生成与状态管理 |
| **服务** | FastAPI | HTTP REST API |
| **HTTP 客户端** | httpx | Agent 间通信 |
| **配置** | PyYAML | 规则外部化配置 |
| **安全求值** | `ast.parse` + 白名单节点遍历 | 联锁表达式安全执行 |
| **并发** | `asyncio` (标准库) | 异步任务调度 |
| **调度** | `asyncio.PriorityQueue` + `asyncio.Lock` | DAG 任务排队与并发控制 |
| **负载均衡** | 前缀哈希 + 最少连接 | LLM 路由 |
| **熔断** | 状态机 + VRAM 水位 | 保护 LLM 实例 |
| **总线模拟** | SpaceWire 协议 | 低带宽遥测约束 |

---

## 版本历史

| 版本 | 日期 | 核心变化 |
|------|------|----------|
| **V7.2** | 2026-04-04 | 分布式网关重构：移除单实例 `llm` 节点 → `gateway.backends[]` 多后端配置阵列；`config.yaml` 驱动 `init_distributed_gateway()`；移除 `create_llm()` LangChain 工厂；修复 `_check_instance` async lock 持有 bug |
| **V7.2** | 2026-04-03 | 内核化重构：提取 `infra/`、`scheduler/`、`interface/`、`labs/`，扁平化目录，`demo_mission.py` 三智能体演示，vLLM/SGLang 前缀感知网关 |
| **V7.1** | 2026-03 | 数据集评测 + LLM 算力埋点 + 物理模拟层 |
| **V7.0** | 2026-03 | DAG 双轨调度 + 理论/实践智能体分离 |
| **V6.2** | 2026-01 | LLM 语义路由替代 BM25 |
| **V5.1** | 2026-01 | 优先级抢占式调度 |
| **V5.0** | 2026-01 | 正交联锁引擎替代单体 FSM |

---

## License

MIT
