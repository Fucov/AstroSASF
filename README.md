# AstroSASF — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · Middleware-First Architecture

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目概述

AstroSASF 是面向空间站科学实验柜的**多智能体协作调度框架**。核心矛盾：大模型推理的 _"概率性/高延迟"_ 与物理硬件控制的 _"确定性/硬实时"_ 之间的冲突。

**V4 设计哲学 — Middleware-First（中间件优先）：**

> 中间件是系统的**核心资产**，LangGraph 和 LLM 只是可替换的应用层工具。
> 网关不写任何业务代码，物理设备主动向中间件注册能力。

---

## 核心特性

| 能力 | 模块 | 描述 |
|------|------|------|
| **SkillRegistry** | `middleware/skill_registry.py` | 实例级技能注册中心，物理设备 `@register` 主动注册 |
| **A2A Protocol** | `middleware/a2a_protocol.py` | 标准消息信封 `A2AMessage` + 路由器审计日志 |
| **Space-MCP 压缩** | `middleware/codec.py` | JSON → 单字节 Token 二进制帧，压缩率 > 85% |
| **SpaceWire 模拟** | `middleware/virtual_bus.py` | 200Kbps 带宽受限总线，按字节延迟 |
| **协议网关** | `middleware/gateway.py` | 纯透传：查询 Registry → 调用 → A2A 记录 |
| **LangGraph** | `cognition/graph_builder.py` | StateGraph 循环工作流 + LLM 错误修正 |
| **配置驱动** | `config.yaml` | 统一配置 LLM/带宽/并发，支持 Ollama + OpenAI 兼容 API |
| **FSM 护栏** | `physics/shadow_fsm.py` | 确定性状态机，零妥协安全拦截 |

---

## 系统架构

### 四层解耦 + Middleware-First

```
┌─────────────────────────────────────────────────────────────┐
│  Core Layer — Orchestrator / LaboratoryEnvironment          │
│  (N 个隔离环境 · config.yaml 驱动)                           │
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

### Skill 注册流

```
Physics Layer                Middleware Layer              Cognition Layer
     │                            │                            │
     │  register_default_skills() │                            │
     │ ─────────────────────────▶ │ SkillRegistry              │
     │   set_temperature          │   .register()              │
     │   move_robotic_arm         │   .register()              │
     │   toggle_vacuum_pump       │   .register()              │
     │                            │                            │
     │                            │ ◀── gateway.list_skills()  │
     │                            │     (注入 LLM prompt)      │
```

### LangGraph 状态图 + A2A 消息流

```
    START ──▶ planner_node ──▶ operator_node ──┬──▶ execute_node ──┐
                  │                │           │                   │
         A2A: TASK_REQUEST    A2A: PLAN_GENERATED                  │
                              A2A: EXECUTION_COMPLETE              │
                                                                   │
                                   ◀───────────── (循环) ──────────┘
                                              A2A: SKILL_INVOCATION
                                              A2A: SKILL_RESULT
```

### Space-MCP 协议链路

```
LangGraph ──JSON──▶ Gateway ──Codec.encode──▶ SpaceWire ──▶ SkillRegistry ──▶ FSM
                                                    │
LangGraph ◀──JSON── Gateway ◀──Codec.decode── SpaceWire ◀──────────────────────┘
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
│   │   ├── orchestrator.py                 # 多实验柜编排器
│   │   └── environment.py                  # 单实验柜装配器
│   ├── middleware/                          # ★ 核心中间件
│   │   ├── skill_registry.py               # 技能注册中心
│   │   ├── a2a_protocol.py                 # A2A 消息标准 + 路由器
│   │   ├── codec.py                        # Space-MCP 二进制编解码
│   │   ├── gateway.py                      # 协议转换网关 (纯透传)
│   │   └── virtual_bus.py                  # SpaceWire 总线模拟
│   ├── cognition/
│   │   ├── state.py                        # LangGraph TypedDict 状态
│   │   └── graph_builder.py                # StateGraph 构建器
│   └── physics/
│       ├── shadow_fsm.py                   # FSM + register_default_skills()
│       └── telemetry_bus.py                # 遥测数据总线
│
└── examples/                               # 【演示用例】(与框架物理隔离)
    └── space_station_demo.py               # 全链路演示脚本
```

---

## 快速开始

### 前置条件

1. **Python 3.10+**
2. **Ollama**（本地推理）或 **DeepSeek/阿里云百炼** API Key（远端推理）

```bash
# 本地 Ollama
ollama serve
ollama pull qwen2.5:7b
```

### 安装与运行

```bash
git clone <repo-url>
cd AstroSASF
pip install -r requirements.txt
python examples/space_station_demo.py
```

### 切换为远端 API

编辑 `config.yaml`：

```yaml
llm:
  provider: "openai_compatible"
  base_url: "https://api.deepseek.com/v1"
  api_key: "sk-xxxxxx"
  model_name: "deepseek-chat"
```

---

## 技术原理

### 1. SkillRegistry — 技能注册中心

**设计问题：** V1-V3 的 Gateway 硬编码了 `set_temperature` 等 Skill 函数，网关同时承担"协议转换"和"业务执行"双重职责，违反了单一职责原则。

**V4 方案：** 引入 `SkillRegistry` 作为中间件暴露的注册端口。物理设备层在初始化时通过 `register_default_skills()` 主动将自身能力注册到 Registry，网关退化为纯协议透传代理。

```python
# physics/shadow_fsm.py 中
registry.register(
    name="set_temperature",
    handler=_set_temperature,      # 闭包捕获 FSM + Bus
    description="设置舱内温度目标值",
    param_schema={"target": "float"},  # 供 LLM function calling
)
```

### 2. A2A Protocol — 消息标准化

**设计问题：** V3 的 LangGraph 节点间通信被框架内部吞没，无法观测和审计。

**V4 方案：** 中间件定义 `A2AMessage` 标准信封，所有节点间通信必须包装为 A2A 消息。`A2ARouter` 统一记录序列号、时间戳和意图类型。

### 3. 配置驱动 LLM

`config_loader.py` 使用工厂模式根据 `config.yaml` 的 `provider` 字段动态创建：
- `"ollama"` → `ChatOllama`（本地推理）
- `"openai_compatible"` → `ChatOpenAI`（兼容 DeepSeek / 阿里云百炼 / OpenAI）

### 4. Space-MCP 静态字典压缩

将高频 JSON 键映射为单字节 Token，数值用 `struct.pack` 紧凑编码：

```
"set_temperature" → 0x01    (15 → 1 字节)
"target"          → 0x10    ( 6 → 1 字节)
float 50.0        → 4 字节  (struct ">f")
```

### 5. FSM 安全护栏

独立于 LLM 推理的确定性状态机。物理安全约束（如温度 ≥ 80℃ 禁止加热）不可被任何上层逻辑绕过。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ (async/await, TypedDict) |
| LLM | Ollama (Qwen2.5) / DeepSeek / 阿里云百炼 |
| 工作流 | LangGraph (StateGraph) |
| LLM 接口 | langchain-ollama / langchain-openai |
| 配置 | PyYAML |
| 二进制编码 | struct (标准库) |
| 并发 | asyncio (标准库) |

---

## License

MIT
