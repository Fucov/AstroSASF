# AstroSASF V8.0 — Astro Scientific Agent Scheduling Framework

> 面向太空实验室的科学智能体调度框架 · **V8.0 内核重写** · **OoO 后台主动扫描** · **五层防死锁协议** · **时间维度挂起** · **纳秒级调度时延** · **意图感知异构路由** · **透明模型降级** · **权威指标评估体系**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-blue.svg)](https://fastapi.tiangolo.com/)
[![asyncio](https://img.shields.io/badge/asyncio-async%2Fawait-purple.svg)](https://docs.python.org/3/library/asyncio.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目概述

AstroSASF V7.3 是专为**空间站科学实验柜**设计的智能体调度框架，解决大语言模型（LLM）推理的**概率性/高延迟**与物理硬件控制的**确定性/硬实时**之间的根本矛盾。

**V7.3 核心变化：LLM 响应净化 + 网关透明修复**
- **ResponseSanitizer**：彻底剥离 DeepSeek-R1 等蒸馏模型的 `<think>...</think>` 推理噪声，上层 Planner/Executor Agent 无需修改任何解析逻辑
- **强制输出指令**：在 User Message 末尾追加 JSON 格式强提醒，防止 Static MCP 前缀锁将关键指令挤出
- **asyncio.create_task 修复**：统一使用 `asyncio.get_running_loop().create_task()`，消除协程上下文缺失导致的潜在 RuntimeError
- **流式 SSE 净化**：在 `_do_streaming_request` 层面逐 chunk 过滤思考标签块

- **令牌桶带宽限流**：读取 `spacewire_bandwidth_kbps`，基于 `asyncio` 的令牌桶按 Byte 发放令牌，允许短时突发但长期速率不超过带宽上限
- **QoS 三级优先级队列**：CRITICAL（无限额优先）/ NORMAL（令牌受限）/ LOW（令牌受限，低优先级日志）
- **AoI 遥测覆写**：NORMAL 队列中同一 `aoi_key` 的高频遥测只保留最新快照，节约带宽，保证最新鲜数据优先发送
- **A2A 语义增量同步**：`A2ASemanticDiff` 对比本地状态快照，只传输增量变更（节点状态变化、遥测差异），废弃全量发送，显著降低 200 Kbps 带宽占用
- **CRITICAL 报警击穿拥塞**：硬件报警包进入 CRITICAL 队列无限额优先发送，平均排队延迟 < 5ms
- **弱网抗性指标**：`bytes_saved_by_aoi`（AoI 覆写节省字节）、`critical_avg_queue_latency_ms`（CRITICAL 包平均排队延迟）、`total_bytes_saved_by_diff`（Diff 节省字节）埋点

- **OoO 乱序执行**（V8.0）：后台 OoO Scanner 协程持续监控 ReadyQueue，当 WorkerPool 有空余 Slot 时主动遍历 BlockedQueue 执行越级提取；资源正交性公式 R(v_k) ∩ R_active = ∅；五层防死锁协议（字母序加锁 / 三重准入门 / 原子预约 / 推进保证 / 超时自动释放）
- **时间维度挂起**（V8.0）：asyncio.Future 驱动的 wait_for_condition，长周期物理 I/O 时立即 yield 控制权，释放推理线程
- **调度时延纳秒级埋点**（V8.0）：每个节点记录 ReadyQueue 入队到真正 Issue 的时间差，支持 P50/P95/P99/P999 分位数统计
- **ActiveResourceTable**（V8.0）：增强版资源占用表，支持超时自动回收（30s 锁超时阈值）、资源正交性 check_orthogonality()、复合键追踪

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

## 实验运行（论文级验证）

AstroSASF 提供两套完整的论文级实验框架，用于验证 OoO-proposed 的性能优势。

### 实验结构总览

| 实验类型 | 入口文件 | 目的 | 输出表格 |
|---------|---------|------|---------|
| **主对比实验** | `experiments/comparator.py` | OoO-proposed vs 基线方法 | Table 1: 主对比 |
| **消融实验** | `experiments/ablator.py` | 验证各内部机制的独立贡献 | Table 3: 消融 |

**三种调度器模式（对应论文方法）：**

| 模式 | 内部名称 | 含义 |
|------|---------|------|
| `Sequential` | `sequential` | 严格顺序执行，理论下界 |
| `Traditional DAG` | `traditional_dag` | 层并发（asyncio.gather），无乱序/事件驱动 |
| `OoO-proposed` | `ooo_proposed` | 完整乱序调度框架（全部机制启用） |

---

### 主对比实验（Main Comparison）

**回答问题：** OoO-proposed 比传统基线好多少？

**运行方式：**

```bash
# 方式一：命令行参数（推荐，快速验证）
uv run python experiments/comparator.py \
    --scenarios heavy_conflict global_shared diamond_deep \
    --episodes 3 \
    --speed 0.1

# 方式二：编程调用（灵活定制）
uv run python -c "
import asyncio
from experiments.comparator import run_main_comparison
asyncio.run(run_main_comparison(
    scenarios=['heavy_conflict', 'global_shared'],
    episodes_per_scenario=3,
    physical_delay_scale=0.1,
    seed=42,
))
"

# 方式三：指定输出目录
uv run python experiments/comparator.py \
    --output-dir results/my_main_exp \
    --scenarios heavy_conflict \
    --episodes 5
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scenarios` | heavy_conflict, global_shared, diamond_deep | 场景类型列表 |
| `--episodes` | 3 | 每个场景采样的 episode 数量 |
| `--speed` | 0.1 | 物理延迟缩放因子（0.05~0.2，越小调度开销越明显） |
| `--output-dir` | results/main_comparison | 输出目录 |
| `--seed` | 42 | 随机种子（保证可复现） |

**可用场景类型：**

| 场景 | 说明 | 区分度 |
|------|------|--------|
| `no_conflict` | 线性 DAG，无资源竞争 | 低 |
| `light_conflict` | 每层 2~3 节点同设备竞争 | 中 |
| `heavy_conflict` | 每层 3~4 节点同设备竞争 | 高 |
| `alarm_recovery` | 含 telemetry_alarm，触发抢占 | 高 |
| `ooo_stress` | 深层宽 DAG，强制同设备竞争 + chaos | 最高 |
| `global_shared` | 全局共享设备竞争（co2_controller） | 最高 |
| `diamond_deep` | 深层钻石形依赖 | 最高 |

---

### 消融实验（Ablation）

**回答问题：** OoO 内部的哪个机制贡献最大？

**运行方式：**

```bash
# 方式一：命令行参数（推荐）
uv run python experiments/ablator.py \
    --scenarios heavy_conflict alarm_recovery diamond_deep \
    --episodes 3 \
    --speed 0.1

# 方式二：编程调用
uv run python -c "
import asyncio
from experiments.ablator import run_targeted_ablation
asyncio.run(run_targeted_ablation(
    scenarios=['heavy_conflict', 'global_shared'],
    episodes_per_scenario=3,
    physical_delay_scale=0.1,
    seed=42,
))
"
```

**消融维度（当前已实现）：**

| 维度 | 消融内容 | 论文展示名称 |
|------|---------|------------|
| `no_event_wakeup` | 禁用 OoO Scanner，改为轮询/阻塞等待 | No Event-Wakeup |
| `no_orthogonality_check` | 禁用资源正交性检查，强制越级发射 | No Orthogonality Check |

**实验结构（每次运行包含）：**

1. **Sequential** 基线（理论下界）
2. **Traditional DAG** 基线（层并发参照）
3. 各**消融变体**（禁用特定机制）
4. **★ Full Proposed** 完整方案（基准）

---

### 输出文件

运行后结果保存在 `results/` 目录下（含时间戳）：

```
results/
├── main_comparison_YYYYMMDD_HHMMSS/
│   ├── all_results.csv     # 逐 episode 详细记录
│   └── summary.json        # 汇总指标（含 speedup_vs_seq）
├── ablation_YYYYMMDD_HHMMSS/
│   ├── ablation_results.csv
│   └── ablation_summary.json
```

**核心指标说明：**

| 指标 | 说明 | 目标方向 |
|------|------|---------|
| `makespan_s` | 总执行时长（秒） | 越小越好 |
| `overlap_ratio` | I/O-Compute 并行效率（0~1） | 越大越好 |
| `ooo_promotion_count` | 越级调度次数 | 越多越好 |
| `conflict_stall_time_ms` | 资源竞争导致的等待时间 | 越小越好 |
| `speedup_vs_seq` | 相对于 Sequential 的加速比 | 越大越好 |
| `success_rate` | 任务成功率 | 越大越好 |

---

### 实验设计原则

1. **物理延迟缩放**：`physical_delay_scale` 控制模拟延迟倍率。设为 0.1 表示物理动作耗时为原始的 10%，调度开销占比更明显；设为 0.05 时物理动作更快，调度开销占比更大
2. **场景选择**：论文主实验推荐使用 `heavy_conflict`、`global_shared`、`diamond_deep` 三个高区分度场景
3. **Episode 数量**：快速验证用 `--episodes 1`，论文级结果建议 `--episodes 5`
4. **随机种子**：`--seed 42` 保证结果可复现，论文投稿前建议用不同 seed 多次验证

---

### 实验示例输出

**主对比表格（Table 1）示例：**

```
========================================================================================================
  Table 1: 主实验结果 (Main Comparison, speed=0.1)
========================================================================================================

Scenario               Method              Makespan   SuccRate  Overlap  Conf.Stall  Promo#  Speedup
----------------------------------------------------------------------------------------------------
heavy_conflict        Sequential           12.340s     100.0%     0.0%    0.0           0    1.00x
                      Traditional DAG      8.210s     100.0%    75.3%  10500.0       0    1.50x
                      ★ Full Proposed      6.850s     100.0%    83.1%   3700.0      24    1.80x

─────────────────────────────────────────────────────────────────────────────────────────────────────
  OoO-proposed 相对于 Traditional DAG 的加速比（Makespan 降低）:
─────────────────────────────────────────────────────────────────────────────────────────────────────
  heavy_conflict        :   -16.6%  (8.210s → 6.850s)
========================================================================================================
```

**消融表格（Table 3）示例：**

```
==============================================================================================================
  Table 3: 消融实验 + 基线对比 (speed=0.1)
==============================================================================================================

  Part 1: 主对比（OoO-proposed vs 基线方法）
--------------------------------------------------------------------------------------------------------------

Variant                Makespan   SuccRate  Overlap  Conf.Stall  Promo#  Speedup
----------------------------------------------------------------------------------------------------
  Sequential           12.340s    100.0%     0.0%    0.0          0    1.00x
  Traditional DAG       8.210s    100.0%    75.3%  10500.0       0    1.50x
  ★ Full Proposed       6.850s    100.0%    83.1%   3700.0      24    1.80x

  Part 2: 消融变体（验证内部机制贡献，相对 ★ Full Proposed）
--------------------------------------------------------------------------------------------------------------

Variant                Makespan    ΔMakespan   SuccRate  Overlap  Conf.Stall  Promo#  Speedup
----------------------------------------------------------------------------------------------------
  No Event-Wakeup      10.520s      +53.6%    100.0%    12.3%   8200.0       0    1.17x
  No Orthogonality Check 8.940s   +30.5%    95.0%    68.0%  11500.0       0    1.38x
==============================================================================================================

★ = Full Proposed 基准行（完整方案）
ΔMakespan: 消融相对于基准的变化（正值=变慢/变差，负值=变快/变好）
Speedup: 相对于 Sequential 基线的加速比（越大越好）
==============================================================================================================
```


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
└── benchmarks/                  # ★ V8.0 权威指标评估体系
    ├── metrics_collector.py     # ★ Makespan / I/O-Compute Overlap /
    │                              #   Scheduling Latency(纳秒) / Resource Utility CV /
    │                              #   Consistency Check + AblationComparator
    ├── astro_concurrency.py     # 并发调度 + Ablation 对比测试
    └── run_dataset_eval.py      # 数据集评测 + 物理模拟
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

### 8. LLM 响应净化器（ResponseSanitizer）

DeepSeek-R1 等蒸馏模型会携带大量 `<think>...</think>` 推理过程。上层 Planner Agent 和 Executor Agent 的 JSON/DAG 解析器收到后会崩溃（生成 `FSM({})` 等垃圾数据）。

```
原始 LLM 输出:
  <think>用户要设置温度，我应该先检查当前状态...
</think>[{"id": "A", "skill": "set_temp", "params": {"temp": 37}}]

净化后:
  [{"id": "A", "skill": "set_temp", "params": {"temp": 37}}]
```

**非流式**：`GatewayProxy.chat()` 返回前，`sanitize()` 一次性替换所有思考标签。

**流式**：`_do_streaming_request` 逐 chunk 过滤，包含不完整 `<think>` 标签的 chunk 会被丢弃，完整标签对的内容会被替换。

**网关透明原则**：上层 Agent 无需修改任何解析逻辑，净化在网关层完成。

---

### 11. 令牌桶带宽管控与 QoS 三级队列（V7.5）

SpaceWire 总线令牌桶按 `spacewire_bandwidth_kbps` 发放令牌（Byte/s），允许短时突发（`burst_capacity_bytes`），但长期速率不超过带宽上限。QoS 发送策略：`CRITICAL` 无限额优先 → `NORMAL` 令牌受限 → `LOW` 令牌受限。

**CRITICAL 报警击穿拥塞**：即使 NORMAL/LLOW 队列积压数千帧，`send_critical()` 仍无限额优先发送，模拟硬件报警的硬实时传输要求。

### 12. AoI 感知的遥测覆写（V7.5）

`send_telemetry(data, aoi_key="sensor:temp")` 高频传感器上报时，若 NORMAL 队列中已存在相同 `aoi_key` 的旧帧，新帧直接覆写旧帧引用（队列中的旧帧在发送时自然跳过）。一旦网络疏通，发送的永远是最新鲜的遥测快照，`bytes_saved_by_aoi` 记录节省的总字节数。

### 13. A2A 语义增量同步（V7.5）

废弃每次全量发送 DAG 或全量发送环境上下文的粗暴做法。`A2ASemanticDiff.from_snapshot(old, new)` 对比两个状态快照，只包含节点状态变化（`COMPLETED`/`FAILED`）、遥测差异等语义变更。`router.apply_incoming_diff()` 对端收到 Diff 后自动合并到本地状态。`bytes_saved` 字段记录单次 Diff 相比全量发送节省的字节数。

### 14. 弱网抗性指标（V7.5）

- `critical_avg_queue_latency_ms`：CRITICAL 包平均排队延迟（应显著低于 NORMAL 包）
- `bytes_saved_by_aoi`：AoI 覆写累计节省字节
- `normal_avg_queue_latency_ms`：NORMAL 包平均排队延迟
- `total_bytes_saved_by_diff`：A2A Diff 累计节省字节

解决"物理 I/O 极慢，LLM 计算极快"的非对称矛盾。当 Worker Pool 有空闲槽位但 ReadyQueue 为空时，调度器主动遍历 BlockedQueue，执行三重准入门检查（逻辑依赖完成 / 资源未被占用 / 联锁不拦截），将符合条件的节点越级推入 ReadyQueue，实现 I/O 与计算的重叠执行。

**五层防死锁策略**：锁顺序协议（字母序加锁）→ 三重准入门 → 资源预约原子性 → 推进保证 → 超时降级兜底。

**ActiveResourceTable** 维护全局硬件资源占用快照，`_extract_required_resources` 从 skill_name 推断所需资源，`_check_interlock` 接入 labs/interlock_engine。

**wait_for_condition** 让 Worker 通过 `await Future` 让出控制权，后台遥测流 `batch_write` 时通过 `future.set_result()` 瞬间唤醒，零轮询开销。

### 16. 异构计算调度 — 意图感知模型路由（V8.0 核心新增）

太空环境的算力极度不对等：大显存卡跑 7B+ 模型做复杂规划，小卡跑 1.5B 模型做极速 Tool Calling。V8.0 引入**异构算力池**和**意图感知路由**，彻底解决算力碎片与浪费问题。

#### V8.0 新增：透明模型重写 + 时间维度优先级切换

当 heavy 实例 VRAM ≥ W_high (0.85) 时，自动执行模型名称重写（查 `_MODEL_DOWNGRADE_MAP`：qwen-7b → qwen-1.5b）并路由至 light 池。时间维度优先级 `_compute_time_priority()` 返回 0=正常 / 1=降级 / 2=高压。复合路由键 `prefix_hash#compute_class` 实现跨节点 KV-Cache 隔离复用。

#### 核心概念

| 概念 | 说明 |
|------|------|
| `compute_class` | 后端实例的算力分级标签：`heavy`（7B+ 大模型）或 `light`（1.5B 轻量模型） |
| `AgentIntent` | Agent 意图枚举：`PLANNER`（复杂规划）/ `EXECUTOR`（工具调用）/ `QA`（问答） |
| 意图检测 | 根据 `agent_id` 前缀或 `tags` 自动识别 Agent 类型，决定分发算力池 |
| 动态降级 | heavy 实例 VRAM/Active Connections 超限 → 透明降级到 light 实例 + Header 标记 |

#### 路由决策流

```
Agent 请求（带 agent_id / tags）
    │
    ▼
┌──────────────────────────────────────────┐
│  1. 意图检测 (detect_agent_intent)         │
│     agent_id 前缀匹配                      │
│     "planner_*" → PLANNER → heavy         │
│     "executor_*" / "worker_*" → EXECUTOR → light │
│     "qa_*" / tags 含 "qa" → QA → light    │
│     默认 → FALLBACK → heavy               │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  2. 亲和路由（Prefix Hash + compute_class 过滤）│
│     Hash 命中 AND 实例 compute_class 匹配   │
│     AND 实例 VRAM < High Watermark        │
│     → 直连历史实例（KV-Cache 复用）         │
│     （不匹配 → 跳过，进入意图降级路由）       │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  3. 意图降级路由（V8.0 核心）               │
│     heavy 请求但 heavy 实例满载             │
│     → 自动降级到 light 实例                │
│     透明重写 model_name（查降级映射表）      │
│     响应标记 downgraded: true              │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  4. 兜底路由（Least-Connections）          │
│     按 compute_class 过滤后最少连接选择     │
└──────────────────────────────────────────┘
```

#### 降级模型映射（config.yaml）

```yaml
intent_routing:
  default_compute_class: "heavy"
  allow_downgrade: true
  downgrade_model_map:         # 降级后使用的替代模型
    "qwen2.5:7b"              : "qwen2.5:1.5b"
    "qwen2.5-7b-instruct"    : "qwen2.5-1.5b-instruct"
    "deepseek-7b"             : "deepseek-1.5b"
```

#### 透明降级示例

当 Normal 优先级的 QA 查询请求 `qwen2.5:7b` 但 heavy 实例 VRAM 超 85% 时：

```python
# 请求（上层 Agent 无需感知降级）
GatewayRequest(
    model="qwen2.5:7b",
    agent_id="qa_agent_01",   # → 意图检测为 QA → light 优先
    priority="NORMAL",
)

# 响应（含降级元信息，上层 Agent 可见）
GatewayResponse(
    model="qwen2.5:1.5b",     # 自动替换为轻量替代模型
    compute_class="light",      # 实际分发的算力级别
    downgraded=True,            # 透明降级标记
    downgrade_reason="算力降级: qwen2.5:7b → qwen2.5:1.5b (高算力实例满载)",
    ...
)
```

#### 异构计算指标埋点（V7.5）

- `compute_downgrade_count`：触发算力降级的总次数
- `light_model_routed_count`：成功分发给低算力小模型的极速请求次数
- `heavy_model_routed_count`：分发给高算力大模型的请求次数
- `light_model_routed_rate`：`light / (light + heavy)` 分发比率
- `downgrade_rate`：`compute_downgrade / total_requests` 降级触发率

### 17. 动态子图挂载（V7.4）

`DAGTaskGraph.mount_sub_dag(parent_node_id, sub_dag)` 在某节点完成后动态注入子图，支持 Planner 的"空闲算力投机预计算"。自动分配节点 ID 前缀（`parent::`）、重建 Kahn 拓扑排序、检测循环依赖。

---

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AstroSASF V8.0 系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    客户端层 (Client)                                    │  │
│  │   航天员 / Agent → HTTP 请求 → REST API → 结果展示                      │  │
│  │   + X-Astro-OoO / X-Astro-Downgraded 响应头（实验记录）                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              北向接口层 (interface/)                                    │  │
│  │    FastAPI Server ← Facade (V8.0) ← Agent (Q&A / Planner / Executor) │  │
│  │    + LLMResponseMetadata（含 X-Astro-* 响应头）                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              分布式 LLM 网关 (infra/gateway/)                          │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  意图检测 (agent_id/tags) → 异构算力路由 (V8.0)                   │  │  │
│  │  │  Planner Agent ──→ heavy (7B+)  │  Executor/QA ──→ light (1.5B)  │  │  │
│  │  │  VRAM ≥ W_high → 透明降级 (qwen-7b → qwen-1.5b) + TimePriority   │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │  PrefixAwareLoadBalancer ──→ VRAMWatermarkBreaker ──→ LLMInstancePool │  │
│  │       (复合路由键 prefix#cc)          (优先级准入+动态降级)    (异构节点池)   │  │
│  │                                                                         │  │
│  │  ┌────────────────────┐  ┌────────────────────┐                       │  │
│  │  │  heavy 实例 (7B+)   │  │  light 实例 (1.5B) │  ← config.yaml        │  │
│  │  │  SGLang / vLLM     │  │  Ollama / 小卡      │                       │  │
│  │  └────────────────────┘  └────────────────────┘                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              内核调度层 (scheduler/) (V8.0)                            │  │
│  │   DAG Orchestrator → PriorityQueue → Worker Pool                       │  │
│  │   ┌────────────────────────────────────────────────────────────────┐ │  │
│  │   │  V8.0 OoO Scanner（后台主动乱序提取）                                │ │  │
│  │   │  ActiveResourceTable（五层防死锁 + 超时自动释放）                    │ │  │
│  │   │  Temporal Yield（asyncio.Future I/O 挂起）                        │ │  │
│  │   │  Scheduling Latency（纳秒级埋点）                                   │ │  │
│  │   └────────────────────────────────────────────────────────────────┘ │  │
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

### config.yaml 示例（V7.5 异构算力池配置）

```yaml
gateway:
  vram_high_watermark: 0.85       # 高水位（限流）
  vram_critical_watermark: 0.92   # 熔断线
  vram_low_watermark: 0.60       # 完全健康

  # V7.5 新增：意图感知路由配置
  intent_routing:
    default_compute_class: "heavy"
    allow_downgrade: true
    downgrade_model_map:
      "qwen2.5:7b"            : "qwen2.5:1.5b"
      "qwen2.5-7b-instruct"   : "qwen2.5-1.5b-instruct"

  backends:
    # ── 低算力极速实例（1.5B）───────────────────────────────────── #
    - url: "http://localhost:11434"
      provider: "ollama"
      model_name: "qwen2.5:1.5b"
      compute_class: "light"       # ★ V7.5 异构核心
      weight: 3
      tags: ["local", "fallback", "1.5b"]

    # ── 高算力大模型实例（7B+）────────────────────────────────── #
    - url: "http://10.244.37.59:8000"
      provider: "sglang"
      model_name: "qwen2.5:7b"
      compute_class: "heavy"       # ★ V7.5 异构核心
      weight: 2
      tags: ["v100", "node-1", "7b"]
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
| **网关流水线** | PromptTransformationPipeline | 流量拦截与 Prompt 重构 |
| **前缀缓存** | RadixTree (SGLang) + SHA-256 排序 | KV-Cache 复用 |
| **负载均衡** | 前缀哈希 + 最少连接 | LLM 路由 |
| **熔断** | 状态机 + VRAM 水位 | 保护 LLM 实例 |
| **总线模拟** | SpaceWire 协议 | 低带宽遥测约束 |

---

## 版本历史

| 版本 | 日期 | 核心变化 |
|------|------|----------|
| **V7.5** | 2026-04-04 | 令牌桶带宽限流（QoS 三级队列）；AoI 遥测覆写（NORMAL 队列去重）；A2A 语义增量同步 `A2ASemanticDiff`（废弃全量发送）；`bytes_saved_by_aoi` / `critical_avg_queue_latency_ms` / `total_bytes_saved_by_diff` 埋点；OoO 乱序越级执行（ActiveResourceTable + 五层防死锁）；`wait_for_condition` 零开销事件驱动 I/O 挂起；动态子图挂载 `mount_sub_dag`；ResponseSanitizer（剥离 <think>/</think> 推理噪声）；**异构计算调度**（意图感知模型路由 Planner→heavy / Executor→light + compute_class 算力分级 + 动态透明降级 + 时间维度优先级切换 + 模型名透明重写 + 复合路由键 `prefix_hash#compute_class` 联动 KV-Cache 隔离复用）；**权威指标评估体系**（Makespan 对比 / I/O-Compute Overlap Rate / Scheduling Latency 纳秒级 / Resource Utility CV 波动 / Consistency Check + AblationComparator）；**Facade X-Astro-\* 响应头**（OoO/Downgraded/Compute-Class/Model/Routing-Strategy/Time-Priority） |
| **V8.0** | 2026-04-07 | V8.0 内核重写：OoO 后台主动扫描协程 + 资源正交性 R(v_k)∩R_active=∅ + 五层防死锁（超时自动释放）+ 时间维度挂起 asyncio.Future + 调度时延纳秒级埋点 + prefix_balancer 透明模型重写 + 时间维度优先级切换 + 权威指标体系（Makespan/I/O-Overlap/Latency/CV/Consistency）+ Facade X-Astro-\* 响应头 |
| **V7.4** | 2026-04-04 | OoO 乱序越级执行（ActiveResourceTable + 三重准入门 + 五层防死锁）；`wait_for_condition` 零开销事件驱动 I/O 挂起（Pub/Sub + asyncio.Future）；动态子图挂载 `mount_sub_dag`（Kahn 拓扑实时重建）；`ooo_execution_count` / `io_compute_overlap_ms` 埋点（V8.0 重写为后台主动扫描 + 超时自动释放 + 纳秒级时延埋点） |
| **V7.3** | 2026-04-04 | ResponseSanitizer（剥离 <think>/</think> 推理噪声）；强制 JSON 输出指令保护；`asyncio.get_running_loop().create_task()` 修复；流式 SSE chunk 净化；PromptTransformationPipeline RAG 确定性重排 + StaticMCP Schema 前缀锁 + SpeculativeWarmer 跨 Agent 预热；Task 0 强制 model 查表覆盖；`cross_agent_cache_hits` / `tool_schema_saved_tokens` 埋点 |
| **V7.2** | 2026-04-04 | 分布式网关重构：移除单实例 `llm` 节点 → `gateway.backends[]` 多后端配置阵列；`config.yaml` 驱动 `init_distributed_gateway()`；移除 `create_llm()` LangChain 工厂；修复 `_check_instance` async lock 持有 bug |
| **V7.1** | 2026-03 | 数据集评测 + LLM 算力埋点 + 物理模拟层 |
| **V7.0** | 2026-03 | DAG 双轨调度 + 理论/实践智能体分离 |
| **V6.2** | 2026-01 | LLM 语义路由替代 BM25 |
| **V5.1** | 2026-01 | 优先级抢占式调度 |
| **V5.0** | 2026-01 | 正交联锁引擎替代单体 FSM |

---

## License

MIT
