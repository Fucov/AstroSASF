"""
AstroSASF · Benchmarks · Benchmark Generator
==========================================
V2 Schema Benchmark 生成器 —— 将 astro_bench.jsonl 升级为支持任务 DAG、
设备作用域、chaos 注入、量化指标的完整格式。

升级要点：
- 每个 episode 包含显式 task_graph（而非纯自然语言 prompt）
- 显式 device_requirements（含 scope: global_shared / cabin_exclusive）
- 完整的 chaos_events（hardware_delay / telemetry_alarm / failure）
- expected_outcomes（成功条件）
- 支持 4-tier × 3-level × 5 episodes = 60 条 benchmark

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import json
import random
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ── 动态项目根路径（支持 uv run / 直接 python / 任意 cwd）──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scheduler.device_model import (
    DEFAULT_DEVICE_REGISTRY,
    DeviceSchema,
    DeviceType,
)

# ────────────────────────────────────────────────────────────────────────────── #
#  Schema Enums                                                                 #
# ────────────────────────────────────────────────────────────────────────────── #

class DifficultyLevel(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class ScenarioType(Enum):
    NO_CONFLICT = "no_conflict"       # 线性 DAG，无资源竞争
    LIGHT_CONFLICT = "light_conflict" # 宽 DAG，每层同设备竞争 <=2 节点
    HEAVY_CONFLICT = "heavy_conflict" # 宽 DAG，每层同设备竞争 2~4 节点
    ALARM_RECOVERY = "alarm_recovery"  # 含 telemetry_alarm，触发抢占
    OOO_STRESS = "ooo_stress"          # 深层宽 DAG，每层强制同设备竞争 + chaos
    # ── 新增极端场景 ──────────────────────────────────────────────────────
    GLOBAL_SHARED = "global_shared"     # 全局共享设备竞争（co2_controller）
    DIAMOND_DEEP = "diamond_deep"      # 深层钻石形依赖（测试 OoO 越级能力）


# ────────────────────────────────────────────────────────────────────────────── #
#  Task Graph Schema                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class TaskNodeDef:
    task_id: str
    skill_name: str
    params: dict[str, Any] = field(default_factory=dict)
    required_devices: list[str] = field(default_factory=list)
    estimated_compute_ms: float = 50.0    # LLM 推理估算时间
    estimated_physical_ms: float = 0.0  # 由 DeviceSchema.compute_latency 计算
    safety_guard: str | None = None
    priority: str = "NORMAL"
    resumable: bool = True
    branchable: bool = False


@dataclass
class TaskGraphDef:
    nodes: list[TaskNodeDef] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (from_id, to_id)


# ────────────────────────────────────────────────────────────────────────────── #
#  Device Requirements Schema                                                   #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class DeviceReqDef:
    device_id: str
    scope: str  # "global_shared" / "cabin_shared" / "cabin_exclusive"
    allowed_cabins: list[str] | None = None


# ────────────────────────────────────────────────────────────────────────────── #
#  Chaos Event Schema                                                           #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class ChaosEventDef:
    trigger_time_sec: float
    type: str   # hardware_delay / telemetry_alarm / failure / timeout / device_steal
    target_tool: str | None = None
    delay_multiplier: float | None = None
    telemetry_key: str | None = None
    override_value: Any | None = None


# ────────────────────────────────────────────────────────────────────────────── #
#  Benchmark Episode                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class BenchmarkEpisode:
    episode_id: str
    scenario_type: ScenarioType
    difficulty: DifficultyLevel
    description: str
    cabins: list[str]
    task_graph: TaskGraphDef
    device_requirements: list[DeviceReqDef]
    initial_telemetry: dict[str, float]
    chaos_events: list[ChaosEventDef]
    expected_outcomes: dict[str, Any] = field(default_factory=dict)
    evaluation_tags: list[str] = field(default_factory=list)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_type": self.scenario_type.value,
            "difficulty": self.difficulty.value,
            "description": self.description,
            "cabins": self.cabins,
            "task_graph": {
                "nodes": [asdict(n) for n in self.task_graph.nodes],
                "edges": [[e[0], e[1]] for e in self.task_graph.edges],
            },
            "device_requirements": [asdict(d) for d in self.device_requirements],
            "initial_telemetry": self.initial_telemetry,
            "chaos_events": [asdict(c) for c in self.chaos_events],
            "expected_outcomes": self.expected_outcomes,
            "evaluation_tags": self.evaluation_tags,
            "seed": self.seed,
        }


# ────────────────────────────────────────────────────────────────────────────── #
#  设备池（benchmark 专用扩展设备）                                              #
# ────────────────────────────────────────────────────────────────────────────── #

BENCHMARK_DEVICE_POOL: dict[str, DeviceSchema] = dict(DEFAULT_DEVICE_REGISTRY)

# 新增 benchmark 专用设备
BENCHMARK_DEVICE_POOL.update({
    "heater_mat": DeviceSchema(
        device_id="heater_mat",
        device_type=DeviceType.HEATER,
        scope=BENCHMARK_DEVICE_POOL["heater_bio"].scope,
        allowed_cabins={"DemoMaterial"},
        jitter_sigma_ms=80.0,
        physics_params={"heat_rate_C_per_sec": 3.0, "max_temperature": 500.0},
    ),
    "heater_plant": DeviceSchema(
        device_id="heater_plant",
        device_type=DeviceType.HEATER,
        scope=BENCHMARK_DEVICE_POOL["heater_bio"].scope,
        allowed_cabins={"DemoPlant"},
        jitter_sigma_ms=30.0,
        physics_params={"heat_rate_C_per_sec": 1.5, "max_temperature": 45.0},
    ),
    "centrifuge_bio": DeviceSchema(
        device_id="centrifuge_bio",
        device_type=DeviceType.CENTRIFUGE,
        scope=BENCHMARK_DEVICE_POOL["heater_bio"].scope,
        allowed_cabins={"DemoBio"},
        jitter_sigma_ms=500.0,
        physics_params={"ramp_up_ms": 3000.0, "ramp_down_ms": 5000.0},
    ),
    "arm_mat": DeviceSchema(
        device_id="arm_mat",
        device_type=DeviceType.ROBOTIC_ARM,
        scope=BENCHMARK_DEVICE_POOL["arm_bio"].scope,
        allowed_cabins={"DemoMaterial"},
        jitter_sigma_ms=150.0,
        physics_params={"speed_deg_per_sec": 20.0, "grip_overhead_ms": 300.0},
    ),
    "arm_plant": DeviceSchema(
        device_id="arm_plant",
        device_type=DeviceType.ROBOTIC_ARM,
        scope=BENCHMARK_DEVICE_POOL["arm_bio"].scope,
        allowed_cabins={"DemoPlant"},
        jitter_sigma_ms=80.0,
        physics_params={"speed_deg_per_sec": 40.0, "grip_overhead_ms": 150.0},
    ),
    "vacuum_mat": DeviceSchema(
        device_id="vacuum_mat",
        device_type=DeviceType.VACUUM,
        scope=BENCHMARK_DEVICE_POOL["vacuum_bio"].scope,
        allowed_cabins={"DemoMaterial"},
        jitter_sigma_ms=300.0,
        physics_params={"pump_speed_kpa_per_sec": 3.0, "seal_check_ms": 3000.0},
    ),
    "pump_plant": DeviceSchema(
        device_id="pump_plant",
        device_type=DeviceType.PUMP,
        scope=BENCHMARK_DEVICE_POOL["pump_fluid"].scope,
        allowed_cabins={"DemoPlant"},
        jitter_sigma_ms=80.0,
        physics_params={"flow_rate_ml_per_sec": 3.0, "startup_ms": 800.0},
    ),
    "scan_bio": DeviceSchema(
        device_id="scan_bio",
        device_type=DeviceType.SCAN,
        scope=BENCHMARK_DEVICE_POOL["heater_bio"].scope,
        allowed_cabins={"DemoBio"},
        jitter_sigma_ms=1000.0,
        physics_params={"sample_prep_ms": 5000.0, "scan_duration_ms": 30000.0},
    ),
})


# ────────────────────────────────────────────────────────────────────────────── #
#  Benchmark Generator                                                         #
# ────────────────────────────────────────────────────────────────────────────── #

class BenchmarkGenerator:
    """可配置的 benchmark episode 生成器。"""

    # 每个舱对应的设备映射（lab_id → [device_ids]）
    LAB_DEVICE_MAP: dict[str, list[str]] = {
        "DemoBio":     ["heater_bio", "vacuum_bio", "arm_bio", "centrifuge_bio", "scan_bio"],
        "DemoFluid":   ["pump_fluid", "valve_fluid"],
        "DemoMaterial":["heater_mat", "arm_mat", "vacuum_mat"],
        "DemoPlant":   ["heater_plant", "arm_plant", "pump_plant"],
    }

    # 全局共享设备
    GLOBAL_SHARED_DEVICES = ["co2_controller", "water_pump_global"]

    # 全局共享舱映射
    GLOBAL_DEVICE_SCOPE: dict[str, list[str]] = {
        "co2_controller":   ["DemoBio", "DemoFluid", "DemoMaterial", "DemoPlant"],
        "water_pump_global":["DemoBio", "DemoFluid", "DemoMaterial", "DemoPlant"],
    }

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._episode_counter = 0

    def _next_id(self) -> str:
        self._episode_counter += 1
        return f"bench-v2-{self._episode_counter:04d}"

    def _random_float(self, lo: float, hi: float) -> float:
        return self._rng.uniform(lo, hi)

    def _random_choice(self, opts: list, k: int = 1) -> Any:
        if k == 1:
            return self._rng.choice(opts)
        return self._rng.sample(opts, k=k)

    # ── Tier-1: 无冲突、短等待 ───────────────────────────────────────────────

    def generate_tier1(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.EASY) -> list[BenchmarkEpisode]:
        """Tier-1: 宽 DAG，无资源竞争（同层节点用不同设备）。"""
        episodes = []
        lab_ids = list(self.LAB_DEVICE_MAP.keys())

        dag_depth_map = {
            DifficultyLevel.EASY:   4,
            DifficultyLevel.MEDIUM: 5,
            DifficultyLevel.HARD:   6,
        }

        dag_depth = dag_depth_map[difficulty]

        for i in range(count):
            lab = self._random_choice(lab_ids)
            devices = list(self.LAB_DEVICE_MAP[lab])
            ep_id = f"t1-wd-{difficulty.value}-{i+1:02d}"

            # 宽 DAG（每层 2~3 节点），但设备随机分布（同层无竞争）
            nodes, edges = self._build_dag(
                lab_id=lab,
                devices=devices,
                depth=dag_depth,
                include_shared=False,
                min_nodes_per_level=2,
                max_nodes_per_level=3,
            )

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.NO_CONFLICT,
                difficulty=difficulty,
                description=f"[Tier-1] {difficulty.value} 宽 DAG 无冲突：{dag_depth}层×2~3节点，验证基础并行调度",
                cabins=[lab],
                task_graph=TaskGraphDef(nodes=nodes, edges=edges),
                device_requirements=self._build_device_reqs(devices, lab, shared=False),
                initial_telemetry=self._default_telemetry(lab),
                chaos_events=[],
                expected_outcomes={"success_rate": 1.0, "ooo_promotion_min": 0},
                evaluation_tags=["tier1", difficulty.value.lower(), "wide_dag", "no_conflict"],
                seed=self._seed + i,
            )
            episodes.append(episode)
        return episodes

    # ── Tier-2: 有冲突、短等待 ───────────────────────────────────────────────

    def generate_tier2(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> list[BenchmarkEpisode]:
        """Tier-2: 宽 DAG + 层内同设备竞争（激发 OoO Scanner 越级调度）。

        关键设计：每层 2~3 个节点故意使用相同设备，在舱内制造资源竞争，
        而非依赖跨舱共享设备。hardware_delay 在中间层触发，测试 OoO Scanner
        在设备持有期间发现并越级处理等待节点的能力。
        """
        episodes = []
        delay_map = {
            DifficultyLevel.EASY:   2.0,
            DifficultyLevel.MEDIUM: 2.5,
            DifficultyLevel.HARD:   3.0,
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   5,
            DifficultyLevel.MEDIUM: 6,
            DifficultyLevel.HARD:   7,
        }

        delay_mult = delay_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=2)

        for i in range(count):
            ep_id = f"t2-ooo-{difficulty.value}-{i+1:02d}"

            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                # 强制同层竞争：每层 2~3 节点，全部用同一个设备类型
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=False,
                    min_nodes_per_level=2,
                    max_nodes_per_level=3,
                    force_competition_per_level=2,  # 每层强制 2 个节点竞争同设备
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.LIGHT_CONFLICT,
                difficulty=difficulty,
                description=f"[Tier-2] {difficulty.value} OoO激发：{dag_depth}层宽DAG，层内同设备竞争，触发越级调度",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=False),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(1.0, 2.0),
                        type="hardware_delay",
                        target_tool="heater_bio",  # 锁定 heater 制造延迟，触发越级
                        delay_multiplier=delay_mult,
                    )
                ],
                expected_outcomes={
                    "success_rate": 0.95,
                    "ooo_promotion_min": 3,
                    "overlap_ratio_min": 0.15,
                },
                evaluation_tags=["tier2", difficulty.value.lower(), "ooo_light", "layer_competition"],
                seed=self._seed + i,
            )
            episodes.append(episode)
        return episodes

    # ── Tier-3: 深层竞争 + 长延迟设备 ─────────────────────────────────────────

    def generate_tier3(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> list[BenchmarkEpisode]:
        """Tier-3: 深层宽 DAG + 钻石形依赖（最强 OoO 激发场景）。

        关键设计：
        - depth 6~8 层，每层 3~4 个节点（大量节点）
        - diamond_mode=True：所有节点依赖上一层所有节点（全连接扇入）
        - 同层竞争 + 跨层钻石形，制造大量"条件满足但资源被占"的场景
        - 长延迟设备 + hardware_delay，延长资源持有时间
        """
        episodes = []
        long_devices = ["vacuum_bio", "centrifuge_bio", "vacuum_mat"]
        delay_map = {
            DifficultyLevel.EASY:   2.5,
            DifficultyLevel.MEDIUM: 3.0,
            DifficultyLevel.HARD:   4.0,
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   6,
            DifficultyLevel.MEDIUM: 7,
            DifficultyLevel.HARD:   8,
        }

        delay_mult = delay_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=2)

        for i in range(count):
            ep_id = f"t3-diamond-{difficulty.value}-{i+1:02d}"
            long_dev = self._random_choice(long_devices)

            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=False,
                    force_device=long_dev,          # 强制使用长延迟设备
                    min_nodes_per_level=3,
                    max_nodes_per_level=4,
                    force_competition_per_level=3,   # 每层 3 个节点强竞争
                    diamond_mode=True,               # 钻石形全连接依赖
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.HEAVY_CONFLICT,
                difficulty=difficulty,
                description=f"[Tier-3] {difficulty.value} 钻石DAG+深层竞争：{dag_depth}层×3~4节点，{long_dev}长延迟，OoO越级核心测试",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=False),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(2.0, 4.0),
                        type="hardware_delay",
                        target_tool=long_dev,
                        delay_multiplier=delay_mult,
                    )
                ],
                expected_outcomes={
                    "success_rate": 0.90,
                    "ooo_promotion_min": 8,
                    "overlap_ratio_min": 0.25,
                },
                evaluation_tags=["tier3", difficulty.value.lower(), "diamond_dag", "heavy_competition"],
                seed=self._seed + i + 100,
            )
            episodes.append(episode)
        return episodes

    # ── Tier-4: OoO 压力测试 ─────────────────────────────────────────────────

    def generate_tier4(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.HARD) -> list[BenchmarkEpisode]:
        """Tier-4: OoO 压力测试（最强竞争 + 告警 + 深层抢占恢复）。

        关键设计：
        - 深度 8~10 层，每层 3~4 节点 + 钻石形
        - hardware_delay 故意在深层触发（让 OoO Scanner 在深度等待队列中越级）
        - telemetry_alarm 触发抢占，验证 OoO Scanner 在告警干扰下的越级能力
        - multi_cabin 逃生路径：多条路径竞争同一关键设备
        """
        episodes = []
        alarm_map = {
            DifficultyLevel.EASY:   ("temperature", 80.0),
            DifficultyLevel.MEDIUM: ("co2_level", 2000.0),
            DifficultyLevel.HARD:   ("smoke_level", 0.9),
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   8,
            DifficultyLevel.MEDIUM: 9,
            DifficultyLevel.HARD:   10,
        }

        alarm_key, alarm_value = alarm_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        delay_mult = {"Easy": 3.0, "Medium": 4.0, "Hard": 5.0}[difficulty.value]
        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=3)

        for i in range(count):
            ep_id = f"t4-ooo-stress-{difficulty.value}-{i+1:02d}"

            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=False,
                    min_nodes_per_level=3,
                    max_nodes_per_level=4,
                    force_competition_per_level=3,
                    diamond_mode=True,  # 钻石形全连接，OoO 越级核心
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.OOO_STRESS,
                difficulty=difficulty,
                description=f"[Tier-4] {difficulty.value} OoO压力：{dag_depth}层钻石DAG，alarm={alarm_key}，深层抢占恢复",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=False),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(dag_depth * 0.3, dag_depth * 0.5),
                        type="hardware_delay",
                        target_tool="heater_bio",
                        delay_multiplier=delay_mult,
                    ),
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(dag_depth * 0.5, dag_depth * 0.7),
                        type="telemetry_alarm",
                        telemetry_key=alarm_key,
                        override_value=alarm_value,
                    ),
                ],
                expected_outcomes={
                    "success_rate": 0.85,
                    "ooo_promotion_min": 12,
                    "overlap_ratio_min": 0.30,
                    "alarm_response_ms": 500.0,
                },
                evaluation_tags=["tier4", difficulty.value.lower(), "ooo_stress", "diamond_dag", "alarm_recovery"],
                seed=self._seed + i + 200,
            )
            episodes.append(episode)
        return episodes

    # ═══════════════════════════════════════════════════════════════════════════ #
    #  Extreme Tier: 极端场景（真正测试 OoO 越级调度能力）                        #
    # ═══════════════════════════════════════════════════════════════════════════ #

    def generate_extreme_shared(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.HARD) -> list[BenchmarkEpisode]:
        """极端场景：全局共享设备竞争。

        设计目标：让 co2_controller 成为全局瓶颈，所有任务竞争这一个设备。
        这迫使 OoO Scanner 必须主动越级调度，否则大部分 Worker 会一直阻塞等待。

        DAG 结构：
        - 深层 DAG（12~15 层）
        - 每层 4~6 个节点
        - 所有节点都需要 co2_controller（全局共享）
        - 钻石形依赖确保大量"等待但可越级"的场景

        预期效果：
        - sequential: 任务串行执行，Worker 大部分时间阻塞在 co2_controller 上
        - ooo_proposed: OoO Scanner 主动发现空闲 Worker 并越级调度其他可用任务
        - 预期提升：30~50% 的 makespan 减少
        """
        episodes = []
        dag_depth_map = {
            DifficultyLevel.EASY:   12,
            DifficultyLevel.MEDIUM: 14,
            DifficultyLevel.HARD:   15,
        }
        nodes_per_level_map = {
            DifficultyLevel.EASY:   4,
            DifficultyLevel.MEDIUM: 5,
            DifficultyLevel.HARD:   6,
        }

        dag_depth = dag_depth_map[difficulty]
        num_per_level = nodes_per_level_map[difficulty]
        delay_mult = {"Easy": 2.0, "Medium": 3.0, "Hard": 4.0}[difficulty.value]

        # 使用一个舱，所有节点都竞争 co2_controller
        lab = "DemoBio"
        devices = ["co2_controller"]  # 全局共享设备

        for i in range(count):
            ep_id = f"ext-shared-{difficulty.value}-{i+1:02d}"

            nodes, edges = self._build_dag(
                lab_id=lab,
                devices=devices,
                depth=dag_depth,
                include_shared=True,              # 启用全局共享设备
                shared_device="co2_controller",  # 全局共享设备
                min_nodes_per_level=num_per_level,
                max_nodes_per_level=num_per_level,  # 每层固定数量
                force_competition_per_level=num_per_level,  # 全层竞争
                diamond_mode=True,                 # 钻石形全连接
            )

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.GLOBAL_SHARED,
                difficulty=difficulty,
                description=f"[Extreme] {difficulty.value} 全局共享设备竞争：{dag_depth}层×{num_per_level}节点，全员竞争co2_controller，OoO越级核心测试",
                cabins=[lab],
                task_graph=TaskGraphDef(nodes=nodes, edges=edges),
                device_requirements=self._build_device_reqs(
                    exclusive_devices=[],
                    cabins=[lab],
                    shared=True,
                    shared_device="co2_controller",
                ),
                initial_telemetry=self._default_telemetry(lab),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(dag_depth * 0.4, dag_depth * 0.6),
                        type="hardware_delay",
                        target_tool="co2_controller",
                        delay_multiplier=delay_mult,
                    ),
                ],
                expected_outcomes={
                    "success_rate": 0.85,
                    "ooo_promotion_min": dag_depth * 2,  # 期望至少 2x 深度的越级次数
                    "overlap_ratio_min": 0.40,
                },
                evaluation_tags=["extreme", difficulty.value.lower(), "global_shared", "co2_bottleneck"],
                seed=self._seed + i + 300,
            )
            episodes.append(episode)
        return episodes

    def generate_extreme_diamond(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.HARD) -> list[BenchmarkEpisode]:
        """极端场景：深层钻石形依赖。

        设计目标：制造大量"条件满足但资源被占"的场景，让 OoO Scanner 必须越级。

        DAG 结构：
        - 超深 DAG（15~20 层）
        - 宽钻石形：每层 4~6 个节点
        - 全连接依赖：每个节点依赖上一层所有节点
        - 故意使用长延迟设备（centrifuge_bio: 3~5 秒）

        关键特性：
        1. 钻石形依赖 → 大量等待节点（上游完成但资源被占）
        2. 长延迟设备 → 资源持有时间长，越级收益高
        3. 超深 DAG → OoO Scanner 有更多越级机会

        预期效果：
        - sequential: 严格串行，centrifuge 延迟主导总时间
        - async_only: 层内并行好，但等待队列堆积
        - ooo_proposed: 主动越级调度，重叠率显著提升
        - 预期提升：40~60% 的 makespan 减少
        """
        episodes = []
        dag_depth_map = {
            DifficultyLevel.EASY:   15,
            DifficultyLevel.MEDIUM: 17,
            DifficultyLevel.HARD:   20,
        }
        nodes_per_level_map = {
            DifficultyLevel.EASY:   4,
            DifficultyLevel.MEDIUM: 5,
            DifficultyLevel.HARD:   6,
        }

        dag_depth = dag_depth_map[difficulty]
        num_per_level = nodes_per_level_map[difficulty]
        delay_mult = {"Easy": 2.5, "Medium": 3.5, "Hard": 5.0}[difficulty.value]

        # 长延迟设备列表
        long_delay_devices = ["centrifuge_bio", "scan_bio", "vacuum_bio"]

        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=2)

        for i in range(count):
            ep_id = f"ext-diamond-{difficulty.value}-{i+1:02d}"
            long_dev = long_delay_devices[i % len(long_delay_devices)]

            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=False,
                    force_device=long_dev,                   # 强制长延迟设备
                    min_nodes_per_level=num_per_level,
                    max_nodes_per_level=num_per_level,
                    force_competition_per_level=num_per_level,  # 全层竞争
                    diamond_mode=True,                         # 钻石形全连接
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.DIAMOND_DEEP,
                difficulty=difficulty,
                description=f"[Extreme] {difficulty.value} 深层钻石DAG：{dag_depth}层×{num_per_level}节点，{long_dev}长延迟，OoO越级极限测试",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=False),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(dag_depth * 0.3, dag_depth * 0.5),
                        type="hardware_delay",
                        target_tool=long_dev,
                        delay_multiplier=delay_mult,
                    ),
                ],
                expected_outcomes={
                    "success_rate": 0.80,
                    "ooo_promotion_min": dag_depth * 3,  # 期望至少 3x 深度的越级次数
                    "overlap_ratio_min": 0.50,
                },
                evaluation_tags=["extreme", difficulty.value.lower(), "deep_diamond", f"long_delay_{long_dev}"],
                seed=self._seed + i + 400,
            )
            episodes.append(episode)
        return episodes

    def generate_extreme_full_suite(self) -> list[BenchmarkEpisode]:
        """生成完整极端场景 benchmark（2 scenario × 3 level × 5 episodes = 30 条）。"""
        episodes = []
        difficulties = [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
        ]
        for diff in difficulties:
            eps = self.generate_extreme_shared(count=5, difficulty=diff)
            episodes.extend(eps)
        for diff in difficulties:
            eps = self.generate_extreme_diamond(count=5, difficulty=diff)
            episodes.extend(eps)
        return episodes

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _build_dag(
        self,
        lab_id: str,
        devices: list[str],
        depth: int,
        include_shared: bool = False,
        shared_device: str | None = None,
        force_device: str | None = None,
        # ── V8.0 宽 DAG 核心参数 ────────────────────────────────────────────
        min_nodes_per_level: int = 1,
        max_nodes_per_level: int = 2,
        # 每层强制使用相同设备的节点数（0=不强制）
        force_competition_per_level: int = 0,
        # 是否使用钻石形依赖（激发 OoO 的关键）
        diamond_mode: bool = False,
    ) -> tuple[list[TaskNodeDef], list[tuple[str, str]]]:
        """构建一个任务 DAG。

        V8.0 核心升级：从「线性链」改为「宽 DAG」，充分激发 OoO 乱序调度。

        设计原则：
        1. 每层 2~4 个节点（而非 1~2 个），增加资源竞争概率
        2. 故意让同层节点使用相同设备，制造资源锁争用
        3. 钻石形依赖：同一层的多个节点共享上游，形成「扇入」结构
        4. chaos event 在设备持有期间触发，测试 OoO Scanner 的越级能力
        5. 全局共享设备（include_shared=True）：所有节点都竞争同一个设备

        示例（宽 DAG + 钻石形）：
            L0: [A] ──→ L1: [C] ──→ L2: [E]
              ↘        ↗ ↘        ↗
               [B] ──→ [D] ──→ [F]
        A 和 B 在 L0 并行竞争 heater；C 和 D 依赖 A/B 但竞争 vacuum；
        OoO Scanner 在 C 等待时主动越级处理 D。
        """
        nodes: list[TaskNodeDef] = []
        edges: list[tuple[str, str]] = []
        rng = random.Random(self._seed + hash(lab_id) % 10000)

        skill_map: dict[str, str] = {
            "heater_bio":     "control_heater",
            "heater_mat":     "set_temperature",
            "heater_plant":   "set_temperature",
            "vacuum_bio":     "toggle_vacuum_pump",
            "vacuum_mat":     "toggle_vacuum_pump",
            "arm_bio":        "move_robotic_arm",
            "arm_mat":        "move_robotic_arm",
            "arm_plant":      "move_robotic_arm",
            "pump_fluid":     "toggle_vacuum_pump",
            "pump_plant":     "inject_nutrient",
            "valve_fluid":    "control_valve",
            "centrifuge_bio": "control_centrifuge",
            "scan_bio":       "read_sensor",
            "co2_controller": "control_co2",
            "water_pump_global": "control_water_pump",
        }

        # V8.0: 按设备类型分组（同类型设备才能在同一层制造竞争）
        device_groups: dict[str, list[str]] = defaultdict(list)
        for dev in devices:
            # 提取设备类型前缀（heater/vacuum/arm/pump/valve/centrifuge/scan）
            base_type = dev.split("_")[0] if "_" in dev else dev
            device_groups[base_type].append(dev)

        prev_level_ids: list[str] = []
        for d in range(depth):
            # ── 确定本层节点数量 ─────────────────────────────────────────
            if force_competition_per_level > 0:
                # OoO stress 模式：强制同设备竞争
                num_in_level = min(force_competition_per_level, len(devices))
            else:
                num_in_level = rng.randint(min_nodes_per_level, max_nodes_per_level)

            # ── 决定本层使用的设备策略（制造竞争）────────────────────────────
            level_devices: list[str] = []
            if diamond_mode and d == 0:
                # 钻石形：第一层用「有多个实例的设备类型」制造竞争
                # 优先选择有多个舱的设备类型：heater / vacuum / arm
                priority_types = ["heater", "vacuum", "arm", "pump", "centrifuge"]
                selected_type = None
                for ptype in priority_types:
                    if ptype in device_groups and len(device_groups[ptype]) >= 2:
                        selected_type = ptype
                        break
                if selected_type:
                    level_devices = device_groups[selected_type][:num_in_level]
                else:
                    level_devices = devices[:num_in_level]
            elif force_competition_per_level > 0:
                # 强制竞争模式：每层都选同一个设备类型，让多个节点竞争它
                # 从优先级设备类型中选择
                priority_types = ["heater", "vacuum", "arm", "pump", "centrifuge"]
                selected_type = None
                for ptype in priority_types:
                    if ptype in device_groups:
                        selected_type = ptype
                        break
                if selected_type:
                    level_device = rng.choice(device_groups[selected_type])
                    level_devices = [level_device] * num_in_level
                else:
                    level_devices = [devices[d % len(devices)]] * num_in_level
            else:
                # 随机选择（可能有竞争，可能没有）
                level_devices = [devices[(d * num_in_level + j) % len(devices)]
                                 for j in range(num_in_level)]

            level_ids: list[str] = []
            for j in range(num_in_level):
                device = level_devices[j]
                if force_device:
                    device = force_device

                skill = skill_map.get(device, "generic_action")
                node_id = f"{lab_id}-L{d}N{j}"

                node = TaskNodeDef(
                    task_id=node_id,
                    skill_name=skill,
                    params=self._device_params(skill, device),
                    required_devices=[device],
                    estimated_compute_ms=rng.uniform(30.0, 100.0),
                    priority=rng.choice(["NORMAL", "NORMAL", "HIGH"]),
                    resumable=True,
                )

                # ── 全局共享设备：所有节点都添加共享设备 ─────────────────────
                if include_shared and shared_device:
                    # 全局共享设备竞争模式：每个节点都竞争同一个设备
                    node.required_devices.append(shared_device)

                nodes.append(node)
                level_ids.append(node_id)

                # ── V8.0 依赖策略 ────────────────────────────────────────
                if diamond_mode and prev_level_ids:
                    # 钻石形：所有节点依赖上一层的所有节点（全连接）
                    for prev_id in prev_level_ids:
                        edges.append((prev_id, node_id))
                elif prev_level_ids:
                    # 普通模式：每个节点依赖上一层的随机一个节点（保留一定并行）
                    dep_target = rng.choice(prev_level_ids)
                    edges.append((dep_target, node_id))

            prev_level_ids = level_ids

        return nodes, edges

    def _device_params(self, skill: str, device: str) -> dict[str, Any]:
        if "heater" in skill or "temperature" in skill:
            return {"temperature": random.Random(self._seed).uniform(37.0, 80.0), "duration": 10}
        if "arm" in skill or "robotic" in skill:
            return {"target_position": self._random_choice(["HOME", "90", "45", "30"])}
        if "vacuum" in skill:
            return {"activate": True}
        if "valve" in skill:
            return {"position": self._random_choice(["OPEN_A", "OPEN_B", "MIXING"])}
        if "pump" in skill or "inject" in skill:
            return {"volume": self._random_float(10.0, 100.0)}
        if "centrifuge" in skill:
            return {"action": "START", "rpm": 3000}
        if "sensor" in skill:
            return {"channel": 1}
        return {}

    def _build_device_reqs(
        self,
        exclusive_devices: list[str],
        cabins: list[str],
        shared: bool = False,
        shared_device: str | None = None,
    ) -> list[DeviceReqDef]:
        reqs = []
        for dev in exclusive_devices:
            reqs.append(DeviceReqDef(device_id=dev, scope="cabin_exclusive", allowed_cabins=cabins))
        if shared and shared_device:
            reqs.append(DeviceReqDef(
                device_id=shared_device,
                scope="global_shared",
                allowed_cabins=list(self.GLOBAL_DEVICE_SCOPE.get(shared_device, cabins)),
            ))
        return reqs

    def _default_telemetry(self, lab_id: str) -> dict[str, float]:
        base = {
            "temperature": 25.0,
            "humidity": 50.0,
            "pressure": 101.325,
            "co2_level": 400.0,
            "smoke_level": 0.0,
            "oxygen_level": 21.0,
            "vacuum_level": 101.325,
        }
        if "Bio" in lab_id:
            base["temperature"] = 25.0
            base["co2_level"] = 0.04
        elif "Material" in lab_id:
            base["temperature"] = 75.0
        return base

    # ── 全量生成 ─────────────────────────────────────────────────────────────

    def generate_full_suite(self) -> list[BenchmarkEpisode]:
        """生成完整 60 条 benchmark（4 tier × 3 level × 5 episodes）。"""
        episodes = []
        difficulties = [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
        ]
        for tier_fn, tier_name in [
            (self.generate_tier1, "tier1"),
            (self.generate_tier2, "tier2"),
            (self.generate_tier3, "tier3"),
            (self.generate_tier4, "tier4"),
        ]:
            for diff in difficulties:
                eps = tier_fn(count=5, difficulty=diff)
                episodes.extend(eps)
        return episodes

    def save(self, episodes: list[BenchmarkEpisode], path: Path) -> None:
        """保存为 JSONL 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        print(f"✅ 已生成 {len(episodes)} 条 benchmark → {path}")

    @classmethod
    def load(cls, path: Path) -> list[BenchmarkEpisode]:
        """从 JSONL 文件加载 benchmark。"""
        episodes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                tg = d.get("task_graph", {})
                nodes = [TaskNodeDef(**n) for n in tg.get("nodes", [])]
                edges = [(e[0], e[1]) for e in tg.get("edges", [])]
                ep = BenchmarkEpisode(
                    episode_id=d["episode_id"],
                    scenario_type=ScenarioType(d["scenario_type"]),
                    difficulty=DifficultyLevel(d["difficulty"]),
                    description=d.get("description", ""),
                    cabins=d.get("cabins", []),
                    task_graph=TaskGraphDef(nodes=nodes, edges=edges),
                    device_requirements=[DeviceReqDef(**dr) for dr in d.get("device_requirements", [])],
                    initial_telemetry=d.get("initial_telemetry", {}),
                    chaos_events=[ChaosEventDef(**ce) for ce in d.get("chaos_events", [])],
                    expected_outcomes=d.get("expected_outcomes", {}),
                    evaluation_tags=d.get("evaluation_tags", []),
                    seed=d.get("seed", 42),
                )
                episodes.append(ep)
        return episodes


__all__ = [
    "BenchmarkGenerator",
    "BenchmarkEpisode",
    "TaskNodeDef",
    "TaskGraphDef",
    "DeviceReqDef",
    "ChaosEventDef",
    "DifficultyLevel",
    "ScenarioType",
    "BENCHMARK_DEVICE_POOL",
]


def main() -> None:
    """CLI 入口：`python -m benchmarks.bench_generator`"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="AstroSASF Benchmark 生成器")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="datasets/astro_bench_v2.jsonl")
    parser.add_argument("--tier", default=None,
                        choices=["tier1", "tier2", "tier3", "tier4", "ext-shared", "ext-diamond", "extreme"])
    parser.add_argument("--count", type=int, default=5,
                        help="每难度级别的 episode 数量")
    args = parser.parse_args()

    gen = BenchmarkGenerator(seed=args.seed)

    if args.tier == "tier1":
        episodes = gen.generate_tier1(count=args.count)
    elif args.tier == "tier2":
        episodes = gen.generate_tier2(count=args.count)
    elif args.tier == "tier3":
        episodes = gen.generate_tier3(count=args.count)
    elif args.tier == "tier4":
        episodes = gen.generate_tier4(count=args.count)
    elif args.tier == "ext-shared":
        episodes = gen.generate_extreme_shared(count=args.count)
    elif args.tier == "ext-diamond":
        episodes = gen.generate_extreme_diamond(count=args.count)
    elif args.tier == "extreme":
        episodes = gen.generate_extreme_full_suite()
    else:
        episodes = gen.generate_full_suite()

    gen.save(episodes, Path(args.output))
    print(f"✅ 已生成 {len(episodes)} 条 benchmark → {args.output}")


if __name__ == "__main__":
    main()

