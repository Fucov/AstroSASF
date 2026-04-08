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
    NO_CONFLICT = "no_conflict"
    LIGHT_CONFLICT = "light_conflict"
    HEAVY_CONFLICT = "heavy_conflict"
    ALARM_RECOVERY = "alarm_recovery"


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
        """Tier-1: 全部 cabin_exclusive，无共享设备竞争。"""
        episodes = []
        lab_ids = list(self.LAB_DEVICE_MAP.keys())

        conflict_device_map = {
            DifficultyLevel.EASY:   0.0,
            DifficultyLevel.MEDIUM: 0.0,
            DifficultyLevel.HARD:   0.0,
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   3,
            DifficultyLevel.MEDIUM: 4,
            DifficultyLevel.HARD:   5,
        }
        num_agents_map = {
            DifficultyLevel.EASY:   2,
            DifficultyLevel.MEDIUM: 3,
            DifficultyLevel.HARD:   4,
        }

        shared_ratio = conflict_device_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        num_agents = num_agents_map[difficulty]

        for i in range(count):
            lab = self._random_choice(lab_ids)
            devices = list(self.LAB_DEVICE_MAP[lab])
            ep_id = self._random_choice(["t1", "no_c"]) + f"-{difficulty.value}-{i+1:02d}"

            # 构建 DAG（每个舱独立，无共享设备）
            nodes, edges = self._build_dag(
                lab_id=lab,
                devices=devices,
                depth=dag_depth,
                include_shared=False,
            )

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.NO_CONFLICT,
                difficulty=difficulty,
                description=f"[Tier-1] {difficulty.value} 无冲突：{num_agents}个舱独立执行，深度{dag_depth}的DAG",
                cabins=[lab],
                task_graph=TaskGraphDef(nodes=nodes, edges=edges),
                device_requirements=self._build_device_reqs(devices, lab, shared=False),
                initial_telemetry=self._default_telemetry(lab),
                chaos_events=[],  # 无 chaos
                expected_outcomes={"success_rate": 1.0, "max_makespan_s": dag_depth * 10.0},
                evaluation_tags=["tier1", difficulty.value.lower(), "no_conflict"],
                seed=self._seed + i,
            )
            episodes.append(episode)
        return episodes

    # ── Tier-2: 有冲突、短等待 ───────────────────────────────────────────────

    def generate_tier2(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> list[BenchmarkEpisode]:
        """Tier-2: 有全局共享设备竞争，hardware_delay（延迟 1.5~2.5x）。"""
        episodes = []
        conflict_map = {
            DifficultyLevel.EASY:   ("co2_controller", 1.5),
            DifficultyLevel.MEDIUM: ("water_pump_global", 2.0),
            DifficultyLevel.HARD:   ("co2_controller", 2.5),
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   4,
            DifficultyLevel.MEDIUM: 5,
            DifficultyLevel.HARD:   6,
        }
        num_agents_map = {
            DifficultyLevel.EASY:   3,
            DifficultyLevel.MEDIUM: 4,
            DifficultyLevel.HARD:   5,
        }

        shared_device, delay_mult = conflict_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        num_agents = num_agents_map[difficulty]
        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=2)

        for i in range(count):
            ep_id = f"t2-conflict-{difficulty.value}-{i+1:02d}"

            # 混合 DAG：两个舱竞争同一个全局设备
            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=True,
                    shared_device=shared_device,
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.LIGHT_CONFLICT,
                difficulty=difficulty,
                description=f"[Tier-2] {difficulty.value} 有冲突：{num_agents}个舱竞争{shared_device}，延迟{delay_mult}x",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=True, shared_device=shared_device),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(0.5, 1.5),
                        type="hardware_delay",
                        target_tool=shared_device,
                        delay_multiplier=delay_mult,
                    )
                ],
                expected_outcomes={"success_rate": 0.95, "conflict_stall_ms": 2000.0},
                evaluation_tags=["tier2", difficulty.value.lower(), "has_conflict", f"device_{shared_device}"],
                seed=self._seed + i,
            )
            episodes.append(episode)
        return episodes

    # ── Tier-3: 有冲突、长等待 ───────────────────────────────────────────────

    def generate_tier3(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> list[BenchmarkEpisode]:
        """Tier-3: 共享设备 + 长延迟设备（vacuum/centrifuge，秒~分钟级）。"""
        episodes = []
        long_devices = ["vacuum_bio", "centrifuge_bio", "vacuum_mat"]
        delay_map = {
            DifficultyLevel.EASY:   2.0,
            DifficultyLevel.MEDIUM: 3.0,
            DifficultyLevel.HARD:   4.0,
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   5,
            DifficultyLevel.MEDIUM: 6,
            DifficultyLevel.HARD:   8,
        }

        shared_device = "co2_controller"
        delay_mult = delay_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=2)

        for i in range(count):
            ep_id = f"t3-long-{difficulty.value}-{i+1:02d}"
            long_dev = self._random_choice(long_devices)

            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=True,
                    shared_device=shared_device,
                    force_device=long_dev,
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.HEAVY_CONFLICT,
                difficulty=difficulty,
                description=f"[Tier-3] {difficulty.value} 长等待：{len(labs)}舱竞争{shared_device}，含{long_dev}秒级设备",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=True, shared_device=shared_device),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(1.0, 3.0),
                        type="hardware_delay",
                        target_tool=shared_device,
                        delay_multiplier=delay_mult,
                    )
                ],
                expected_outcomes={"success_rate": 0.90, "makespan_s": dag_depth * 30.0},
                evaluation_tags=["tier3", difficulty.value.lower(), "long_wait", "has_conflict"],
                seed=self._seed + i + 100,
            )
            episodes.append(episode)
        return episodes

    # ── Tier-4: 有冲突 + 告警 + 恢复 ───────────────────────────────────────

    def generate_tier4(self, count: int = 5, difficulty: DifficultyLevel = DifficultyLevel.HARD) -> list[BenchmarkEpisode]:
        """Tier-4: telemetry_alarm + checkpoint resume + 多舱逃生路径。"""
        episodes = []
        alarm_map = {
            DifficultyLevel.EASY:   ("temperature", 80.0),
            DifficultyLevel.MEDIUM: ("co2_level", 2000.0),
            DifficultyLevel.HARD:   ("smoke_level", 0.9),
        }
        dag_depth_map = {
            DifficultyLevel.EASY:   6,
            DifficultyLevel.MEDIUM: 8,
            DifficultyLevel.HARD:   10,
        }

        alarm_key, alarm_value = alarm_map[difficulty]
        dag_depth = dag_depth_map[difficulty]
        delay_mult = {"Easy": 2.0, "Medium": 3.0, "Hard": 4.0}[difficulty.value]
        labs = self._random_choice(list(self.LAB_DEVICE_MAP.keys()), k=3)

        for i in range(count):
            ep_id = f"t4-alarm-{difficulty.value}-{i+1:02d}"
            shared_device = self._random_choice(self.GLOBAL_SHARED_DEVICES)

            all_nodes = []
            all_edges = []
            for lab in labs:
                devices = list(self.LAB_DEVICE_MAP[lab])
                nodes, edges = self._build_dag(
                    lab_id=lab,
                    devices=devices,
                    depth=dag_depth,
                    include_shared=True,
                    shared_device=shared_device,
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            episode = BenchmarkEpisode(
                episode_id=ep_id,
                scenario_type=ScenarioType.ALARM_RECOVERY,
                difficulty=difficulty,
                description=f"[Tier-4] {difficulty.value} 告警恢复：{len(labs)}舱，alarm={alarm_key}≥{alarm_value}",
                cabins=labs,
                task_graph=TaskGraphDef(nodes=all_nodes, edges=all_edges),
                device_requirements=self._build_device_reqs([], labs, shared=True, shared_device=shared_device),
                initial_telemetry=self._default_telemetry(labs[0]),
                chaos_events=[
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(2.0, 5.0),
                        type="hardware_delay",
                        target_tool=shared_device,
                        delay_multiplier=delay_mult,
                    ),
                    ChaosEventDef(
                        trigger_time_sec=self._random_float(5.0, 8.0),
                        type="telemetry_alarm",
                        telemetry_key=alarm_key,
                        override_value=alarm_value,
                    ),
                ],
                expected_outcomes={"success_rate": 0.85, "alarm_response_ms": 500.0},
                evaluation_tags=["tier4", difficulty.value.lower(), "alarm_recovery", "checkpoint_resume"],
                seed=self._seed + i + 200,
            )
            episodes.append(episode)
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
    ) -> tuple[list[TaskNodeDef], list[tuple[str, str]]]:
        """构建一个深度为 depth 的任务 DAG。"""
        nodes = []
        edges = []
        rng = random.Random(self._seed + hash(lab_id) % 10000)

        skill_map: dict[str, str] = {
            "heater_bio":    "control_heater",
            "heater_mat":    "set_temperature",
            "heater_plant":  "set_temperature",
            "vacuum_bio":    "toggle_vacuum_pump",
            "vacuum_mat":    "toggle_vacuum_pump",
            "arm_bio":       "move_robotic_arm",
            "arm_mat":       "move_robotic_arm",
            "arm_plant":     "move_robotic_arm",
            "pump_fluid":    "toggle_vacuum_pump",
            "pump_plant":    "inject_nutrient",
            "valve_fluid":   "control_valve",
            "centrifuge_bio":"control_centrifuge",
            "scan_bio":      "read_sensor",
        }

        # 每个 depth 层 1~2 个节点
        prev_ids: list[str] = []
        for d in range(depth):
            num_in_level = rng.randint(1, 2)
            level_ids = []
            for j in range(num_in_level):
                dev_idx = (d * num_in_level + j) % len(devices)
                device = devices[dev_idx]
                if force_device and d == depth // 2:
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

                # 全局共享设备
                if include_shared and shared_device and rng.random() < 0.5:
                    node.required_devices.append(shared_device)

                nodes.append(node)
                level_ids.append(node_id)

                # 连接到上一层的节点
                if prev_ids:
                    for prev_id in prev_ids:
                        edges.append((prev_id, node_id))

            prev_ids = level_ids

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
                        choices=["tier1", "tier2", "tier3", "tier4"])
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
    else:
        episodes = gen.generate_full_suite()

    gen.save(episodes, Path(args.output))
    print(f"✅ 已生成 {len(episodes)} 条 benchmark → {args.output}")


if __name__ == "__main__":
    main()

