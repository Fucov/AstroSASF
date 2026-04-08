"""
AstroSASF · Scheduler · Device Model
====================================
物理设备统一数据模型 —— scope / latency model / chaos 参数。

这是 Part 2 设备时延仿真层的数据结构基础。
所有设备类型的参数化延迟模型在此统一实现。

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ────────────────────────────────────────────────────────────────────────────── #
#  设备类型 & 作用域 & 锁模式                                                   #
# ────────────────────────────────────────────────────────────────────────────── #

class DeviceType(Enum):
    HEATER = "heater"
    ROBOTIC_ARM = "robotic_arm"
    VALVE = "valve"
    PUMP = "pump"
    CENTRIFUGE = "centrifuge"
    SCAN = "scan"
    VACUUM = "vacuum"
    GENERIC = "generic"


class DeviceScope(Enum):
    """设备作用域：
    - GLOBAL_SHARED: 所有舱共同竞争（如 CO2 控制器）
    - CABIN_SHARED:   同舱内共享（多任务竞争）
    - CABIN_EXCLUSIVE: 舱级独占（无竞争）"""
    GLOBAL_SHARED = "global_shared"
    CABIN_SHARED = "cabin_shared"
    CABIN_EXCLUSIVE = "cabin_exclusive"


class LockMode(Enum):
    """锁获取策略。"""
    EXCLUSIVE = "exclusive"
    SHARED_READ = "shared_read"
    PRIORITY_AWARE = "priority_aware"


class RecoveryPolicy(Enum):
    """设备故障后的恢复策略。"""
    RETRY_3 = "retry_3"
    RETRY_5 = "retry_5"
    FAIL_FAST = "fail_fast"
    CHECKPOINT_RESUME = "checkpoint_resume"


# ────────────────────────────────────────────────────────────────────────────── #
#  DeviceSchema — 设备元信息描述符                                               #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class DeviceSchema:
    """物理设备的完整元信息描述符。"""
    device_id: str
    device_type: DeviceType
    scope: DeviceScope
    allowed_cabins: set[str] | None = None
    lock_mode: LockMode = LockMode.EXCLUSIVE
    base_latency_ms: float = 100.0
    jitter_sigma_ms: float = 0.0
    timeout_ms: float = 30000.0
    failure_rate: float = 0.0
    preemptible: bool = True
    recovery_policy: RecoveryPolicy = RecoveryPolicy.RETRY_3
    physics_params: dict[str, float] = field(default_factory=dict)

    def is_allowed_for(self, cabin_id: str) -> bool:
        if self.allowed_cabins is None:
            return True
        return cabin_id in self.allowed_cabins

    def compute_latency(
        self,
        action_params: dict[str, Any],
        current_telemetry: dict[str, Any],
    ) -> float:
        """根据设备类型和动作参数计算物理延迟（毫秒）。

        这是参数化物理仿真的核心函数。
        延迟与温度差、体积、距离等物理参数相关，而非固定值。

        Parameters
        ----------
        action_params : dict
            动作参数，如 {"temperature": 50.0, "volume_ml": 100.0}
        current_telemetry : dict
            当前遥测状态，用于计算起点条件（如当前温度）

        Returns
        -------
        float
            延迟毫秒数（最小 1ms）
        """
        params = dict(self.physics_params)
        params.update(action_params)

        # ── heater ──────────────────────────────────────────────────────────────
        if self.device_type == DeviceType.HEATER:
            T_target = params.get("temperature", 37.0)
            T_current = current_telemetry.get("temperature", 25.0)
            heat_rate = params.get("heat_rate_C_per_sec", 2.0)
            hold_ms = params.get("hold_duration_ms", 0.0)
            ramp_ms = abs(T_target - T_current) / max(heat_rate, 0.01) * 1000.0
            return max(500.0, ramp_ms + hold_ms)

        # ── robotic_arm ─────────────────────────────────────────────────────────
        if self.device_type == DeviceType.ROBOTIC_ARM:
            pos_from = current_telemetry.get("arm_angle", 0.0)
            pos_to = params.get("target_position", 0.0)
            if isinstance(pos_to, str):
                pos_map = {"HOME": 0.0, "90": 90.0, "45": 45.0, "30": 30.0, "0": 0.0}
                pos_to = pos_map.get(pos_to, 0.0)
            try:
                pos_to = float(pos_to)
            except (ValueError, TypeError):
                pos_to = 0.0
            distance_deg = abs(pos_to - pos_from)
            speed = params.get("speed_deg_per_sec", 30.0)
            move_ms = distance_deg / max(speed, 0.1) * 1000.0
            action = params.get("action_type", "move")
            overhead_ms = params.get("grip_overhead_ms", 200.0)
            return move_ms + (overhead_ms if action != "move" else 0.0)

        # ── valve ───────────────────────────────────────────────────────────────
        if self.device_type == DeviceType.VALVE:
            return params.get("switch_time_ms", 300.0)

        # ── pump ────────────────────────────────────────────────────────────────
        if self.device_type == DeviceType.PUMP:
            volume = params.get("volume_ml", 0.0)
            flow_rate = params.get("flow_rate_ml_per_sec", 5.0)
            startup_ms = params.get("startup_ms", 500.0)
            if volume > 0:
                return startup_ms + (volume / max(flow_rate, 0.1) * 1000.0)
            return startup_ms

        # ── centrifuge ───────────────────────────────────────────────────────────
        if self.device_type == DeviceType.CENTRIFUGE:
            ramp_up = params.get("ramp_up_ms", 3000.0)
            hold = params.get("hold_ms", 60000.0)
            ramp_down = params.get("ramp_down_ms", 5000.0)
            return ramp_up + hold + ramp_down

        # ── vacuum ───────────────────────────────────────────────────────────────
        if self.device_type == DeviceType.VACUUM:
            current_pressure = current_telemetry.get("pressure", 101.325)
            target_pressure = params.get("target_pressure_kpa", 10.0)
            pump_speed = params.get("pump_speed_kpa_per_sec", 5.0)
            seal_check = params.get("seal_check_ms", 2000.0)
            if current_pressure > target_pressure:
                evac_ms = (current_pressure - target_pressure) / max(pump_speed, 0.01) * 1000.0
            else:
                evac_ms = 0.0
            return evac_ms + seal_check

        # ── scan ────────────────────────────────────────────────────────────────
        if self.device_type == DeviceType.SCAN:
            prep = params.get("sample_prep_ms", 5000.0)
            scan = params.get("scan_duration_ms", 30000.0)
            processing = params.get("processing_ms", 2000.0)
            return prep + scan + processing

        # ── generic ─────────────────────────────────────────────────────────────
        return self.base_latency_ms

    def apply_jitter(self, latency_ms: float) -> float:
        if self.jitter_sigma_ms <= 0:
            return latency_ms
        noise = random.gauss(0.0, self.jitter_sigma_ms)
        return max(1.0, latency_ms + noise)

    def should_fail(self) -> bool:
        return random.random() < self.failure_rate


# ────────────────────────────────────────────────────────────────────────────── #
#  默认设备注册表                                                               #
# ────────────────────────────────────────────────────────────────────────────── #

# 所有新增设备应在 benchmarks/bench_generator.py 的 _DEVICE_POOL 中定义。
# 此处仅保留 demo 演示用设备。
DEFAULT_DEVICE_REGISTRY: dict[str, DeviceSchema] = {
    # ── DemoBio 舱内独占设备 ─────────────────────────────────────────────────
    "heater_bio": DeviceSchema(
        device_id="heater_bio",
        device_type=DeviceType.HEATER,
        scope=DeviceScope.CABIN_EXCLUSIVE,
        allowed_cabins={"DemoBio"},
        jitter_sigma_ms=50.0,
        physics_params={"heat_rate_C_per_sec": 2.0, "max_temperature": 80.0},
    ),
    "vacuum_bio": DeviceSchema(
        device_id="vacuum_bio",
        device_type=DeviceType.VACUUM,
        scope=DeviceScope.CABIN_EXCLUSIVE,
        allowed_cabins={"DemoBio"},
        jitter_sigma_ms=200.0,
        physics_params={"pump_speed_kpa_per_sec": 5.0, "seal_check_ms": 2000.0},
    ),
    "arm_bio": DeviceSchema(
        device_id="arm_bio",
        device_type=DeviceType.ROBOTIC_ARM,
        scope=DeviceScope.CABIN_EXCLUSIVE,
        allowed_cabins={"DemoBio"},
        jitter_sigma_ms=100.0,
        physics_params={"speed_deg_per_sec": 30.0, "grip_overhead_ms": 200.0},
    ),
    # ── DemoFluid 舱内独占设备 ──────────────────────────────────────────────
    "pump_fluid": DeviceSchema(
        device_id="pump_fluid",
        device_type=DeviceType.PUMP,
        scope=DeviceScope.CABIN_EXCLUSIVE,
        allowed_cabins={"DemoFluid"},
        jitter_sigma_ms=100.0,
        physics_params={"flow_rate_ml_per_sec": 5.0, "startup_ms": 500.0},
    ),
    "valve_fluid": DeviceSchema(
        device_id="valve_fluid",
        device_type=DeviceType.VALVE,
        scope=DeviceScope.CABIN_EXCLUSIVE,
        allowed_cabins={"DemoFluid"},
        jitter_sigma_ms=10.0,
    ),
    # ── 全局共享设备（跨舱竞争）────────────────────────────────────────────────
    "co2_controller": DeviceSchema(
        device_id="co2_controller",
        device_type=DeviceType.GENERIC,
        scope=DeviceScope.GLOBAL_SHARED,
        allowed_cabins=None,
        lock_mode=LockMode.PRIORITY_AWARE,
        jitter_sigma_ms=50.0,
    ),
    "water_pump_global": DeviceSchema(
        device_id="water_pump_global",
        device_type=DeviceType.PUMP,
        scope=DeviceScope.GLOBAL_SHARED,
        allowed_cabins=None,
        lock_mode=LockMode.PRIORITY_AWARE,
        jitter_sigma_ms=150.0,
        physics_params={"flow_rate_ml_per_sec": 10.0, "startup_ms": 1000.0},
    ),
}


__all__ = [
    "DeviceSchema",
    "DeviceType",
    "DeviceScope",
    "LockMode",
    "RecoveryPolicy",
    "DEFAULT_DEVICE_REGISTRY",
]
