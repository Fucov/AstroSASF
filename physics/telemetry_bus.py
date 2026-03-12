"""
AstroSASF · Physics · TelemetryBus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模拟 1553B 总线，维护单个实验柜内所有物理硬件的遥测状态。

每个 LaboratoryEnvironment 持有一个独立实例，确保多系统并发时
内存状态完全隔离。
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Default telemetry snapshot for a fresh laboratory                           #
# --------------------------------------------------------------------------- #

def _default_telemetry() -> dict[str, Any]:
    """出厂遥测快照，覆盖实验柜内主要硬件指标。"""
    return {
        "temperature": 22.0,          # ℃ — 舱内温度
        "pressure": 101.3,            # kPa — 舱内气压
        "robotic_arm_angle": 0.0,     # ° — 机械臂关节角
        "vacuum_pump_active": False,  # 真空泵开关
        "heater_active": False,       # 加热器开关
        "coolant_flow_rate": 0.0,     # L/min — 冷却液流速
    }


# --------------------------------------------------------------------------- #
#  TelemetryBus                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class TelemetryBus:
    """实例级遥测总线 —— 1553B 总线的软件影子。

    所有读写通过 ``asyncio.Lock`` 做细粒度互斥，保障同一个
    LaboratoryEnvironment 内 Planner / Operator 并发安全。
    """

    lab_id: str
    _state: dict[str, Any] = field(default_factory=_default_telemetry)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # ---- 只读快照 --------------------------------------------------------- #

    async def snapshot(self) -> dict[str, Any]:
        """返回当前遥测状态的 **深拷贝**，防止外部直接篡改。"""
        async with self._lock:
            return deepcopy(self._state)

    # ---- 单键读写 --------------------------------------------------------- #

    async def read(self, key: str) -> Any:
        """读取单个遥测指标。"""
        async with self._lock:
            if key not in self._state:
                raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
            return self._state[key]

    async def write(self, key: str, value: Any) -> None:
        """写入单个遥测指标（仅由已通过 FSM 校验的操作调用）。"""
        async with self._lock:
            if key not in self._state:
                raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
            old = self._state[key]
            self._state[key] = value
            logger.info(
                "[%s] 遥测更新: %s  %r → %r",
                self.lab_id, key, old, value,
            )

    # ---- 批量写入 --------------------------------------------------------- #

    async def batch_write(self, updates: dict[str, Any]) -> None:
        """原子批量更新多个指标。"""
        async with self._lock:
            for key, value in updates.items():
                if key not in self._state:
                    raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
                self._state[key] = value
            logger.info(
                "[%s] 遥测批量更新: %s",
                self.lab_id, list(updates.keys()),
            )
