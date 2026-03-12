"""
AstroSASF · Physics · TelemetryBus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模拟 1553B 总线，维护单个实验柜内所有物理硬件的遥测状态。
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _default_telemetry() -> dict[str, Any]:
    """出厂遥测快照。"""
    return {
        "temperature": 22.0,
        "pressure": 101.3,
        "robotic_arm_angle": 0.0,
        "vacuum_pump_active": False,
        "heater_active": False,
        "coolant_flow_rate": 0.0,
    }


@dataclass
class TelemetryBus:
    """实例级遥测总线 —— 1553B 总线的软件影子。"""

    lab_id: str
    _state: dict[str, Any] = field(default_factory=_default_telemetry)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return deepcopy(self._state)

    async def read(self, key: str) -> Any:
        async with self._lock:
            if key not in self._state:
                raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
            return self._state[key]

    async def write(self, key: str, value: Any) -> None:
        async with self._lock:
            if key not in self._state:
                raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
            old = self._state[key]
            self._state[key] = value
            logger.info("[%s] 遥测更新: %s  %r → %r", self.lab_id, key, old, value)

    async def batch_write(self, updates: dict[str, Any]) -> None:
        async with self._lock:
            for key, value in updates.items():
                if key not in self._state:
                    raise KeyError(f"[{self.lab_id}] 未知遥测指标: {key}")
                self._state[key] = value
            logger.info("[%s] 遥测批量更新: %s", self.lab_id, list(updates.keys()))
