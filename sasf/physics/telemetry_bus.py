"""
AstroSASF · Physics · TelemetryBus (V4.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模拟 1553B 总线 —— 通用遥测数据存储。

V4.3 变化：
- 初始遥测由外部注入（构造参数），不再硬编码业务默认值
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)



@dataclass
class TelemetryBus:
    """实例级遥测总线 —— 通用 1553B 总线影子。

    Parameters
    ----------
    lab_id : str
    initial_state : dict, optional
        初始遥测数据，由应用层注入。
    """

    lab_id: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    _state: dict[str, Any] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self._state = dict(self.initial_state)

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
            old = self._state.get(key)
            self._state[key] = value
            logger.info("[%s] 遥测更新: %s  %r → %r", self.lab_id, key, old, value)

    async def batch_write(self, updates: dict[str, Any]) -> None:
        async with self._lock:
            for key, value in updates.items():
                self._state[key] = value
            logger.info("[%s] 遥测批量更新: %s", self.lab_id, list(updates.keys()))

