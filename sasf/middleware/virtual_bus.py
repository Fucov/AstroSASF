"""
AstroSASF · Middleware · VirtualSpaceWire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模拟 SpaceWire / 1553B 低带宽航天总线。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VirtualSpaceWire:
    """虚拟 SpaceWire 总线 —— 带宽受限的传输信道模拟。"""

    lab_id: str
    bandwidth_kbps: float = 200.0
    _total_bytes_transmitted: int = field(default=0, init=False)
    _total_frames: int = field(default=0, init=False)
    _total_latency_ms: float = field(default=0.0, init=False)

    async def transmit(self, data: bytearray | bytes) -> bytearray:
        byte_count = len(data)
        latency_sec = self._calculate_latency(byte_count)
        latency_ms = latency_sec * 1000.0

        logger.info(
            "[%s] 🛰️  SpaceWire TX: %d Bytes | 带宽 %.0f Kbps | 延迟 %.3f ms",
            self.lab_id, byte_count, self.bandwidth_kbps, latency_ms,
        )
        await asyncio.sleep(latency_sec)

        self._total_bytes_transmitted += byte_count
        self._total_frames += 1
        self._total_latency_ms += latency_ms
        return bytearray(data)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_frames": self._total_frames,
            "total_bytes": self._total_bytes_transmitted,
            "total_latency_ms": round(self._total_latency_ms, 3),
            "bandwidth_kbps": self.bandwidth_kbps,
        }

    def _calculate_latency(self, byte_count: int) -> float:
        return (byte_count * 8) / (self.bandwidth_kbps * 1000.0)
