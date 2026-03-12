"""
AstroSASF · Middleware · VirtualSpaceWire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模拟 SpaceWire / 1553B 低带宽航天总线。

在 ``SpaceMCPGateway`` 与 ``ShadowFSM`` 之间引入一个物理总线仿真层，
按二进制帧的字节长度计算传输耗时（``asyncio.sleep``），并记录累计的
流量统计数据。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  VirtualSpaceWire                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class VirtualSpaceWire:
    """虚拟 SpaceWire 总线 —— 带宽受限的传输信道模拟。

    Parameters
    ----------
    lab_id : str
        所属实验柜标识。
    bandwidth_kbps : float
        总线带宽（千比特/秒），默认 200 Kbps。
    """

    lab_id: str
    bandwidth_kbps: float = 200.0

    # ---- 内部统计 --------------------------------------------------------- #
    _total_bytes_transmitted: int = field(default=0, init=False)
    _total_frames: int = field(default=0, init=False)
    _total_latency_ms: float = field(default=0.0, init=False)

    # ---- 核心传输 --------------------------------------------------------- #

    async def transmit(self, data: bytearray | bytes) -> bytearray:
        """模拟经总线传输二进制数据。

        根据数据长度和总线带宽计算传输延迟，使用 ``asyncio.sleep``
        模拟真实的物理传输耗时。

        Parameters
        ----------
        data : bytearray | bytes
            待传输的二进制帧。

        Returns
        -------
        bytearray
            原样透传的数据（模拟无损信道）。
        """
        byte_count = len(data)
        latency_sec = self._calculate_latency(byte_count)
        latency_ms = latency_sec * 1000.0

        logger.info(
            "[%s] 🛰️  SpaceWire TX: %d Bytes | "
            "带宽 %.0f Kbps | 传输延迟 %.3f ms",
            self.lab_id, byte_count, self.bandwidth_kbps, latency_ms,
        )

        # 模拟传输延迟
        await asyncio.sleep(latency_sec)

        # 更新统计
        self._total_bytes_transmitted += byte_count
        self._total_frames += 1
        self._total_latency_ms += latency_ms

        return bytearray(data)

    # ---- 统计 ------------------------------------------------------------- #

    @property
    def stats(self) -> dict[str, Any]:
        """返回总线传输统计。"""
        return {
            "total_frames": self._total_frames,
            "total_bytes": self._total_bytes_transmitted,
            "total_latency_ms": round(self._total_latency_ms, 3),
            "bandwidth_kbps": self.bandwidth_kbps,
        }

    # ---- 内部 ------------------------------------------------------------- #

    def _calculate_latency(self, byte_count: int) -> float:
        """计算传输延迟（秒）。

        latency = (byte_count * 8) / (bandwidth_kbps * 1000)
        """
        bits = byte_count * 8
        bandwidth_bps = self.bandwidth_kbps * 1000.0
        return bits / bandwidth_bps
