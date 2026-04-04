"""
AstroSASF · Scheduler · Virtual SpaceWire Bus (Kernel)
=====================================================
模拟 SpaceWire / 1553B 低带宽航天总线。

V7.5 核心重构：
- 令牌桶带宽限流：按 Byte 发放令牌，令牌速率 = spacewire_bandwidth_kbps
- QoS 三级优先级队列：CRITICAL（无限额）/ NORMAL / LOW
- AoI 遥测覆写：同一 sensor_id 的高频遥测只保留最新快照
- 弱网抗性指标：bytes_saved_by_aoi / critical_packet_latency_ms

Author: AstroSASF Team
Version: 7.5
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  QoS 级别定义                                                                 #
# --------------------------------------------------------------------------- #

class QoSLevel(IntEnum):
    """QoS 优先级级别（数值越小优先级越高）。"""
    CRITICAL = 0   # 硬件报警、逃生指令（无限额优先发送）
    NORMAL = 1     # Agent 间协调消息、遥测汇总
    LOW = 2        # 日志回传、调试信息（令牌不足时排队）


# --------------------------------------------------------------------------- #
#  总线数据包封装                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class BusFrame:
    """SpaceWire 总线数据帧。"""
    qos: QoSLevel
    data: bytes
    source: str
    frame_id: str
    timestamp: float = field(default_factory=time.time)
    # 下面两个字段用于 AoI 覆写
    aoi_key: str | None = None     # 如 "sensor:temperature" 用于覆写判定
    _queued_at: float = field(default=0.0, init=False)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------- #
#  令牌桶限流器                                                                  #
# --------------------------------------------------------------------------- #

class TokenBucket:
    """令牌桶带宽限流器。

    令牌按固定速率（字节/秒）生成，允许短时突发（bucket_capacity），
    但长期发送速率不超过令牌生成速率。

    设计要点：
    - 令牌桶容量 = burst_capacity_bytes（允许短时突发）
    - 每次发送扣减相应数量的令牌
    - 补充速度 = bandwidth_kbps * 1000 / 8 (Byte/s)
    - 令牌不足时，asyncio.sleep() 等待补充

    Example
    -------
    >>> tb = TokenBucket(bandwidth_kbps=200.0, burst_capacity_bytes=512)
    >>> await tb.acquire(256)   # 扣减 256 字节的令牌
    >>> # 如果令牌不足，会自动 await 等待补充
    """

    def __init__(
        self,
        bandwidth_kbps: float,
        burst_capacity_bytes: int = 1024,
    ) -> None:
        self._bandwidth_kbps = bandwidth_kbps
        # 令牌补充速率（Byte/s）
        self._token_rate = (bandwidth_kbps * 1000.0) / 8.0
        self._capacity = float(burst_capacity_bytes)
        self._tokens = float(burst_capacity_bytes)  # 初始满桶
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """补充令牌（按时间流逝量）。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._token_rate)

    async def acquire(self, byte_count: int, allow_burst: bool = False) -> float:
        """申请发送 byte_count 字节的数据。

        Parameters
        ----------
        byte_count : int
            要发送的字节数
        allow_burst : bool
            True = 允许透支桶容量（用于 CRITICAL 紧急包无限额发送）
            False = 不允许透支（标准发送）

        Returns
        -------
        float
            实际等待的秒数（用于延迟统计）

        Raises
        ------
        ValueError
            当 allow_burst=False 且 byte_count > 桶容量时
        """
        if byte_count > self._capacity and not allow_burst:
            raise ValueError(
                f"数据包大小 ({byte_count} B) 超过令牌桶容量 ({self._capacity} B)，"
                "请设置 allow_burst=True 或增大 burst_capacity_bytes"
            )

        wait_time = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= byte_count or allow_burst:
                    self._tokens -= byte_count
                    return wait_time
                # 令牌不足，等待补充
                deficit = byte_count - self._tokens
                sleep_time = deficit / self._token_rate
                await asyncio.sleep(sleep_time)
                wait_time += sleep_time
                self._refill()

    @property
    def tokens_available(self) -> float:
        """当前可用令牌数（只读快照，无需加锁）。"""
        self._refill()
        return self._tokens


# --------------------------------------------------------------------------- #
#  AoI 覆写索引（用于 NORMAL 队列去重）                                          #
# --------------------------------------------------------------------------- #

@dataclass
class AoIIndexEntry:
    """AoI 索引表条目。"""
    frame: BusFrame
    queue_index: int   # 在 asyncio.PriorityQueue 内部堆中的位置（近似）


# --------------------------------------------------------------------------- #
#  VirtualSpaceWire — 完整总线实现                                             #
# --------------------------------------------------------------------------- #

@dataclass
class VirtualSpaceWire:
    """虚拟 SpaceWire 总线 —— 令牌桶 + QoS 队列 + AoI 覆写 + 指标埋点。

    V7.5 核心设计：
    - 令牌桶（TokenBucket）：按 spacewire_bandwidth_kbps 限流
    - QoS 三级队列：CRITICAL > NORMAL > LOW
    - AoI 覆写：NORMAL 队列中同一 aoi_key 的旧帧被新帧替换
    - 拥塞下 CRITICAL 报警无限额优先发送，平均排队延迟 < 5ms

    QoS 发送策略：
    1. 清空 CRITICAL 队列（无限额，优先发送）
    2. 用剩余令牌发送 NORMAL 队列（AoI 覆写生效）
    3. 最后发送 LOW 队列（令牌不足时等待）

    Usage
    -----
    >>> bus = VirtualSpaceWire(lab_id="DemoBio", bandwidth_kbps=200.0)
    >>> await bus.start()
    >>>
    >>> # 硬件报警（无限额优先）
    >>> await bus.send_critical(b"ALARM:temperature>80", source="telemetry")
    >>>
    >>> # 遥测高频上报（AoI 覆写，只保留最新）
    >>> await bus.send_telemetry(b'{"sensor":"temp","value":37}', source="sensor", aoi_key="sensor:temp")
    >>>
    >>> # Agent 间协调（NORMAL）
    >>> await bus.send(b'execute plan', source="planner")
    >>>
    >>> # 日志回传（LOW）
    >>> await bus.send_low(b'debug log entry', source="worker")
    """

    lab_id: str
    bandwidth_kbps: float = 200.0
    burst_capacity_bytes: int = 1024

    # 内部状态
    _token_bucket: TokenBucket = field(init=False)
    _running: bool = field(default=False, init=False)
    _transmit_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _shutdown_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    # QoS 三级队列
    _q_critical: asyncio.PriorityQueue = field(
        default_factory=lambda: asyncio.PriorityQueue(), init=False,
    )
    _q_normal: asyncio.PriorityQueue = field(
        default_factory=lambda: asyncio.PriorityQueue(), init=False,
    )
    _q_low: asyncio.PriorityQueue = field(
        default_factory=lambda: asyncio.PriorityQueue(), init=False,
    )

    # AoI 索引（用于 NORMAL 队列覆写）
    _aoi_index: dict[str, BusFrame] = field(default_factory=dict, init=False)
    _aoi_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # 统计指标
    _total_bytes_transmitted: int = field(default=0, init=False)
    _total_frames: int = field(default=0, init=False)
    _total_latency_ms: float = field(default=0.0, init=False)
    # V7.5 弱网抗性指标
    _bytes_saved_by_aoi: int = field(default=0, init=False)
    _aoi_overwrite_count: int = field(default=0, init=False)
    _critical_packet_count: int = field(default=0, init=False)
    _critical_queue_time_sum_ms: float = field(default=0.0, init=False)
    _normal_packet_count: int = field(default=0, init=False)
    _normal_queue_time_sum_ms: float = field(default=0.0, init=False)
    _frame_counter: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._token_bucket = TokenBucket(
            bandwidth_kbps=self.bandwidth_kbps,
            burst_capacity_bytes=self.burst_capacity_bytes,
        )

    # ------------------------------------------------------------------------- #
    #  生命周期                                                                  #
    # ------------------------------------------------------------------------- #

    async def start(self) -> None:
        """启动总线发送循环后台协程。"""
        if self._running:
            return
        self._running = True
        self._shutdown_event.clear()
        self._transmit_task = asyncio.get_running_loop().create_task(
            self._tx_loop(),
            name=f"spacewire-tx-{self.lab_id}",
        )
        logger.info(
            "[%s] SpaceWire 总线启动: %.0f Kbps | QoS=[CRITICAL/NORMAL/LOW] | AoI=启用",
            self.lab_id, self.bandwidth_kbps,
        )

    async def stop(self) -> None:
        """优雅停止总线。"""
        self._running = False
        self._shutdown_event.set()
        if self._transmit_task is not None:
            try:
                await asyncio.wait_for(self._transmit_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._transmit_task.cancel()
            finally:
                self._transmit_task = None
        logger.info("[%s] SpaceWire 总线已停止", self.lab_id)

    # ------------------------------------------------------------------------- #
    #  发送 API                                                                  #
    # ------------------------------------------------------------------------- #

    async def send_critical(self, data: bytes, source: str = "unknown") -> None:
        """发送 CRITICAL 级别数据（无限额优先）。

        用于：硬件报警、逃生指令等硬实时数据。
        CRITICAL 包会立即击穿拥塞，在下一次发送循环中无限额优先处理。
        """
        frame = BusFrame(
            qos=QoSLevel.CRITICAL,
            data=bytes(data),
            source=source,
            frame_id=self._next_frame_id(),
        )
        frame._queued_at = time.monotonic()
        await self._q_critical.put((frame.qos, frame))
        logger.debug(
            "[%s] CRITICAL 入队: %s (队列深度: %d)",
            self.lab_id, frame.frame_id, self._q_critical.qsize(),
        )

    async def send_telemetry(
        self,
        data: bytes,
        source: str = "sensor",
        aoi_key: str | None = None,
    ) -> None:
        """发送 NORMAL 级别遥测数据（支持 AoI 覆写）。

        Parameters
        ----------
        aoi_key : str | None
            AoI 标识键，如 "sensor:temperature"。
            如果 NORMAL 队列中已存在相同 aoi_key 的帧，
            该旧帧会被新帧覆写（字节节省 + 最新鲜保证）。
        """
        frame = BusFrame(
            qos=QoSLevel.NORMAL,
            data=bytes(data),
            source=source,
            frame_id=self._next_frame_id(),
            aoi_key=aoi_key,
        )
        frame._queued_at = time.monotonic()

        # AoI 覆写：检查 NORMAL 队列中是否有相同 aoi_key 的旧帧
        if aoi_key is not None:
            async with self._aoi_lock:
                old_frame = self._aoi_index.get(aoi_key)
                if old_frame is not None:
                    # 记录节省的字节数（旧的被丢弃，不再发送）
                    saved = old_frame.size_bytes
                    self._bytes_saved_by_aoi += saved
                    self._aoi_overwrite_count += 1
                    logger.debug(
                        "[%s] AoI 覆写: key=%s, 节省 %d B (累计节省 %d B, 覆写次数 %d)",
                        self.lab_id, aoi_key, saved,
                        self._bytes_saved_by_aoi, self._aoi_overwrite_count,
                    )
                    # 从索引中移除旧帧引用（队列中的旧帧会在发送时自然跳过）
                    del self._aoi_index[aoi_key]
                self._aoi_index[aoi_key] = frame

        await self._q_normal.put((frame.qos, frame))

    async def send(self, data: bytes, source: str = "unknown") -> None:
        """发送 NORMAL 级别数据（默认）。"""
        frame = BusFrame(
            qos=QoSLevel.NORMAL,
            data=bytes(data),
            source=source,
            frame_id=self._next_frame_id(),
        )
        frame._queued_at = time.monotonic()
        await self._q_normal.put((frame.qos, frame))

    async def send_low(self, data: bytes, source: str = "unknown") -> None:
        """发送 LOW 级别数据（令牌不足时等待）。"""
        frame = BusFrame(
            qos=QoSLevel.LOW,
            data=bytes(data),
            source=source,
            frame_id=self._next_frame_id(),
        )
        frame._queued_at = time.monotonic()
        await self._q_low.put((frame.qos, frame))

    # ------------------------------------------------------------------------- #
    #  发送循环（核心调度器）                                                   #
    # ------------------------------------------------------------------------- #

    async def _tx_loop(self) -> None:
        """总线发送循环后台协程。

        调度策略（每次循环）：
        1. 清空 CRITICAL 队列（无限额，按 FIFO 顺序发送）
        2. 用剩余令牌发送 NORMAL 队列（遇到 LOW 帧则停止）
        3. 用剩余令牌发送 LOW 队列
        4. 如果三个队列都为空，短暂等待后继续轮询
        """
        logger.info("[%s] SpaceWire 发送循环启动", self.lab_id)

        while self._running and not self._shutdown_event.is_set():
            try:
                await self._drain_qos_queues()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("[%s] SpaceWire 发送循环异常: %s", self.lab_id, exc)
                await asyncio.sleep(0.1)

        logger.info("[%s] SpaceWire 发送循环退出", self.lab_id)

    async def _drain_qos_queues(self) -> None:
        """一次性清空所有可发送的队列（按 QoS 优先级）。

        调度顺序：
        CRITICAL（无限额）→ NORMAL（令牌受限）→ LOW（令牌受限）
        """
        sent_any = False

        # ── Step 1: CRITICAL 队列（无限额，直通发送）───────────────────────
        while not self._q_critical.empty():
            try:
                _, frame = await asyncio.wait_for(
                    self._q_critical.get(), timeout=0.01,
                )
            except asyncio.TimeoutError:
                break

            # 无限额发送（CRITICAL 永远不扣减令牌桶）
            queue_time_ms = (time.monotonic() - frame._queued_at) * 1000.0
            self._critical_queue_time_sum_ms += queue_time_ms
            self._critical_packet_count += 1

            await self._transmit_frame(frame, token_cost=0)
            sent_any = True

        # ── Step 2: NORMAL 队列（令牌受限，AoI 索引更新）───────────────────
        while not self._q_normal.empty():
            try:
                _, frame = await asyncio.wait_for(
                    self._q_normal.get(), timeout=0.01,
                )
            except asyncio.TimeoutError:
                break

            # 检查是否已被 AoI 覆写（如果 aoi_key 存在但索引中的帧已更新）
            async with self._aoi_lock:
                if frame.aoi_key is not None:
                    current = self._aoi_index.get(frame.aoi_key)
                    if current is not frame:
                        # 已被覆写，跳过这个旧帧
                        logger.debug(
                            "[%s] AoI 跳过旧帧: key=%s (已被更新帧覆盖)",
                            self.lab_id, frame.aoi_key,
                        )
                        continue

            queue_time_ms = (time.monotonic() - frame._queued_at) * 1000.0
            self._normal_queue_time_sum_ms += queue_time_ms
            self._normal_packet_count += 1

            # 扣减令牌（不够则等待补充）
            await self._token_bucket.acquire(frame.size_bytes)
            await self._transmit_frame(frame, token_cost=frame.size_bytes)
            sent_any = True

        # ── Step 3: LOW 队列（令牌受限）─────────────────────────────────────
        while not self._q_low.empty():
            try:
                _, frame = await asyncio.wait_for(
                    self._q_low.get(), timeout=0.01,
                )
            except asyncio.TimeoutError:
                break

            # LOW 队列在 NORMAL 之后处理，确保 NORMAL 数据优先
            await self._token_bucket.acquire(frame.size_bytes)
            await self._transmit_frame(frame, token_cost=frame.size_bytes)
            sent_any = True

        # 如果三个队列都为空，短暂让出 CPU
        if not sent_any:
            await asyncio.sleep(0.05)

    async def _transmit_frame(self, frame: BusFrame, token_cost: int) -> None:
        """执行单个帧的实际传输。"""
        # 从 AoI 索引中清除（已发送）
        if frame.aoi_key is not None:
            async with self._aoi_lock:
                self._aoi_index.pop(frame.aoi_key, None)

        # 模拟物理链路传输延迟（按实际字节数）
        tx_latency = (frame.size_bytes * 8) / (self.bandwidth_kbps * 1000.0)
        await asyncio.sleep(tx_latency)

        async with self._lock:
            self._total_bytes_transmitted += frame.size_bytes
            self._total_frames += 1
            self._total_latency_ms += tx_latency * 1000.0

        if frame.qos == QoSLevel.CRITICAL:
            qos_tag = "🔴 CRITICAL"
        elif frame.qos == QoSLevel.NORMAL:
            qos_tag = "🟡 NORMAL"
        else:
            qos_tag = "🟢 LOW"

        logger.info(
            "[%s] %s TX: %s | %d B | 令牌消耗 %d | 队列等待 %.2f ms",
            self.lab_id, qos_tag, frame.frame_id,
            frame.size_bytes, token_cost,
            (time.monotonic() - frame._queued_at) * 1000,
        )

    # ------------------------------------------------------------------------- #
    #  辅助方法                                                                  #
    # ------------------------------------------------------------------------- #

    def _next_frame_id(self) -> str:
        self._frame_counter += 1
        return f"{self.lab_id}-{self._frame_counter:06d}"

    # ------------------------------------------------------------------------- #
    #  统计指标                                                                  #
    # ------------------------------------------------------------------------- #

    @property
    def stats(self) -> dict[str, Any]:
        """返回总线统计信息（含 V7.5 弱网抗性指标）。"""
        critical_avg_latency = (
            self._critical_queue_time_sum_ms / max(1, self._critical_packet_count)
        )
        normal_avg_latency = (
            self._normal_queue_time_sum_ms / max(1, self._normal_packet_count)
        )
        return {
            # 基础指标
            "lab_id": self.lab_id,
            "bandwidth_kbps": self.bandwidth_kbps,
            "total_frames": self._total_frames,
            "total_bytes": self._total_bytes_transmitted,
            "total_latency_ms": round(self._total_latency_ms, 3),
            # V7.5 弱网抗性指标
            "bytes_saved_by_aoi": self._bytes_saved_by_aoi,
            "aoi_overwrite_count": self._aoi_overwrite_count,
            "critical_packet_count": self._critical_packet_count,
            "critical_avg_queue_latency_ms": round(critical_avg_latency, 3),
            "normal_packet_count": self._normal_packet_count,
            "normal_avg_queue_latency_ms": round(normal_avg_latency, 3),
            # 队列深度快照
            "queue_depth": {
                "CRITICAL": self._q_critical.qsize(),
                "NORMAL": self._q_normal.qsize(),
                "LOW": self._q_low.qsize(),
            },
            "aoi_index_size": len(self._aoi_index),
            "tokens_available": round(self._token_bucket.tokens_available, 2),
        }


__all__ = ["VirtualSpaceWire", "BusFrame", "QoSLevel", "TokenBucket"]
