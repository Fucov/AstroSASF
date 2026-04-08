"""
AstroSASF · Scheduler · Device Runtime
======================================
物理设备统一调用层 —— 参数化 latency + chaos injection + metrics 埋点。

这是 Part 2 的核心工程落点：
- 替代所有 custom_tools.py 中的 asyncio.sleep()
- 统一输出 start_ts / end_ts / wait_reason / lock_owner / latency_components
- 支持 chaos event 注入
- 支持 shared device contention + priority-aware lock
- 支持 cabin-exclusive device isolation

Author: AstroSASF Team
Version: 8.0
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from scheduler.device_model import (
    DEFAULT_DEVICE_REGISTRY,
    DeviceSchema,
    DeviceScope,
    DeviceType,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────── #
#  Chaos Event Types                                                            #
# ────────────────────────────────────────────────────────────────────────────── #

class ChaosType(Enum):
    HARDWARE_DELAY = "hardware_delay"
    TELEMETRY_ALARM = "telemetry_alarm"
    DEVICE_STEAL = "device_steal"
    TIMEOUT = "timeout"
    FAILURE = "failure"


# ────────────────────────────────────────────────────────────────────────────── #
#  Chaos Event & Engine                                                         #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class ChaosEvent:
    """单个 chaos 事件定义（在 benchmark episode 中预定义）。"""
    trigger_time_sec: float
    type: ChaosType
    target_device: str | None = None
    delay_multiplier: float | None = None
    telemetry_key: str | None = None
    override_value: Any | None = None
    consumed: bool = field(default=False, init=False)


class ChaosEngine:
    """chaos event 注入引擎。"""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._events: list[ChaosEvent] = []
        self._elapsed_sec: float = 0.0
        self._injected: list[ChaosEvent] = []

    def load_events(self, events: list[ChaosEvent]) -> None:
        self._events = sorted(events, key=lambda e: e.trigger_time_sec)
        self._injected.clear()

    def advance_time(self, delta_sec: float) -> None:
        self._elapsed_sec += delta_sec

    @property
    def elapsed_sec(self) -> float:
        return self._elapsed_sec

    def check_and_inject(
        self,
        device_id: str,
        task_id: str,
    ) -> tuple[bool, ChaosEvent | None]:
        """检查并注入 chaos。触发时标记 consumed=True。"""
        for ev in self._events:
            if ev.consumed:
                continue
            if ev.trigger_time_sec <= self._elapsed_sec:
                if ev.target_device is None or ev.target_device == device_id:
                    ev.consumed = True
                    self._injected.append(ev)
                    logger.warning(
                        "[Chaos] 注入: device=%s type=%s task=%s at %.3fs",
                        device_id, ev.type.value, task_id, self._elapsed_sec,
                    )
                    return True, ev
        return False, None

    def apply_delay_multiplier(
        self,
        base_ms: float,
        multiplier: float | None,
    ) -> float:
        if multiplier is None or multiplier <= 0:
            return base_ms
        return base_ms * multiplier

    @property
    def injected_events(self) -> list[ChaosEvent]:
        return list(self._injected)

    def reset(self) -> None:
        self._elapsed_sec = 0.0
        self._injected.clear()
        for ev in self._events:
            ev.consumed = False


# ────────────────────────────────────────────────────────────────────────────── #
#  Device Result                                                                #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class DeviceResult:
    """设备调用结果的完整记录（用于 metrics 采集）。"""
    device_id: str
    action: str
    params: dict[str, Any]
    task_id: str
    lab_id: str
    start_ts: float
    end_ts: float
    wait_reason: str | None = None
    lock_owner: str | None = None
    latency_components: dict[str, float] = field(default_factory=dict)
    chaos_injected: bool = False
    chaos_type: ChaosType | None = None
    status: str = "ok"  # ok / timeout / failure / interlock / scope_denied

    @property
    def total_latency_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "action": self.action,
            "params": self.params,
            "task_id": self.task_id,
            "lab_id": self.lab_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "wait_reason": self.wait_reason,
            "lock_owner": self.lock_owner,
            "latency_components": {
                k: round(v, 3) for k, v in self.latency_components.items()
            },
            "chaos_injected": self.chaos_injected,
            "chaos_type": self.chaos_type.value if self.chaos_type else None,
            "status": self.status,
        }


# ────────────────────────────────────────────────────────────────────────────── #
#  Device Lock Manager                                                           #
# ────────────────────────────────────────────────────────────────────────────── #

@dataclass
class DeviceLockEntry:
    owner_task_id: str
    owner_lab_id: str
    acquired_at: float
    priority: int = 2


class DeviceLockManager:
    """全局设备锁管理器（支持 priority-aware lock）。"""

    def __init__(self) -> None:
        self._locks: dict[str, DeviceLockEntry] = {}
        self._lock = asyncio.Lock()
        self._total_contention_events: int = 0
        self._total_contention_wait_ms: float = 0.0

    async def acquire(
        self,
        device_id: str,
        task_id: str,
        lab_id: str,
        priority: int = 2,
        timeout: float = 30.0,
    ) -> tuple[bool, str | None]:
        """尝试原子获取设备锁。

        Returns
        -------
        tuple[bool, str | None]
            (acquired, wait_reason)
            acquired=True → 成功
            acquired=False, "device_busy" → 设备被占用
            acquired=False, "scope_denied" → 舱无权限（由 DeviceRuntime 在调用前检查）
        """
        async with self._lock:
            entry = self._locks.get(device_id)
            if entry is not None:
                self._total_contention_events += 1
                return False, "device_busy"

            self._locks[device_id] = DeviceLockEntry(
                owner_task_id=task_id,
                owner_lab_id=lab_id,
                acquired_at=time.monotonic(),
                priority=priority,
            )
            logger.debug("[Lock] 获取锁: device=%s task=%s", device_id, task_id)
            return True, None

    async def wait_for_unlock(
        self,
        device_id: str,
        task_id: str,
        timeout: float = 30.0,
    ) -> tuple[bool, float]:
        """轮询等待设备锁释放（被 asyncio.sleep 打断时立即返回 False）。

        Returns
        -------
        tuple[bool, float]
            (success, waited_ms) — 成功获取锁时返回等待时长（毫秒）
        """
        wait_start = time.monotonic()
        total_wait = 0.0
        while True:
            async with self._lock:
                if device_id not in self._locks:
                    waited_ms = total_wait * 1000.0
                    self._total_contention_wait_ms += waited_ms
                    return True, waited_ms
            elapsed = time.monotonic() - wait_start
            if elapsed >= timeout:
                waited_ms = total_wait * 1000.0
                return False, waited_ms
            sleep_t = min(0.05, timeout - elapsed)
            await asyncio.sleep(sleep_t)
            total_wait += sleep_t

    def release(self, device_id: str, task_id: str) -> None:
        """释放设备锁（仅当持有者匹配时）。"""
        entry = self._locks.get(device_id)
        if entry is not None and entry.owner_task_id == task_id:
            del self._locks[device_id]
            logger.debug("[Lock] 释放锁: device=%s task=%s held=%.2fs",
                         device_id, task_id,
                         time.monotonic() - entry.acquired_at)

    def is_locked(self, device_id: str) -> bool:
        return device_id in self._locks

    def get_lock_owner(self, device_id: str) -> str | None:
        entry = self._locks.get(device_id)
        return entry.owner_task_id if entry else None

    @property
    def contention_stats(self) -> dict[str, Any]:
        return {
            "total_contention_events": self._total_contention_events,
            "total_contention_wait_ms": round(self._total_contention_wait_ms, 3),
            "currently_locked": list(self._locks.keys()),
        }


# ────────────────────────────────────────────────────────────────────────────── #
#  Device Runtime（核心 wrapper）                                                #
# ────────────────────────────────────────────────────────────────────────────── #

class DeviceRuntime:
    """所有物理设备动作的统一调用 wrapper。

    使用方式（替换原有 custom_tools.py 中的 asyncio.sleep）：
    ```python
    async def set_temperature(ctx: MCPToolContext, temperature: float) -> dict:
        result = await ctx.device_runtime.invoke(
            device_id="heater_bio",
            action="set_temperature",
            params={"temperature": temperature, "hold_duration_ms": 2000.0},
            task_id=ctx.current_task_id,
            lab_id=ctx.lab_id,
            cabin_id=ctx.lab_id,
            telemetry_snapshot=await ctx.bus.snapshot(),
        )
        await ctx.bus.write("temperature", temperature)
        return {"status": "ok", "detail": f"温度设置完成，耗时 {result.total_latency_ms:.1f}ms"}
    ```

    相比 asyncio.sleep 的优势：
    1. 延迟与温度差/体积/距离等物理参数相关（非固定值）
    2. 输出完整的 DeviceResult 用于 metrics 采集
    3. 支持 chaos 注入（hardware_delay / failure / timeout）
    4. 支持 cabin-exclusive isolation
    5. 支持 shared device contention + priority-aware lock
    """

    def __init__(
        self,
        device_registry: dict[str, DeviceSchema] | None = None,
        chaos: ChaosEngine | None = None,
        seed: int | None = None,
        physical_delay_scale: float = 1.0,
    ) -> None:
        self._registry: dict[str, DeviceSchema] = (
            device_registry if device_registry is not None
            else dict(DEFAULT_DEVICE_REGISTRY)
        )
        self._chaos = chaos if chaos is not None else ChaosEngine(seed=seed)
        self._lock_mgr = DeviceLockManager()
        self._seed = seed
        self._rng = random.Random(seed)
        self._invocation_count: int = 0
        self._total_physical_ms: float = 0.0
        self._physical_delay_scale: float = max(0.0, min(1.0, physical_delay_scale))

    @property
    def chaos(self) -> ChaosEngine:
        return self._chaos

    @property
    def lock_manager(self) -> DeviceLockManager:
        return self._lock_mgr

    def register_device(self, schema: DeviceSchema) -> None:
        self._registry[schema.device_id] = schema
        logger.debug("[DeviceRuntime] 注册设备: %s (scope=%s)",
                     schema.device_id, schema.scope.name)

    def get_device_schema(self, device_id: str) -> DeviceSchema | None:
        return self._registry.get(device_id)

    # ── 核心调用入口 ──────────────────────────────────────────────────────────

    async def invoke(
        self,
        device_id: str,
        action: str,
        params: dict[str, Any],
        task_id: str,
        lab_id: str,
        cabin_id: str,
        telemetry_snapshot: dict[str, Any] | None = None,
    ) -> DeviceResult:
        """统一设备调用入口。

        完整调用链：
        1. scope 检查（cabin_exclusive / white-list）
        2. 资源锁获取（shared contention）
        3. chaos 注入检查
        4. 参数化 latency 计算
        5. asyncio.sleep（真实让出控制权）
        6. 资源锁释放
        7. 返回 DeviceResult（含 latency_components）
        """
        start_ts = time.monotonic()
        wait_reason: str | None = None
        lock_owner: str | None = None
        chaos_injected = False
        chaos_type: ChaosType | None = None
        latency_components: dict[str, float] = {}
        status = "ok"
        telemetry = telemetry_snapshot or {}

        # ── Step 0: 查找设备 schema ─────────────────────────────────────────
        schema = self._registry.get(device_id)
        if schema is None:
            logger.warning("[DeviceRuntime] 未知设备 '%s'，使用默认延迟", device_id)
            await asyncio.sleep(0.05)
            end_ts = time.monotonic()
            return DeviceResult(
                device_id=device_id,
                action=action,
                params=params,
                task_id=task_id,
                lab_id=lab_id,
                start_ts=start_ts,
                end_ts=end_ts,
                wait_reason="unknown_device",
                status="ok",
            )

        # ── Step 1: Scope 检查 ───────────────────────────────────────────────
        if not schema.is_allowed_for(cabin_id):
            logger.warning(
                "[DeviceRuntime] 舱 '%s' 无权访问设备 '%s' (scope=%s)",
                cabin_id, device_id, schema.scope.name,
            )
            return DeviceResult(
                device_id=device_id,
                action=action,
                params=params,
                task_id=task_id,
                lab_id=lab_id,
                start_ts=start_ts,
                end_ts=time.monotonic(),
                wait_reason="scope_denied",
                status="scope_denied",
            )

        # ── Step 2: 获取资源锁 ─────────────────────────────────────────────────
        acquired, _ = await self._lock_mgr.acquire(
            device_id=device_id,
            task_id=task_id,
            lab_id=lab_id,
        )
        contention_wait_ms = 0.0
        if not acquired:
            lock_owner = self._lock_mgr.get_lock_owner(device_id)
            wait_reason = "device_busy"
            got_lock, contention_wait_ms = await self._lock_mgr.wait_for_unlock(device_id, task_id)
            if not got_lock:
                return DeviceResult(
                    device_id=device_id,
                    action=action,
                    params=params,
                    task_id=task_id,
                    lab_id=lab_id,
                    start_ts=start_ts,
                    end_ts=time.monotonic(),
                    wait_reason="timeout",
                    lock_owner=lock_owner,
                    status="timeout",
                )
            lock_owner = None

        try:
            # ── Step 3: Chaos 注入检查 ──────────────────────────────────────────
            should_chaos, chaos_ev = self._chaos.check_and_inject(device_id, task_id)
            if should_chaos and chaos_ev is not None:
                chaos_injected = True
                chaos_type = chaos_ev.type

                if chaos_ev.type == ChaosType.FAILURE:
                    return DeviceResult(
                        device_id=device_id,
                        action=action,
                        params=params,
                        task_id=task_id,
                        lab_id=lab_id,
                        start_ts=start_ts,
                        end_ts=time.monotonic(),
                        wait_reason="chaos_failure",
                        lock_owner=lock_owner,
                        chaos_injected=True,
                        chaos_type=chaos_type,
                        status="failure",
                    )

                if chaos_ev.type == ChaosType.TIMEOUT:
                    await asyncio.sleep(schema.timeout_ms / 1000.0)
                    return DeviceResult(
                        device_id=device_id,
                        action=action,
                        params=params,
                        task_id=task_id,
                        lab_id=lab_id,
                        start_ts=start_ts,
                        end_ts=time.monotonic(),
                        wait_reason="chaos_timeout",
                        lock_owner=lock_owner,
                        chaos_injected=True,
                        chaos_type=chaos_type,
                        status="timeout",
                    )

            # ── Step 4: 参数化 latency 计算 ───────────────────────────────────
            base_ms = schema.compute_latency(params, telemetry)
            latency_components["base_ms"] = base_ms

            chaos_mult = 1.0
            if chaos_injected and chaos_ev is not None:
                chaos_mult = chaos_ev.delay_multiplier or 1.0

            actual_ms = schema.apply_jitter(base_ms)
            actual_ms = self._chaos.apply_delay_multiplier(actual_ms, chaos_mult)
            latency_components["jitter_ms"] = max(0.0, actual_ms - base_ms)
            latency_components["chaos_delay_ms"] = (
                (chaos_mult - 1.0) * base_ms if chaos_injected else 0.0
            )
            latency_components["contention_wait_ms"] = contention_wait_ms  # 锁竞争等待时间
            scaled_ms = actual_ms * self._physical_delay_scale
            latency_components["total_physical_ms"] = actual_ms  # 原始物理时间（metrics 用）
            latency_components["scaled_physical_ms"] = scaled_ms  # 实际等待时间

            # ── Step 5: 执行物理动作（真实 sleep，让出控制权）────────────────
            # physical_delay_scale ∈ [0,1]，用于实验变量控制（paper 实验设为 1.0）
            await asyncio.sleep(scaled_ms / 1000.0)

            self._invocation_count += 1
            self._total_physical_ms += actual_ms

        except asyncio.CancelledError:
            self._lock_mgr.release(device_id, task_id)
            raise
        finally:
            # ── Step 6: 释放资源锁 ─────────────────────────────────────────────
            self._lock_mgr.release(device_id, task_id)

        end_ts = time.monotonic()

        return DeviceResult(
            device_id=device_id,
            action=action,
            params=params,
            task_id=task_id,
            lab_id=lab_id,
            start_ts=start_ts,
            end_ts=end_ts,
            wait_reason=wait_reason,
            lock_owner=lock_owner,
            latency_components=latency_components,
            chaos_injected=chaos_injected,
            chaos_type=chaos_type,
            status=status,
        )

    # ── 便捷方法 ─────────────────────────────────────────────────────────────

    async def invoke_batch(
        self,
        calls: list[dict[str, Any]],
        telemetry_snapshot: dict[str, Any] | None = None,
    ) -> list[DeviceResult]:
        """批量调用多个设备动作（串行执行）。"""
        results = []
        for call in calls:
            result = await self.invoke(
                device_id=call["device_id"],
                action=call["action"],
                params=call.get("params", {}),
                task_id=call.get("task_id", "batch"),
                lab_id=call.get("lab_id", "default"),
                cabin_id=call.get("cabin_id", "default"),
                telemetry_snapshot=telemetry_snapshot,
            )
            results.append(result)
        return results

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "invocation_count": self._invocation_count,
            "total_physical_ms": round(self._total_physical_ms, 3),
            "avg_physical_ms": (
                round(self._total_physical_ms / max(1, self._invocation_count), 3)
            ),
            "chaos_injected_count": len(self._chaos.injected_events),
            "lock_contention": self._lock_mgr.contention_stats,
            "elapsed_sec": round(self._chaos.elapsed_sec, 3),
        }


__all__ = [
    "DeviceRuntime",
    "DeviceResult",
    "DeviceLockManager",
    "DeviceLockEntry",
    "ChaosEngine",
    "ChaosEvent",
    "ChaosType",
]
