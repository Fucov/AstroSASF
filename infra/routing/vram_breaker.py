"""
AstroSASF · Infra · VRAM Watermark Breaker (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
高压显存水线与请求熔断器。

针对 V100-32G 显存瓶颈，结合任务优先级系统：

1. **显存水线检测**：当实例 VRAM > High Watermark (85%) 时触发限流
2. **优先级感知的熔断策略**：

   | 优先级   | 允许最高 VRAM | 策略 |
   |----------|---------------|------|
   | CRITICAL | 100%          | 强制放行（硬件报警响应必须执行） |
   | HIGH     | 92%           | 超过 Critical 则拒绝 |
   | NORMAL   | 85%           | 超过 High 则拒绝 |
   | LOW      | 60%           | 超过 Low 则拒绝 |

3. **自适应限流**：根据显存压力动态调整请求速率
4. **熔断器状态机**：CLOSED → OPEN → HALF_OPEN → CLOSED

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from infra.llm.instance_pool import (
    InstanceMetrics,
    LLMInstancePool,
    VRAM_HIGH_WATERMARK,
    VRAM_CRITICAL_WATERMARK,
    VRAM_LOW_WATERMARK,
    InstanceStatus,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Enums & Constants                                                          #
# --------------------------------------------------------------------------- #

class CircuitState(Enum):
    """熔断器状态。"""
    CLOSED = auto()      # 正常：允许请求
    OPEN = auto()        # 熔断：拒绝请求
    HALF_OPEN = auto()   # 半开：尝试放行一个测试请求


class AdmissionResult(Enum):
    """准入决策。"""
    ADMITTED = auto()         # 允许进入
    REJECTED = auto()        # 拒绝（资源不足）
    QUEUED = auto()           # 排队等待
    REDIRECTED = auto()      # 重定向到其他实例
    CIRCUIT_OPEN = auto()     # 熔断器开启


DEFAULT_HIGH_WATERMARK: float = VRAM_HIGH_WATERMARK
DEFAULT_CRITICAL_WATERMARK: float = VRAM_CRITICAL_WATERMARK
DEFAULT_LOW_WATERMARK: float = VRAM_LOW_WATERMARK

DEFAULT_CIRCUIT_BREAKER_THRESHOLD: int = 3
DEFAULT_CIRCUIT_BREAKER_TIMEOUT: float = 30.0
DEFAULT_CIRCUIT_BREAKER_HALF_OPEN_TEST_COUNT: int = 3


# --------------------------------------------------------------------------- #
#  Admission Decision                                                         #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AdmissionDecision:
    """准入决策结果。"""
    result: AdmissionResult
    instance_url: str
    priority: str
    vram_ratio: float
    wait_seconds: float | None
    reason: str
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class BreakerConfig:
    """熔断器配置。"""
    high_watermark: float = DEFAULT_HIGH_WATERMARK
    critical_watermark: float = DEFAULT_CRITICAL_WATERMARK
    low_watermark: float = DEFAULT_LOW_WATERMARK
    circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    circuit_breaker_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT
    half_open_test_count: int = DEFAULT_CIRCUIT_BREAKER_HALF_OPEN_TEST_COUNT


# --------------------------------------------------------------------------- #
#  Per-Instance Circuit Breaker                                               #
# --------------------------------------------------------------------------- #

@dataclass
class InstanceCircuitBreaker:
    """单个实例的熔断器。"""
    url: str
    config: BreakerConfig = field(default_factory=BreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    consecutive_rejections: int = 0
    last_rejection_time: float = field(default_factory=time.monotonic)
    last_success_time: float = field(default_factory=time.monotonic)
    half_open_test_remaining: int = 0

    def can_attempt(self) -> bool:
        """判断是否可以尝试请求。"""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_test_remaining > 0
        if time.monotonic() - self.last_rejection_time >= self.config.circuit_breaker_timeout:
            return True
        return False

    def record_rejection(self) -> bool:
        """记录一次拒绝，返回是否触发了状态变更。"""
        self.consecutive_rejections += 1
        self.last_rejection_time = time.monotonic()

        if self.state == CircuitState.CLOSED:
            if self.consecutive_rejections >= self.config.circuit_breaker_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "[CircuitBreaker] 实例 %s 熔断开启 (连续 %d 次拒绝)",
                    self.url, self.consecutive_rejections
                )
                return True
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.half_open_test_remaining = 0
            logger.warning("[CircuitBreaker] 实例 %s 半开测试失败，重新熔断", self.url)
            return True

        return False

    def record_success(self) -> bool:
        """记录一次成功，返回是否触发了状态变更。"""
        self.consecutive_rejections = 0
        self.last_success_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_test_remaining -= 1
            if self.half_open_test_remaining <= 0:
                self.state = CircuitState.CLOSED
                logger.info(
                    "[CircuitBreaker] 实例 %s 熔断恢复 (半开测试通过)",
                    self.url
                )
                return True

        return False

    def attempt_half_open(self) -> bool:
        """尝试进入半开状态。"""
        if self.state != CircuitState.OPEN:
            return False

        elapsed = time.monotonic() - self.last_rejection_time
        if elapsed >= self.config.circuit_breaker_timeout:
            self.state = CircuitState.HALF_OPEN
            self.half_open_test_remaining = self.config.half_open_test_count
            self.consecutive_rejections = 0
            logger.info(
                "[CircuitBreaker] 实例 %s 进入半开状态 (timeout=%.1fs elapsed)",
                self.url, elapsed
            )
            return True
        return False


# --------------------------------------------------------------------------- #
#  VRAMWatermarkBreaker                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class VRAMWatermarkBreaker:
    """高压显存水线熔断器（内核模块）。

    结合任务优先级系统，对每个实例的显存使用进行监控和熔断。

    优先级策略：
    | 优先级   | 允许最高 VRAM | 策略 |
    |----------|---------------|------|
    | CRITICAL | 100%          | 强制放行（即使临界） |
    | HIGH     | 92%           | 超过 Critical 则拒绝 |
    | NORMAL   | 85%           | 超过 High 则拒绝 |
    | LOW      | 60%           | 超过 Low 则拒绝 |
    """

    instance_pool: LLMInstancePool
    config: BreakerConfig = field(default_factory=BreakerConfig)
    _circuit_breakers: dict[str, InstanceCircuitBreaker] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "admitted": 0,
        "rejected": 0,
        "queued": 0,
        "redirected": 0,
        "circuit_open": 0,
    })
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.instance_pool.on_status_change(self._on_instance_status_change)

    # ------------------------------------------------------------------------ #
    #  Priority-Aware Admission Control                                          #
    # ------------------------------------------------------------------------ #

    def check_admission(
        self,
        instance: InstanceMetrics,
        priority: str = "NORMAL",
    ) -> AdmissionDecision:
        """检查请求是否允许进入实例。"""
        url = instance.url
        vram_ratio = instance.vram_ratio

        breaker = self._get_or_create_breaker(url)

        # 1. 检查熔断器
        if not breaker.can_attempt():
            breaker.attempt_half_open()
            if breaker.state == CircuitState.OPEN:
                self._stats["circuit_open"] += 1
                return AdmissionDecision(
                    result=AdmissionResult.CIRCUIT_OPEN,
                    instance_url=url,
                    priority=priority,
                    vram_ratio=vram_ratio,
                    wait_seconds=self.config.circuit_breaker_timeout,
                    reason=f"熔断器开启，预计 {self.config.circuit_breaker_timeout}s 后恢复",
                )

        # 2. 确定该优先级允许的最高 VRAM 水线
        max_vram = self._get_priority_max_vram(priority)

        # 3. 检查 VRAM 是否超过水线
        if vram_ratio >= max_vram:
            # CRITICAL 特殊处理：即使超过临界也放行
            if priority == "CRITICAL" and vram_ratio < 1.0:
                logger.warning(
                    "[WatermarkBreaker] CRITICAL 任务强制放行: %s (VRAM: %.1f%% >= %.1f%%)",
                    url, vram_ratio * 100, max_vram * 100
                )
                self._stats["admitted"] += 1
                breaker.record_success()
                return AdmissionDecision(
                    result=AdmissionResult.ADMITTED,
                    instance_url=url,
                    priority=priority,
                    vram_ratio=vram_ratio,
                    wait_seconds=None,
                    reason="CRITICAL 任务强制放行（显存超限但紧急任务优先）",
                )

            # 记录拒绝
            breaker.record_rejection()
            self._stats["rejected"] += 1

            if vram_ratio >= self.config.critical_watermark:
                return AdmissionDecision(
                    result=AdmissionResult.REJECTED,
                    instance_url=url,
                    priority=priority,
                    vram_ratio=vram_ratio,
                    wait_seconds=None,
                    reason=f"VRAM 超过临界水线 ({vram_ratio * 100:.1f}% >= {self.config.critical_watermark * 100:.1f}%)",
                )
            else:
                return AdmissionDecision(
                    result=AdmissionResult.REJECTED,
                    instance_url=url,
                    priority=priority,
                    vram_ratio=vram_ratio,
                    wait_seconds=None,
                    reason=f"VRAM 超过优先级允许水线 ({vram_ratio * 100:.1f}% >= {max_vram * 100:.1f}%)",
                )

        # 4. 通过检查
        breaker.record_success()
        self._stats["admitted"] += 1

        return AdmissionDecision(
            result=AdmissionResult.ADMITTED,
            instance_url=url,
            priority=priority,
            vram_ratio=vram_ratio,
            wait_seconds=None,
            reason=f"准入通过 (VRAM: {vram_ratio * 100:.1f}% < {max_vram * 100:.1f}%)",
        )

    def _get_priority_max_vram(self, priority: str) -> float:
        """根据优先级确定允许的最高 VRAM 水线。"""
        priority_vram_map = {
            "CRITICAL": 1.0,
            "HIGH": self.config.critical_watermark,
            "NORMAL": self.config.high_watermark,
            "LOW": self.config.low_watermark,
        }
        return priority_vram_map.get(priority.upper(), self.config.high_watermark)

    def _get_or_create_breaker(self, url: str) -> InstanceCircuitBreaker:
        """获取或创建实例的熔断器。"""
        if url not in self._circuit_breakers:
            self._circuit_breakers[url] = InstanceCircuitBreaker(url=url, config=self.config)
        return self._circuit_breakers[url]

    # ------------------------------------------------------------------------ #
    #  Multi-Instance Admission Control                                          #
    # ------------------------------------------------------------------------ #

    def find_admissible_instance(
        self,
        priority: str = "NORMAL",
        tags: list[str] | None = None,
    ) -> tuple[InstanceMetrics | None, AdmissionDecision]:
        """寻找一个允许准入的实例（按 VRAM 从低到高遍历）。"""
        candidates = self.instance_pool.available_instances

        if tags:
            configs = self.instance_pool.configs
            candidates = [
                inst for inst in candidates
                if all(tag in configs[inst.url].tags for tag in tags)
            ]

        candidates = sorted(candidates, key=lambda inst: inst.vram_ratio)

        for instance in candidates:
            decision = self.check_admission(instance, priority)
            if decision.result == AdmissionResult.ADMITTED:
                return instance, decision

        if candidates:
            decision = self.check_admission(candidates[-1], priority)
            return None, decision

        return None, None

    # ------------------------------------------------------------------------ #
    #  Request Lifecycle Integration                                             #
    # ------------------------------------------------------------------------ #

    async def on_request_start(self, url: str, priority: str) -> bool:
        """请求开始时的准入检查。"""
        instance = self.instance_pool.get_instance(url)
        if not instance:
            return False

        decision = self.check_admission(instance, priority)
        return decision.result == AdmissionResult.ADMITTED

    async def on_request_success(self, url: str) -> None:
        """请求成功完成。"""
        breaker = self._get_or_create_breaker(url)
        breaker.record_success()

    async def on_request_failure(self, url: str) -> None:
        """请求失败。"""
        breaker = self._get_or_create_breaker(url)
        breaker.record_rejection()

    # ------------------------------------------------------------------------ #
    #  Instance Status Change Handler                                            #
    # ------------------------------------------------------------------------ #

    def _on_instance_status_change(self, url: str, old_status: InstanceStatus, new_status: InstanceStatus) -> None:
        """实例状态变更回调。当实例变为 UNHEALTHY 时，重置该实例的熔断器。"""
        if new_status == InstanceStatus.UNHEALTHY:
            breaker = self._get_or_create_breaker(url)
            breaker.state = CircuitState.OPEN
            breaker.consecutive_rejections = breaker.config.circuit_breaker_threshold
            breaker.last_rejection_time = time.monotonic()
            logger.warning("[WatermarkBreaker] 实例 %s 不可用，重置熔断器", url)

    # ------------------------------------------------------------------------ #
    #  Stats & Debug                                                            #
    # ------------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。"""
        total = sum(self._stats.values())
        rejection_rate = (
            self._stats["rejected"] / max(1, total) * 100
        )
        return {
            "total_requests": total,
            "admitted": self._stats["admitted"],
            "rejected": self._stats["rejected"],
            "rejection_rate": f"{rejection_rate:.1f}%",
            "circuit_open_count": self._stats["circuit_open"],
            "active_circuit_breakers": sum(
                1 for b in self._circuit_breakers.values()
                if b.state == CircuitState.OPEN
            ),
            "breaker_states": {
                url: breaker.state.name
                for url, breaker in self._circuit_breakers.items()
            },
        }
