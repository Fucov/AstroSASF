"""
AstroSASF · Middleware · LLMInstancePool (V7.2 — Distributed Gateway)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
跨节点 LLM 实例池管理器。

支持：
- 多 IP:Port 实例的注册与健康检查
- KV-Cache 显存比例探测（通过 SGLang/vLLM 的 /memory 分析接口）
- 权重加权路由

V7.2 新增：
- Async health check coroutine
- VRAM watermark reporting
- Connection pooling per instance

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

import httpx

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Enums & Constants                                                          #
# --------------------------------------------------------------------------- #

class InstanceStatus(Enum):
    """实例健康状态。"""
    HEALTHY = auto()      # 完全可用
    DEGRADED = auto()     # 显存告警，降级运行
    UNHEALTHY = auto()    # 不可达或严重故障
    STARTING = auto()     # 正在启动


# V100-32G 显存水线配置
VRAM_HIGH_WATERMARK: float = 0.85    # 85% 以上：高水位告警
VRAM_CRITICAL_WATERMARK: float = 0.92 # 92% 以上：熔断禁入
VRAM_LOW_WATERMARK: float = 0.60     # 60% 以下：完全健康

# 健康检查配置
DEFAULT_HEALTH_CHECK_INTERVAL: float = 10.0   # 健康检查周期（秒）
DEFAULT_HEALTH_CHECK_TIMEOUT: float = 5.0     # 健康检查超时（秒）
DEFAULT_UNHEALTHY_THRESHOLD: int = 3           # 连续失败次数阈值


# --------------------------------------------------------------------------- #
#  Data Classes                                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LLMInstanceConfig:
    """LLM 实例配置。"""
    url: str                           # 例如 "http://192.168.1.10:8000"
    weight: int = 1                   # 路由权重
    model_name: str = ""               # 模型名称（SGLang/vLLM 部署的模型）
    tags: list[str] = field(default_factory=list)  # 标签，用于路由筛选


@dataclass
class InstanceMetrics:
    """实例运行时指标。"""
    url: str
    status: InstanceStatus = InstanceStatus.STARTING
    # VRAM metrics
    vram_used_gb: float = 0.0
    vram_total_gb: float = 32.0       # V100-32G 默认值
    vram_ratio: float = 0.0          # vram_used / vram_total
    # Request metrics
    active_connections: int = 0      # 当前处理中的请求数
    total_requests: int = 0           # 累计请求数
    failed_requests: int = 0         # 失败请求数
    avg_latency_ms: float = 0.0      # 平均响应延迟
    # Health check
    consecutive_failures: int = 0
    last_check_time: float = field(default_factory=time.monotonic)
    last_success_time: float | None = None
    # Timestamps
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    @property
    def vram_category(self) -> str:
        """返回显存健康分类。"""
        if self.vram_ratio >= VRAM_CRITICAL_WATERMARK:
            return "CRITICAL"
        elif self.vram_ratio >= VRAM_HIGH_WATERMARK:
            return "HIGH"
        elif self.vram_ratio >= VRAM_LOW_WATERMARK:
            return "NORMAL"
        else:
            return "LOW"

    @property
    def is_available(self) -> bool:
        """判断实例是否可接受新请求。"""
        return (
            self.status in (InstanceStatus.HEALTHY, InstanceStatus.DEGRADED)
            and self.vram_ratio < VRAM_CRITICAL_WATERMARK
            and self.consecutive_failures < DEFAULT_UNHEALTHY_THRESHOLD
        )

    @property
    def health_score(self) -> float:
        """计算健康评分（0-1，越高越健康）。"""
        if self.status == InstanceStatus.UNHEALTHY:
            return 0.0
        if self.status == InstanceStatus.DEGRADED:
            return 0.5
        # 基于显存和连接数计算
        vram_score = 1.0 - self.vram_ratio
        connection_score = 1.0 - (self.active_connections / 10.0)  # 假设最大 10 并发
        return max(0.0, min(1.0, (vram_score + connection_score) / 2.0))


# --------------------------------------------------------------------------- #
#  LLMInstancePool                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class LLMInstancePool:
    """跨节点 LLM 实例池管理器。

    管理多个远端 LLM 实例（基于 SGLang/vLLM），提供：
    - 异步健康检查协程
    - KV-Cache 显存比例探测
    - 实例状态跟踪

    Example
    -------
    >>> pool = LLMInstancePool()
    >>> pool.register_instance(LLMInstanceConfig(
    ...     url="http://192.168.1.10:8000",
    ...     weight=1,
    ...     model_name="qwen2.5-7b",
    ...     tags=["v100", "node-1"]
    ... ))
    >>> await pool.start()
    >>> # 获取最健康的实例
    >>> instance = pool.get_least_loaded_instance()
    >>> await pool.stop()
    """

    instances: dict[str, InstanceMetrics] = field(default_factory=dict)
    configs: dict[str, LLMInstanceConfig] = field(default_factory=dict)
    _health_check_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    # HTTP client（复用连接池）
    _http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    # Callbacks
    _on_status_change: list[callable] = field(default_factory=list, init=False, repr=False)

    def register_instance(self, config: LLMInstanceConfig) -> None:
        """注册一个 LLM 实例。

        Parameters
        ----------
        config : LLMInstanceConfig
            实例配置
        """
        if config.url in self.instances:
            logger.warning("[InstancePool] 实例已存在: %s，将更新配置", config.url)
        else:
            logger.info("[InstancePool] 注册新实例: %s (weight=%d, model=%s)",
                        config.url, config.weight, config.model_name)

        self.configs[config.url] = config
        self.instances[config.url] = InstanceMetrics(
            url=config.url,
            status=InstanceStatus.STARTING,
        )

    def unregister_instance(self, url: str) -> bool:
        """注销一个 LLM 实例。

        Returns
        -------
        bool
            是否成功注销
        """
        if url not in self.instances:
            return False

        del self.instances[url]
        del self.configs[url]
        logger.info("[InstancePool] 注销实例: %s", url)
        return True

    @property
    def available_instances(self) -> list[InstanceMetrics]:
        """获取所有可用实例。"""
        return [inst for inst in self.instances.values() if inst.is_available]

    @property
    def healthy_instances(self) -> list[InstanceMetrics]:
        """获取所有健康实例。"""
        return [inst for inst in self.instances.values()
                if inst.status == InstanceStatus.HEALTHY]

    def get_instance(self, url: str) -> InstanceMetrics | None:
        """根据 URL 获取实例。"""
        return self.instances.get(url)

    def get_least_loaded_instance(
        self,
        tags: list[str] | None = None,
        require_healthy: bool = False,
    ) -> InstanceMetrics | None:
        """获取负载最轻的可用实例（Least-Connections 策略）。

        Parameters
        ----------
        tags : list[str] | None
            标签过滤，仅返回包含所有指定标签的实例
        require_healthy : bool
            是否仅返回 HEALTHY 状态的实例

        Returns
        -------
        InstanceMetrics | None
            负载最轻的实例，若无可用实例返回 None
        """
        candidates = self.available_instances

        if tags:
            configs = self.configs
            candidates = [
                inst for inst in candidates
                if all(tag in configs[inst.url].tags for tag in tags)
            ]

        if require_healthy:
            candidates = [inst for inst in candidates
                          if inst.status == InstanceStatus.HEALTHY]

        if not candidates:
            return None

        # 选择健康评分最高且连接数最少的
        return min(candidates, key=lambda inst: (1 - inst.health_score, inst.active_connections))

    async def start(self) -> None:
        """启动实例池（启动健康检查协程）。"""
        if self._running:
            logger.warning("[InstancePool] 实例池已在运行中")
            return

        self._running = True

        # 创建共享的 HTTP 客户端（连接池）
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_HEALTH_CHECK_TIMEOUT),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

        # 启动健康检查协程
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("[InstancePool] 实例池已启动，共 %d 个实例", len(self.instances))

    async def stop(self) -> None:
        """停止实例池。"""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("[InstancePool] 实例池已停止")

    # ------------------------------------------------------------------------ #
    #  Health Check Loop                                                       #
    # ------------------------------------------------------------------------ #

    async def _health_check_loop(self) -> None:
        """健康检查协程主循环。"""
        logger.info("[InstancePool] 健康检查协程已启动")

        while self._running:
            try:
                await self._check_all_instances()
                await asyncio.sleep(DEFAULT_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[InstancePool] 健康检查循环异常: %s", e)
                await asyncio.sleep(DEFAULT_HEALTH_CHECK_INTERVAL)

    async def _check_all_instances(self) -> None:
        """并发检查所有实例。"""
        if not self._http_client:
            return

        tasks = [
            self._check_instance(url)
            for url in self.instances
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_instance(self, url: str) -> None:
        """检查单个实例的健康状态。

        检查内容：
        1. 基础连通性（GET /health）
        2. 显存使用情况（GET /memory 或 /metrics）
        """
        if not self._http_client:
            return

        metrics = self.instances.get(url)
        if not metrics:
            return

        async with self._lock:
            start_time = time.monotonic()

            try:
                # 1. 基础健康检查
                health_response = await self._http_client.get(f"{url}/health")
                basic_ok = health_response.status_code == 200

                # 2. 显存检查（SGLang/vLLM 的 /memory 或 /metrics 接口）
                vram_ratio = 0.0
                vram_used = 0.0

                try:
                    # 尝试 SGLang 的 /memory 接口
                    mem_response = await self._http_client.get(f"{url}/memory")
                    if mem_response.status_code == 200:
                        mem_data = mem_response.json()
                        # SGLang 格式：{"mem_used": 1234567890, "mem_total": 34359738368}
                        if "mem_used" in mem_data and "mem_total" in mem_data:
                            vram_used = mem_data["mem_used"] / (1024 ** 3)  # bytes → GB
                            vram_total = mem_data["mem_total"] / (1024 ** 3)
                            vram_ratio = vram_used / vram_total if vram_total > 0 else 0.0
                except Exception:
                    # 尝试 vLLM 的 /metrics 接口（Prometheus 格式解析）
                    try:
                        metrics_response = await self._http_client.get(f"{url}/metrics")
                        if metrics_response.status_code == 200:
                            # 简单解析 vLLM GPU 显存指标
                            text = metrics_response.text
                            for line in text.split("\n"):
                                if "vllm_gpu_memory_usage" in line:
                                    parts = line.split()
                                    for i, part in enumerate(parts):
                                        if part == "value" and i + 1 < len(parts):
                                            vram_ratio = float(parts[i + 1])
                                            break
                    except Exception:
                        pass

                elapsed = time.monotonic() - start_time
                old_status = metrics.status

                # 更新指标
                metrics.vram_ratio = vram_ratio
                metrics.vram_used_gb = vram_used
                metrics.consecutive_failures = 0
                metrics.last_check_time = time.monotonic()
                metrics.last_success_time = time.monotonic()

                # 更新状态
                if not basic_ok:
                    metrics.status = InstanceStatus.UNHEALTHY
                elif vram_ratio >= VRAM_HIGH_WATERMARK:
                    metrics.status = InstanceStatus.DEGRADED
                else:
                    metrics.status = InstanceStatus.HEALTHY

                metrics.updated_at = time.monotonic()

                # 触发状态变更回调
                if old_status != metrics.status:
                    self._notify_status_change(url, old_status, metrics.status)
                    logger.warning(
                        "[InstancePool] 实例状态变更: %s %s -> %s (VRAM: %.1f%%)",
                        url, old_status.name, metrics.status.name, vram_ratio * 100
                    )

                logger.debug(
                    "[InstancePool] 实例健康检查 OK: %s (VRAM: %.1f%%, %.1fms)",
                    url, vram_ratio * 100, elapsed * 1000
                )

            except httpx.TimeoutException:
                metrics.consecutive_failures += 1
                if metrics.consecutive_failures >= DEFAULT_UNHEALTHY_THRESHOLD:
                    old_status = metrics.status
                    metrics.status = InstanceStatus.UNHEALTHY
                    if old_status != InstanceStatus.UNHEALTHY:
                        self._notify_status_change(url, old_status, InstanceStatus.UNHEALTHY)
                        logger.error("[InstancePool] 实例不可达: %s (%d 次连续失败)",
                                     url, metrics.consecutive_failures)
                metrics.updated_at = time.monotonic()

            except Exception as e:
                metrics.consecutive_failures += 1
                metrics.updated_at = time.monotonic()
                logger.warning("[InstancePool] 健康检查异常: %s -> %s", url, e)

    # ------------------------------------------------------------------------ #
    #  Connection Management                                                    #
    # ------------------------------------------------------------------------ #

    async def acquire_connection(self, url: str) -> None:
        """增加实例的活跃连接计数。

        在发起请求前调用，请求结束后调用 release_connection。
        """
        async with self._lock:
            if url in self.instances:
                self.instances[url].active_connections += 1

    async def release_connection(self, url: str, success: bool = True, latency_ms: float = 0.0) -> None:
        """减少实例的活跃连接计数并更新指标。

        Parameters
        ----------
        url : str
            实例 URL
        success : bool
            请求是否成功
        latency_ms : float
            请求延迟（毫秒）
        """
        async with self._lock:
            if url not in self.instances:
                return

            metrics = self.instances[url]
            metrics.active_connections = max(0, metrics.active_connections - 1)
            metrics.total_requests += 1

            if not success:
                metrics.failed_requests += 1

            # 滑动平均更新延迟
            if latency_ms > 0:
                n = metrics.total_requests
                metrics.avg_latency_ms = (metrics.avg_latency_ms * (n - 1) + latency_ms) / n

    # ------------------------------------------------------------------------ #
    #  Status Change Callbacks                                                  #
    # ------------------------------------------------------------------------ #

    def on_status_change(self, callback: callable) -> None:
        """注册状态变更回调。

        回调签名：callback(url: str, old_status: InstanceStatus, new_status: InstanceStatus)
        """
        self._on_status_change.append(callback)

    def _notify_status_change(self, url: str, old_status: InstanceStatus, new_status: InstanceStatus) -> None:
        """通知所有状态变更回调。"""
        for callback in self._on_status_change:
            try:
                callback(url, old_status, new_status)
            except Exception as e:
                logger.warning("[InstancePool] 状态变更回调异常: %s", e)

    # ------------------------------------------------------------------------ #
    #  Stats & Info                                                             #
    # ------------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """获取实例池统计信息。"""
        return {
            "total_instances": len(self.instances),
            "healthy_instances": len(self.healthy_instances),
            "available_instances": len(self.available_instances),
            "instances": {
                url: {
                    "status": m.status.name,
                    "vram_ratio": f"{m.vram_ratio * 100:.1f}%",
                    "vram_used_gb": f"{m.vram_used_gb:.1f}",
                    "active_connections": m.active_connections,
                    "health_score": f"{m.health_score:.2f}",
                    "avg_latency_ms": f"{m.avg_latency_ms:.1f}",
                    "last_success": m.last_success_time,
                }
                for url, m in self.instances.items()
            },
        }

    def __len__(self) -> int:
        """返回实例数量。"""
        return len(self.instances)
