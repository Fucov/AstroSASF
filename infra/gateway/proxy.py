"""
AstroSASF · Infra · Distributed Gateway Proxy (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分布式 LLM 网关反向代理。

整合 LLMInstancePool、PrefixAwareLoadBalancer、VRAMWatermarkBreaker，
实现对 AstroSASF 原单机 /api/v1/llm/chat 请求的反向代理。

核心功能：
1. **请求拦截**：拦截 /api/v1/llm/* 请求
2. **前缀感知路由**：计算 Prompt 前缀哈希，复用 KV-Cache（支持 SGLang RadixAttention）
3. **显存熔断**：结合优先级进行准入控制
4. **流式响应透传**：将 SGLang/vLLM 的流式响应原样透传给客户端
5. **透明集成**：对上层多智能体完全透明

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

from infra.llm.instance_pool import (
    LLMInstanceConfig,
    LLMInstancePool,
)
from infra.routing.prefix_balancer import (
    PrefixAwareLoadBalancer,
    RoutingDecision,
    RoutingStrategy,
)
from infra.routing.vram_breaker import (
    VRAMWatermarkBreaker,
    AdmissionResult,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Request/Response Models                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class GatewayRequest:
    """网关请求。"""
    messages: list[dict[str, str]]   # [{"role": "user", "content": "..."}]
    model: str = ""                  # 模型名称
    temperature: float = 0.1
    max_tokens: int = 2048
    stream: bool = False
    priority: str = "NORMAL"         # NORMAL | HIGH | CRITICAL | LOW
    tags: list[str] = field(default_factory=list)
    request_id: str = ""
    arrival_time: float = field(default_factory=time.monotonic)


@dataclass
class GatewayResponse:
    """网关响应。"""
    content: str | None    # 非流式响应的内容
    model: str
    usage: dict[str, int] | None
    finish_reason: str | None
    request_id: str
    routed_to: str
    routing_strategy: str
    prefix_hash: str
    latency_ms: float
    streamed: bool = False


@dataclass
class GatewayError:
    """网关错误。"""
    error: str
    error_code: str
    routed_to: str | None = None
    retry_after: float | None = None


# --------------------------------------------------------------------------- #
#  GatewayProxy                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class GatewayProxy:
    """分布式 LLM 网关反向代理（内核模块）。

    整合三层组件：
    - LLMInstancePool：实例管理
    - PrefixAwareLoadBalancer：前缀感知路由（SGLang RadixAttention KV-Cache 复用）
    - VRAMWatermarkBreaker：优先级感知准入控制

    请求流程：
    1. 路由决策 → load_balancer.route()
    2. 准入检查 → watermark_breaker.check_admission()
       2a. 熔断拒绝 → _try_fallback_routing()
       2b. VRAM 拒绝 → _try_fallback_routing()
    3. 记录连接 → instance_pool.acquire_connection()
    4. 发送请求 → _do_request() / _do_streaming_request()
    5. 记录成功 → release_connection() + record_route_success()
    """

    instance_pool: LLMInstancePool
    load_balancer: PrefixAwareLoadBalancer | None = None
    watermark_breaker: VRAMWatermarkBreaker | None = None
    _http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    # 统计
    _stats: dict[str, int] = field(default_factory=lambda: {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "prefix_cached_requests": 0,
        "circuit_broken_requests": 0,
        "vram_rejected_requests": 0,
    })

    def __post_init__(self) -> None:
        if self.load_balancer is None:
            self.load_balancer = PrefixAwareLoadBalancer(instance_pool=self.instance_pool)
        if self.watermark_breaker is None:
            self.watermark_breaker = VRAMWatermarkBreaker(instance_pool=self.instance_pool)

    # ------------------------------------------------------------------------ #
    #  Lifecycle                                                                 #
    # ------------------------------------------------------------------------ #

    async def start(self) -> None:
        """启动网关代理。"""
        if self._running:
            return
        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        await self.load_balancer.start()
        await self.instance_pool.start()
        logger.info("[GatewayProxy] 网关代理已启动")

    async def stop(self) -> None:
        """停止网关代理。"""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        await self.load_balancer.stop()
        await self.instance_pool.stop()
        logger.info("[GatewayProxy] 网关代理已停止")

    # ------------------------------------------------------------------------ #
    #  Routing & Admission Control                                               #
    # ------------------------------------------------------------------------ #

    async def _route(self, request: GatewayRequest) -> tuple[RoutingDecision, str]:
        """执行路由决策和准入检查。

        Returns
        -------
        tuple[RoutingDecision, str]
            (路由决策, 目标实例 URL)
        """
        system_prompt = None
        user_prompt = ""

        for msg in request.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content

        if not user_prompt:
            user_prompt = str(request.messages[-1].get("content", ""))

        # 1. 路由决策
        decision = await self.load_balancer.route(
            prompt=user_prompt,
            system_prompt=system_prompt,
            priority=request.priority,
            tags=request.tags,
            prefer_cached=True,
        )

        # 2. 准入检查
        admission = self.watermark_breaker.check_admission(
            instance=self.instance_pool.get_instance(decision.url),
            priority=request.priority,
        )

        if admission.result != AdmissionResult.ADMITTED:
            # 尝试降级重路由
            return await self._try_fallback_routing(request, admission.reason)

        return decision, decision.url

    async def _try_fallback_routing(
        self,
        request: GatewayRequest,
        reason: str,
    ) -> tuple[RoutingDecision, str]:
        """尝试降级重路由（Least-Connections）。"""
        self._stats["vram_rejected_requests"] += 1

        logger.warning(
            "[GatewayProxy] 路由被拒绝 (%s)，尝试降级重路由", reason
        )

        instance, admission = self.watermark_breaker.find_admissible_instance(
            priority=request.priority,
            tags=request.tags,
        )

        if instance is None:
            raise RuntimeError(f"无可用实例: {admission.reason if admission else '未知原因'}")

        decision = RoutingDecision(
            url=instance.url,
            strategy=RoutingStrategy.LEAST_CONNECTIONS,
            reason=f"降级重路由: {reason}",
            prefix_hash="",
            vram_ratio=instance.vram_ratio,
            active_connections=instance.active_connections,
            is_cached=False,
        )

        return decision, instance.url

    # ------------------------------------------------------------------------ #
    #  HTTP Request Handling                                                     #
    # ------------------------------------------------------------------------ #

    async def chat(self, request: GatewayRequest) -> GatewayResponse | GatewayError:
        """同步 Chat 接口。

        完整链路：路由 → 准入 → 连接 → 请求 → 响应 → 记录
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        start_time = time.monotonic()
        self._stats["total_requests"] += 1

        try:
            # 1. 路由 + 准入
            decision, url = await self._route(request)

            # 2. 获取连接
            await self.instance_pool.acquire_connection(url)

            try:
                # 3. 发送请求
                result = await self._do_request(url, request)
                elapsed_ms = (time.monotonic() - start_time) * 1000

                # 4. 记录成功
                await self.instance_pool.release_connection(url, success=True, latency_ms=elapsed_ms)
                await self.watermark_breaker.on_request_success(url)
                await self.load_balancer.record_route_success(decision)

                self._stats["successful_requests"] += 1
                if decision.is_cached:
                    self._stats["prefix_cached_requests"] += 1

                logger.info(
                    "[GatewayProxy] 请求完成: %s → %s [%s] (%.1fms)",
                    request.request_id[:8], url,
                    decision.strategy.value, elapsed_ms,
                )

                return GatewayResponse(
                    content=result["content"],
                    model=result.get("model", ""),
                    usage=result.get("usage"),
                    finish_reason=result.get("finish_reason"),
                    request_id=request.request_id,
                    routed_to=url,
                    routing_strategy=decision.strategy.value,
                    prefix_hash=decision.prefix_hash,
                    latency_ms=elapsed_ms,
                    streamed=False,
                )

            except Exception as e:
                await self.instance_pool.release_connection(url, success=False)
                await self.watermark_breaker.on_request_failure(url)
                await self.load_balancer.record_route_failure(decision)
                self._stats["failed_requests"] += 1
                raise

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            error_code = "CIRCUIT_OPEN" if "熔断" in str(e) else "REQUEST_FAILED"
            return GatewayError(
                error=str(e),
                error_code=error_code,
                retry_after=5.0 if error_code == "CIRCUIT_OPEN" else None,
            )

    async def stream_chat(self, request: GatewayRequest) -> AsyncGenerator[bytes, None]:
        """流式 Chat 接口。

        直接透传 SGLang/vLLM 的 SSE 流式响应。
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        self._stats["total_requests"] += 1

        try:
            decision, url = await self._route(request)
            await self.instance_pool.acquire_connection(url)

            try:
                start_time = time.monotonic()
                async for chunk in self._do_streaming_request(url, request):
                    yield chunk

                elapsed_ms = (time.monotonic() - start_time) * 1000
                await self.instance_pool.release_connection(url, success=True, latency_ms=elapsed_ms)
                await self.watermark_breaker.on_request_success(url)
                await self.load_balancer.record_route_success(decision)

                self._stats["successful_requests"] += 1
                if decision.is_cached:
                    self._stats["prefix_cached_requests"] += 1

                logger.info(
                    "[GatewayProxy] 流式请求完成: %s → %s [%s] (%.1fms)",
                    request.request_id[:8], url,
                    decision.strategy.value, elapsed_ms,
                )

            except Exception:
                await self.instance_pool.release_connection(url, success=False)
                await self.watermark_breaker.on_request_failure(url)
                await self.load_balancer.record_route_failure(decision)
                self._stats["failed_requests"] += 1
                raise

        except Exception as e:
            error_msg = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_msg}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    # ------------------------------------------------------------------------ #
    #  Internal HTTP Methods                                                     #
    # ------------------------------------------------------------------------ #

    async def _do_request(self, url: str, request: GatewayRequest) -> dict[str, Any]:
        """发送非流式请求到 LLM 实例。"""
        if not self._http_client:
            raise RuntimeError("HTTP 客户端未初始化")

        payload = {
            "model": request.model or "qwen2.5-7b",
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        response = await self._http_client.post(
            f"{url}/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise RuntimeError(f"LLM 请求失败: HTTP {response.status_code} - {response.text}")

        data = response.json()

        content = ""
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")

        return {
            "content": content,
            "model": data.get("model", ""),
            "usage": data.get("usage"),
            "finish_reason": data["choices"][0].get("finish_reason") if "choices" in data else None,
        }

    async def _do_streaming_request(
        self,
        url: str,
        request: GatewayRequest,
    ) -> AsyncGenerator[bytes, None]:
        """发送流式请求到 LLM 实例（透传 SSE）。"""
        if not self._http_client:
            raise RuntimeError("HTTP 客户端未初始化")

        payload = {
            "model": request.model or "qwen2.5-7b",
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        async with self._http_client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(120.0),
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"LLM 流式请求失败: HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if line.strip() and line.startswith("data: "):
                    yield f"{line}\n".encode("utf-8")

            yield b"data: [DONE]\n\n"

    # ------------------------------------------------------------------------ #
    #  Instance Management                                                       #
    # ------------------------------------------------------------------------ #

    def register_instance(self, config: LLMInstanceConfig) -> None:
        """注册一个 LLM 实例。"""
        self.instance_pool.register_instance(config)

    # ------------------------------------------------------------------------ #
    #  Stats                                                                     #
    # ------------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """获取聚合统计信息。"""
        total = self._stats["total_requests"]
        successful = self._stats["successful_requests"]

        return {
            "gateway": {
                "total_requests": total,
                "successful_requests": successful,
                "failed_requests": self._stats["failed_requests"],
                "success_rate": f"{successful / max(1, total) * 100:.1f}%",
                "prefix_cached_rate": f"{self._stats['prefix_cached_requests'] / max(1, total) * 100:.1f}%",
                "circuit_broken_requests": self._stats["circuit_broken_requests"],
                "vram_rejected_requests": self._stats["vram_rejected_requests"],
            },
            "instance_pool": self.instance_pool.get_stats(),
            "load_balancer": self.load_balancer.get_stats(),
            "watermark_breaker": self.watermark_breaker.get_stats(),
        }
