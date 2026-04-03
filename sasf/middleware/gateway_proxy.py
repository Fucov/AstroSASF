"""
AstroSASF · Middleware · GatewayProxy (V7.2 — Distributed Gateway)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分布式 LLM 网关反向代理。

整合 LLMInstancePool、PrefixAwareLoadBalancer、VRAMWatermarkBreaker，
实现对 AstroSASF 原单机 /api/v1/llm/chat 请求的反向代理。

核心功能：
1. **请求拦截**：拦截 /api/v1/llm/* 请求
2. **前缀感知路由**：计算 Prompt 前缀哈希，复用 KV-Cache
3. **显存熔断**：结合优先级进行准入控制
4. **流式响应透传**：将 SGLang/vLLM 的流式响应原样透传给客户端
5. **透明集成**：对上层多智能体完全透明

V7.2 新增：
- Distributed HTTP reverse proxy
- Streaming response passthrough
- Prefix-aware routing integration

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

from sasf.middleware.llm_instance_pool import (
    LLMInstanceConfig,
    LLMInstancePool,
)
from sasf.middleware.prefix_aware_load_balancer import (
    PrefixAwareLoadBalancer,
    RoutingDecision,
    RoutingStrategy,
)
from sasf.middleware.vram_watermark_breaker import (
    VRAMWatermarkBreaker,
    AdmissionDecision,
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
    tags: list[str] = field(default_factory=list)  # 标签筛选
    # 内部字段
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
    routed_to: str  # 实际路由到的实例 URL
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
    retry_after: float | None = None  # 重试建议延迟（秒）


# --------------------------------------------------------------------------- #
#  GatewayProxy                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class GatewayProxy:
    """分布式 LLM 网关反向代理。

    工作流程：
    1. 接收请求（/api/v1/llm/chat）
    2. 提取 Prompt 前缀，计算 Hash
    3. 通过 PrefixAwareLoadBalancer 选择实例
    4. 通过 VRAMWatermarkBreaker 进行准入检查
    5. 发起 HTTP 请求到选中的 SGLang/vLLM 实例
    6. 透传流式响应 or 聚合非流式响应

    Example
    -------
    >>> pool = LLMInstancePool()
    >>> pool.register_instance(LLMInstanceConfig(
    ...     url="http://192.168.1.10:8000",
    ...     weight=1,
    ...     model_name="qwen2.5-7b"
    ... ))
    >>> await pool.start()
    >>>
    >>> proxy = GatewayProxy(instance_pool=pool)
    >>> await proxy.start()
    >>>
    >>> # 处理请求
    >>> request = GatewayRequest(
    ...     messages=[{"role": "user", "content": "SOP:bio_culture\\n请帮我开始细胞培养"}],
    ...     model="qwen2.5-7b",
    ...     priority="NORMAL",
    ... )
    >>> response = await proxy.chat(request)
    >>> print(f"路由到: {response.routed_to}")
    >>> print(f"内容: {response.content}")
    """

    instance_pool: LLMInstancePool
    load_balancer: PrefixAwareLoadBalancer | None = None
    watermark_breaker: VRAMWatermarkBreaker | None = None

    # HTTP client
    _http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    # SGLang/vLLM API 路径
    _chat_api_path: str = "/v1/chat/completions"
    _health_api_path: str = "/health"
    _running: bool = field(default=False, init=False)
    # 超时配置
    _request_timeout: float = 120.0
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
        # 初始化子组件
        if self.load_balancer is None:
            self.load_balancer = PrefixAwareLoadBalancer(
                instance_pool=self.instance_pool
            )
        if self.watermark_breaker is None:
            self.watermark_breaker = VRAMWatermarkBreaker(
                instance_pool=self.instance_pool
            )

    async def start(self) -> None:
        """启动网关。"""
        if self._running:
            return

        self._running = True

        # 创建 HTTP 客户端
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._request_timeout),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

        # 启动子组件
        await self.instance_pool.start()
        await self.load_balancer.start()

        logger.info("[GatewayProxy] 分布式 LLM 网关已启动")

    async def stop(self) -> None:
        """停止网关。"""
        self._running = False

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        await self.load_balancer.stop()
        await self.instance_pool.stop()

        logger.info("[GatewayProxy] 分布式 LLM 网关已停止")

    # ------------------------------------------------------------------------ #
    #  Core Chat Interface                                                      #
    # ------------------------------------------------------------------------ #

    async def chat(self, request: GatewayRequest) -> GatewayResponse | GatewayError:
        """处理 LLM Chat 请求。

        Parameters
        ----------
        request : GatewayRequest
            网关请求

        Returns
        -------
        GatewayResponse | GatewayError
            成功返回响应，失败返回错误
        """
        self._stats["total_requests"] += 1
        request_id = f"req-{int(time.time() * 1000)}"
        request.request_id = request_id

        start_time = time.monotonic()

        try:
            # Step 1: 路由决策
            routing_decision = await self._route(request)
            if routing_decision is None:
                return GatewayError(
                    error="无可用 LLM 实例",
                    error_code="NO_INSTANCES",
                    retry_after=10.0,
                )

            url = routing_decision.url
            prefix_hash = routing_decision.prefix_hash
            strategy = routing_decision.strategy

            if routing_decision.is_cached:
                self._stats["prefix_cached_requests"] += 1

            # Step 2: 准入检查（跳过 CRITICAL 任务的强制放行检查）
            if request.priority != "CRITICAL":
                admission = self.watermark_breaker.check_admission(
                    instance=self.instance_pool.get_instance(url),
                    priority=request.priority,
                )

                if admission.result == AdmissionResult.CIRCUIT_OPEN:
                    self._stats["circuit_broken_requests"] += 1
                    logger.warning(
                        "[GatewayProxy] 请求 %s 被熔断器拒绝: %s",
                        request_id, url
                    )
                    # 尝试重路由
                    fallback = await self._try_fallback_routing(request)
                    if fallback:
                        return fallback
                    return GatewayError(
                        error=f"实例 {url} 熔断器开启",
                        error_code="CIRCUIT_OPEN",
                        routed_to=url,
                        retry_after=30.0,
                    )

                if admission.result == AdmissionResult.REJECTED:
                    self._stats["vram_rejected_requests"] += 1
                    logger.warning(
                        "[GatewayProxy] 请求 %s 被显存限制拒绝: %s (%.1f%%)",
                        request_id, url, admission.vram_ratio * 100
                    )
                    # 尝试重路由
                    fallback = await self._try_fallback_routing(request)
                    if fallback:
                        return fallback
                    return GatewayError(
                        error=admission.reason,
                        error_code="VRAM_LIMIT",
                        routed_to=url,
                        retry_after=5.0,
                    )

            # Step 3: 记录连接
            await self.instance_pool.acquire_connection(url)

            # Step 4: 发送请求
            try:
                if request.stream:
                    # 流式响应
                    content, usage, finish_reason = await self._do_streaming_request(
                        url=url,
                        request=request,
                    )
                else:
                    # 非流式响应
                    content, usage, finish_reason = await self._do_request(
                        url=url,
                        request=request,
                    )

                latency_ms = (time.monotonic() - start_time) * 1000

                # Step 5: 记录成功
                await self.instance_pool.release_connection(url, success=True, latency_ms=latency_ms)
                await self.load_balancer.record_route_success(routing_decision)
                self._stats["successful_requests"] += 1

                return GatewayResponse(
                    content=content,
                    model=request.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    request_id=request_id,
                    routed_to=url,
                    routing_strategy=strategy.value,
                    prefix_hash=prefix_hash,
                    latency_ms=latency_ms,
                    streamed=request.stream,
                )

            except Exception as e:
                # Step 5: 记录失败
                await self.instance_pool.release_connection(url, success=False)
                await self.load_balancer.record_route_failure(routing_decision)
                self._stats["failed_requests"] += 1

                # 尝试重路由
                fallback = await self._try_fallback_routing(request)
                if fallback and not isinstance(fallback, GatewayError):
                    # 重路由成功（返回了响应），但仍然记录原始失败
                    return fallback

                logger.exception("[GatewayProxy] 请求 %s 失败: %s", request_id, e)
                return GatewayError(
                    error=f"LLM 请求失败: {str(e)}",
                    error_code="LLM_ERROR",
                    routed_to=url,
                )

        except Exception as e:
            logger.exception("[GatewayProxy] 网关异常: %s", e)
            self._stats["failed_requests"] += 1
            return GatewayError(
                error=f"网关异常: {str(e)}",
                error_code="GATEWAY_ERROR",
            )

    # ------------------------------------------------------------------------ #
    #  Routing                                                                  #
    # ------------------------------------------------------------------------ #

    async def _route(self, request: GatewayRequest) -> RoutingDecision | None:
        """执行路由决策。"""
        try:
            return await self.load_balancer.route(
                prompt=self._extract_prompt(request),
                system_prompt=self._extract_system_prompt(request),
                priority=request.priority,
                tags=request.tags,
                prefer_cached=True,
            )
        except RuntimeError:
            return None

    async def _try_fallback_routing(self, request: GatewayRequest) -> GatewayResponse | GatewayError | None:
        """尝试备用路由（Least-Connections 降级）。"""
        try:
            # 强制使用 Least-Connections
            decision = await self.load_balancer.route(
                prompt=self._extract_prompt(request),
                system_prompt=self._extract_system_prompt(request),
                priority=request.priority,
                tags=request.tags,
                prefer_cached=False,  # 强制降级
            )

            url = decision.url

            # 跳过准入检查（紧急情况下放行）
            await self.instance_pool.acquire_connection(url)

            try:
                if request.stream:
                    content, usage, finish_reason = await self._do_streaming_request(
                        url=url, request=request,
                    )
                else:
                    content, usage, finish_reason = await self._do_request(
                        url=url, request=request,
                    )

                await self.instance_pool.release_connection(url, success=True)
                logger.info("[GatewayProxy] 备用路由成功: %s", url)

                return GatewayResponse(
                    content=content,
                    model=request.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    request_id=request.request_id,
                    routed_to=url,
                    routing_strategy=RoutingStrategy.LEAST_CONNECTIONS.value,
                    prefix_hash=decision.prefix_hash,
                    latency_ms=0.0,
                    streamed=request.stream,
                )

            except Exception:
                await self.instance_pool.release_connection(url, success=False)
                return None

        except RuntimeError:
            return None

    # ------------------------------------------------------------------------ #
    #  HTTP Request Helpers                                                     #
    # ------------------------------------------------------------------------ #

    async def _do_request(
        self,
        url: str,
        request: GatewayRequest,
    ) -> tuple[str, dict[str, int] | None, str | None]:
        """发送非流式请求到 LLM 实例。

        Returns
        -------
        tuple[content, usage, finish_reason]
        """
        if not self._http_client:
            raise RuntimeError("HTTP client 未初始化")

        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        api_url = f"{url}{self._chat_api_path}"
        response = await self._http_client.post(api_url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"LLM API 返回错误: {response.status_code} - {response.text}")

        data = response.json()

        # 解析响应
        content = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice:
                content = choice["message"].get("content", "")
                finish_reason = choice.get("finish_reason")
            else:
                content = choice.get("text", "")
                finish_reason = choice.get("finish_reason")
        else:
            finish_reason = None

        usage = data.get("usage")

        return content, usage, finish_reason

    async def _do_streaming_request(
        self,
        url: str,
        request: GatewayRequest,
    ) -> tuple[str, dict[str, int] | None, str | None]:
        """发送流式请求到 LLM 实例，并聚合响应。

        对于流式请求，我们聚合所有 chunk 以提供与非流式相同的接口。

        Returns
        -------
        tuple[content, usage, finish_reason]
        """
        if not self._http_client:
            raise RuntimeError("HTTP client 未初始化")

        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        api_url = f"{url}{self._chat_api_path}"

        # 异步上下文管理器确保连接释放
        async with self._http_client.stream("POST", api_url, json=payload) as response:
            if response.status_code != 200:
                raise RuntimeError(f"LLM API 返回错误: {response.status_code}")

            full_content = ""
            finish_reason = None
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # 解析 SGLang/vLLM 的流式格式
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
                    finish_reason = chunk["choices"][0].get("finish_reason")

                # 累加 usage
                if "usage" in chunk:
                    for key in total_usage:
                        total_usage[key] += chunk["usage"].get(key, 0)

            return full_content, total_usage, finish_reason

    async def stream_chat(
        self,
        request: GatewayRequest,
    ) -> AsyncGenerator[str, None]:
        """流式 Chat 响应生成器。

        用于 FastAPI 的 StreamingResponse。

        Yields
        ------
        str
            SSE 格式的响应 chunk
        """
        request.stream = True

        # 获取路由
        routing_decision = await self._route(request)
        if routing_decision is None:
            yield 'data: {"error": "无可用 LLM 实例"}\n\n'
            return

        url = routing_decision.url

        # 准入检查
        if request.priority != "CRITICAL":
            admission = self.watermark_breaker.check_admission(
                instance=self.instance_pool.get_instance(url),
                priority=request.priority,
            )
            if admission.result != AdmissionResult.ADMITTED:
                yield f'data: {{"error": "{admission.reason}"}}\n\n'
                return

        await self.instance_pool.acquire_connection(url)

        try:
            if not self._http_client:
                raise RuntimeError("HTTP client 未初始化")

            payload = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }

            api_url = f"{url}{self._chat_api_path}"

            async with self._http_client.stream("POST", api_url, json=payload) as response:
                if response.status_code != 200:
                    yield f'data: {{"error": "LLM API 返回 {response.status_code}"}}\n\n'
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # 直接透传 SSE 行
                    if line.startswith("data:"):
                        yield line + "\n\n"

                # 发送完成信号
                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("[GatewayProxy] 流式请求异常: %s", e)
            yield f'data: {{"error": "{str(e)}"}}\n\n'
        finally:
            await self.instance_pool.release_connection(url, success=True)

    # ------------------------------------------------------------------------ #
    #  Request Parsing Helpers                                                  #
    # ------------------------------------------------------------------------ #

    def _extract_prompt(self, request: GatewayRequest) -> str:
        """从请求中提取用户 prompt。"""
        for msg in request.messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _extract_system_prompt(self, request: GatewayRequest) -> str | None:
        """从请求中提取系统 prompt。"""
        for msg in request.messages:
            if msg.get("role") == "system":
                return msg.get("content")
        return None

    # ------------------------------------------------------------------------ #
    #  Admin & Stats                                                            #
    # ------------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """获取网关统计信息。"""
        total = self._stats["total_requests"]
        success_rate = (
            self._stats["successful_requests"] / max(1, total) * 100
        )
        return {
            "gateway": {
                "total_requests": total,
                "successful_requests": self._stats["successful_requests"],
                "failed_requests": self._stats["failed_requests"],
                "success_rate": f"{success_rate:.1f}%",
                "prefix_cached_rate": f"{self._stats['prefix_cached_requests'] / max(1, total) * 100:.1f}%",
            },
            "instance_pool": self.instance_pool.get_stats(),
            "load_balancer": self.load_balancer.get_stats(),
            "watermark_breaker": self.watermark_breaker.get_stats(),
        }

    def register_instance(self, config: LLMInstanceConfig) -> None:
        """注册 LLM 实例。"""
        self.instance_pool.register_instance(config)
