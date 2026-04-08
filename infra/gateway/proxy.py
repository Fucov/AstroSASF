"""
AstroSASF · Infra · Gateway Proxy (Kernel)
==========================================
分布式 LLM 网关反向代理。

V7.5 核心新增：
- 意图感知异构模型路由（Planner → heavy / Executor → light）
- 算力超载动态降级：VRAMWatermarkBreaker 高负载时透明降级到 1.5B 模型
- compute_downgrade_count / light_model_routed_count 异构指标埋点
- GatewayResponse.downgraded / GatewayResponse.compute_class 透明降级标记

Author: AstroSASF Team
Version: 7.5
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

from infra.llm.config_loader import (
    GatewayConfig,
    GatewayBackendConfig,
    IntentRoutingConfig,
    SASFConfig,
    load_config,
)
from infra.llm.instance_pool import LLMInstancePool, LLMInstanceConfig
from infra.routing import (
    PrefixAwareLoadBalancer,
    RoutingStrategy,
    RoutingDecision,
    VRAMWatermarkBreaker,
    AdmissionResult,
)
from infra.gateway.transform_pipeline import (
    PromptTransformationPipeline,
    RAGReorderMiddleware,
    StaticMCPMiddleware,
    PrefixBuilder,
    SpeculativeWarmer,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Gateway Data Models                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class GatewayRequest:
    """LLM 网关请求（V7.5 增强：agent_id 字段用于意图检测）。"""
    model: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048
    request_id: str | None = None
    priority: str = "NORMAL"
    tags: list[str] = field(default_factory=list)
    # V7.5 新增：Agent 标识符（用于意图感知路由）
    agent_id: str | None = None


@dataclass
class GatewayResponse:
    """LLM 网关响应（V7.5 增强：异构降级标记）。"""
    content: str
    model: str
    request_id: str
    routed_to: str
    routing_strategy: str
    prefix_hash: str
    latency_ms: float
    usage: dict | None = None
    finish_reason: str | None = None
    streamed: bool = False
    # V7.5 新增：异构算力降级标记
    downgraded: bool = False                       # 是否经历透明算力降级
    compute_class: str = "heavy"                  # 实际分发的算力级别
    downgrade_reason: str | None = None            # 降级原因描述


@dataclass
class GatewayError:
    error: str
    error_code: str = "UNKNOWN"
    retry_after: float | None = None


# --------------------------------------------------------------------------- #
#  Response Sanitizer                                                          #
# --------------------------------------------------------------------------- #

class ResponseSanitizer:
    """LLM 响应净化器。

    剥离 DeepSeek-R1 等蒸馏模型的 <think>...</think> 推理噪声，
    防止上层 Planner/Executor Agent 的 JSON 解析器崩溃。
    """

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def sanitize(self, content: str) -> str:
        """一次性净化响应内容。"""
        result = content
        result = result.replace(self.THINK_OPEN, "")
        result = result.replace(self.THINK_CLOSE, "")
        result = result.strip()
        return result

    def stream_sanitize_chunk(self, chunk_json: str) -> str | None:
        """流式净化单个 SSE chunk。"""
        try:
            data = json.loads(chunk_json)
        except json.JSONDecodeError:
            return chunk_json

        content = (
            data.get("choices", [{}])[0]
            .get("delta", {})
            .get("content", "")
        )

        has_open = self.THINK_OPEN in content
        has_close = self.THINK_CLOSE in content

        if has_open and not has_close:
            return None  # 不完整的开标签 → 丢弃

        if has_close and not content.startswith(self.THINK_OPEN):
            idx = content.find(self.THINK_CLOSE)
            return json.dumps({
                "choices": [{
                    "delta": {"content": content[idx + len(self.THINK_CLOSE):]},
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                }]
            })

        cleaned = content.replace(self.THINK_OPEN, "").replace(self.THINK_CLOSE, "")
        if cleaned != content:
            return json.dumps({
                "choices": [{
                    "delta": {"content": cleaned},
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                }]
            })

        return chunk_json


# --------------------------------------------------------------------------- #
#  GatewayProxy                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class GatewayProxy:
    """分布式 LLM 网关代理（V7.5 异构计算感知版）。

    V7.5 异构计算增强链路：
    1. **意图检测**：从 request.agent_id / tags 识别 Agent 类型
    2. **异构路由**：Intent → compute_class（Planner → heavy / Executor → light）
    3. **Prefix Hash 亲和**：KV-Cache 复用（相同前缀 → 历史实例）
    4. **动态降级**：heavy 实例 VRAM/连接数超限 → 自动降级到 light 实例
       - 透明重写 model_name（降级后使用轻量模型）
       - 响应 Header/Meta 标记 `downgraded: true`
    5. **透明输出**：GatewayResponse 返回降级元信息，上层 Agent 无需感知
    """

    instance_pool: LLMInstancePool = field(default=None)
    load_balancer: PrefixAwareLoadBalancer = field(default=None)
    watermark_breaker: VRAMWatermarkBreaker = field(default=None)
    config: GatewayConfig | None = None

    _http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _pipeline: PromptTransformationPipeline | None = field(default=None, init=False)
    # V7.5 新增：异构算力指标
    _stats: dict[str, int] = field(default_factory=lambda: {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "circuit_broken_requests": 0,
        "vram_rejected_requests": 0,
        "cross_agent_cache_hits": 0,
        "tool_schema_saved_tokens": 0,
        "prefix_cached_requests": 0,
        # V7.5 异构计算指标
        "compute_downgrade_count": 0,    # 触发算力降级的总次数
        "light_model_routed_count": 0,   # 成功分发给低算力小模型的极速请求次数
        "heavy_model_routed_count": 0,   # 分发给高算力大模型的请求次数
    })
    _sanitizer: ResponseSanitizer = field(
        default_factory=ResponseSanitizer, init=False, repr=False,
    )
    # V7.5 新增：降级模型映射表（config.yaml intent_routing.downgrade_model_map）
    _downgrade_model_map: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.load_balancer is None:
            self.load_balancer = PrefixAwareLoadBalancer(instance_pool=self.instance_pool)
        if self.watermark_breaker is None:
            self.watermark_breaker = VRAMWatermarkBreaker(instance_pool=self.instance_pool)

        # 构建 Prompt 转换流水线
        self.pipeline = PromptTransformationPipeline(
            rag_middleware=RAGReorderMiddleware(),
            mcp_middleware=StaticMCPMiddleware(stats_ref=self._stats),
            prefix_builder=PrefixBuilder(),
            warmer=None,  # 启动时注入，见 start()
        )

        # 注入意图路由配置到 LoadBalancer
        if self.config and self.config.intent_routing:
            self.load_balancer.set_intent_config(
                default_compute_class=self.config.intent_routing.default_compute_class,
                allow_downgrade=self.config.intent_routing.allow_downgrade,
            )
            self._downgrade_model_map = dict(self.config.intent_routing.downgrade_model_map)

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

        # 注入 URL → model_name 映射
        url_to_model: dict[str, str] = {
            url: cfg.model_name
            for url, cfg in self.instance_pool.configs.items()
        }

        # 启动推测性预热后台协程
        if self.pipeline is not None:
            self.pipeline.warmer = SpeculativeWarmer(
                http_client=self._http_client,
                prefix_builder=self.pipeline.prefix_builder,
                instance_pool=self.instance_pool,
                stats_ref=self._stats,
            )
            if self.pipeline.warmer is not None:
                await self.pipeline.warmer.start()
                self.pipeline.warmer.set_url_model_map(url_to_model)

        logger.info(
            "[GatewayProxy] 网关代理已启动（V7.5 异构算力感知版）"
            " | downgrade_model_map=%s",
            self._downgrade_model_map,
        )

    async def stop(self) -> None:
        """停止网关代理。"""
        self._running = False
        if self.pipeline and self.pipeline.warmer:
            await self.pipeline.warmer.stop()
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
        """执行路由决策和准入检查（V7.5 增强：agent_id 传入意图路由）。

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

        # 1. 路由决策（V7.5：传入 agent_id 用于意图检测）
        decision = await self.load_balancer.route(
            prompt=user_prompt,
            system_prompt=system_prompt,
            priority=request.priority,
            tags=request.tags,
            prefer_cached=True,
            agent_id=request.agent_id,    # V7.5 新增
        )

        # 2. 准入检查（VRAMWatermarkBreaker）
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

        cfg = self.instance_pool.configs.get(instance.url)
        cc = cfg.compute_class if cfg else "heavy"

        decision = RoutingDecision(
            url=instance.url,
            strategy=RoutingStrategy.LEAST_CONNECTIONS,
            reason=f"降级重路由: {reason}",
            prefix_hash="",
            vram_ratio=instance.vram_ratio,
            active_connections=instance.active_connections,
            is_cached=False,
            compute_class=cc,
        )

        return decision, instance.url

    def _resolve_model_for_decision(
        self,
        request: GatewayRequest,
        decision: RoutingDecision,
    ) -> tuple[str, str | None]:
        """V7.5 新增：解析请求应使用的真实模型名称。

        若路由决策发生了算力降级（heavy → light），则通过降级模型映射表
        将原模型名替换为轻量替代模型名。

        Returns
        -------
        tuple[str, str | None]
            (实际使用的 model_name, 降级原因描述)
            降级原因仅在发生降级时非 None
        """
        instance_config = self.instance_pool.configs.get(decision.url)
        if instance_config is None:
            return request.model or "qwen2.5:7b", None

        base_model = request.model or instance_config.model_name
        downgrade_reason: str | None = None

        # Case 1: 路由决策本身已触发降级（负载均衡器直接路由到 light 实例）
        if decision.is_downgraded and decision.required_compute_class == "heavy":
            # 查降级映射表，尝试将原始模型替换为轻量替代模型
            if base_model in self._downgrade_model_map:
                fallback_model = self._downgrade_model_map[base_model]
                downgrade_reason = (
                    f"算力降级: {base_model} → {fallback_model} "
                    f"(高算力实例满载，透明路由至 {decision.url})"
                )
                return fallback_model, downgrade_reason
            else:
                # 无映射表，直接使用 light 实例的默认模型
                downgrade_reason = (
                    f"算力降级: 高算力实例满载，透明路由至 light 实例 {decision.url}"
                    f"（使用实例默认模型: {instance_config.model_name}）"
                )
                return instance_config.model_name, downgrade_reason

        # Case 2: 正常路由，使用实例默认模型
        return instance_config.model_name, None

    # ------------------------------------------------------------------------ #
    #  HTTP Request Handling                                                     #
    # ------------------------------------------------------------------------ #

    async def chat(self, request: GatewayRequest) -> GatewayResponse | GatewayError:
        """同步 Chat 接口（V7.5 异构计算增强）。

        完整链路：
        Transform → 意图检测 → 异构路由 → Prefix Hash 亲和 →
        VRAM 准入 → 连接池 → HTTP 请求 → 响应净化 → 预热触发

        V7.5 异构增强：
        - agent_id 传入 LoadBalancer 进行意图检测
        - 算力降级时透明替换 model_name
        - GatewayResponse.downgraded 标记降级状态
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        start_time = time.monotonic()
        self._stats["total_requests"] += 1

        try:
            # 0. Prompt 转换（流量拦截与重构）
            agent_type = "experiment"
            if self.pipeline:
                agent_type = self.pipeline.infer_agent_type(
                    request.messages, request.tags,
                )
                transformed_messages, transform_ctx = await self.pipeline.transform(
                    request, agent_type=agent_type,
                )
                request.messages = transformed_messages

            # 1. 路由 + 准入（V7.5：含意图检测）
            decision, url = await self._route(request)

            # 2. 解析实际使用的模型（V7.5：降级时透明替换 model_name）
            real_model, downgrade_reason = self._resolve_model_for_decision(request, decision)

            # 3. 记录异构指标（V7.5 新增）
            self._record_compute_metrics(decision, downgrade_reason)

            # 4. 获取连接
            await self.instance_pool.acquire_connection(url)

            try:
                # 5. 发送请求
                result = await self._do_request(url, real_model, request)
                elapsed_ms = (time.monotonic() - start_time) * 1000

                # 5a. 响应净化：剥离 <think>/</think> 推理噪声
                raw_content = result.get("content", "")
                clean_content = self._sanitizer.sanitize(raw_content)
                if clean_content != raw_content:
                    logger.debug(
                        "[GatewayProxy] 剥离思考标签: request_id=%s, 原始长度=%d → 清理后=%d",
                        request.request_id[:8], len(raw_content), len(clean_content),
                    )
                    result["content"] = clean_content

                # 6. 记录成功
                await self.instance_pool.release_connection(url, success=True, latency_ms=elapsed_ms)
                await self.watermark_breaker.on_request_success(url)
                await self.load_balancer.record_route_success(decision)

                self._stats["successful_requests"] += 1
                if decision.is_cached:
                    self._stats["prefix_cached_requests"] += 1

                # 7. 触发推测性预热（后台，fire-and-forget）
                if self.pipeline:
                    self.pipeline.trigger_warmup(request, agent_type, url)

                logger.info(
                    "[GatewayProxy] 请求完成: %s → %s [%s] model=%s compute_class=%s "
                    "downgraded=%s (%.1fms)",
                    request.request_id[:8], url,
                    decision.strategy.value, real_model,
                    decision.compute_class, decision.is_downgraded, elapsed_ms,
                )

                return GatewayResponse(
                    content=result["content"],
                    model=real_model,
                    usage=result.get("usage"),
                    finish_reason=result.get("finish_reason"),
                    request_id=request.request_id,
                    routed_to=url,
                    routing_strategy=decision.strategy.value,
                    prefix_hash=decision.prefix_hash,
                    latency_ms=elapsed_ms,
                    streamed=False,
                    # V7.5 新增：异构降级标记
                    downgraded=decision.is_downgraded,
                    compute_class=decision.compute_class,
                    downgrade_reason=downgrade_reason,
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
        """流式 Chat 接口（V7.5 异构计算增强）。

        完整链路：Transform → 意图检测 → 异构路由 → 流式请求 → SSE 净化
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        self._stats["total_requests"] += 1

        try:
            # 0. Prompt 转换
            agent_type = "experiment"
            if self.pipeline:
                agent_type = self.pipeline.infer_agent_type(
                    request.messages, request.tags,
                )
                transformed_messages, _ = await self.pipeline.transform(
                    request, agent_type=agent_type,
                )
                request.messages = transformed_messages

            # 1. 路由 + 准入
            decision, url = await self._route(request)

            # 2. 解析实际模型（V7.5：降级时透明替换）
            real_model, downgrade_reason = self._resolve_model_for_decision(request, decision)

            # 3. 记录异构指标
            self._record_compute_metrics(decision, downgrade_reason)

            await self.instance_pool.acquire_connection(url)

            try:
                start_time = time.monotonic()
                # 流式净化：逐 chunk 过滤 <think>/</think> 思考标签
                async for chunk in self._do_streaming_request(url, real_model, request, sanitizer=self._sanitizer):
                    yield chunk

                elapsed_ms = (time.monotonic() - start_time) * 1000
                await self.instance_pool.release_connection(url, success=True, latency_ms=elapsed_ms)
                await self.watermark_breaker.on_request_success(url)
                await self.load_balancer.record_route_success(decision)

                self._stats["successful_requests"] += 1
                if decision.is_cached:
                    self._stats["prefix_cached_requests"] += 1

                # 触发推测性预热
                if self.pipeline:
                    self.pipeline.trigger_warmup(request, agent_type, url)

                logger.info(
                    "[GatewayProxy] 流式请求完成: %s → %s [%s] model=%s compute_class=%s "
                    "downgraded=%s (%.1fms)",
                    request.request_id[:8], url,
                    decision.strategy.value, real_model,
                    decision.compute_class, decision.is_downgraded, elapsed_ms,
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

    def _record_compute_metrics(
        self,
        decision: RoutingDecision,
        downgrade_reason: str | None,
    ) -> None:
        """V7.5 新增：记录异构算力路由指标。"""
        if decision.is_downgraded:
            self._stats["compute_downgrade_count"] += 1
            logger.info(
                "[GatewayProxy] 算力降级触发: url=%s reason=%s",
                decision.url, downgrade_reason,
            )

        if decision.compute_class == "light":
            self._stats["light_model_routed_count"] += 1
        else:
            self._stats["heavy_model_routed_count"] += 1

    # ------------------------------------------------------------------------ #
    #  Internal HTTP Methods                                                     #
    # ------------------------------------------------------------------------ #

    async def _do_request(self, url: str, real_model: str, request: GatewayRequest) -> dict[str, Any]:
        """发送非流式请求到 LLM 实例。"""
        if not self._http_client:
            raise RuntimeError("HTTP 客户端未初始化")

        payload = {
            "model": real_model,
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
            "model": real_model,
            "usage": data.get("usage"),
            "finish_reason": data["choices"][0].get("finish_reason") if "choices" in data else None,
        }

    async def _do_streaming_request(
        self,
        url: str,
        real_model: str,
        request: GatewayRequest,
        sanitizer: ResponseSanitizer | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """发送流式请求到 LLM 实例（透传 SSE）。"""
        if not self._http_client:
            raise RuntimeError("HTTP 客户端未初始化")

        payload = {
            "model": real_model,
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
                    raw_chunk = line[6:].strip()
                    if sanitizer is not None:
                        cleaned = sanitizer.stream_sanitize_chunk(raw_chunk)
                        if cleaned is None:
                            continue
                        yield f"data: {cleaned}\n".encode("utf-8")
                    else:
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
        """获取聚合统计信息（含 V7.5 异构计算指标）。"""
        total = self._stats["total_requests"]
        successful = self._stats["successful_requests"]
        light_count = self._stats["light_model_routed_count"]
        heavy_count = self._stats["heavy_model_routed_count"]
        downgrade_count = self._stats["compute_downgrade_count"]

        return {
            "gateway": {
                "total_requests": total,
                "successful_requests": successful,
                "failed_requests": self._stats["failed_requests"],
                "success_rate": f"{successful / max(1, total) * 100:.1f}%",
                "prefix_cached_rate": f"{self._stats['prefix_cached_requests'] / max(1, total) * 100:.1f}%",
                "circuit_broken_requests": self._stats["circuit_broken_requests"],
                "vram_rejected_requests": self._stats["vram_rejected_requests"],
                "cross_agent_cache_hits": self._stats["cross_agent_cache_hits"],
                "tool_schema_saved_tokens": self._stats["tool_schema_saved_tokens"],
                # V7.5 异构算力指标
                "compute_downgrade_count": downgrade_count,     # 算力降级触发次数
                "light_model_routed_count": light_count,       # 低算力极速分发次数
                "heavy_model_routed_count": heavy_count,       # 高算力大模型分发次数
                "light_model_routed_rate": f"{light_count / max(1, heavy_count + light_count) * 100:.1f}%",
                "downgrade_rate": f"{downgrade_count / max(1, total) * 100:.1f}%",
            },
            "instance_pool": self.instance_pool.get_stats(),
            "load_balancer": self.load_balancer.get_stats(),
            "watermark_breaker": self.watermark_breaker.get_stats(),
        }
