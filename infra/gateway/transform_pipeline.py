"""
AstroSASF · Infra · Prompt Transformation Pipeline (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
流量拦截与 Prompt 重构流水线。

目标：最大化 vLLM / SGLang 的 Prefix Caching（RadixTree KV-Cache 复用）。

设计原则
─────────
- 所有 Middleware 无状态（可并发调用）
- 传入的 messages 列表就地重排（避免复制开销）
- SpeculativeWarmer 运行于独立后台协程，与请求处理完全解耦
- 三类 Agent 的 Prompt 模板被静态化，即使运行时内容变化，也会被拦截到末尾
- ResponseSanitizer 运行于 GatewayProxy 层，剥离 LLM 返回的 <think>/</think> 推理噪声

Author: AstroSASF Team
Version: 7.3
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infra.gateway.proxy import GatewayRequest
    from infra.llm.instance_pool import LLMInstanceConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Constants & Patterns                                                        #
# --------------------------------------------------------------------------- #

# RAG Context 边界标签
RAG_OPEN_TAG = "<rag_context>"
RAG_CLOSE_TAG = "</rag_context>"

# Agent 类型识别（从 messages 或请求 tags 推断）
AGENT_TYPE_QA = "qa"
AGENT_TYPE_PLANNER = "planner"
AGENT_TYPE_EXPERIMENT = "experiment"

# SOP ID 正则（用于 Speculative Warming 触发检测）
SOP_ID_PATTERN = re.compile(r"SOP[s]?:?\s*([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)

# DeepSeek / 蒸馏模型强制输出指令（追加到 User Message 末尾）
FORCED_JSON_OUTPUT_REMINDER = (
    "\n\n请直接输出执行计划的 JSON 数组，不要包含任何额外的 markdown 标记或解释。"
)

# 思考标签（用于 ResponseSanitizer — 彻底剔除 LLM 推理噪声）
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
_THINK_PATTERN = re.compile(
    re.escape(THINK_OPEN_TAG) + r".*?" + re.escape(THINK_CLOSE_TAG),
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
#  ResponseSanitizer — 剥离 LLM 推理噪声                                        #
# --------------------------------------------------------------------------- #

class ResponseSanitizer:
    """LLM 响应净化器。

    背景：
    DeepSeek-R1 等蒸馏模型在输出中会携带大量 <think>...</think> 推理过程。
    上层 Planner Agent 和 Executor Agent 的 JSON/DAG 解析器收到这类内容会崩溃
    （生成类似 FSM({}) 的垃圾数据）。

    设计原则：
    - 网关透明：上层 Agent 无需修改任何解析逻辑
    - 非流式（chat）：在 GatewayProxy.chat() 返回前完整替换内容
    - 流式（stream_chat）：在 SSE chunk 层面过滤（见 stream_sanitize_chunk）

    性能备注：
    - 非流式：使用 re.DOTALL 模式的 DOTALL 模式贪婪匹配，高效一次性替换
    - 流式：逐 chunk 检查，包含标签的 chunk 会被整块丢弃

    Example
    -------
    >>> sanitizer = ResponseSanitizer()
    >>> raw = "让我想想。<think>用户要设置温度</think>[{\"id\": \"A\", \"skill\": \"set_temp\"}]"
    >>> clean = sanitizer.sanitize(raw)
    >>> print(clean)
    让我想想。[{"id": "A", "skill": "set_temp"}]
    """

    __slots__ = ("_compiled",)

    def __init__(self) -> None:
        # 编译一次，复用（_THINK_PATTERN 在模块级别已编译）
        self._compiled: re.Pattern = _THINK_PATTERN

    def sanitize(self, content: str) -> str:
        """非流式场景：一次性剔除所有 <think>...</think> 标签及内容。

        Parameters
        ----------
        content : str
            原始 LLM 返回内容

        Returns
        -------
        str
            剔除思考标签后的干净内容
        """
        if THINK_OPEN_TAG not in content:
            return content
        return self._compiled.sub("", content)

    @staticmethod
    def stream_sanitize_chunk(chunk: str) -> str | None:
        """流式场景：检查单个 SSE chunk 是否需要过滤。

        流式处理的复杂性：
        - <think>...</think> 标签可能跨越多个 SSE chunk 边界
        - 一个 chunk 可能只包含 <think> 或 仅包含</think>
        - 同一轮对话中可能出现多次 <think>...</think>

        简化策略（状态机）：
        - 维护一个 "是否处于标签内" 的状态
        - 如果 chunk 完整包含标签，则删除标签内容
        - 如果 chunk 包含开标签且没有闭标签，丢弃从开标签到下一个换行前的内容
        - 如果 chunk 包含闭标签且没有开标签，丢弃该标签
        - 如果 chunk 跨越多个标签，仅保留最后一个闭标签之后的内容

        注意：由于流式 SSE 中 DeepSeek 的 <think> 内容通常单独作为
        某几个 chunk 存在，本函数主要处理单个 chunk 内包含完整标签的情况。
        对于跨 chunk 的情况，建议在流结束后进行一次 sanitize() 调用。

        Parameters
        ----------
        chunk : str
            单个 SSE data 行（不含 "data: " 前缀）

        Returns
        -------
        str | None
            清理后的 chunk 内容；如果该 chunk 应被完全丢弃则返回 None
        """
        # 快速路径：没有任何思考标签 → 直接透传
        if THINK_OPEN_TAG not in chunk and THINK_CLOSE_TAG not in chunk:
            return chunk

        # 如果 chunk 包含完整标签对：一次替换
        if THINK_OPEN_TAG in chunk and THINK_CLOSE_TAG in chunk:
            cleaned = _THINK_PATTERN.sub("", chunk)
            # 如果清理后为空（纯思考 chunk），丢弃
            return cleaned if cleaned.strip() else None

        # 跨越标签不完整：丢弃开标签到行尾，或行首到闭标签
        # 这种边界情况在流式 SSE 中相对少见，此处保守处理
        if THINK_OPEN_TAG in chunk:
            # 丢弃开标签及其后续内容
            return None
        if THINK_CLOSE_TAG in chunk:
            # 丢弃该标签
            return chunk.replace(THINK_CLOSE_TAG, "")

        return chunk


# --------------------------------------------------------------------------- #
#  Middleware Interface                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class TransformContext:
    """一次完整请求的 Transform 上下文（在整个流水线中传递）。"""
    # 原始请求引用
    request_id: str
    agent_type: str                   # "qa" | "planner" | "experiment"
    sop_id: str | None = None         # 提取到的 SOP ID（如有）
    prefix_built: str = ""            # PrefixBuilder 构建的固定前缀（用于预热）
    tool_schema_saved_tokens: int = 0  # 本次拦截 StaticMCP 节省的 Prefill Tokens
    is_dummy: bool = False            # 是否为 SpeculativeWarmer 发出的 Dummy 请求
    tags: list[str] = field(default_factory=list)


class PromptMiddleware(ABC):
    """Prompt 转换中间件基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """中间件名称（用于日志和调试）。"""
        ...

    @abstractmethod
    async def apply(
        self,
        messages: list[dict[str, str]],
        context: TransformContext,
        instance_config: LLMInstanceConfig | None,
    ) -> list[dict[str, str]]:
        """对 messages 进行转换，返回转换后的 messages 列表。"""
        ...


# --------------------------------------------------------------------------- #
#  Middleware 1: 确定性 RAG 上下文重排 (Deterministic RAG Reordering)          #
# --------------------------------------------------------------------------- #

class RAGReorderMiddleware(PromptMiddleware):
    """确定性 RAG 上下文重排中间件。

    工作原理：
    1. 拦截 QA 和 Planner 的请求
    2. 在 User Message 中寻找 <rag_context>...</rag_context> 包裹的检索片段
    3. 提取所有片段，按 SHA-256 哈希值升序排列，再塞回原位
    4. 保证相同检索词在不同 Agent 轮次中产生完全一致的 Prompt 前缀

    理论依据：即使两个 Agent 检索到的 Chunk 语义相似但顺序不同，
    重排后底层 SGLang RadixAttention 看到的字符串前缀完全一致，
    完美命中 RadixTree 分支，复用 KV-Cache。
    """

    def __init__(self) -> None:
        # 编译一次，复用
        self._rag_pattern = re.compile(
            re.escape(RAG_OPEN_TAG) + r"(.*?)" + re.escape(RAG_CLOSE_TAG),
            re.DOTALL,
        )

    @property
    def name(self) -> str:
        return "RAGReorder"

    async def apply(
        self,
        messages: list[dict[str, str]],
        context: TransformContext,
        instance_config: LLMInstanceConfig | None,
    ) -> list[dict[str, str]]:
        if context.agent_type not in (AGENT_TYPE_QA, AGENT_TYPE_PLANNER):
            return messages

        modified = False
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role != "user" or not content:
                continue

            # 检查是否有 RAG 上下文包裹
            if RAG_OPEN_TAG not in content or RAG_CLOSE_TAG not in content:
                continue

            new_content = self._reorder_rag_context(content)
            if new_content != content:
                msg["content"] = new_content
                modified = True

        if modified:
            logger.debug(
                "[%s] RAG 上下文重排完成: request_id=%s",
                self.name, context.request_id[:8],
            )

        return messages

    def _reorder_rag_context(self, content: str) -> str:
        """提取 RAG 片段，计算 SHA-256，按字母序重排。

        支持以下两种格式：
        1. 单 RAG 块（常见）：<rag_context>chunk1\\nchunk2\\nchunk3</rag_context>
        2. 多 RAG 块：<rag_context>...</rag_context> 前后可含其他文字

        重排后所有 chunks 按 SHA-256 升序合并为一个块。
        """
        chunks: list[tuple[str, str]] = []   # (sha256_hex, chunk_text)

        def replacer(match: re.Match) -> str:
            chunk_text = match.group(1).strip()
            if not chunk_text:
                return ""
            sha = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append((sha, chunk_text))
            # 使用唯一占位符（避免与内容冲突）
            return f"\x00RAG_CHUNK_{len(chunks) - 1}\x00"

        placeholder_content = self._rag_pattern.sub(replacer, content)

        if not chunks:
            return content

        # SHA-256 升序排列
        chunks.sort(key=lambda x: x[0])

        # 合并为重排后的 RAG 块
        reordered_block = (
            RAG_OPEN_TAG + "\n" +
            "\n".join(c[1] for c in chunks) + "\n" +
            RAG_CLOSE_TAG
        )

        # 填回占位符（全部替换为同一个重排后的块）
        for i in range(len(chunks)):
            placeholder_content = placeholder_content.replace(
                f"\x00RAG_CHUNK_{i}\x00",
                reordered_block,
            )

        return placeholder_content


# --------------------------------------------------------------------------- #
#  Middleware 2: 静态 MCP Schema 前缀锁 (Static MCP Prefix Lock)              #
# --------------------------------------------------------------------------- #

class StaticMCPMiddleware(PromptMiddleware):
    """静态化 MCP Schema 前缀锁中间件。

    针对 Experiment Agent 的高频 tool_calling 请求设计。

    设计目标：
    - 将 "工具定义" 和 "系统级 SOP 指令" 锁定为静态前缀
    - 将所有高频变化的动态遥测数据（温度、气压、实时状态）推到 messages 末尾
    - 底层 vLLM 通过 Jinja 渲染 tools，只要 tools 数组内容+顺序不变，
      渲染出的 String 前缀就绝对固定

    效果：
    - 即使温度从 25°C 变到 37°C，前缀不变 → 整树重新 Prefill 次数大幅降低
    - Schema 长度 * 拦截次数 = 累计节省的 Prefill Tokens
    """

    def __init__(self, stats_ref: dict[str, int]) -> None:
        """
        Parameters
        ----------
        stats_ref : dict[str, int]
            GatewayProxy._stats 的引用，用于累计 tool_schema_saved_tokens。
        """
        self._stats_ref = stats_ref

    @property
    def name(self) -> str:
        return "StaticMCP"

    async def apply(
        self,
        messages: list[dict[str, str]],
        context: TransformContext,
        instance_config: LLMInstanceConfig | None,
    ) -> list[dict[str, str]]:
        if context.agent_type != AGENT_TYPE_EXPERIMENT:
            return messages

        # 寻找含 tools 的 assistant message（通常是工具定义轮次）
        tool_def_msg_idx: int | None = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tools" in msg:
                tool_def_msg_idx = i
                break

        if tool_def_msg_idx is None:
            return messages

        tool_def_msg = messages[tool_def_msg_idx]
        raw_tools = tool_def_msg.get("tools", [])

        if not raw_tools:
            return messages

        # 计算 Schema 总 Token 数（按字符数粗估：中文 ~2 chars/token，英文 ~4 chars/token）
        schema_text = self._serialize_tools(raw_tools)
        estimated_tokens = self._estimate_tokens(schema_text)
        self._stats_ref["tool_schema_saved_tokens"] += estimated_tokens
        context.tool_schema_saved_tokens = estimated_tokens

        # 收集所有动态内容（遥测状态、具体数值）并移除
        static_messages: list[dict[str, str]] = []
        dynamic_tail: list[dict[str, str]] = []

        for i, msg in enumerate(messages):
            if i == tool_def_msg_idx:
                # 工具定义消息：只保留 tools 字段，去掉 content（如果有动态描述）
                static_messages.append({
                    "role": "assistant",
                    "tools": raw_tools,
                })
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            # 动态内容：数值、遥测、具体参数 → 推到末尾
            if role == "user" and self._contains_dynamic_content(content):
                dynamic_tail.append({"role": "user", "content": content})
            elif role == "system":
                # system prompt 本身静态
                static_messages.append(msg)
            elif role == "assistant":
                static_messages.append(msg)
            else:
                dynamic_tail.append(msg)

        result = static_messages + dynamic_tail

        logger.debug(
            "[%s] MCP Schema 静态化: request_id=%s, 节省 ~%d tokens",
            self.name, context.request_id[:8], estimated_tokens,
        )

        return result

    def _serialize_tools(self, tools: list[dict[str, Any]]) -> str:
        """将 tools 列表序列化为字符串（用于 token 估算）。"""
        import json
        return json.dumps(tools, ensure_ascii=False, sort_keys=True)

    def _estimate_tokens(self, text: str) -> int:
        """简单 Token 估算（中英混合文本）。"""
        chinese_chars = sum(1 for c in text if ord(c) > 127)
        ascii_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.5 + ascii_chars * 0.25)

    def _contains_dynamic_content(self, content: str) -> bool:
        """判断 content 是否包含动态遥测/数值内容。"""
        if not content:
            return False
        # 常见动态指标模式：温度、气压、数值范围、时间戳
        dynamic_patterns = [
            r"\d+\.\d+\s*°?[Cc]",    # 温度：37.5°C
            r"\d+\.\d+\s*kPa",       # 气压：101.3kPa
            r"\d+\s*rpm",             # 转速
            r"\d+\s*ml",              # 体积
            r"当前|实时|现在|此时",     # 实时状态
            r"温度[是为：:]\s*\d+",    # 温度设置为 37
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}",  # 时间戳
            r"UTC|GMT",
        ]
        return any(re.search(p, content) for p in dynamic_patterns)


# --------------------------------------------------------------------------- #
#  PrefixBuilder — 统一前缀构建器                                              #
# --------------------------------------------------------------------------- #

class PrefixBuilder:
    """统一前缀构建器。

    确保 Dummy 预热请求与真实请求的前缀绝对一致（包括空格和换行）。

    固定顺序：
    1. System Prompt（含 SOP 设定）
    2. SOP 文本内容
    3. Agent 特异性指令（System role 的 content 末尾追加）

    Example
    -------
    >>> pb = PrefixBuilder()
    >>> prefix = pb.build(
    ...     system_prompt="你是实验柜助手。",
    ...     sop_id="bio_culture",
    ...     sop_text="细胞培养 SOP：...",
    ...     agent_instruction="仅负责 QA，不执行操作。"
    ... )
    >>> print(prefix)  # 包含所有三层的固定字符串
    """

    def __init__(self) -> None:
        # SOP ID → SOP 文本内容的缓存（避免每次重读）
        self._sop_cache: dict[str, str] = {}
        # 已注册的 SOP 文本
        self._sop_registry: dict[str, str] = {}

    def register_sop(self, sop_id: str, sop_text: str) -> None:
        """注册 SOP 文本（通常在框架初始化时调用）。"""
        self._sop_registry[sop_id] = sop_text
        self._sop_cache[sop_id] = sop_text
        logger.debug("[PrefixBuilder] 注册 SOP: %s (len=%d)", sop_id, len(sop_text))

    def build(
        self,
        system_prompt: str | None,
        sop_id: str | None,
        agent_instruction: str | None = None,
        extra_static_fields: dict[str, str] | None = None,
    ) -> str:
        """构建固定前缀。

        Parameters
        ----------
        system_prompt : str | None
            系统级通用设定（通常来自 messages 的 system role）
        sop_id : str | None
            SOP 标识符（用于从缓存中获取 SOP 文本）
        agent_instruction : str | None
            Agent 特异性追加指令（追加到 system_prompt 末尾）
        extra_static_fields : dict[str, str] | None
            额外静态字段（以 key\\nvalue\\n 格式追加）

        Returns
        -------
        str
            拼接后的固定前缀字符串
        """
        parts: list[str] = []

        # Layer 1: System 通用设定
        if system_prompt:
            parts.append(system_prompt.rstrip())

        # Layer 2: SOP 文本（从注册表获取，确保 Dummy/真实请求完全一致）
        if sop_id and sop_id in self._sop_registry:
            sop_text = self._sop_registry[sop_id]
            parts.append(sop_text)
        elif sop_id:
            parts.append(f"[SOP:{sop_id}]")

        # Layer 3: Agent 特异性指令（强制在 system prompt 末尾追加）
        if agent_instruction:
            if parts:
                parts[-1] = parts[-1].rstrip() + "\n\n" + agent_instruction
            else:
                parts.append(agent_instruction)

        # Layer 4: 额外静态字段
        if extra_static_fields:
            for key, value in sorted(extra_static_fields.items()):
                parts.append(f"{key}\n{value}")

        return "\n\n".join(parts)

    def build_messages(
        self,
        system_prompt: str | None,
        sop_id: str | None,
        agent_instruction: str | None = None,
        extra_static_fields: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """构建完整的 messages 列表（前缀 + 单条 user 占位）。"""
        prefix = self.build(system_prompt, sop_id, agent_instruction, extra_static_fields)
        return [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "<DUMMY_TOKEN_PLACEHOLDER>"},
        ]

    def extract_sop_id(self, messages: list[dict[str, str]]) -> str | None:
        """从 messages 中提取 SOP ID（供 SpeculativeWarmer 使用）。"""
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue
            match = SOP_ID_PATTERN.search(content)
            if match:
                return match.group(1)
        return None


# --------------------------------------------------------------------------- #
#  SpeculativeWarmer — 跨智能体推测性预热                                        #
# --------------------------------------------------------------------------- #

class SpeculativeWarmer:
    """推测性预热后台协程。

    工作原理：
    1. 监听 QA 智能体的高频请求流
    2. 检测到 SOP ID 后，自动构造 Dummy Request 发往目标实例
    3. Dummy 请求的 messages 序列必须与后续 Planner 请求的绝对起始部分完全一致
       （通过 PrefixBuilder 保证）
    4. Dummy 请求设置 max_tokens=1，触发底层 vLLM 完成 Prefill 但几乎不生成 tokens

    与 PrefixBuilder 的配合：
    - PrefixBuilder.build_messages() 保证 Dummy 和真实请求的 [system + sop_text] 前缀完全相同
    - RadixTree 命中后，Planner 请求的 Prefill 阶段被完全省略 → TTFT 降低

    设计约束：
    - Dummy 请求不改变任何状态（只读，无副作用）
    - 并发预热任务有上限（max_concurrent_warmers=3）
    - 已预热的 SOP 有冷却时间（sop_warmup_cooldown=30s）
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        prefix_builder: PrefixBuilder,
        instance_pool: Any,          # LLMInstancePool
        stats_ref: dict[str, int],
        max_concurrent: int = 3,
        warmup_cooldown: float = 30.0,
    ) -> None:
        self._http_client = http_client
        self._prefix_builder = prefix_builder
        self._instance_pool = instance_pool
        self._stats_ref = stats_ref
        self._max_concurrent = max_concurrent
        self._warmup_cooldown = warmup_cooldown

        # SOP ID → 上次预热时间戳
        self._last_warmup: dict[str, float] = {}
        # 当前正在进行的预热任务
        self._active_tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

        self._running = False

    @property
    def name(self) -> str:
        return "SpeculativeWarmer"

    async def start(self) -> None:
        self._running = True
        logger.info("[%s] 推测性预热后台协程已启动 (max_concurrent=%d)", self.name, self._max_concurrent)

    async def stop(self) -> None:
        self._running = False
        async with self._lock:
            for task in list(self._active_tasks):
                task.cancel()
        logger.info("[%s] 推测性预热后台协程已停止", self.name)

    async def on_request(self, request: GatewayRequest, agent_type: str, url: str) -> None:
        """请求入口：检测是否需要触发预热。

        在 GatewayProxy.chat() 中每次请求后调用（后台 fire-and-forget）。
        """
        if not self._running:
            return

        if agent_type != AGENT_TYPE_QA:
            return

        # 提取 SOP ID
        sop_id = self._prefix_builder.extract_sop_id(request.messages)
        if not sop_id:
            return

        # 冷却期检查
        now = time.monotonic()
        last = self._last_warmup.get(sop_id, 0.0)
        if now - last < self._warmup_cooldown:
            return

        # 并发上限检查
        async with self._lock:
            if len(self._active_tasks) >= self._max_concurrent:
                return

        # 触发预热（fire-and-forget）
        task = asyncio.create_task(
            self._do_warmup(sop_id, request, url),
            name=f"warmup-{sop_id}",
        )
        async with self._lock:
            self._active_tasks.add(task)
        task.add_done_callback(
            lambda t: asyncio.create_task(self._cleanup_task(t))
        )

    async def _do_warmup(
        self,
        sop_id: str,
        request: GatewayRequest,
        url: str,
    ) -> None:
        """执行一次预热。"""
        # 构建与真实请求前缀完全一致的 Dummy messages
        system_prompt = self._extract_system_prompt(request.messages)
        dummy_messages = self._prefix_builder.build_messages(
            system_prompt=system_prompt,
            sop_id=sop_id,
            agent_instruction=None,   # Dummy 不含 agent 特异性指令
        )

        payload = {
            "model": self._get_model_for_url(url),
            "messages": dummy_messages,
            "temperature": 0.0,
            "max_tokens": 1,          # 最小生成，触发 Prefill 即可
        }

        try:
            response = await self._http_client.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(10.0),
            )

            if response.status_code == 200:
                self._last_warmup[sop_id] = time.monotonic()
                self._stats_ref["cross_agent_cache_hits"] += 1
                logger.info(
                    "[%s] 预热成功: SOP=%s → %s",
                    self.name, sop_id, url,
                )
            else:
                logger.warning(
                    "[%s] 预热失败: SOP=%s → %s HTTP %d",
                    self.name, sop_id, url, response.status_code,
                )

        except Exception as e:
            logger.debug("[%s] 预热异常: SOP=%s → %s: %s", self.name, sop_id, url, e)

    async def _cleanup_task(self, task: asyncio.Task) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            async with self._lock:
                self._active_tasks.discard(task)

    def _extract_system_prompt(self, messages: list[dict[str, str]]) -> str | None:
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content")
        return None

    def _get_model_for_url(self, url: str) -> str:
        """查表获取实例对应的真实模型名称（Task 0 能力）。"""
        # 在 SpeculativeWarmer 初始化时由外部注入
        return getattr(self, "_url_to_model", {}).get(url, "qwen2.5:7b")

    def set_url_model_map(self, mapping: dict[str, str]) -> None:
        """注入 URL → model_name 映射表（由 GatewayProxy 启动时调用）。"""
        self._url_to_model = mapping


# --------------------------------------------------------------------------- #
#  PromptTransformationPipeline — 流水线入口                                   #
# --------------------------------------------------------------------------- #

@dataclass
class PromptTransformationPipeline:
    """Prompt 转换流水线。

    将多个 Middleware 按顺序串联，最终返回转换后的 messages 和上下文。

    使用示例
    ---------
    >>> pipeline = PromptTransformationPipeline(
    ...     rag_middleware=RAGReorderMiddleware(),
    ...     mcp_middleware=StaticMCPMiddleware(stats_ref=proxy._stats),
    ...     prefix_builder=PrefixBuilder(),
    ...     warmer=SpeculativeWarmer(...),
    ... )
    >>> new_messages, ctx = await pipeline.transform(request, agent_type, instance_config)
    """

    rag_middleware: RAGReorderMiddleware = field(default_factory=RAGReorderMiddleware)
    mcp_middleware: StaticMCPMiddleware | None = None
    prefix_builder: PrefixBuilder = field(default_factory=PrefixBuilder)
    warmer: SpeculativeWarmer | None = None

    def infer_agent_type(self, messages: list[dict[str, str]], tags: list[str]) -> str:
        """从 messages 或 tags 推断 Agent 类型。"""
        # 优先从 tags 推断
        for tag in tags:
            tag_lower = tag.lower()
            if "qa" in tag_lower or "question" in tag_lower:
                return AGENT_TYPE_QA
            if "planner" in tag_lower or "plan" in tag_lower:
                return AGENT_TYPE_PLANNER
            if "experiment" in tag_lower or "executor" in tag_lower or "tool" in tag_lower:
                return AGENT_TYPE_EXPERIMENT

        # 从 messages 内容推断
        combined = " ".join(m.get("content", "") for m in messages)
        if "<rag_context>" in combined or "检索" in combined or "知识库" in combined:
            return AGENT_TYPE_QA
        if "SOP:" in combined or "计划" in combined or "执行步骤" in combined:
            return AGENT_TYPE_PLANNER
        if "tool" in combined or "工具" in combined or "调用" in combined:
            return AGENT_TYPE_EXPERIMENT

        return AGENT_TYPE_EXPERIMENT  # 默认为高频工具调用场景

    async def transform(
        self,
        request: GatewayRequest,
        agent_type: str | None = None,
        instance_config: LLMInstanceConfig | None = None,
    ) -> tuple[list[dict[str, str]], TransformContext]:
        """执行完整转换流水线。

        Parameters
        ----------
        request : GatewayRequest
            原始网关请求
        agent_type : str | None
            Agent 类型（None 时自动推断）
        instance_config : LLMInstanceConfig | None
            目标实例配置（用于 provider 适配）

        Returns
        -------
        tuple[list[dict[str, str]], TransformContext]
            (转换后的 messages, Transform 上下文)
        """
        if agent_type is None:
            agent_type = self.infer_agent_type(request.messages, request.tags)

        # 提取 SOP ID（供 PrefxBuilder 和 Warmer 使用）
        sop_id = self.prefix_builder.extract_sop_id(request.messages)

        ctx = TransformContext(
            request_id=request.request_id or "",
            agent_type=agent_type,
            sop_id=sop_id,
            tags=request.tags,
        )

        messages = request.messages  # 引用，不复制（Middleware 内部处理）

        # 3. 追加强制输出指令（保护 Prompt 中的 JSON 输出要求不被 Static MCP 前缀挤压）
        # 注意：这条指令必须在动态 Task 拼接之后、请求发出之前追加
        messages = self._append_forced_output_reminder(messages)

        # 4. RAG 重排（仅 QA / Planner）
        if self.rag_middleware:
            messages = await self.rag_middleware.apply(messages, ctx, instance_config)

        # 5. 静态 MCP 前缀锁（仅 Experiment）
        if self.mcp_middleware:
            messages = await self.mcp_middleware.apply(messages, ctx, instance_config)

        logger.debug(
            "[Pipeline] 转换完成: agent=%s, sop=%s, saved_tokens=%d",
            agent_type, sop_id, ctx.tool_schema_saved_tokens,
        )

        return messages, ctx

    # ------------------------------------------------------------------------ #
    #  Forced Output Reminder                                                    #
    # ------------------------------------------------------------------------ #

    @staticmethod
    def _append_forced_output_reminder(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """在 User Message 末尾追加强制 JSON 输出指令。

        背景：Static MCP Prefix Lock 会将动态遥测推到末尾，可能导致系统提示词中的
        "请务必输出 JSON 格式" 等关键指令被挤出到不重要的位置。
        本方法在最后一条 User Message 末尾追加一句不可绕过的强提醒，
        确保 DeepSeek 等蒸馏模型的输出严格符合 JSON 格式规范。

        注意：本方法仅追加文本，不修改 messages 的 role 结构。
        """
        if not messages:
            return messages

        # 找到最后一条 User Message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                original = messages[i].get("content", "")
                messages[i]["content"] = (
                    original.rstrip() + FORCED_JSON_OUTPUT_REMINDER
                )
                logger.debug(
                    "[Pipeline] 强制输出指令已追加到 user message（长度=%d → %d）",
                    len(original), len(messages[i]["content"]),
                )
                return messages

        # 没有 user message：追加一条新的
        messages.append({
            "role": "user",
            "content": FORCED_JSON_OUTPUT_REMINDER.strip(),
        })
        logger.debug("[Pipeline] 强制输出指令已追加（新 user message）")
        return messages

    async def trigger_warmup(
        self,
        request: GatewayRequest,
        agent_type: str,
        url: str,
    ) -> None:
        """触发推测性预热（后台调用，无阻塞）。

        实现要点：
        - 必须使用 asyncio.get_running_loop().create_task() 而非 asyncio.create_task()
        - asyncio.create_task() 要求在协程内部调用（否则抛出 RuntimeError）
        - GatewayProxy.chat() 是协程，故此处可以安全使用
        - 本方法本身是协程，保证在协程上下文中执行
        """
        if self.warmer:
            # asyncio.get_running_loop() 在协程内调用始终安全
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.warmer.on_request(request, agent_type, url),
                name=f"warmup-trigger-{request.request_id[:8]}",
            )
