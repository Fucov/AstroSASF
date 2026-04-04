"""
AstroSASF · Infra · Config Loader (Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
统一配置加载器 —— 解析 ``config.yaml`` 并提供强类型配置对象。

V7.2 核心变化：
- 移除旧的单一 ``llm`` 节点，引入 ``gateway.backends[]`` 多后端阵列
- AstroSASF 不管理 LLM 进程生命周期，只通过 HTTP 连接已运行的推理服务
- 所有 LLM 调用统一经由内部 GatewayProxy，不在框架内直接实例化 LangChain

支持的 Provider：
- ``ollama``   — 本地 Ollama（http://localhost:11434）
- ``sglang``   — SGLang 推理服务器（http://<host>:8000）
- ``vllm``     — vLLM 推理服务器（http://<host>:8000）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Default config path                                                         #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
#  Config Dataclasses                                                          #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GatewayBackendConfig:
    """LLM 推理后端配置（对应 config.yaml 中的一个 backend 条目）。"""
    url: str                            # e.g. "http://192.168.1.10:8000"
    provider: str = "ollama"            # "ollama" | "sglang" | "vllm"
    model_name: str = "qwen2.5:7b"     # 推理部署的模型名称
    weight: int = 1                     # 路由权重（weight 越高分到越多请求）
    tags: list[str] = field(default_factory=list)  # 标签（如 ["v100", "node-1"]）
    enabled: bool = True                # 是否启用（False 则跳过注册）
    # 可选的 VRAM 水线覆盖（留空则使用 GatewayConfig 的全局值）
    vram_high_watermark: float | None = None
    vram_critical_watermark: float | None = None


@dataclass(frozen=True)
class GatewayConfig:
    """分布式 LLM 网关配置。"""
    backends: list[GatewayBackendConfig] = field(default_factory=list)
    vram_high_watermark: float = 0.85
    vram_critical_watermark: float = 0.92
    vram_low_watermark: float = 0.60


@dataclass(frozen=True)
class OrchestratorConfig:
    """编排器配置。"""
    max_concurrent_labs: int = 3
    dag_execution_timeout: float = 3600.0


@dataclass(frozen=True)
class MiddlewareConfig:
    """中间件配置。"""
    spacewire_bandwidth_kbps: float = 200.0
    enable_space_mcp_compression: bool = True


@dataclass(frozen=True)
class SASFConfig:
    """AstroSASF 全局配置（内核配置）。"""
    gateway: GatewayConfig
    orchestrator: OrchestratorConfig
    middleware: MiddlewareConfig


# --------------------------------------------------------------------------- #
#  Loader                                                                      #
# --------------------------------------------------------------------------- #

def load_config(path: str | Path | None = None) -> SASFConfig:
    """加载并解析 YAML 配置文件。

    Parameters
    ----------
    path : str | Path | None
        配置文件路径，为 ``None`` 时使用项目根目录下的 ``config.yaml``。

    Returns
    -------
    SASFConfig
        解析后的强类型配置对象。
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning("配置文件不存在: %s，使用默认值", config_path)
        return _default_config()

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    logger.info("已加载配置: %s", config_path)

    # ── Gateway ────────────────────────────────────────────────────────────── #
    gw_raw = raw.get("gateway", {})
    gw_global = gw_raw.get("vram_high_watermark", 0.85)
    gw_critical = gw_raw.get("vram_critical_watermark", 0.92)
    gw_low = gw_raw.get("vram_low_watermark", 0.60)

    backends_raw: list[dict[str, Any]] = gw_raw.get("backends", [])
    backends: list[GatewayBackendConfig] = []
    for b in backends_raw:
        if not b.get("enabled", True):
            logger.debug("[ConfigLoader] 跳过已禁用的后端: %s", b.get("url"))
            continue
        backends.append(GatewayBackendConfig(
            url=b.get("url", ""),
            provider=b.get("provider", "ollama"),
            model_name=b.get("model_name", "qwen2.5:7b"),
            weight=b.get("weight", 1),
            tags=b.get("tags", []),
            enabled=b.get("enabled", True),
            vram_high_watermark=b.get("vram_high_watermark"),
            vram_critical_watermark=b.get("vram_critical_watermark"),
        ))

    gateway_cfg = GatewayConfig(
        backends=backends,
        vram_high_watermark=gw_global,
        vram_critical_watermark=gw_critical,
        vram_low_watermark=gw_low,
    )

    # ── Middleware ──────────────────────────────────────────────────────────── #
    mw_raw = raw.get("middleware", {})
    middleware_cfg = MiddlewareConfig(
        spacewire_bandwidth_kbps=mw_raw.get("spacewire_bandwidth_kbps", 200.0),
        enable_space_mcp_compression=mw_raw.get("enable_space_mcp_compression", True),
    )

    # ── Orchestrator ───────────────────────────────────────────────────────── #
    orch_raw = raw.get("orchestrator", {})
    orchestrator_cfg = OrchestratorConfig(
        max_concurrent_labs=orch_raw.get("max_concurrent_labs", 3),
        dag_execution_timeout=orch_raw.get("dag_execution_timeout", 3600.0),
    )

    logger.info(
        "[ConfigLoader] Gateway 配置: %d 个后端, VRAM 水线 [%.0f%% / %.0f%% / %.0f%%]",
        len(backends),
        gw_low * 100, gw_global * 100, gw_critical * 100,
    )
    for b in backends:
        logger.info(
            "  · %s (%s, model=%s, weight=%d, tags=%s)",
            b.url, b.provider, b.model_name, b.weight, b.tags,
        )

    return SASFConfig(
        gateway=gateway_cfg,
        orchestrator=orchestrator_cfg,
        middleware=middleware_cfg,
    )


def _default_config() -> SASFConfig:
    """返回全默认配置（仅包含 localhost Ollama）。"""
    return SASFConfig(
        gateway=GatewayConfig(
            backends=[
                GatewayBackendConfig(
                    url="http://localhost:11434",
                    provider="ollama",
                    model_name="qwen2.5:7b",
                    weight=1,
                    tags=["local", "fallback"],
                ),
            ],
            vram_high_watermark=0.85,
            vram_critical_watermark=0.92,
            vram_low_watermark=0.60,
        ),
        orchestrator=OrchestratorConfig(
            max_concurrent_labs=3,
            dag_execution_timeout=3600.0,
        ),
        middleware=MiddlewareConfig(
            spacewire_bandwidth_kbps=200.0,
            enable_space_mcp_compression=True,
        ),
    )
