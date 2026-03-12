"""
AstroSASF · Core · ConfigLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
统一配置加载器 —— 解析 ``config.yaml`` 并提供 LLM 工厂方法。

支持两种 LLM Provider：
- ``ollama``            → ``langchain_ollama.ChatOllama`` (本地推理)
- ``openai_compatible`` → ``langchain_openai.ChatOpenAI``  (DeepSeek / 阿里云百炼 Qwen 等)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
class LLMConfig:
    """LLM 配置。"""
    provider: str       # "ollama" | "openai_compatible"
    base_url: str
    api_key: str
    model_name: str
    temperature: float


@dataclass(frozen=True)
class MiddlewareConfig:
    """中间件配置。"""
    spacewire_bandwidth_kbps: float
    enable_space_mcp_compression: bool


@dataclass(frozen=True)
class OrchestratorConfig:
    """编排器配置。"""
    max_concurrent_labs: int


@dataclass(frozen=True)
class SASFConfig:
    """AstroSASF 全局配置。"""
    llm: LLMConfig
    middleware: MiddlewareConfig
    orchestrator: OrchestratorConfig


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

    llm_raw = raw.get("llm", {})
    mw_raw = raw.get("middleware", {})
    orch_raw = raw.get("orchestrator", {})

    return SASFConfig(
        llm=LLMConfig(
            provider=llm_raw.get("provider", "ollama"),
            base_url=llm_raw.get("base_url", "http://localhost:11434"),
            api_key=llm_raw.get("api_key", ""),
            model_name=llm_raw.get("model_name", "qwen2.5:7b"),
            temperature=llm_raw.get("temperature", 0.1),
        ),
        middleware=MiddlewareConfig(
            spacewire_bandwidth_kbps=mw_raw.get("spacewire_bandwidth_kbps", 200.0),
            enable_space_mcp_compression=mw_raw.get("enable_space_mcp_compression", True),
        ),
        orchestrator=OrchestratorConfig(
            max_concurrent_labs=orch_raw.get("max_concurrent_labs", 3),
        ),
    )


def _default_config() -> SASFConfig:
    """返回全默认配置。"""
    return SASFConfig(
        llm=LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            api_key="",
            model_name="qwen2.5:7b",
            temperature=0.1,
        ),
        middleware=MiddlewareConfig(
            spacewire_bandwidth_kbps=200.0,
            enable_space_mcp_compression=True,
        ),
        orchestrator=OrchestratorConfig(
            max_concurrent_labs=3,
        ),
    )


# --------------------------------------------------------------------------- #
#  LLM Factory                                                                 #
# --------------------------------------------------------------------------- #

def create_llm(config: LLMConfig) -> Any:
    """根据配置动态创建 LLM 实例。

    Parameters
    ----------
    config : LLMConfig
        LLM 配置。

    Returns
    -------
    BaseChatModel
        LangChain 聊天模型实例。
    """
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=config.model_name,
            base_url=config.base_url,
            temperature=config.temperature,
        )
        logger.info(
            "LLM 创建: ChatOllama(model=%s, base_url=%s)",
            config.model_name, config.base_url,
        )
        return llm

    if config.provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.model_name,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
        )
        logger.info(
            "LLM 创建: ChatOpenAI(model=%s, base_url=%s)",
            config.model_name, config.base_url,
        )
        return llm

    raise ValueError(f"不支持的 LLM Provider: {config.provider}")
