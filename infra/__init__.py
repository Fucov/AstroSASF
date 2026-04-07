"""
AstroSASF Infra — 核心计算与存储管理内核
=========================================

本目录为框架内核层，包含：

- llm/       — LLM 实例管理、配置加载
- routing/   — 前缀感知负载均衡、VRAM 熔断
- gateway/   — 分布式 LLM 网关代理

所有模块均不含任何业务词汇（Bio/Fluid等），保持框架通用性。
"""

__version__ = "8.0.0"
__all__ = ["llm", "routing", "gateway"]
