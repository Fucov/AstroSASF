"""
AstroSASF Interface — 向上暴露的 API 接口
===========================================

本目录包含：

- gateway.py  — Space-MCP 协议网关
- server.py   — FastAPI 服务端入口

提供统一的 HTTP API 接口，供外部系统调用。
"""

__version__ = "8.0.0"
__all__ = ["gateway", "server", "facade", "state"]
