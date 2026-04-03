"""
AstroSASF Labs — 物理模拟与 MCP 注册
=====================================

本目录为物理执行层，包含：

- mcp_registry.py    — MCP 工具注册中心
- interlock_engine.py — 正交联锁引擎
- mcp_codec.py       — Space-MCP 编解码器

【注意】此处可以有业务词汇（如 Lab-Fluid 的泵控制），
因为这是与具体实验舱交互的物理层。内核化层级不出现业务词汇。
"""

__version__ = "7.2.0"
__all__ = ["mcp_registry", "interlock_engine", "mcp_codec", "lab_loader"]
