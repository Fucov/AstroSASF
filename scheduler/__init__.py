"""
AstroSASF Scheduler — 任务编排与协议控制内核
=============================================

本目录为框架调度层，包含：

- core.py          — DAG 双轨调度器
- models.py        — DAG 数据结构
- a2a_protocol.py  — Agent-to-Agent 协议
- virtual_bus.py   — 虚拟 SpaceWire 总线
- telemetry.py     — 遥测总线（1553B 协议模拟）

所有模块均不含任何业务词汇，保持框架通用性。
"""

__version__ = "7.5.0"
__all__ = ["core", "models", "a2a_protocol", "virtual_bus", "telemetry"]
