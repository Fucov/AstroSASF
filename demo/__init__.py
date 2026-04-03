"""
AstroSASF Demo — 演示资产与模拟智能体
=====================================

本目录包含：

- demo_mission.py — 单一演示文件（3 Agent 并发协作）
- assets/         — 演示所需的文档和技能资产
- agents/         — Agent 逻辑定义

演示场景：模拟三个智能体（Q&A、Planner、Executor）
通过 interface/server.py 暴露的本地接口进行通信协作。
"""

__version__ = "7.2.0"
__all__ = ["demo_mission", "assets", "agents"]
