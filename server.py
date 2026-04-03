"""
AstroSASF V7.2 · Server (兼容层)
================================
本文件为 V7.1 → V7.2 迁移兼容层。

推荐使用：
    uvicorn interface.server:app --reload --host 0.0.0.0 --port 8000

旧启动方式（仍可用）：
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Author: AstroSASF Team
Version: 7.2 (deprecated, redirects to interface.server)
"""

from __future__ import annotations

# V7.2: 重导出到新的 interface.server
# 所有路由和逻辑已迁移至 interface/server.py
from interface.server import app  # noqa: F401, E402

__all__ = ["app"]
