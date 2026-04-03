"""
AstroSASF · Interface · Global State
====================================
全局应用状态 — 由 FastAPI lifespan 管理生命周期。

所有需要访问 loader/facade/config/gateway 的模块应从本模块导入 state。

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    """应用全局状态（lifespan 管理）。"""
    facade: Any = field(default=None)
    loader: Any = field(default=None)
    config: Any = field(default=None)
    gateway_proxy: Any = field(default=None)   # infra.gateway.proxy.GatewayProxy
    startup_time: float = field(default_factory=time.time)


# 全局单例
state = AppState()
