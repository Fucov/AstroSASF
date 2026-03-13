"""
AstroSASF · Core · Orchestrator (V4.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
实验系统管理器 —— 崩溃安全的统计保全机制。

V4.1 修复:
- 环境崩溃时，从 env.collect_stats() 安全提取已产生的统计数据。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from sasf.core.config_loader import SASFConfig
from sasf.core.environment import LaboratoryEnvironment

logger = logging.getLogger(__name__)


@dataclass
class Orchestrator:
    """多实验柜编排器 (V4.1 崩溃安全)。"""

    config: SASFConfig
    _labs: dict[str, tuple[LaboratoryEnvironment, list[str]]] = field(
        default_factory=dict, init=False,
    )

    def spawn_laboratory(
        self,
        lab_id: str,
        fsm: Any,
        tasks: list[str] | None = None,
        tool_registrar: Any = None,
        initial_telemetry: dict[str, Any] | None = None,
    ) -> LaboratoryEnvironment:
        """创建并注册实验柜。

        Parameters
        ----------
        lab_id : str
        fsm : ShadowFSM
            外部构造的通用 FSM 实例。
        tasks : list[str], optional
        tool_registrar : callable, optional
            业务 MCP Tool 注册函数。
        initial_telemetry : dict, optional
            初始遥测数据。
        """
        if lab_id in self._labs:
            raise ValueError(f"实验柜 '{lab_id}' 已存在")

        max_labs = self.config.orchestrator.max_concurrent_labs
        if len(self._labs) >= max_labs:
            raise RuntimeError(
                f"已达最大并发数 {max_labs}，无法创建 '{lab_id}'"
            )

        env = LaboratoryEnvironment(
            lab_id=lab_id,
            config=self.config,
            fsm=fsm,
            tool_registrar=tool_registrar,
            initial_telemetry=initial_telemetry or {},
        )
        self._labs[lab_id] = (env, tasks or [])
        logger.info("Orchestrator: 注册 '%s', 任务数=%d", lab_id, len(tasks or []))
        return env

    async def run_all(self) -> list[dict[str, Any]]:
        """并发运行所有实验柜。崩溃时保全已产生的统计数据。"""
        if not self._labs:
            logger.warning("Orchestrator: 没有实验柜")
            return []

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Orchestrator: 并发启动 %d 个实验柜", len(self._labs))
        logger.info("╚" + "═" * 58 + "╝")

        coroutines = [env.run(tasks) for env, tasks in self._labs.values()]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        processed: list[dict[str, Any]] = []
        for (lab_id, (env, _tasks)), result in zip(self._labs.items(), results):
            if isinstance(result, Exception):
                logger.error(
                    "Orchestrator: '%s' 运行异常: %s", lab_id, result,
                )
                # ── 崩溃安全：从环境实例中抢救统计数据 ── #
                salvaged = env.collect_stats()
                salvaged["status"] = "error"
                salvaged["detail"] = str(result)
                salvaged["task_results"] = []

                # 尝试获取遥测快照
                try:
                    import asyncio as _aio
                    loop = _aio.get_event_loop()
                    if loop.is_running():
                        salvaged["final_telemetry"] = {}
                    else:
                        salvaged["final_telemetry"] = loop.run_until_complete(
                            env.get_telemetry()
                        )
                except Exception:
                    salvaged["final_telemetry"] = {}

                processed.append(salvaged)
            else:
                processed.append(result)

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Orchestrator: 所有实验柜运行完毕")
        logger.info("╚" + "═" * 58 + "╝")
        return processed

    @property
    def lab_ids(self) -> list[str]:
        return list(self._labs.keys())
