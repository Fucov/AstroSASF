"""
AstroSASF · Core · Orchestrator (V5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
多实验柜编排器。

V5 变化：
- ``ShadowFSM`` → ``InterlockEngine``
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
    """多实验柜编排器 (V5)。"""

    config: SASFConfig
    _labs: dict[str, tuple[LaboratoryEnvironment, list[str]]] = field(
        default_factory=dict, init=False,
    )

    def spawn_laboratory(
        self,
        lab_id: str,
        engine: Any,
        tasks: list[str] | None = None,
        tool_registrar: Any = None,
        macro_registrar: Any = None,
        initial_telemetry: dict[str, Any] | None = None,
    ) -> LaboratoryEnvironment:
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
            engine=engine,
            tool_registrar=tool_registrar,
            macro_registrar=macro_registrar,
            initial_telemetry=initial_telemetry or {},
        )
        self._labs[lab_id] = (env, tasks or [])
        logger.info("Orchestrator: 注册 '%s', 任务数=%d", lab_id, len(tasks or []))
        return env

    async def run_all(self) -> list[dict[str, Any]]:
        if not self._labs:
            logger.warning("Orchestrator: 没有实验柜")
            return []

        coroutines = [env.run(tasks) for env, tasks in self._labs.values()]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        processed: list[dict[str, Any]] = []
        for (lab_id, (env, _)), result in zip(self._labs.items(), results):
            if isinstance(result, Exception):
                logger.error("Orchestrator: '%s' 异常: %s", lab_id, result)
                salvaged = env.collect_stats()
                salvaged["status"] = "error"
                salvaged["detail"] = str(result)
                processed.append(salvaged)
            else:
                processed.append(result)

        return processed

    @property
    def lab_ids(self) -> list[str]:
        return list(self._labs.keys())
