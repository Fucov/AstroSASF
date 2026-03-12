"""
AstroSASF · Core · Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
实验系统管理器 —— 管理 N 个完全隔离的 LaboratoryEnvironment。
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
    """多实验柜编排器。"""

    config: SASFConfig
    _labs: dict[str, tuple[LaboratoryEnvironment, list[str]]] = field(
        default_factory=dict, init=False,
    )

    def spawn_laboratory(
        self,
        lab_id: str,
        tasks: list[str] | None = None,
    ) -> LaboratoryEnvironment:
        if lab_id in self._labs:
            raise ValueError(f"实验柜 '{lab_id}' 已存在")

        max_labs = self.config.orchestrator.max_concurrent_labs
        if len(self._labs) >= max_labs:
            raise RuntimeError(
                f"已达最大并发数 {max_labs}，无法创建 '{lab_id}'"
            )

        env = LaboratoryEnvironment(lab_id=lab_id, config=self.config)
        self._labs[lab_id] = (env, tasks or [])
        logger.info("Orchestrator: 注册 '%s', 任务数=%d", lab_id, len(tasks or []))
        return env

    async def run_all(self) -> list[dict[str, Any]]:
        if not self._labs:
            logger.warning("Orchestrator: 没有实验柜")
            return []

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Orchestrator: 并发启动 %d 个实验柜", len(self._labs))
        logger.info("╚" + "═" * 58 + "╝")

        coroutines = [env.run(tasks) for env, tasks in self._labs.values()]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        processed: list[dict[str, Any]] = []
        for (lab_id, _), result in zip(self._labs.items(), results):
            if isinstance(result, Exception):
                logger.error("Orchestrator: '%s' 异常: %s", lab_id, result)
                processed.append({"lab_id": lab_id, "status": "error", "detail": str(result)})
            else:
                processed.append(result)

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Orchestrator: 所有实验柜运行完毕")
        logger.info("╚" + "═" * 58 + "╝")
        return processed

    @property
    def lab_ids(self) -> list[str]:
        return list(self._labs.keys())
