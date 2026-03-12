"""
AstroSASF · Core · Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
实验系统管理器 —— 管理 N 个完全隔离的 LaboratoryEnvironment。

使用 ``asyncio.gather`` 实现真正的多系统并发：每个实验柜拥有独立
的 EventBus、FSM、AgentNexus 和智能体实例，彼此之间零干扰。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from core.environment import LaboratoryEnvironment

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Orchestrator                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Orchestrator:
    """多实验柜编排器。

    Example
    -------
    >>> orch = Orchestrator()
    >>> orch.spawn_laboratory("Lab-Alpha", tasks=["heat_and_move"])
    >>> results = await orch.run_all()
    """

    _labs: dict[str, tuple[LaboratoryEnvironment, list[str]]] = field(
        default_factory=dict, init=False,
    )

    # ---- 环境管理 --------------------------------------------------------- #

    def spawn_laboratory(
        self,
        lab_id: str,
        tasks: list[str] | None = None,
    ) -> LaboratoryEnvironment:
        """创建并注册一个新的实验柜环境。

        Parameters
        ----------
        lab_id : str
            实验柜唯一标识。
        tasks : list[str], optional
            分配给该实验柜的任务列表。

        Returns
        -------
        LaboratoryEnvironment
            创建的环境实例。
        """
        if lab_id in self._labs:
            raise ValueError(f"实验柜 '{lab_id}' 已存在，请勿重复创建")

        env = LaboratoryEnvironment(lab_id=lab_id)
        self._labs[lab_id] = (env, tasks or [])
        logger.info(
            "Orchestrator: 注册实验柜 '%s', 任务数=%d",
            lab_id, len(tasks or []),
        )
        return env

    # ---- 并发运行 --------------------------------------------------------- #

    async def run_all(self) -> list[dict[str, Any]]:
        """并发启动所有已注册的实验柜环境。

        Returns
        -------
        list[dict]
            每个实验柜的运行结果摘要。
        """
        if not self._labs:
            logger.warning("Orchestrator: 没有已注册的实验柜")
            return []

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Orchestrator: 并发启动 %d 个实验柜环境", len(self._labs))
        logger.info("╚" + "═" * 58 + "╝")

        # asyncio.gather 实现真正的并发
        coroutines = [
            env.run(tasks) for env, tasks in self._labs.values()
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # 处理可能的异常
        processed: list[dict[str, Any]] = []
        for (lab_id, _), result in zip(self._labs.items(), results):
            if isinstance(result, Exception):
                logger.error(
                    "Orchestrator: 实验柜 '%s' 运行异常: %s", lab_id, result,
                )
                processed.append({
                    "lab_id": lab_id,
                    "status": "error",
                    "detail": str(result),
                })
            else:
                processed.append(result)

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Orchestrator: 所有实验柜运行完毕")
        logger.info("╚" + "═" * 58 + "╝")

        return processed

    # ---- 辅助 ------------------------------------------------------------- #

    @property
    def lab_ids(self) -> list[str]:
        """已注册的实验柜 ID 列表。"""
        return list(self._labs.keys())
