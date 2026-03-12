"""
AstroSASF · Middleware · SkillRegistry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
技能注册中心 —— 中间件暴露给物理设备层的标准注册端口。

核心设计理念：网关 (Gateway) **不硬编码任何业务逻辑**。底层物理设备
通过 ``SkillRegistry.register()`` 主动将自己的能力注册为标准 MCP
技能，网关收到请求时去注册中心查找并执行，实现纯粹的协议透传。

每个 ``LaboratoryEnvironment`` 持有独立的 ``SkillRegistry`` 实例，
确保多系统并发时注册表互不干扰。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Type Aliases                                                                #
# --------------------------------------------------------------------------- #

class SkillContext:
    """Skill 执行上下文 —— 封装 Skill Handler 运行时所需的依赖。

    由 Gateway 在调用 Skill 前构造，注入 FSM / TelemetryBus 等运行时依赖，
    使 Skill Handler 无需直接依赖 Gateway 类型。
    """

    def __init__(self, fsm: Any, bus: Any, lab_id: str) -> None:
        self.fsm = fsm
        self.bus = bus
        self.lab_id = lab_id


# Skill Handler 签名：(context, params) → result dict
SkillHandler = Callable[[SkillContext, dict[str, Any]], Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------------- #
#  Skill Descriptor                                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SkillDescriptor:
    """已注册技能的元信息。"""
    name: str
    description: str
    param_schema: dict[str, Any]
    handler: SkillHandler


# --------------------------------------------------------------------------- #
#  SkillRegistry                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class SkillRegistry:
    """实例级技能注册中心。

    每个 LaboratoryEnvironment 持有自己的 SkillRegistry，
    物理设备层初始化时通过 ``register()`` 将硬件操作注册为
    标准 MCP 技能。

    Example
    -------
    >>> registry = SkillRegistry(lab_id="Lab-Alpha")
    >>> registry.register(
    ...     name="set_temperature",
    ...     handler=set_temp_handler,
    ...     description="设置舱内温度",
    ...     param_schema={"target": "float"},
    ... )
    """

    lab_id: str
    _skills: dict[str, SkillDescriptor] = field(default_factory=dict, init=False)

    # ---- 注册 ------------------------------------------------------------- #

    def register(
        self,
        name: str,
        handler: SkillHandler,
        description: str = "",
        param_schema: dict[str, Any] | None = None,
    ) -> None:
        """将一个异步函数注册为 MCP Skill。

        Parameters
        ----------
        name : str
            技能唯一标识（如 ``"set_temperature"``）。
        handler : SkillHandler
            异步处理函数 ``(context, params) → result``。
        description : str
            技能描述（供 LLM function_calling 使用）。
        param_schema : dict, optional
            参数 JSON Schema（用于 LLM 提示和类型校验）。
        """
        if name in self._skills:
            logger.warning(
                "[%s] SkillRegistry: 技能 '%s' 被重复注册，覆盖旧定义",
                self.lab_id, name,
            )
        descriptor = SkillDescriptor(
            name=name,
            description=description,
            param_schema=param_schema or {},
            handler=handler,
        )
        self._skills[name] = descriptor
        logger.info(
            "[%s] SkillRegistry: ✅ 注册技能 '%s' — %s",
            self.lab_id, name, description,
        )

    # ---- 查询 ------------------------------------------------------------- #

    def has_skill(self, name: str) -> bool:
        return name in self._skills

    def get_skill(self, name: str) -> SkillDescriptor | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, Any]]:
        """返回所有已注册技能的摘要（供 LLM prompt 注入）。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "param_schema": s.param_schema,
            }
            for s in self._skills.values()
        ]

    # ---- 调用 ------------------------------------------------------------- #

    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        context: SkillContext,
    ) -> dict[str, Any]:
        """查找并调用已注册的 Skill。

        Parameters
        ----------
        name : str
            技能名。
        params : dict
            调用参数。
        context : SkillContext
            运行时上下文（FSM, Bus 等）。

        Returns
        -------
        dict
            包含 ``status`` 字段的执行结果。
        """
        descriptor = self._skills.get(name)
        if descriptor is None:
            return {
                "skill": name,
                "status": "error",
                "detail": f"SkillRegistry: 未注册的技能 '{name}'",
            }

        return await descriptor.handler(context, params)

    # ---- 统计 ------------------------------------------------------------- #

    @property
    def count(self) -> int:
        return len(self._skills)
