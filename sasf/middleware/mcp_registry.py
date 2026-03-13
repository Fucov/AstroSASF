"""
AstroSASF · Middleware · MCPToolRegistry (V4.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP 工具注册中心 —— **自动反射生成 JSON Schema**。

V4.2 核心区分：
- **MCP Tools** = 底层原子操作接口（此模块管理），由 ``@registry.mcp_tool`` 注册
- **OpenAI Skills** = 认知层 SOP 知识套件（由 ``cognition/skill_loader.py`` 管理）

``@mcp_tool`` 装饰器通过 ``inspect`` + ``typing`` 自动反射目标函数的
Type Hints 和 Docstring，动态生成兼容 **OpenAI Function Calling** 的
JSON Schema。物理开发者 **无需手写任何 Schema 字典**。
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Type Hints → JSON Schema 映射                                               #
# --------------------------------------------------------------------------- #

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    float: "number",
    int: "integer",
    str: "string",
    bool: "boolean",
}


def _type_to_json_schema(py_type: type) -> str:
    """将 Python 类型注解映射为 JSON Schema 类型字符串。"""
    return _PYTHON_TYPE_TO_JSON.get(py_type, "string")


# --------------------------------------------------------------------------- #
#  MCPToolContext — 替代旧 SkillContext                                         #
# --------------------------------------------------------------------------- #

class MCPToolContext:
    """MCP Tool 执行上下文 —— 封装 Handler 运行时所需的依赖。

    由 Gateway 在调用 Tool 前构造，注入 FSM / TelemetryBus 等运行时依赖，
    使 Tool Handler 无需直接依赖 Gateway 类型。
    """

    def __init__(self, fsm: Any, bus: Any, lab_id: str) -> None:
        self.fsm = fsm
        self.bus = bus
        self.lab_id = lab_id


# MCP Tool Handler 签名：(context, **kwargs) → result dict
MCPToolHandler = Callable[..., Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------------- #
#  ToolDescriptor                                                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ToolDescriptor:
    """已注册 MCP Tool 的元信息。"""
    name: str
    description: str
    json_schema: dict[str, Any]     # OpenAI Function Calling 兼容 schema
    param_keys: list[str]           # 参数键名列表（供 Codec 动态字典）
    handler: MCPToolHandler


# --------------------------------------------------------------------------- #
#  MCPToolRegistry                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class MCPToolRegistry:
    """实例级 MCP 工具注册中心（V4.2 自动反射版本）。

    提供 ``@mcp_tool`` 声明式装饰器。装饰器通过 ``inspect`` 自动反射
    目标函数的 Type Hints 和 Docstring，动态生成 JSON Schema。

    Example
    -------
    >>> registry = MCPToolRegistry(lab_id="Lab-Alpha")
    >>>
    >>> @registry.mcp_tool
    ... async def set_temperature(ctx: MCPToolContext, target: float) -> dict:
    ...     \"\"\"设置舱内温度目标值（℃）\"\"\"
    ...     ...
    """

    lab_id: str
    _tools: dict[str, ToolDescriptor] = field(default_factory=dict, init=False)

    # ---- 装饰器 ----------------------------------------------------------- #

    def mcp_tool(self, func: MCPToolHandler) -> MCPToolHandler:
        """声明式 MCP Tool 注册装饰器。

        自动从函数签名中提取：
        - 函数名 → tool name
        - Docstring → description
        - Type Hints → JSON Schema (排除 ctx 参数和 return)

        Parameters
        ----------
        func : MCPToolHandler
            目标异步函数，签名为 ``(ctx: MCPToolContext, **业务参数) → dict``。

        Returns
        -------
        MCPToolHandler
            原函数（不做包装，仅注册侧效应）。
        """
        name = func.__name__
        description = (inspect.getdoc(func) or "").strip()

        sig = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []
        param_keys: list[str] = []

        for param_name, param in sig.parameters.items():
            # 跳过 ctx (第一个参数) 和 self
            if param_name in ("ctx", "self", "return"):
                continue

            # 使用相对简单的启发式方法解析注解，避免 get_type_hints() 因为局部 import 引发 NameError
            # (由于 from __future__ import annotations，annotation 可能是字符串)
            ann = param.annotation
            json_type = "string"
            if ann is not inspect.Parameter.empty:
                if isinstance(ann, str):
                    ann_lower = ann.lower()
                    if "float" in ann_lower:
                        json_type = "number"
                    elif "int" in ann_lower:
                        json_type = "integer"
                    elif "bool" in ann_lower:
                        json_type = "boolean"
                else:
                    json_type = _type_to_json_schema(ann)

            param_keys.append(param_name)
            prop: dict[str, str] = {"type": json_type}
            properties[param_name] = prop

            # 没有默认值 → required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        json_schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

        descriptor = ToolDescriptor(
            name=name,
            description=description,
            json_schema=json_schema,
            param_keys=param_keys,
            handler=func,
        )

        if name in self._tools:
            logger.warning(
                "[%s] MCPToolRegistry: Tool '%s' 被重复注册，覆盖旧定义",
                self.lab_id, name,
            )

        self._tools[name] = descriptor

        logger.info(
            "[%s] MCPToolRegistry: ✅ 注册 Tool '%s' — %s  "
            "Schema 自动生成: %s",
            self.lab_id, name, description, list(properties.keys()),
        )

        return func

    # ---- 查询 ------------------------------------------------------------- #

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """返回所有已注册 Tool 的摘要（供 LLM prompt 和 Function Calling 注入）。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "json_schema": t.json_schema,
            }
            for t in self._tools.values()
        ]

    def all_tool_names(self) -> list[str]:
        """返回所有 Tool 名称。"""
        return list(self._tools.keys())

    def all_param_keys(self) -> list[str]:
        """返回所有 Tool 的参数键名（去重、确定性排序）。"""
        seen: set[str] = set()
        keys: list[str] = []
        for t in self._tools.values():
            for k in t.param_keys:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return sorted(keys)

    def all_vocabulary(self) -> list[str]:
        """返回 Codec 协商所需的完整词汇表 —— Tool 名 + 参数键 + 常见状态值。

        按字母序排列，确保跨实例字典一致性。
        """
        vocab: set[str] = set()
        for t in self._tools.values():
            vocab.add(t.name)
            vocab.update(t.param_keys)
        # 添加协议级常见字符串
        vocab.update(["skill", "status", "detail", "fsm_state", "success", "error"])
        return sorted(vocab)

    # ---- 调用 ------------------------------------------------------------- #

    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        context: MCPToolContext,
    ) -> dict[str, Any]:
        """查找并调用已注册的 MCP Tool。

        Handler 签名为 ``(ctx, **params)`` — 参数以关键字方式传入。
        """
        descriptor = self._tools.get(name)
        if descriptor is None:
            return {
                "skill": name,
                "status": "error",
                "detail": f"MCPToolRegistry: 未注册的 Tool '{name}'",
            }

        return await descriptor.handler(context, **params)


    # ---- 统计 ------------------------------------------------------------- #

    @property
    def count(self) -> int:
        return len(self._tools)
