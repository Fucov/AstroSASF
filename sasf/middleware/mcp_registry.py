"""
AstroSASF · Middleware · MCPToolRegistry (V5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP 工具注册中心 —— Guard 装饰器 + Macro 参数预绑定。

V5 核心变化：
- ``@mcp_tool(forbid_states=..., require_states=..., telemetry_rules=...)``
  声明式前置 Guard，invoke 前自动校验
- ``bind_macro(macro_name, target_tool, preset_params)``
  参数预绑定宏指令，注册为独立 Tool 暴露给 LLM
- ``all_vocabulary()`` 自动包含 Macro 名
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
    return _PYTHON_TYPE_TO_JSON.get(py_type, "string")


# --------------------------------------------------------------------------- #
#  MCPToolContext                                                               #
# --------------------------------------------------------------------------- #

class MCPToolContext:
    """MCP Tool 执行上下文。

    V5: ``fsm`` 更名为 ``engine`` (InterlockEngine)，同时保留 ``fsm``
    属性作为向后兼容别名。
    """

    def __init__(self, engine: Any, bus: Any, lab_id: str) -> None:
        self.engine = engine
        self.bus = bus
        self.lab_id = lab_id

    @property
    def fsm(self) -> Any:
        """向后兼容别名。"""
        return self.engine


MCPToolHandler = Callable[..., Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------------- #
#  ToolGuard — 声明式前置守卫                                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ToolGuard:
    """MCP Tool 的声明式安全守卫。

    Attributes
    ----------
    require_states : dict[str, str]
        子系统必须处于的状态 ``{subsystem: required_state}``
    forbid_states : dict[str, str]
        子系统禁止处于的状态 ``{subsystem: forbidden_state}``
    telemetry_rules : list[str]
        遥测条件表达式（满足时**放行**，不满足时拦截）
    """
    require_states: dict[str, str] = field(default_factory=dict)
    forbid_states: dict[str, str] = field(default_factory=dict)
    telemetry_rules: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  ToolDescriptor                                                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ToolDescriptor:
    """已注册 MCP Tool 的元信息。"""
    name: str
    description: str
    json_schema: dict[str, Any]
    param_keys: list[str]
    handler: MCPToolHandler
    guard: ToolGuard | None = None
    is_macro: bool = False
    macro_target: str | None = None        # Macro 指向的底层 Tool
    macro_preset: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  MCPToolRegistry                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class MCPToolRegistry:
    """实例级 MCP 工具注册中心 (V5)。

    Features
    --------
    - ``@mcp_tool(...)`` — 带 Guard 的声明式装饰器
    - ``bind_macro()`` — 参数预绑定宏指令
    - ``invoke()`` — 调用前自动 Guard 校验

    Example
    -------
    >>> registry = MCPToolRegistry(lab_id="Lab-01")
    >>>
    >>> @registry.mcp_tool(forbid_states={"vacuum": "ACTIVE"})
    ... async def measure_sensor(ctx: MCPToolContext, channel: int) -> dict:
    ...     \\"\\"\\"读取传感器通道值\\"\\"\\"
    ...     ...
    >>>
    >>> registry.bind_macro("quick_measure", "measure_sensor", {"channel": 1})
    """

    lab_id: str
    _tools: dict[str, ToolDescriptor] = field(default_factory=dict, init=False)

    # ---- 装饰器 ----------------------------------------------------------- #

    def mcp_tool(
        self,
        func: MCPToolHandler | None = None,
        *,
        require_states: dict[str, str] | None = None,
        forbid_states: dict[str, str] | None = None,
        telemetry_rules: list[str] | None = None,
    ) -> MCPToolHandler | Callable[[MCPToolHandler], MCPToolHandler]:
        """声明式 MCP Tool 注册装饰器 (V5)。

        可无参使用 ``@mcp_tool`` 或带参使用
        ``@mcp_tool(forbid_states={"vacuum": "ACTIVE"})``。
        """
        guard = None
        if require_states or forbid_states or telemetry_rules:
            guard = ToolGuard(
                require_states=require_states or {},
                forbid_states=forbid_states or {},
                telemetry_rules=telemetry_rules or [],
            )

        def decorator(fn: MCPToolHandler) -> MCPToolHandler:
            self._register_function(fn, guard=guard)
            return fn

        if func is not None:
            # @mcp_tool (无参)
            return decorator(func)
        # @mcp_tool(...) (带参)
        return decorator

    def _register_function(
        self,
        func: MCPToolHandler,
        guard: ToolGuard | None = None,
    ) -> None:
        """内部注册逻辑：反射签名 → 生成 Schema → 存储描述符。"""
        name = func.__name__
        description = (inspect.getdoc(func) or "").strip()

        sig = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []
        param_keys: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("ctx", "self", "return"):
                continue

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
            properties[param_name] = {"type": json_type}
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
            guard=guard,
        )

        if name in self._tools:
            logger.warning(
                "[%s] MCPToolRegistry: Tool '%s' 被重复注册，覆盖旧定义",
                self.lab_id, name,
            )

        self._tools[name] = descriptor

        guard_info = ""
        if guard:
            parts = []
            if guard.require_states:
                parts.append(f"require={guard.require_states}")
            if guard.forbid_states:
                parts.append(f"forbid={guard.forbid_states}")
            if guard.telemetry_rules:
                parts.append(f"rules={guard.telemetry_rules}")
            guard_info = f"  Guard: {', '.join(parts)}"

        logger.info(
            "[%s] MCPToolRegistry: ✅ 注册 Tool '%s' — %s  "
            "Schema: %s%s",
            self.lab_id, name, description,
            list(properties.keys()), guard_info,
        )

    # ---- Macro 绑定 ------------------------------------------------------- #

    def bind_macro(
        self,
        macro_name: str,
        target_tool: str,
        preset_params: dict[str, Any],
        description: str | None = None,
    ) -> None:
        """将底层 Tool 绑定为参数预设的宏指令。

        Macro 作为独立 Tool 暴露给 LLM（无参或少参调用）。
        """
        target = self._tools.get(target_tool)
        if target is None:
            raise ValueError(
                f"[{self.lab_id}] bind_macro: 目标 Tool '{target_tool}' 未注册"
            )

        # 计算剩余参数（未被预绑定的）
        remaining_params = {
            k: v
            for k, v in target.json_schema["function"]["parameters"]["properties"].items()
            if k not in preset_params
        }
        remaining_required = [
            k for k in target.json_schema["function"]["parameters"].get("required", [])
            if k not in preset_params
        ]

        macro_desc = description or f"宏指令: {target_tool}({preset_params})"

        json_schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": macro_name,
                "description": macro_desc,
                "parameters": {
                    "type": "object",
                    "properties": remaining_params,
                    "required": remaining_required,
                },
            },
        }

        descriptor = ToolDescriptor(
            name=macro_name,
            description=macro_desc,
            json_schema=json_schema,
            param_keys=list(remaining_params.keys()),
            handler=target.handler,
            guard=target.guard,
            is_macro=True,
            macro_target=target_tool,
            macro_preset=dict(preset_params),
        )

        self._tools[macro_name] = descriptor

        logger.info(
            "[%s] MCPToolRegistry: 🔗 绑定 Macro '%s' → %s(%s)  "
            "剩余参数: %s",
            self.lab_id, macro_name, target_tool,
            preset_params, list(remaining_params.keys()) or "(无)",
        )

    # ---- 查询 ------------------------------------------------------------- #

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "json_schema": t.json_schema,
                "is_macro": t.is_macro,
            }
            for t in self._tools.values()
        ]

    def all_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def all_param_keys(self) -> list[str]:
        seen: set[str] = set()
        keys: list[str] = []
        for t in self._tools.values():
            for k in t.param_keys:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return sorted(keys)

    def all_vocabulary(self) -> list[str]:
        """返回 Codec 所需的完整词汇表 (含 Macro 名)。"""
        vocab: set[str] = set()
        for t in self._tools.values():
            vocab.add(t.name)
            vocab.update(t.param_keys)
        vocab.update(["skill", "status", "detail", "fsm_state", "success", "error"])
        return sorted(vocab)

    def get_macros(self) -> dict[str, dict[str, Any]]:
        """返回所有 Macro 的 target→preset 映射（供 SkillLoader 使用）。"""
        return {
            t.name: {
                "target": t.macro_target,
                "preset": t.macro_preset,
                "description": t.description,
            }
            for t in self._tools.values()
            if t.is_macro
        }

    # ---- 调用 (含 Guard 校验) --------------------------------------------- #

    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        context: MCPToolContext,
    ) -> dict[str, Any]:
        """查找并调用 MCP Tool（含 Guard 前置校验）。"""
        from sasf.physics.interlock_engine import SecurityGuardrailException

        descriptor = self._tools.get(name)
        if descriptor is None:
            return {
                "skill": name,
                "status": "error",
                "detail": f"MCPToolRegistry: 未注册的 Tool '{name}'",
            }

        # ── Guard 前置校验 ── #
        if descriptor.guard:
            guard = descriptor.guard
            engine = context.engine

            # 1) require_states 校验
            for subsystem, required_state in guard.require_states.items():
                try:
                    current = engine.get_subsystem_state(subsystem)
                    if current != required_state:
                        raise SecurityGuardrailException(
                            f"[{self.lab_id}] Guard 拦截 '{name}': "
                            f"子系统 '{subsystem}' 需要 '{required_state}' "
                            f"但当前为 '{current}'"
                        )
                except KeyError:
                    raise SecurityGuardrailException(
                        f"[{self.lab_id}] Guard 拦截 '{name}': "
                        f"未知子系统 '{subsystem}'"
                    )

            # 2) forbid_states 校验
            for subsystem, forbidden_state in guard.forbid_states.items():
                try:
                    current = engine.get_subsystem_state(subsystem)
                    if current == forbidden_state:
                        raise SecurityGuardrailException(
                            f"[{self.lab_id}] Guard 拦截 '{name}': "
                            f"子系统 '{subsystem}' 处于禁止状态 '{forbidden_state}'"
                        )
                except KeyError:
                    raise SecurityGuardrailException(
                        f"[{self.lab_id}] Guard 拦截 '{name}': "
                        f"未知子系统 '{subsystem}'"
                    )

            # 3) telemetry_rules 校验（表达式为 True 时放行）
            if guard.telemetry_rules:
                from sasf.physics.interlock_engine import safe_eval_bool
                telemetry = await context.bus.snapshot()
                for rule_expr in guard.telemetry_rules:
                    try:
                        if not safe_eval_bool(rule_expr, telemetry):
                            raise SecurityGuardrailException(
                                f"[{self.lab_id}] Guard 拦截 '{name}': "
                                f"遥测条件不满足: {rule_expr}"
                            )
                    except SecurityGuardrailException:
                        raise
                    except Exception as exc:
                        raise SecurityGuardrailException(
                            f"[{self.lab_id}] Guard 校验异常 '{name}': {exc}"
                        ) from exc

        # ── Macro: 合并预设参数 ── #
        actual_params = dict(params)
        if descriptor.is_macro and descriptor.macro_preset:
            merged = dict(descriptor.macro_preset)
            merged.update(actual_params)
            actual_params = merged

        return await descriptor.handler(context, **actual_params)

    # ---- 统计 ------------------------------------------------------------- #

    @property
    def count(self) -> int:
        return len(self._tools)

    @property
    def macro_count(self) -> int:
        return sum(1 for t in self._tools.values() if t.is_macro)
