"""
AstroSASF · Labs · MCP Registry
================================
MCP 工具注册中心 —— Guard 装饰器 + Macro 参数预绑定。

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    float: "number",
    int: "integer",
    str: "string",
    bool: "boolean",
}


def _type_to_json_schema(py_type: type) -> str:
    return _PYTHON_TYPE_TO_JSON.get(py_type, "string")


class MCPToolContext:
    """MCP Tool 执行上下文（含 DeviceRuntime 支持）。

    V8.0 新增字段：
    - device_runtime : DeviceRuntime | None  — 物理设备统一调用层
    - current_task_id : str | None          — 当前执行任务的 ID（用于 metrics）
    - metrics        : MetricsCollector | None — 指标采集器（可选）
    """

    def __init__(
        self,
        engine: Any,
        bus: Any,
        lab_id: str,
        device_runtime: Any = None,
        current_task_id: str | None = None,
        metrics: Any = None,
    ) -> None:
        self.engine = engine
        self.bus = bus
        self.lab_id = lab_id
        self.device_runtime = device_runtime
        self.current_task_id = current_task_id
        self.metrics = metrics

    @property
    def fsm(self) -> Any:
        """向后兼容别名。"""
        return self.engine


MCPToolHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolGuard:
    """MCP Tool 的声明式安全守卫。"""
    require_states: dict[str, str] = field(default_factory=dict)
    forbid_states: dict[str, str] = field(default_factory=dict)
    telemetry_rules: list[str] = field(default_factory=list)


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
    macro_target: str | None = None
    macro_preset: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolRegistry:
    """实例级 MCP 工具注册中心 (V5)。"""

    lab_id: str
    _tools: dict[str, ToolDescriptor] = field(default_factory=dict, init=False)

    def mcp_tool(
        self,
        func: MCPToolHandler | None = None,
        *,
        require_states: dict[str, str] | None = None,
        forbid_states: dict[str, str] | None = None,
        telemetry_rules: list[str] | None = None,
    ) -> MCPToolHandler | Callable[[MCPToolHandler], MCPToolHandler]:
        """声明式 MCP Tool 注册装饰器。"""
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
            return decorator(func)
        return decorator

    def _register_function(
        self,
        func: MCPToolHandler,
        guard: ToolGuard | None = None,
    ) -> None:
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

    def bind_macro(
        self,
        macro_name: str,
        target_tool: str,
        preset_params: dict[str, Any],
        description: str | None = None,
    ) -> None:
        """将底层 Tool 绑定为参数预设的宏指令。"""
        target = self._tools.get(target_tool)
        if target is None:
            raise ValueError(
                f"[{self.lab_id}] bind_macro: 目标 Tool '{target_tool}' 未注册"
            )

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
            "[%s] MCPToolRegistry: 🔗 绑定 Macro '%s' → %s(%s)  剩余参数: %s",
            self.lab_id, macro_name, target_tool,
            preset_params, list(remaining_params.keys()) or "(无)",
        )

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
        vocab: set[str] = set()
        for t in self._tools.values():
            vocab.add(t.name)
            vocab.update(t.param_keys)
        vocab.update(["skill", "status", "detail", "fsm_state", "success", "error"])
        return sorted(vocab)

    def get_macros(self) -> dict[str, dict[str, Any]]:
        return {
            t.name: {
                "target": t.macro_target,
                "preset": t.macro_preset,
                "description": t.description,
            }
            for t in self._tools.values()
            if t.is_macro
        }

    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        context: MCPToolContext,
    ) -> dict[str, Any]:
        """查找并调用 MCP Tool（含 Guard 前置校验）。"""
        from labs.interlock_engine import SecurityGuardrailException, safe_eval_bool

        descriptor = self._tools.get(name)
        if descriptor is None:
            return {
                "skill": name,
                "status": "error",
                "detail": f"MCPToolRegistry: 未注册的 Tool '{name}'",
            }

        if descriptor.guard:
            guard = descriptor.guard
            engine = context.engine

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
                        f"[{self.lab_id}] Guard 拦截 '{name}': 未知子系统 '{subsystem}'"
                    )

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
                        f"[{self.lab_id}] Guard 拦截 '{name}': 未知子系统 '{subsystem}'"
                    )

            if guard.telemetry_rules:
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

        actual_params = dict(params)
        if descriptor.is_macro and descriptor.macro_preset:
            merged = dict(descriptor.macro_preset)
            merged.update(actual_params)
            actual_params = merged

        return await descriptor.handler(context, **actual_params)

    @property
    def count(self) -> int:
        return len(self._tools)

    @property
    def macro_count(self) -> int:
        return sum(1 for t in self._tools.values() if t.is_macro)


__all__ = [
    "MCPToolContext",
    "MCPToolRegistry",
    "MCPToolHandler",
    "ToolGuard",
    "ToolDescriptor",
]
