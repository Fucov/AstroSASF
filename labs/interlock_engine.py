"""
AstroSASF · Labs · Interlock Engine
===================================
正交子系统状态管理 + 跨系统联锁规则引擎（物理层）。

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import ast
import asyncio
import logging
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class SecurityGuardrailException(Exception):
    """安全联锁拦截异常 —— 通用，不含业务词汇。"""
    pass


_ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Constant, ast.Name, ast.Load,
)

_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _validate_ast(node: ast.AST) -> None:
    if not isinstance(node, _ALLOWED_AST_NODES):
        raise SecurityGuardrailException(
            f"联锁表达式含非法语法节点: {type(node).__name__}"
        )
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise SecurityGuardrailException(
                f"联锁表达式引用了未知变量: '{node.id}'"
            )
        return env[node.id]
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_ast(v, env) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_ast(v, env) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_ast(node.operand, env)
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, env)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator, env)
            op_func = _CMP_OPS.get(type(op_node))
            if op_func is None:
                raise SecurityGuardrailException(
                    f"不支持的比较运算符: {type(op_node).__name__}"
                )
            if not op_func(left, right):
                return False
            left = right
        return True
    raise SecurityGuardrailException(f"不支持的 AST 节点类型: {type(node).__name__}")


def safe_eval_bool(expression: str, env: dict[str, Any]) -> bool:
    """安全布尔表达式求值（仅允许数学比较运算符）。"""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        _validate_ast(tree.body)
        return bool(_eval_ast(tree.body, env))
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        raise SecurityGuardrailException(f"表达式求值错误: '{expression}' -> {e}")


class InterlockRule:
    """单条联锁规则。"""

    def __init__(
        self,
        rule_id: str,
        condition: str,
        message: str,
        scope: str | None = None,
    ):
        self.rule_id = rule_id
        self.condition = condition
        self.message = message
        self.scope = scope  # 可选：仅对特定 tool 生效（None=全局）

    def evaluate(self, env: dict[str, Any]) -> bool:
        return safe_eval_bool(self.condition, env)


@dataclass
class InterlockEngine:
    """正交子系统状态管理 + 联锁规则引擎（物理层模块）。

    框架零业务词汇：子系统名、状态名、规则全部由外部 YAML 注入。
    """

    lab_id: str
    subsystems: dict[str, list[str]] = field(default_factory=dict)
    initial_states: dict[str, str] = field(default_factory=dict)
    interlocks: list[InterlockRule] = field(default_factory=list)

    _states: dict[str, str] = field(default_factory=dict, init=False)
    _bus: Any = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        for subsystem, initial in self.initial_states.items():
            allowed = self.subsystems.get(subsystem, [])
            if initial not in allowed:
                raise ValueError(
                    f"子系统 '{subsystem}' 初始状态 '{initial}' "
                    f"不在允许状态集 {allowed} 中"
                )
        self._states = dict(self.initial_states)

    @classmethod
    def from_yaml(cls, lab_id: str, yaml_path: str | Path) -> "InterlockEngine":
        """从 YAML 文件加载联锁配置。"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"联锁配置文件不存在: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        raw_interlocks = config.get("interlocks", [])
        rules = [
            InterlockRule(
                rule_id=f"rule_{i}",
                condition=r["condition"],
                message=r.get("message", ""),
                scope=r.get("scope"),
            )
            for i, r in enumerate(raw_interlocks)
        ]

        engine = cls(
            lab_id=lab_id,
            subsystems=config.get("subsystems", {}),
            initial_states=config.get("initial_states", {}),
            interlocks=rules,
        )

        logger.info(
            "[%s] InterlockEngine: %d 子系统, %d 联锁规则",
            lab_id, len(engine.subsystems), len(engine.interlocks),
        )
        return engine

    def bind_telemetry_bus(self, bus: Any) -> None:
        self._bus = bus

    async def set_subsystem_state(
        self,
        subsystem: str,
        new_state: str,
    ) -> None:
        """设置子系统状态（含合法性校验）。"""
        async with self._lock:
            if subsystem not in self.subsystems:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] 未知子系统: '{subsystem}'"
                )
            allowed = self.subsystems[subsystem]
            if new_state not in allowed:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] '{subsystem}' 禁止状态 '{new_state}' "
                    f"(允许: {allowed})"
                )
            old = self._states.get(subsystem, "UNKNOWN")
            self._states[subsystem] = new_state
            logger.info(
                "[%s] FSM: %s: %s → %s",
                self.lab_id, subsystem, old, new_state,
            )

    def get_subsystem_state(self, subsystem: str) -> str:
        """查询单个子系统状态。"""
        if subsystem not in self._states:
            raise KeyError(f"[{self.lab_id}] 未知子系统: '{subsystem}'")
        return self._states[subsystem]

    @property
    def current_states(self) -> dict[str, str]:
        """获取当前所有子系统状态。"""
        return dict(self._states)

    async def check_interlocks(
        self,
        tool_name: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """校验联锁规则（条件为 True 时抛出异常）。"""
        env: dict[str, Any] = dict(self._states)
        if telemetry:
            env.update(telemetry)

        for rule in self.interlocks:
            if rule.scope and rule.scope != tool_name:
                continue
            try:
                if rule.evaluate(env):
                    raise SecurityGuardrailException(
                        f"[{self.lab_id}] 联锁拦截 [{tool_name}]: {rule.message}"
                    )
            except SecurityGuardrailException:
                raise
            except Exception as exc:
                logger.warning(
                    "[%s] 规则 '%s' 求值异常: %s",
                    self.lab_id, rule.rule_id, exc,
                )


__all__ = [
    "InterlockEngine",
    "InterlockRule",
    "SecurityGuardrailException",
    "safe_eval_bool",
]
