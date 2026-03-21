"""
AstroSASF · Physics · InterlockEngine (V5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
正交子系统状态管理 + 跨系统联锁规则引擎。

V5 核心设计：
- **正交状态**：每个子系统独立维护自己的状态集合，彻底消灭状态爆炸
- **联锁规则**：通过安全求值器（``ast`` 白名单）执行布尔表达式，
  跨子系统校验禁止条件
- **框架零业务词汇**：子系统名、状态名、规则全部由外部 YAML 注入
"""

from __future__ import annotations

import ast
import asyncio
import logging
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Exception                                                                   #
# --------------------------------------------------------------------------- #

class SecurityGuardrailException(Exception):
    """安全联锁拦截异常 —— 通用，不含业务词汇。"""


# --------------------------------------------------------------------------- #
#  Safe Expression Evaluator                                                   #
# --------------------------------------------------------------------------- #

# 允许的 AST 节点白名单（仅布尔 / 比较 / 字面量）
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Constant, ast.Name, ast.Load,
    # Python 3.12+ 可能产生 BinOp 用于字符串拼接，不允许
)

_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _validate_ast(node: ast.AST) -> None:
    """递归验证 AST 节点是否在白名单中。"""
    if not isinstance(node, _ALLOWED_AST_NODES):
        raise SecurityGuardrailException(
            f"联锁表达式含非法语法节点: {type(node).__name__}"
        )
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    """递归求值白名单 AST。"""
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

    raise SecurityGuardrailException(
        f"联锁表达式含不可求值节点: {type(node).__name__}"
    )


def safe_eval_bool(expression: str, env: dict[str, Any]) -> bool:
    """安全求值布尔表达式（仅允许比较/逻辑运算）。

    Parameters
    ----------
    expression : str
        布尔表达式，例如 ``"vacuum == 'ACTIVE' and arm != 'IDLE'"``
    env : dict
        变量名→值映射

    Returns
    -------
    bool
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SecurityGuardrailException(
            f"联锁表达式语法错误: {expression!r} → {exc}"
        ) from exc

    _validate_ast(tree)
    return bool(_eval_ast(tree, env))


# --------------------------------------------------------------------------- #
#  InterlockRule                                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InterlockRule:
    """单条联锁规则。"""
    condition: str                         # 布尔表达式
    message: str                           # 拦截时的人类可读消息
    scope: str | None = None               # 可选：仅对特定 tool 生效（None=全局）
    _compiled: ast.Expression = field(      # 预编译 AST（启动时校验语法）
        default=None, init=False, repr=False, compare=False, hash=False,
    )

    def __post_init__(self) -> None:
        # 预编译并校验 AST
        try:
            tree = ast.parse(self.condition, mode="eval")
            _validate_ast(tree)
            object.__setattr__(self, "_compiled", tree)
        except (SyntaxError, SecurityGuardrailException) as exc:
            raise ValueError(
                f"联锁规则编译失败: {self.condition!r} → {exc}"
            ) from exc


# --------------------------------------------------------------------------- #
#  InterlockEngine                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class InterlockEngine:
    """正交子系统状态管理 + 跨系统联锁规则引擎。

    Parameters
    ----------
    lab_id : str
    subsystems : dict[str, list[str]]
        子系统定义 ``{subsystem_name: [allowed_states]}``
    initial_states : dict[str, str]
        初始状态 ``{subsystem_name: initial_state}``
    interlocks : list[InterlockRule]
        联锁规则列表
    """

    lab_id: str
    subsystems: dict[str, list[str]] = field(default_factory=dict)
    initial_states: dict[str, str] = field(default_factory=dict)
    interlocks: list[InterlockRule] = field(default_factory=list)

    _states: dict[str, str] = field(default_factory=dict, init=False)
    _bus: Any = field(default=None, init=False, repr=False)  # TelemetryBus (可选)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        # 校验初始状态合法性
        for subsystem, initial in self.initial_states.items():
            allowed = self.subsystems.get(subsystem, [])
            if initial not in allowed:
                raise ValueError(
                    f"子系统 '{subsystem}' 初始状态 '{initial}' "
                    f"不在允许状态集 {allowed} 中"
                )
        self._states = dict(self.initial_states)

    # ---- 工厂方法 --------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path, lab_id: str = "default") -> InterlockEngine:
        """从 YAML 配置加载。

        YAML 格式::

            subsystems:
              thermal: [IDLE, HEATING, COOLING]
              vacuum: [IDLE, ACTIVE]
            initial_states:
              thermal: IDLE
              vacuum: IDLE
            interlocks:
              - condition: "vacuum == 'ACTIVE' and arm != 'IDLE'"
                message: "真空激活时禁止移动机械臂"
                scope: move_robotic_arm    # 可选
        """
        import yaml
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        subsystems = rules.get("subsystems", {})
        initial = rules.get("initial_states", {})
        raw_interlocks = rules.get("interlocks", [])

        interlocks = [
            InterlockRule(
                condition=r["condition"],
                message=r.get("message", "联锁规则被触发"),
                scope=r.get("scope"),
            )
            for r in raw_interlocks
        ]

        engine = cls(
            lab_id=lab_id,
            subsystems=subsystems,
            initial_states=initial,
            interlocks=interlocks,
        )
        logger.info(
            "[%s] InterlockEngine: %d 子系统, %d 联锁规则 (来源: %s)",
            lab_id, len(subsystems), len(interlocks), path.name,
        )
        for name, allowed in subsystems.items():
            logger.info(
                "[%s]    子系统 %-12s: %s (初始: %s)",
                lab_id, name, allowed, initial.get(name, "?"),
            )
        for rule in interlocks:
            logger.info(
                "[%s]    联锁: [%s] %s → %s",
                lab_id, rule.scope or "*", rule.condition, rule.message,
            )
        return engine

    # ---- 状态查询 / 修改 -------------------------------------------------- #

    @property
    def current_states(self) -> dict[str, str]:
        """返回所有子系统当前状态的快照。"""
        return dict(self._states)

    def get_subsystem_state(self, subsystem: str) -> str:
        """查询单个子系统当前状态。"""
        if subsystem not in self._states:
            raise KeyError(f"[{self.lab_id}] 未知子系统: '{subsystem}'")
        return self._states[subsystem]

    async def set_subsystem_state(
        self, subsystem: str, state: str,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """设置子系统状态（含合法性校验 + 联锁校验）。

        若未传入 telemetry 但引擎绑定了 TelemetryBus，会自动获取快照。
        """
        async with self._lock:
            # 1) 子系统存在性
            allowed = self.subsystems.get(subsystem)
            if allowed is None:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] 未知子系统: '{subsystem}'"
                )
            # 2) 状态合法性
            if state not in allowed:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] 非法状态 '{state}' "
                    f"(子系统 '{subsystem}' 允许: {allowed})"
                )
            # 3) 自动获取遥测（如果未显式传入）
            if telemetry is None and self._bus is not None:
                try:
                    telemetry = await self._bus.snapshot()
                except Exception:
                    telemetry = None

            # 4) 模拟设置 → 校验联锁
            old = self._states[subsystem]
            self._states[subsystem] = state
            try:
                self._check_interlocks_sync(telemetry=telemetry)
            except SecurityGuardrailException:
                self._states[subsystem] = old  # 回滚
                raise

            logger.info(
                "[%s] 🔄 子系统 '%s': %s → %s",
                self.lab_id, subsystem, old, state,
            )

    # ---- 联锁校验 --------------------------------------------------------- #

    def _build_eval_env(
        self,
        tool_name: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构建联锁表达式的求值环境。"""
        env: dict[str, Any] = {}
        # 子系统状态作为变量
        env.update(self._states)
        # 遥测数据作为变量
        if telemetry:
            env.update(telemetry)
        # 当前 tool 名（供 scope 过滤）
        if tool_name:
            env["__tool__"] = tool_name
        return env

    def _check_interlocks_sync(
        self,
        tool_name: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """同步联锁校验（内部使用）。"""
        env = self._build_eval_env(tool_name, telemetry)
        for rule in self.interlocks:
            if rule.scope and tool_name and rule.scope != tool_name:
                continue
            try:
                if safe_eval_bool(rule.condition, env):
                    raise SecurityGuardrailException(
                        f"[{self.lab_id}] 联锁拦截: {rule.message} "
                        f"(条件: {rule.condition})"
                    )
            except SecurityGuardrailException as exc:
                # 区分：联锁拦截 vs 变量缺失
                if "联锁拦截" in str(exc):
                    raise  # 真正的联锁触发，必须传播
                # 变量缺失 → 跳过此规则（缺少遥测数据时不误拦截）
                logger.debug(
                    "[%s] 联锁规则跳过 (缺少变量): %s → %s",
                    self.lab_id, rule.condition, exc,
                )

    async def check_interlocks(
        self,
        tool_name: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """异步联锁校验（Tools 调用前调用）。"""
        async with self._lock:
            self._check_interlocks_sync(tool_name, telemetry)

    # ---- 便捷属性 --------------------------------------------------------- #

    @property
    def subsystem_names(self) -> list[str]:
        return list(self.subsystems.keys())

    def bind_telemetry_bus(self, bus: Any) -> None:
        """绑定遥测总线，使 set_subsystem_state 自动获取遥测快照。"""
        self._bus = bus
        logger.info("[%s] InterlockEngine: 已绑定 TelemetryBus", self.lab_id)
