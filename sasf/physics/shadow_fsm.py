"""
AstroSASF · Physics · ShadowFSM (V4.3 — 通用规则引擎)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
通用有限状态机引擎 —— **框架内零业务词汇**。

V4.3 核心变化：
- 删除所有硬编码枚举（``DeviceState``, ``Action``）
- 状态 / 转移规则 / 安全约束由外部注入（dict 或 YAML 文件）
- 框架只负责 **规则校验 + 状态迁移 + 异常拦截**
"""

from __future__ import annotations

import asyncio
import logging
import operator as op_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Exception                                                                   #
# --------------------------------------------------------------------------- #

class SecurityGuardrailException(Exception):
    """FSM 安全护栏异常 —— 通用拦截，不含业务词汇。"""


# --------------------------------------------------------------------------- #
#  Operator Lookup                                                             #
# --------------------------------------------------------------------------- #

_OP_MAP: dict[str, Any] = {
    ">=": op_mod.ge, "<=": op_mod.le,
    ">":  op_mod.gt, "<":  op_mod.lt,
    "==": op_mod.eq, "!=": op_mod.ne,
}


# --------------------------------------------------------------------------- #
#  ShadowFSM — 通用规则引擎                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class ShadowFSM:
    """通用有限状态机引擎。

    Parameters
    ----------
    lab_id : str
        实验柜标识。
    states : list[str]
        所有合法状态名。
    initial_state : str
        初始状态。
    transitions : dict[tuple[str, str], str]
        转移规则表 ``{(from_state, action): to_state}``。
    constraints : list[dict]
        安全约束列表，每项 ``{"action", "metric", "operator", "threshold"}``。
    """

    lab_id: str
    states: list[str] = field(default_factory=list)
    initial_state: str = "IDLE"
    transitions: dict[tuple[str, str], str] = field(default_factory=dict)
    constraints: list[dict[str, Any]] = field(default_factory=list)

    _state: str = field(default="", init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.initial_state not in self.states:
            if self.states:
                raise ValueError(
                    f"初始状态 '{self.initial_state}' 不在状态集合 {self.states} 中"
                )
        self._state = self.initial_state

    # ---- 工厂方法 --------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path, lab_id: str = "default") -> ShadowFSM:
        """从 YAML 规则文件创建 FSM 实例。

        YAML 格式::

            states: [STATE_A, STATE_B, ...]
            initial_state: STATE_A
            transitions:
              - {from: STATE_A, action: DO_SOMETHING, to: STATE_B}
            safety_constraints:
              - {action: DO_SOMETHING, metric: sensor_value, operator: ">=", threshold: 100.0}
        """
        import yaml  # noqa: delayed import

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        states = rules.get("states", [])
        initial = rules.get("initial_state", states[0] if states else "IDLE")

        transitions: dict[tuple[str, str], str] = {}
        for t in rules.get("transitions", []):
            key = (t["from"], t["action"])
            transitions[key] = t["to"]

        constraints = rules.get("safety_constraints", [])

        fsm = cls(
            lab_id=lab_id,
            states=states,
            initial_state=initial,
            transitions=transitions,
            constraints=constraints,
        )
        logger.info(
            "[%s] FSM 引擎加载: %d 状态, %d 转移规则, %d 安全约束  (来源: %s)",
            lab_id, len(states), len(transitions), len(constraints), path.name,
        )
        return fsm

    @classmethod
    def from_dict(cls, rules: dict[str, Any], lab_id: str = "default") -> ShadowFSM:
        """从字典创建 FSM 实例（与 YAML 结构相同）。"""
        states = rules.get("states", [])
        initial = rules.get("initial_state", states[0] if states else "IDLE")

        transitions: dict[tuple[str, str], str] = {}
        for t in rules.get("transitions", []):
            key = (t["from"], t["action"])
            transitions[key] = t["to"]

        constraints = rules.get("safety_constraints", [])

        return cls(
            lab_id=lab_id,
            states=states,
            initial_state=initial,
            transitions=transitions,
            constraints=constraints,
        )

    # ---- 状态查询 --------------------------------------------------------- #

    @property
    def current_state(self) -> str:
        return self._state

    # ---- 校验与转移 ------------------------------------------------------- #

    async def validate_and_transition(
        self,
        action: str,
        params: dict[str, Any] | None = None,
        telemetry_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """校验转移合法性 + 安全约束 → 执行状态迁移。

        Parameters
        ----------
        action : str
            要执行的动作名。
        params : dict, optional
            动作参数（传递给约束检查）。
        telemetry_snapshot : dict, optional
            当前遥测快照（用于安全约束校验）。

        Returns
        -------
        str
            迁移后的新状态名。

        Raises
        ------
        SecurityGuardrailException
            转移不合法或安全约束被触发时抛出。
        """
        params = params or {}
        telemetry_snapshot = telemetry_snapshot or {}

        async with self._lock:
            transition_key = (self._state, action)

            # 1) 转移合法性校验
            if transition_key not in self.transitions:
                raise SecurityGuardrailException(
                    f"[{self.lab_id}] FSM 拒绝: 状态 '{self._state}' "
                    f"下不允许执行 '{action}'"
                )

            # 2) 安全约束校验
            for constraint in self.constraints:
                if constraint.get("action") != action:
                    continue
                metric_key = constraint["metric"]
                op_str = constraint["operator"]
                threshold = constraint["threshold"]
                current_value = telemetry_snapshot.get(metric_key)
                if current_value is not None:
                    op_func = _OP_MAP.get(op_str)
                    if op_func and op_func(current_value, threshold):
                        raise SecurityGuardrailException(
                            f"[{self.lab_id}] 安全拦截: {metric_key}={current_value} "
                            f"{op_str} {threshold}，禁止执行 '{action}'"
                        )

            # 3) 状态迁移
            old_state = self._state
            self._state = self.transitions[transition_key]
            logger.info(
                "[%s] FSM 状态迁移: %s -[%s]-> %s",
                self.lab_id, old_state, action, self._state,
            )
            return self._state
