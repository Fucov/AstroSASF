"""
AstroSASF · Labs · LabLoader (V7.2 — Kernel-Centric Lab Discovery)
===================================================================
基于 `labs_catalog/` 目录的实验舱自动发现与加载器。

设计理念：
- **零硬编码**：所有实验舱配置由文件系统驱动
- **动态注册**：启动时自动扫描、加载、实例化
- **模块化隔离**：每个实验舱有独立的配置、工具和宏定义
- **内核解耦**：仅创建内核级组件（Registry、Engine、Bus），
  不依赖认知层（Cognition/Graph）；认知编排由 Interface 层负责

目录结构：
```
labs_catalog/
├── shared/                    # 公共工具（可选）
│   └── shared_tools.py
├── Lab-Alpha/                # 实验舱 A
│   ├── lab_config.yaml        # FSM、宏、技能配置
│   └── custom_tools.py        # 专属工具（可选）
└── Lab-Beta/                 # 实验舱 B
     └── ...
```

Author: AstroSASF Team
Version: 7.2
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from infra.llm.config_loader import SASFConfig

from labs.interlock_engine import InterlockEngine, InterlockRule
from labs.mcp_registry import MCPToolRegistry
from scheduler.telemetry import TelemetryBus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Exceptions                                                                  #
# --------------------------------------------------------------------------- #

class LabLoaderError(Exception):
    """实验室加载器基础异常。"""
    pass


class LabConfigNotFoundError(LabLoaderError):
    """实验舱配置文件未找到。"""
    pass


class LabToolImportError(LabLoaderError):
    """工具模块导入失败。"""
    pass


# --------------------------------------------------------------------------- #
#  LabDescriptor — 实验室元数据描述符                                           #
# --------------------------------------------------------------------------- #

@dataclass
class LabDescriptor:
    """实验舱元数据描述符（从 lab_config.yaml 解析）。"""
    lab_id: str
    name: str
    description: str
    subsystem_configs: dict[str, list[str]]
    initial_states: dict[str, str]
    interlocks: list[dict[str, Any]]
    macros: dict[str, dict[str, Any]]
    skills: list[str]
    config_path: Path
    custom_tools_path: Path | None = None


# --------------------------------------------------------------------------- #
#  LabContext — 内核级实验舱上下文（不含认知层）                               #
# --------------------------------------------------------------------------- #

@dataclass
class LabContext:
    """内核级实验舱运行时上下文 (V7.2)。

    仅包含调度/内核组件：
    - registry : MCPToolRegistry — 工具注册（含 Guard 校验）
    - engine   : InterlockEngine  — 正交联锁引擎
    - bus      : TelemetryBus     — 遥测数据总线

    认知层（LangGraph、SkillCatalog）由 Interface 层按需注入。
    """
    descriptor: LabDescriptor
    registry: MCPToolRegistry
    engine: InterlockEngine
    bus: TelemetryBus


# --------------------------------------------------------------------------- #
#  LabLoader — 实验室目录加载器                                               #
# --------------------------------------------------------------------------- #

@dataclass
class LabLoader:
    """实验舱目录加载器 (V7.2)。

    扫描 `labs_catalog/` 目录，自动发现并加载所有实验舱内核组件。

    Example
    -------
    >>> loader = LabLoader(catalog_dir="labs_catalog")
    >>> loaded_labs = await loader.discover_and_load(config)
    >>> for lab in loaded_labs.values():
    ...     print(f"已加载: {lab.descriptor.lab_id}")
    """

    catalog_dir: Path
    _discovered_labs: dict[str, LabDescriptor] = field(default_factory=dict, init=False)
    _loaded_labs: dict[str, LabContext] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.catalog_dir = Path(self.catalog_dir)

    # --------------------------------------------------------------------------- #
    #  发现阶段 (Discovery)                                                        #
    # --------------------------------------------------------------------------- #

    def discover_labs(self) -> dict[str, LabDescriptor]:
        """扫描目录，发现所有实验舱配置。

        Returns
        -------
        dict[str, LabDescriptor]
            lab_id → LabDescriptor 映射
        """
        self._discovered_labs.clear()
        # 支持两种调用方式：
        #   create_loader("labs_catalog")              → 扫描 labs_catalog/
        #   create_loader("demo/assets/labs")         → 扫描 demo/assets/labs/
        #   create_loader("demo/assets/labs/DemoBio")  → 只加载 DemoBio
        labs_path = self.catalog_dir
        if not labs_path.exists():
            raise LabLoaderError(f"实验室目录不存在: {labs_path}")

        for entry in labs_path.iterdir():
            if not entry.is_dir():
                continue
            if entry.name == "shared":
                continue

            config_path = entry / "lab_config.yaml"
            if not config_path.exists():
                logger.warning("[LabLoader] 跳过 %s：缺少 lab_config.yaml", entry.name)
                continue

            try:
                descriptor = self._parse_lab_config(entry.name, config_path)
                self._discovered_labs[entry.name] = descriptor
                logger.info(
                    "[LabLoader] 发现实验舱: %s (%s)",
                    entry.name, descriptor.description,
                )
            except Exception as exc:
                logger.error("[LabLoader] 解析 %s 失败: %s", entry.name, exc)
                continue

        logger.info("[LabLoader] 发现 %d 个实验舱", len(self._discovered_labs))
        return self._discovered_labs

    def _parse_lab_config(self, lab_id: str, config_path: Path) -> LabDescriptor:
        """解析实验舱配置文件。"""
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        name = raw_config.get("name", lab_id)
        description = raw_config.get("description", "")

        subsystems_raw = raw_config.get("subsystems", {})
        subsystem_configs: dict[str, list[str]] = {}
        for subsystem, states in subsystems_raw.items():
            if isinstance(states, list):
                subsystem_configs[subsystem] = states
            elif isinstance(states, dict):
                subsystem_configs[subsystem] = states.get("states", [])

        initial_states = raw_config.get("initial_states", {})
        interlocks = raw_config.get("interlocks", [])
        macros = raw_config.get("macros", {})
        skills = raw_config.get("skills", [])

        custom_tools_path = None
        custom_tools_file = config_path.parent / "custom_tools.py"
        if custom_tools_file.exists():
            custom_tools_path = custom_tools_file

        return LabDescriptor(
            lab_id=lab_id,
            name=name,
            description=description,
            subsystem_configs=subsystem_configs,
            initial_states=initial_states,
            interlocks=interlocks,
            macros=macros,
            skills=skills,
            config_path=config_path,
            custom_tools_path=custom_tools_path,
        )

    # --------------------------------------------------------------------------- #
    #  加载阶段 (Loading)                                                         #
    # --------------------------------------------------------------------------- #

    async def load_lab(
        self,
        descriptor: LabDescriptor,
        shared_tools_module: Any = None,
    ) -> LabContext:
        """加载单个实验舱内核组件。

        Parameters
        ----------
        descriptor : LabDescriptor
            实验舱描述符
        shared_tools_module : module, optional
            共享工具模块

        Returns
        -------
        LabContext
            已加载的内核级实验舱上下文
        """
        logger.info("[LabLoader] 加载实验舱: %s", descriptor.lab_id)

        # ── Step 1: 创建 InterlockEngine ── #
        interlocks = [
            InterlockRule(
                rule_id=f"rule_{i}",
                condition=rule["condition"],
                message=rule.get("message", ""),
                scope=rule.get("scope"),
            )
            for i, rule in enumerate(descriptor.interlocks)
        ]

        engine = InterlockEngine(
            lab_id=descriptor.lab_id,
            subsystems=descriptor.subsystem_configs,
            initial_states=descriptor.initial_states,
            interlocks=interlocks,
        )

        # ── Step 2: 创建 TelemetryBus ── #
        initial_telemetry = {
            "oxygen_level": 21.0,
            "co2_level": 0.04,
            "pressure": 101.325,
        }
        bus = TelemetryBus(
            lab_id=descriptor.lab_id,
            initial_state=initial_telemetry,
        )
        engine.bind_telemetry_bus(bus)

        # ── Step 3: 创建 MCPToolRegistry ── #
        registry = MCPToolRegistry(lab_id=descriptor.lab_id)

        # ── Step 4: 注册共享工具 ── #
        if shared_tools_module is not None:
            self._register_tools_from_module(registry, shared_tools_module, descriptor.lab_id)

        # ── Step 5: 注册自定义工具 ── #
        if descriptor.custom_tools_path:
            custom_module = self._import_tools_module(descriptor.custom_tools_path)
            if custom_module:
                self._register_tools_from_module(
                    registry, custom_module, descriptor.lab_id,
                )

        # ── Step 6: 绑定宏 ── #
        for macro_name, macro_info in descriptor.macros.items():
            try:
                registry.bind_macro(
                    macro_name=macro_name,
                    target_tool=macro_info["target"],
                    preset_params=macro_info.get("preset", {}),
                    description=macro_info.get("description"),
                )
            except Exception as exc:
                logger.warning(
                    "[LabLoader] 绑定宏 '%s' 失败: %s",
                    macro_name, exc,
                )

        logger.info(
            "[LabLoader] ✅ 实验舱 '%s' 加载完成: %d 工具, %d 宏",
            descriptor.lab_id,
            registry.count,
            registry.macro_count,
        )

        return LabContext(
            descriptor=descriptor,
            registry=registry,
            engine=engine,
            bus=bus,
        )

    def _import_tools_module(self, tools_path: Path) -> Any | None:
        """动态导入工具模块。"""
        try:
            module_name = (
                f"labs_tools_{tools_path.parent.name}_{tools_path.stem}"
            )
            spec = importlib.util.spec_from_file_location(module_name, tools_path)
            if spec is None or spec.loader is None:
                raise LabToolImportError(f"无法加载模块: {tools_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        except Exception as exc:
            logger.error(
                "[LabLoader] 导入工具模块失败 %s: %s",
                tools_path, exc,
            )
            return None

    def _register_tools_from_module(
        self,
        registry: MCPToolRegistry,
        module: Any,
        lab_id: str,
    ) -> None:
        """从模块中注册所有 async 函数为 MCP Tools。"""
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if not asyncio.iscoroutinefunction(obj):
                continue
            registry._register_function(obj)
            logger.debug("[LabLoader] [%s] 注册工具: %s", lab_id, name)

    # --------------------------------------------------------------------------- #
    #  批量加载                                                                  #
    # --------------------------------------------------------------------------- #

    async def discover_and_load(
        self,
    ) -> dict[str, LabContext]:
        """发现并加载所有实验舱。

        Returns
        -------
        dict[str, LabContext]
            lab_id → LabContext 映射
        """
        # 加载共享工具模块
        shared_tools_path = self.catalog_dir / "shared" / "shared_tools.py"
        shared_tools_module = None
        if shared_tools_path.exists():
            shared_tools_module = self._import_tools_module(shared_tools_path)
            if shared_tools_module:
                logger.info("[LabLoader] 已加载共享工具")

        # 发现实验室
        if not self._discovered_labs:
            self.discover_labs()

        # 加载每个实验室
        self._loaded_labs.clear()
        for lab_id, descriptor in self._discovered_labs.items():
            try:
                ctx = await self.load_lab(descriptor, shared_tools_module)
                self._loaded_labs[lab_id] = ctx
            except Exception as exc:
                logger.error("[LabLoader] 加载实验舱 '%s' 失败: %s", lab_id, exc)
                continue

        logger.info(
            "[LabLoader] 加载完成: %d/%d 个实验舱",
            len(self._loaded_labs), len(self._discovered_labs),
        )
        return self._loaded_labs

    # --------------------------------------------------------------------------- #
    #  查询接口                                                                  #
    # --------------------------------------------------------------------------- #

    @property
    def loaded_labs(self) -> dict[str, LabContext]:
        """获取所有已加载的实验舱。"""
        return self._loaded_labs

    def get_lab(self, lab_id: str) -> LabContext | None:
        """获取指定实验舱。"""
        return self._loaded_labs.get(lab_id)

    def list_lab_ids(self) -> list[str]:
        """列出所有已加载的实验舱 ID。"""
        return list(self._loaded_labs.keys())

    def get_lab_metadata(self, lab_id: str) -> dict[str, Any] | None:
        """获取实验舱元数据（用于 API 响应）。"""
        ctx = self._loaded_labs.get(lab_id)
        if ctx is None:
            return None

        descriptor = ctx.descriptor
        registry = ctx.registry

        return {
            "lab_id": descriptor.lab_id,
            "name": descriptor.name,
            "description": descriptor.description,
            "tools": registry.list_tools(),
            "macros": [
                {
                    "name": name,
                    "target": info["target"],
                    "preset": info.get("preset", {}),
                    "description": info.get("description", ""),
                }
                for name, info in descriptor.macros.items()
            ],
            "fsm": {
                "subsystems": descriptor.subsystem_configs,
                "initial_states": descriptor.initial_states,
                "interlocks": [
                    {
                        "condition": rule["condition"],
                        "message": rule.get("message", ""),
                        "scope": rule.get("scope"),
                    }
                    for rule in descriptor.interlocks
                ],
                "current_states": ctx.engine.current_states,
            },
            "skills": descriptor.skills,
            "stats": {
                "tool_count": registry.count,
                "macro_count": registry.macro_count,
            },
        }


# --------------------------------------------------------------------------- #
#  工厂函数                                                                   #
# --------------------------------------------------------------------------- #

def create_loader(catalog_dir: str | Path | None = None) -> LabLoader:
    """创建实验室加载器。"""
    if catalog_dir is None:
        catalog_dir = Path(__file__).parent.parent.parent / "labs_catalog"
    return LabLoader(catalog_dir=Path(catalog_dir))


__all__ = [
    "LabLoader",
    "LabDescriptor",
    "LabContext",
    "LabLoaderError",
    "LabConfigNotFoundError",
    "LabToolImportError",
    "create_loader",
]
