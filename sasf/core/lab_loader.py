"""
AstroSASF · Core · LabLoader (V7.1 — Catalog-Based Lab Discovery)
================================================================
基于 `labs_catalog/` 目录的实验舱自动发现与加载器。

设计理念：
- **零硬编码**：所有实验舱配置由文件系统驱动
- **动态注册**：启动时自动扫描、加载、实例化
- **模块化隔离**：每个实验舱有独立的配置、工具和宏定义

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
Version: 7.1
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sasf.core.config_loader import SASFConfig, load_config
from sasf.core.environment import LaboratoryEnvironment
from sasf.physics.interlock_engine import InterlockEngine, InterlockRule

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


@dataclass
class LoadedLab:
    """已加载的实验舱完整实例。"""
    descriptor: LabDescriptor
    environment: LaboratoryEnvironment
    interlock_engine: InterlockEngine


# --------------------------------------------------------------------------- #
#  LabLoader — 实验室目录加载器                                               #
# --------------------------------------------------------------------------- #

@dataclass
class LabLoader:
    """实验舱目录加载器 (V7.1)。

    扫描 `labs_catalog/` 目录，自动发现并加载所有实验舱配置。

    Example
    -------
    >>> loader = LabLoader(catalog_dir="labs_catalog")
    >>> loaded_labs = await loader.discover_and_load(config)
    >>> for lab in loaded_labs:
    ...     print(f"已加载: {lab.descriptor.lab_id}")
    """

    catalog_dir: Path
    _discovered_labs: dict[str, LabDescriptor] = field(default_factory=dict, init=False)
    _loaded_labs: dict[str, LoadedLab] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.catalog_dir = Path(self.catalog_dir)
        if not self.catalog_dir.exists():
            raise LabLoaderError(f"实验室目录不存在: {self.catalog_dir}")

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
        labs_path = self.catalog_dir / "labs_catalog"

        if not labs_path.exists():
            # 兼容：也可能直接在 catalog_dir 下
            labs_path = self.catalog_dir

        for entry in labs_path.iterdir():
            if not entry.is_dir():
                continue

            # 跳过 shared 目录
            if entry.name == "shared":
                continue

            # 检查配置文件
            config_path = entry / "lab_config.yaml"
            if not config_path.exists():
                logger.warning("[LabLoader] 跳过 %s：缺少 lab_config.yaml", entry.name)
                continue

            try:
                descriptor = self._parse_lab_config(entry.name, config_path)
                self._discovered_labs[entry.name] = descriptor
                logger.info("[LabLoader] 发现实验舱: %s (%s)", entry.name, descriptor.description)
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

        # 解析子系统配置
        subsystems_raw = raw_config.get("subsystems", {})
        subsystem_configs: dict[str, list[str]] = {}
        for subsystem, states in subsystems_raw.items():
            if isinstance(states, list):
                subsystem_configs[subsystem] = states
            elif isinstance(states, dict):
                subsystem_configs[subsystem] = states.get("states", [])

        # 解析初始状态
        initial_states = raw_config.get("initial_states", {})

        # 解析联锁规则
        interlocks = raw_config.get("interlocks", [])

        # 解析宏定义
        macros = raw_config.get("macros", {})

        # 解析技能列表
        skills = raw_config.get("skills", [])

        # 检查自定义工具
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
        config: SASFConfig,
        shared_tools_module: Any = None,
    ) -> LoadedLab:
        """加载单个实验舱。

        Parameters
        ----------
        descriptor : LabDescriptor
            实验舱描述符
        config : SASFConfig
            全局配置
        shared_tools_module : module, optional
            共享工具模块

        Returns
        -------
        LoadedLab
            已加载的实验舱实例
        """
        logger.info("[LabLoader] 加载实验舱: %s", descriptor.lab_id)

        # ── Step 1: 创建 InterlockEngine ── #
        interlocks = [
            InterlockRule(
                condition=rule["condition"],
                message=rule.get("message", ""),
                scope=rule.get("scope"),
            )
            for rule in descriptor.interlocks
        ]

        engine = InterlockEngine(
            lab_id=descriptor.lab_id,
            subsystems=descriptor.subsystem_configs,
            initial_states=descriptor.initial_states,
            interlocks=interlocks,
        )

        # ── Step 2: 创建实验舱环境 ── #
        # 初始化遥测数据（包含公共变量）
        initial_telemetry = {
            "oxygen_level": 21.0,
            "co2_level": 0.04,
            "pressure": 101.325,
        }

        env = LaboratoryEnvironment(
            lab_id=descriptor.lab_id,
            config=config,
            engine=engine,
            initial_telemetry=initial_telemetry,
        )

        # ── Step 3: 注册工具 ── #
        # 注册共享工具（如果提供）
        if shared_tools_module is not None:
            self._register_tools_from_module(
                env.registry,
                shared_tools_module,
                descriptor.lab_id,
            )

        # 注册自定义工具
        if descriptor.custom_tools_path:
            custom_module = self._import_tools_module(descriptor.custom_tools_path)
            if custom_module:
                self._register_tools_from_module(
                    env.registry,
                    custom_module,
                    descriptor.lab_id,
                )

        # ── Step 4: 绑定宏 ── #
        for macro_name, macro_info in descriptor.macros.items():
            try:
                env.registry.bind_macro(
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

        # ── Step 5: 注册技能 ── #
        # 技能列表已存储在 descriptor.skills 中，由 SkillLoader 加载

        logger.info(
            "[LabLoader] ✅ 实验舱 '%s' 加载完成: %d 工具, %d 宏",
            descriptor.lab_id,
            env.registry.count,
            env.registry.macro_count,
        )

        return LoadedLab(
            descriptor=descriptor,
            environment=env,
            interlock_engine=engine,
        )

    def _import_tools_module(self, tools_path: Path) -> Any | None:
        """动态导入工具模块。"""
        try:
            module_name = f"labs_tools_{tools_path.parent.name}_{tools_path.stem}"
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
        registry: Any,
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

            # 注册为 MCP Tool
            registry._register_function(obj)
            logger.debug("[LabLoader] [%s] 注册工具: %s", lab_id, name)

    # --------------------------------------------------------------------------- #
    #  批量加载                                                                  #
    # --------------------------------------------------------------------------- #

    async def discover_and_load(
        self,
        config: SASFConfig | None = None,
        config_path: str | Path | None = None,
    ) -> dict[str, LoadedLab]:
        """发现并加载所有实验舱。

        Parameters
        ----------
        config : SASFConfig, optional
            全局配置
        config_path : str | Path, optional
            配置文件路径

        Returns
        -------
        dict[str, LoadedLab]
            lab_id → LoadedLab 映射
        """
        # 加载配置
        if config is None:
            config = load_config(config_path)

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
                loaded = await self.load_lab(descriptor, config, shared_tools_module)
                self._loaded_labs[lab_id] = loaded
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
    def loaded_labs(self) -> dict[str, LoadedLab]:
        """获取所有已加载的实验舱。"""
        return self._loaded_labs

    def get_lab(self, lab_id: str) -> LoadedLab | None:
        """获取指定实验舱。"""
        return self._loaded_labs.get(lab_id)

    def list_lab_ids(self) -> list[str]:
        """列出所有已加载的实验舱 ID。"""
        return list(self._loaded_labs.keys())

    def get_lab_metadata(self, lab_id: str) -> dict[str, Any] | None:
        """获取实验舱元数据（用于 API 响应）。"""
        loaded = self._loaded_labs.get(lab_id)
        if loaded is None:
            return None

        descriptor = loaded.descriptor
        env = loaded.environment

        return {
            "lab_id": descriptor.lab_id,
            "name": descriptor.name,
            "description": descriptor.description,
            "tools": env.registry.list_tools(),
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
                "current_states": loaded.interlock_engine.current_states,
            },
            "skills": descriptor.skills,
            "stats": {
                "tool_count": env.registry.count,
                "macro_count": env.registry.macro_count,
            },
        }


# --------------------------------------------------------------------------- #
#  工厂函数                                                                   #
# --------------------------------------------------------------------------- #

def create_loader(catalog_dir: str | Path | None = None) -> LabLoader:
    """创建实验室加载器。"""
    if catalog_dir is None:
        # 默认使用项目根目录下的 labs_catalog
        catalog_dir = Path(__file__).parent.parent.parent / "labs_catalog"

    return LabLoader(catalog_dir=Path(catalog_dir))


__all__ = [
    "LabLoader",
    "LabDescriptor",
    "LoadedLab",
    "LabLoaderError",
    "LabConfigNotFoundError",
    "LabToolImportError",
    "create_loader",
]
