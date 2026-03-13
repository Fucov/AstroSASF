"""
AstroSASF · Cognition · SkillLoader (V4.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI 标准 Skill 加载器 —— 解析 ``SKILL.md`` 知识套件。

V4.2 核心区分：
- **MCP Tools** = 底层原子操作接口（由 ``middleware/mcp_registry.py`` 管理）
- **OpenAI Skills** = 认知层 SOP 知识套件（**此模块管理**）

``OpenAISkillCatalog`` 扫描指定目录下的 ``SKILL.md`` 文件，解析
YAML Frontmatter（name, description）和 Markdown 正文（Workflow, When to use），
生成可注入 LLM System Prompt 的结构化知识片段。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  SkillEntry — 单个 Skill 知识项                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SkillEntry:
    """解析后的 OpenAI Skill 知识条目。"""
    name: str
    description: str
    workflow: str           # Markdown 正文（SOP 工作流）
    source_path: str        # SKILL.md 路径


# --------------------------------------------------------------------------- #
#  YAML Frontmatter 解析                                                       #
# --------------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析 SKILL.md 的 YAML Frontmatter + Markdown 正文。

    Returns
    -------
    tuple[dict, str]
        (frontmatter 键值对, markdown 正文)
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    yaml_block = match.group(1)
    body = text[match.end():]

    # 简易 YAML 解析（key: value 格式）
    metadata: dict[str, str] = {}
    for line in yaml_block.strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            metadata[key.strip()] = val.strip().strip('"').strip("'")

    return metadata, body.strip()


# --------------------------------------------------------------------------- #
#  OpenAISkillCatalog                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class OpenAISkillCatalog:
    """OpenAI Skill 知识目录 —— 扫描并加载 ``SKILL.md`` 文件。

    Parameters
    ----------
    catalog_dir : str | Path
        Skill 目录路径（如 ``./skills_catalog/``）。
    """

    catalog_dir: Path
    _skills: dict[str, SkillEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.catalog_dir = Path(self.catalog_dir)
        if self.catalog_dir.exists():
            self._scan()
        else:
            logger.warning(
                "SkillCatalog: 目录不存在: %s", self.catalog_dir,
            )

    def _scan(self) -> None:
        """递归扫描目录下所有 SKILL.md 文件。"""
        skill_files = list(self.catalog_dir.rglob("SKILL.md"))
        logger.info(
            "📚 SkillCatalog: 扫描 '%s' → 发现 %d 个 SKILL.md",
            self.catalog_dir, len(skill_files),
        )

        for skill_path in sorted(skill_files):
            try:
                text = skill_path.read_text(encoding="utf-8")
                metadata, body = _parse_frontmatter(text)

                name = metadata.get("name", skill_path.parent.name)
                description = metadata.get("description", "")

                entry = SkillEntry(
                    name=name,
                    description=description,
                    workflow=body,
                    source_path=str(skill_path),
                )
                self._skills[name] = entry

                logger.info(
                    "📚 SkillCatalog: ✅ 加载 Skill '%s' — %s",
                    name, description,
                )
                logger.info(
                    "📚    来源: %s  |  SOP 正文 %d 字符",
                    skill_path.relative_to(self.catalog_dir),
                    len(body),
                )

            except Exception as exc:
                logger.warning(
                    "📚 SkillCatalog: 加载 '%s' 失败: %s",
                    skill_path, exc,
                )

    # ---- 查询 ------------------------------------------------------------- #

    def get_skill(self, name: str) -> SkillEntry | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, str]]:
        """返回所有已加载 Skill 的摘要。"""
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def get_skill_context(self, skill_name: str) -> str:
        """生成可注入 LLM System Prompt 的知识上下文。

        返回格式化的 Markdown 片段，包含 Skill 描述和 SOP 工作流。
        """
        entry = self._skills.get(skill_name)
        if entry is None:
            return ""

        return (
            f"## Skill: {entry.name}\n"
            f"**Description**: {entry.description}\n\n"
            f"### Standard Operating Procedure (SOP)\n"
            f"{entry.workflow}\n"
        )

    def get_all_skills_context(self) -> str:
        """生成所有已加载 Skill 的完整知识上下文。"""
        if not self._skills:
            return ""

        sections: list[str] = [
            "# 已加载的 OpenAI Skills (标准操作程序)\n"
            "以下 Skill 告诉你**如何**组合调用底层 MCP Tools 来完成复杂任务。\n"
        ]
        for entry in self._skills.values():
            sections.append(self.get_skill_context(entry.name))

        return "\n---\n".join(sections)

    @property
    def count(self) -> int:
        return len(self._skills)
