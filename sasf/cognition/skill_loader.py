"""
AstroSASF · Cognition · SkillLoader (V6.2 — Semantic Routing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI Skill 知识套件加载器。

V6.2 变化：
- **删除** BM25/TF-IDF 算法（语义路由由 LLM router_node 承担）
- 保留纯知识目录：扫描 → 解析 → 按名称提供 SOP 内容
- Macro 感知上下文保留
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  SkillEntry                                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SkillEntry:
    """解析后的 OpenAI Skill 知识条目。"""
    name: str
    description: str
    workflow: str
    source_path: str


# --------------------------------------------------------------------------- #
#  YAML Frontmatter 解析                                                       #
# --------------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    yaml_block = match.group(1)
    body = text[match.end():]

    metadata: dict[str, str] = {}
    for line in yaml_block.strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            metadata[key.strip()] = val.strip().strip('"').strip("'")

    return metadata, body.strip()


# --------------------------------------------------------------------------- #
#  OpenAISkillCatalog (V6.2 — Pure Knowledge Directory)                        #
# --------------------------------------------------------------------------- #

@dataclass
class OpenAISkillCatalog:
    """OpenAI Skill 知识目录 (V6.2)。

    不再包含检索算法，仅负责：
    1. 扫描并解析 ``skills_catalog/*/SKILL.md``
    2. 按名称提供 SOP 全文
    3. 生成 Macro 感知提示

    语义路由由 ``graph_builder.router_node`` 中的 LLM 承担。
    """

    catalog_dir: Path
    registry: object | None = None
    _skills: dict[str, SkillEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.catalog_dir = Path(self.catalog_dir)
        if self.catalog_dir.exists():
            self._scan()
        else:
            logger.warning("SkillCatalog: 目录不存在: %s", self.catalog_dir)

    def _scan(self) -> None:
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

            except Exception as exc:
                logger.warning(
                    "📚 SkillCatalog: 加载 '%s' 失败: %s", skill_path, exc,
                )

    # ---- 查询 ------------------------------------------------------------- #

    def get_skill(self, name: str) -> SkillEntry | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def get_skill_names_and_descriptions(self) -> list[dict[str, str]]:
        """返回所有 Skill 的 name + description（供 Router LLM 使用）。"""
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def get_skill_context(self, skill_name: str) -> str:
        """获取指定 Skill 的完整 SOP 上下文。"""
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
        """生成所有 Skill 的知识上下文 + Macro 映射提示。"""
        if not self._skills:
            return ""

        sections: list[str] = []
        for entry in self._skills.values():
            sections.append(self.get_skill_context(entry.name))

        context = "\n---\n".join(sections)

        macro_hint = self._build_macro_hint()
        if macro_hint:
            context += "\n\n" + macro_hint

        return context

    def _build_macro_hint(self) -> str:
        if self.registry is None:
            return ""
        try:
            macros = self.registry.get_macros()  # type: ignore[attr-defined]
        except AttributeError:
            return ""
        if not macros:
            return ""

        lines = [
            "## 🔗 可用宏指令 (Macro)\n"
            "以下 Macro 是底层 MCP Tool 的参数预绑定快捷方式。\n"
            "**优先使用 Macro** 代替手动指定参数的底层 Tool：\n"
        ]
        for macro_name, info in macros.items():
            lines.append(
                f"- `{macro_name}` → {info['target']}({info['preset']}) "
                f"— {info['description']}"
            )

        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._skills)
