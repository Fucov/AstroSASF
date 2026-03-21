"""
AstroSASF · Cognition · SkillLoader (V6.0 — Edge-RAG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI Skill 知识套件加载 + **轻量级边缘 RAG 检索器**。

V6.0 核心变化:
- 纯 Python 标准库 BM25-lite 相似度打分（零第三方依赖）
- ``retrieve_relevant_skills(query, top_k)`` 动态检索相关 SOP
- Macro 感知上下文保留
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
#  BM25-Lite: 零依赖轻量级文本相关性评分                                          #
# --------------------------------------------------------------------------- #

# 中文 / 英文分词正则（按非字母数字汉字拆分）
_TOKENIZE_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

# BM25 超参数
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """简易分词：按词边界拆分，统一小写。"""
    return [t.lower() for t in _TOKENIZE_RE.findall(text)]


@dataclass
class _BM25Index:
    """纯 Python BM25 索引（零第三方依赖）。

    针对太空站边缘计算节点极度受限算力设计。
    """
    doc_ids: list[str] = field(default_factory=list)
    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    avg_dl: float = 0.0
    idf_cache: dict[str, float] = field(default_factory=dict)
    _n_docs: int = 0

    def add_document(self, doc_id: str, text: str) -> None:
        """添加文档到索引。"""
        tokens = _tokenize(text)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(tokens)
        self.doc_lens.append(len(tokens))
        self._n_docs = len(self.doc_ids)
        # 重算 avg_dl
        self.avg_dl = sum(self.doc_lens) / self._n_docs if self._n_docs else 0
        # 清缓存（文档变化后需重算 IDF）
        self.idf_cache.clear()

    def _compute_idf(self, term: str) -> float:
        """IDF = ln((N - df + 0.5) / (df + 0.5) + 1)"""
        if term in self.idf_cache:
            return self.idf_cache[term]

        df = sum(1 for tokens in self.doc_tokens if term in tokens)
        idf = math.log(
            (self._n_docs - df + 0.5) / (df + 0.5) + 1.0
        )
        self.idf_cache[term] = idf
        return idf

    def score(self, query: str) -> list[tuple[str, float]]:
        """对 query 计算每个文档的 BM25 分数。

        Returns
        -------
        list of (doc_id, score) 按分数降序排列
        """
        query_tokens = _tokenize(query)
        if not query_tokens or not self._n_docs:
            return [(did, 0.0) for did in self.doc_ids]

        results: list[tuple[str, float]] = []
        for i, (doc_id, tokens) in enumerate(zip(self.doc_ids, self.doc_tokens)):
            dl = self.doc_lens[i]
            tf_map = Counter(tokens)
            doc_score = 0.0

            for term in query_tokens:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._compute_idf(term)
                # BM25 公式
                numerator = tf * (_BM25_K1 + 1)
                denominator = tf + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * dl / self.avg_dl
                )
                doc_score += idf * (numerator / denominator)

            results.append((doc_id, doc_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# --------------------------------------------------------------------------- #
#  OpenAISkillCatalog (V6.0)                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class OpenAISkillCatalog:
    """OpenAI Skill 知识目录 + Edge-RAG 检索器 (V6.0)。

    Parameters
    ----------
    catalog_dir : str | Path
    registry : MCPToolRegistry | None
        可选，用于获取 Macro 映射信息。
    """

    catalog_dir: Path
    registry: object | None = None
    _skills: dict[str, SkillEntry] = field(default_factory=dict, init=False)
    _bm25: _BM25Index = field(default_factory=_BM25Index, init=False)

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

                # 建立 BM25 索引（合并 name + description + workflow）
                index_text = f"{name} {description} {body}"
                self._bm25.add_document(name, index_text)

                logger.info(
                    "📚 SkillCatalog: ✅ 加载 Skill '%s' — %s",
                    name, description,
                )

            except Exception as exc:
                logger.warning(
                    "📚 SkillCatalog: 加载 '%s' 失败: %s", skill_path, exc,
                )

        logger.info(
            "📚 SkillCatalog: BM25 索引构建完成 (%d 文档, avg_dl=%.0f tokens)",
            self._bm25._n_docs, self._bm25.avg_dl,
        )

    # ---- Edge-RAG 检索 --------------------------------------------------- #

    def retrieve_relevant_skills(
        self,
        query: str,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """基于 BM25-lite 的轻量级边缘 RAG 检索。

        Parameters
        ----------
        query : str
            用户任务描述或自然语言查询。
        top_k : int
            返回相关度最高的 K 个 SOP。

        Returns
        -------
        list[dict]
            ``[{"name", "description", "score", "context"}]``
        """
        if not self._skills:
            return []

        scored = self._bm25.score(query)
        results: list[dict[str, Any]] = []

        for doc_id, score in scored[:top_k]:
            entry = self._skills.get(doc_id)
            if entry is None:
                continue
            results.append({
                "name": entry.name,
                "description": entry.description,
                "score": round(score, 4),
                "context": self.get_skill_context(entry.name),
            })

        return results

    # ---- 查询 ------------------------------------------------------------- #

    def get_skill(self, name: str) -> SkillEntry | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def get_skill_context(self, skill_name: str) -> str:
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

        sections: list[str] = [
            "# 已加载的 OpenAI Skills (标准操作程序)\n"
            "以下 Skill 告诉你**如何**组合调用底层 MCP Tools 来完成复杂任务。\n"
        ]
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
