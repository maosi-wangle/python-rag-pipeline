from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import KnowledgeBaseProfile


@dataclass(slots=True)
class KnowledgeBaseCatalogItem:
    kb_id: str
    name: str
    description: str = ""
    language: str = "zh"
    retrieval_modes: list[str] = field(default_factory=list)
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    priority: int = 0

    @classmethod
    def from_profile(cls, profile: KnowledgeBaseProfile) -> "KnowledgeBaseCatalogItem":
        metadata = dict(profile.metadata or {})
        return cls(
            kb_id=profile.kb_id,
            name=profile.name,
            description=profile.description,
            language=profile.language,
            retrieval_modes=list(profile.retrieval_modes),
            domain=str(metadata.get("domain") or ""),
            tags=_list_of_str(metadata.get("tags")),
            examples=_list_of_str(metadata.get("examples")),
            priority=int(metadata.get("priority") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "retrieval_modes": self.retrieval_modes,
            "domain": self.domain,
            "tags": self.tags,
            "examples": self.examples,
            "priority": self.priority,
        }

    def to_prompt_line(self) -> str:
        parts = [
            f"- {self.kb_id}: {self.name}",
            f"  描述: {self.description or '(无)'}",
            f"  语言: {self.language}",
            f"  检索模式: {', '.join(self.retrieval_modes) or '(默认)'}",
        ]
        if self.domain:
            parts.append(f"  领域: {self.domain}")
        if self.tags:
            parts.append(f"  标签: {', '.join(self.tags)}")
        if self.examples:
            parts.append(f"  适合问题: {'; '.join(self.examples)}")
        return "\n".join(parts)


def build_catalog_prompt(items: list[KnowledgeBaseCatalogItem]) -> str:
    if not items:
        return "可用知识库: (无)"
    sorted_items = sorted(items, key=lambda item: item.priority, reverse=True)
    return "\n\n".join(item.to_prompt_line() for item in sorted_items)


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
