from __future__ import annotations

from ..text_utils import extract_recent_terms, normalize_whitespace, unique_preserve_order


class QueryRewriteTool:
    name = "rewrite_query"

    def rewrite(
        self,
        query: str,
        history: list[str] | None = None,
        modes: list[str] | None = None,
    ) -> list[str]:
        history = history or []
        modes = modes or []
        normalized_query = normalize_whitespace(query)
        variants = [normalized_query]
        recent_terms = extract_recent_terms(history)

        if "specific" in modes and recent_terms:
            variants.append(f"{normalized_query} {' '.join(recent_terms)}")

        if "general" in modes:
            variants.append(self._generalize(normalized_query))

        if "chunk_like" in modes:
            suffix = "定义 原因 方法 注意事项"
            if suffix not in normalized_query:
                variants.append(f"{normalized_query} {suffix}")

        return unique_preserve_order(variants)

    def _generalize(self, query: str) -> str:
        trimmed = query.replace("请详细说明", "").replace("详细", "").strip()
        return trimmed or query
