from __future__ import annotations

import re

from ..text_utils import has_multi_intent_markers, normalize_whitespace, unique_preserve_order


class QueryDecomposeTool:
    name = "decompose_query"

    def decompose(self, query: str) -> list[str]:
        normalized_query = normalize_whitespace(query)
        if not has_multi_intent_markers(normalized_query):
            return []

        candidates = re.split(r"[，,；;、]", normalized_query)
        subqueries: list[str] = []
        for candidate in candidates:
            candidate = normalize_whitespace(candidate)
            if len(candidate) < 4:
                continue
            subqueries.append(candidate)

        if len(subqueries) <= 1 and "以及" in normalized_query:
            pieces = [part.strip() for part in normalized_query.split("以及") if part.strip()]
            subqueries.extend(pieces)

        return unique_preserve_order(subqueries)[:4]
