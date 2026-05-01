from __future__ import annotations

from typing import Any

from ..llm import LLMClient
from ..text_utils import (
    extract_recent_terms,
    has_multi_intent_markers,
    normalize_whitespace,
    unique_preserve_order,
)


class QueryRewriteTool:
    name = "query_transform"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def transform(
        self,
        *,
        query: str,
        history: list[str] | None = None,
        rewrite_mode: str = "none",
        should_decompose: bool = False,
    ) -> dict[str, object]:
        history = history or []
        normalized_query = normalize_whitespace(query)
        if self.llm and self.llm.available:
            try:
                return self._llm_transform(
                    query=normalized_query,
                    history=history,
                    rewrite_mode=rewrite_mode,
                    should_decompose=should_decompose,
                )
            except Exception as exc:
                payload = self._heuristic_transform(
                    query=normalized_query,
                    history=history,
                    rewrite_mode=rewrite_mode,
                    should_decompose=should_decompose,
                )
                payload["used_fallback"] = True
                payload["error"] = self._format_error(exc)
                return payload
        return self._heuristic_transform(
            query=normalized_query,
            history=history,
            rewrite_mode=rewrite_mode,
            should_decompose=should_decompose,
        )

    def _llm_transform(
        self,
        *,
        query: str,
        history: list[str],
        rewrite_mode: str,
        should_decompose: bool,
    ) -> dict[str, object]:
        history_text = "\n".join(history[-6:]) if history else "(empty)"
        user_prompt = f"""
User query:
{query}

Conversation memory:
{history_text}

Requested rewrite mode:
{rewrite_mode}

Should decompose:
{str(should_decompose).lower()}

Return JSON with:
{{
  "queries": ["..."],
  "retrieval_modes": [["semantic", "keyword"], ["semantic", "keyword", "grep"]],
  "rationale": "short explanation"
}}

Rules:
- Keep the original user intent.
- If rewrite_mode is "specific", make the query more precise with useful context.
- If rewrite_mode is "general", broaden wording while preserving topic.
- If rewrite_mode is "chunk_like", rewrite it closer to knowledge-base phrasing.
- If rewrite_mode is "hybrid", combine precision and chunk-like phrasing.
- If should_decompose is true, split a multi-intent query into independent retrieval-ready subqueries.
- Each query must be directly retrievable.
- retrieval_modes for each query can only use semantic, keyword, grep.
"""
        payload = self.llm.generate_json(
            system_prompt="You rewrite retrieval queries for a modular RAG system. Return JSON only.",
            user_prompt=user_prompt,
            max_tokens=1200,
        )
        queries = [
            normalize_whitespace(str(item))
            for item in payload.get("queries", [])
            if normalize_whitespace(str(item))
        ]
        if not queries:
            queries = [query]

        raw_modes = payload.get("retrieval_modes") or []
        plans: list[dict[str, object]] = []
        for index, item in enumerate(queries):
            modes = []
            if index < len(raw_modes) and isinstance(raw_modes[index], list):
                modes = [str(mode) for mode in raw_modes[index] if str(mode) in {"semantic", "keyword", "grep"}]
            if not modes:
                modes = self._select_retrieval_modes(item)
            plans.append({"query": item, "retrieval_modes": modes})

        return {
            "transform_applied": rewrite_mode != "none" or should_decompose,
            "transform_type": self._transform_type(rewrite_mode, should_decompose),
            "queries": queries,
            "plans": plans,
            "rationale": str(payload.get("rationale") or "").strip(),
            "used_fallback": False,
            "error": None,
        }

    def _heuristic_transform(
        self,
        *,
        query: str,
        history: list[str],
        rewrite_mode: str,
        should_decompose: bool,
    ) -> dict[str, object]:
        queries = self._rewrite_variants(query, history=history, rewrite_mode=rewrite_mode)
        if should_decompose:
            queries = self._decompose_queries(queries)

        plans = [
            {
                "query": item,
                "retrieval_modes": self._select_retrieval_modes(item),
            }
            for item in queries
        ]
        return {
            "transform_applied": rewrite_mode != "none" or should_decompose,
            "transform_type": self._transform_type(rewrite_mode, should_decompose),
            "queries": queries,
            "plans": plans,
            "rationale": "Heuristic query transformation completed.",
            "used_fallback": not (self.llm and self.llm.available),
            "error": "LLM client is not configured." if not (self.llm and self.llm.available) else None,
        }

    def _rewrite_variants(
        self,
        query: str,
        *,
        history: list[str],
        rewrite_mode: str,
    ) -> list[str]:
        variants = [query]
        recent_terms = extract_recent_terms(history)

        if rewrite_mode in {"specific", "hybrid"} and recent_terms:
            variants.append(f"{query} {' '.join(recent_terms[:4])}")

        if rewrite_mode == "general":
            variants.append(self._generalize(query))

        if rewrite_mode in {"chunk_like", "hybrid"}:
            chunk_like_query = f"{query} 相关定义 背景 条件 流程 约束"
            if chunk_like_query not in variants:
                variants.append(chunk_like_query)

        return unique_preserve_order([normalize_whitespace(item) for item in variants if item])

    def _generalize(self, query: str) -> str:
        simplified = query.replace("怎么", "").replace("如何", "").replace("为什么", "").strip()
        return simplified or query

    def _decompose_queries(self, queries: list[str]) -> list[str]:
        expanded: list[str] = []
        separators = ["；", ";", "，", ",", "、", "以及", "并且", "同时"]
        for query in queries:
            if not has_multi_intent_markers(query):
                expanded.append(query)
                continue

            split_queries = [query]
            for separator in separators:
                if separator in query:
                    split_queries = [piece.strip() for piece in query.split(separator) if piece.strip()]
                    break
            expanded.extend(split_queries)

        return unique_preserve_order(expanded)[:6]

    def _select_retrieval_modes(self, query: str) -> list[str]:
        normalized_query = normalize_whitespace(query)
        modes = ["semantic", "keyword"]
        if len(normalized_query) <= 24 or any(token in normalized_query for token in ("定义", "命令", "参数", "路径", "配置")):
            modes.append("grep")
        return unique_preserve_order(modes)

    def _transform_type(self, rewrite_mode: str, should_decompose: bool) -> str:
        if should_decompose and rewrite_mode != "none":
            return "hybrid"
        if should_decompose:
            return "decompose"
        if rewrite_mode != "none":
            return "rewrite"
        return "none"

    def _format_error(self, exc: Exception) -> str:
        message = str(exc).strip()
        if len(message) > 600:
            message = f"{message[:600]}..."
        return f"{exc.__class__.__name__}: {message}"
