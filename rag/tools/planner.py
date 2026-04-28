from __future__ import annotations

from typing import Any

from ..llm import LLMClient
from ..prompts import PLANNER_SYSTEM_PROMPT
from ..text_utils import has_multi_intent_markers, normalize_whitespace, unique_preserve_order


class QueryPlannerTool:
    name = "plan_query"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def plan(self, query: str, history: list[str] | None = None) -> dict[str, Any]:
        history = history or []
        if self.llm and self.llm.available:
            try:
                return self._llm_plan(query, history)
            except Exception:
                pass
        return self._heuristic_plan(query, history)

    def _llm_plan(self, query: str, history: list[str]) -> dict[str, Any]:
        history_text = "\n".join(history[-6:]) if history else "(empty)"
        user_prompt = f"""
User query:
{query}

Conversation memory:
{history_text}

Return JSON with:
- rewritten_queries: string[]
- subqueries: string[]
- retrievers: string[] chosen from ["semantic", "keyword", "grep"]
- rationale: string
- next_focus: string
Constraints:
- Keep rewritten_queries <= 4
- Keep subqueries <= 4
- Preserve the original meaning
- Prefer parallel-friendly plans
"""
        payload = self.llm.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1200,
        )
        return self._normalize_plan(query, payload)

    def _heuristic_plan(self, query: str, history: list[str]) -> dict[str, Any]:
        normalized_query = normalize_whitespace(query)
        rewritten_queries = [normalized_query]
        if history and len(normalized_query) <= 16:
            rewritten_queries.append(
                f"{normalized_query} {' '.join(item for item in history[-2:] if item)[:60]}"
            )
        suffix = "定义 原因 方法 注意事项"
        if suffix not in normalized_query:
            rewritten_queries.append(f"{normalized_query} {suffix}")

        subqueries: list[str] = []
        if has_multi_intent_markers(normalized_query):
            for separator in ("，", ",", "；", ";", "、"):
                if separator in normalized_query:
                    subqueries = [piece.strip() for piece in normalized_query.split(separator) if piece.strip()]
                    break

        retrievers = ["semantic", "keyword", "grep"]
        return {
            "rewritten_queries": unique_preserve_order(rewritten_queries)[:4],
            "subqueries": unique_preserve_order(subqueries)[:4],
            "retrievers": retrievers,
            "rationale": "Heuristic planner selected all retrievers.",
            "next_focus": normalized_query,
        }

    def _normalize_plan(self, query: str, payload: dict[str, Any]) -> dict[str, Any]:
        rewritten_queries = payload.get("rewritten_queries") or [query]
        subqueries = payload.get("subqueries") or []
        retrievers = payload.get("retrievers") or ["semantic", "keyword", "grep"]
        retrievers = [item for item in retrievers if item in {"semantic", "keyword", "grep"}]
        if not retrievers:
            retrievers = ["semantic", "keyword", "grep"]
        return {
            "rewritten_queries": unique_preserve_order([query] + [str(item) for item in rewritten_queries])[:4],
            "subqueries": unique_preserve_order([str(item) for item in subqueries])[:4],
            "retrievers": retrievers,
            "rationale": str(payload.get("rationale") or ""),
            "next_focus": str(payload.get("next_focus") or query),
        }
