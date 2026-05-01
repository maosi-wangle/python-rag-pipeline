from __future__ import annotations


class FinishTool:
    name = "finish"

    def finish(
        self,
        *,
        response: str,
        query: str,
        grounded: bool,
        completeness: str,
        if_multi_turn: bool,
        rationale: str,
        next_focus: str | None = None,
        retrieved_chunk_ids: list[str] | None = None,
        used_queries: list[str] | None = None,
        tool_rounds: int = 0,
    ) -> dict[str, object]:
        normalized_completeness = "yes" if str(completeness).lower() == "yes" else "no"
        return {
            "response": str(response).strip(),
            "query": str(query).strip(),
            "grounded": bool(grounded),
            "completeness": normalized_completeness,
            "if_multi_turn": bool(if_multi_turn),
            "need_followup": bool(if_multi_turn),
            "rationale": str(rationale).strip(),
            "next_focus": str(next_focus).strip() if next_focus else None,
            "retrieved_chunk_ids": list(retrieved_chunk_ids or []),
            "used_queries": list(used_queries or []),
            "tool_rounds": int(tool_rounds),
        }
