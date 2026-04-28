from __future__ import annotations

from ..llm import LLMClient
from ..prompts import JUDGE_SYSTEM_PROMPT
from ..schemas import JudgeResult, RetrievalHit
from ..text_utils import lexical_overlap_score, tokenize, unique_preserve_order


class AnswerJudgeTool:
    name = "judge_answer"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def judge(
        self,
        *,
        query: str,
        response: str,
        hits: list[RetrievalHit],
        subqueries: list[str],
        round_index: int,
        max_rounds: int,
    ) -> JudgeResult:
        if self.llm and self.llm.available:
            try:
                return self._llm_judge(
                    query=query,
                    response=response,
                    hits=hits,
                    subqueries=subqueries,
                    round_index=round_index,
                    max_rounds=max_rounds,
                )
            except Exception:
                pass

        if not hits:
            return JudgeResult(
                completeness="no",
                relevance_score=0.0,
                support_score=0.0,
                need_followup=round_index < max_rounds,
                next_query=query,
                missing_aspects=tokenize(query)[:4],
                reason="No retrieval hits were found.",
            )

        query_terms = tokenize(query)
        response_score = lexical_overlap_score(query, response)
        support_score = min(sum(hit.final_score() for hit in hits[:3]) / max(len(hits[:3]), 1), 1.0)

        covered_subqueries = 0
        for subquery in subqueries:
            if any(lexical_overlap_score(subquery, hit.text) > 0 for hit in hits[:5]):
                covered_subqueries += 1

        coverage_ratio = response_score
        if subqueries:
            coverage_ratio = max(coverage_ratio, covered_subqueries / len(subqueries))

        missing_aspects = self._missing_aspects(query_terms, hits)
        completeness = "yes" if coverage_ratio >= 0.55 and support_score >= 0.2 else "no"
        need_followup = completeness == "no" and round_index < max_rounds
        next_query = None
        if need_followup:
            next_query = self._build_followup_query(query, missing_aspects)

        return JudgeResult(
            completeness=completeness,
            relevance_score=coverage_ratio,
            support_score=support_score,
            need_followup=need_followup,
            next_query=next_query,
            missing_aspects=missing_aspects,
            reason=(
                "Coverage and support passed thresholds."
                if completeness == "yes"
                else "Coverage or support is still insufficient."
            ),
        )

    def _llm_judge(
        self,
        *,
        query: str,
        response: str,
        hits: list[RetrievalHit],
        subqueries: list[str],
        round_index: int,
        max_rounds: int,
    ) -> JudgeResult:
        context_preview = "\n\n".join(
            [f"{hit.chunk_id}: {hit.chunk.context[:280]}" for hit in hits[:6]]
        )
        user_prompt = f"""
User query:
{query}

Draft answer:
{response}

Subqueries:
{subqueries}

Retrieved chunk preview:
{context_preview}

Return JSON:
{{
  "completeness": "yes or no",
  "relevance_score": 0.0,
  "support_score": 0.0,
  "need_followup": true,
  "next_query": "string or null",
  "missing_aspects": ["..."],
  "reason": "short explanation"
}}

Rules:
- completeness must be yes only if the answer is supported and covers the user question
- if evidence is weak or some subquestions are missing, set completeness=no
- if round_index == max_rounds, need_followup must be false
"""
        payload = self.llm.generate_json(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1000,
        )
        completeness = "yes" if str(payload.get("completeness", "")).lower() == "yes" else "no"
        need_followup = bool(payload.get("need_followup")) and round_index < max_rounds
        next_query = payload.get("next_query")
        if next_query is not None:
            next_query = str(next_query).strip() or None
        return JudgeResult(
            completeness=completeness,
            relevance_score=float(payload.get("relevance_score", 0.0)),
            support_score=float(payload.get("support_score", 0.0)),
            need_followup=need_followup,
            next_query=next_query if need_followup else None,
            missing_aspects=[str(item) for item in payload.get("missing_aspects", [])][:4],
            reason=str(payload.get("reason") or ""),
        )

    def _missing_aspects(self, query_terms: list[str], hits: list[RetrievalHit]) -> list[str]:
        all_text = " ".join(hit.text for hit in hits[:5])
        missing = [term for term in query_terms if term not in all_text]
        return unique_preserve_order(missing)[:4]

    def _build_followup_query(self, query: str, missing_aspects: list[str]) -> str:
        if not missing_aspects:
            suffix = "定义 原因 方法 注意事项"
            if suffix not in query:
                return f"{query} {suffix}"
            return f"{query} 相关细节"
        return f"{query} {' '.join(missing_aspects)}"
