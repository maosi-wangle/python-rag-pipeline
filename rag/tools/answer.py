from __future__ import annotations

from ..llm import LLMClient
from ..prompts import ANSWER_SYSTEM_PROMPT
from ..schemas import RetrievalHit
from ..text_utils import truncate_text


class AnswerGenerationTool:
    name = "generate_answer"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def generate(
        self,
        *,
        query: str,
        hits: list[RetrievalHit],
        history: list[str] | None = None,
    ) -> str:
        history = history or []
        if self.llm and self.llm.available:
            try:
                return self._llm_generate(query=query, hits=hits, history=history)
            except Exception:
                pass
        return self._fallback_generate(query, hits)

    def _llm_generate(
        self,
        *,
        query: str,
        hits: list[RetrievalHit],
        history: list[str],
    ) -> str:
        context_blocks = []
        for hit in hits[:8]:
            context_blocks.append(
                f"Chunk ID: {hit.chunk_id}\n"
                f"Source: {hit.chunk.source or 'unknown'}\n"
                f"Content: {hit.chunk.context}"
            )
        user_prompt = f"""
User query:
{query}

Conversation memory:
{chr(10).join(history[-6:]) if history else "(empty)"}

Retrieved chunks:
{chr(10).join(context_blocks)}

Return JSON:
{{
  "answer": "final answer in Chinese with chunk citations",
  "confidence_note": "short note"
}}
"""
        payload = self.llm.generate_json(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1400,
        )
        answer = str(payload.get("answer") or "").strip()
        if answer:
            return answer
        return self._fallback_generate(query, hits)

    def _fallback_generate(self, query: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return f"未找到足够证据回答该问题：{query}"
        sentences = []
        for hit in hits[:3]:
            sentences.append(f"[{hit.chunk_id}] {truncate_text(hit.chunk.context, max_chars=180)}")
        return " ".join(sentences)
