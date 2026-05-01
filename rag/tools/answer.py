from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMClient
from ..prompts import ANSWER_SYSTEM_PROMPT
from ..schemas import RetrievalHit
from ..text_utils import truncate_text


@dataclass(slots=True)
class GenerationResult:
    answer: str
    used_fallback: bool = False
    error: str | None = None
    llm_context: dict[str, object] | None = None
    raw_output: str | None = None


class AnswerGenerationTool:
    name = "generate"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def generate(
        self,
        *,
        query: str,
        hits: list[RetrievalHit],
        history: list[str] | None = None,
        instruction: str | None = None,
        source_mode: str = "retrieval",
    ) -> GenerationResult:
        history = history or []
        source_mode = self._normalize_source_mode(source_mode)
        llm_context = self._build_llm_context(
            query=query,
            hits=hits,
            history=history,
            instruction=instruction,
            source_mode=source_mode,
        )
        if self.llm and self.llm.available:
            try:
                raw_output = self._llm_generate(llm_context)
                answer = raw_output.strip()
                if answer:
                    return GenerationResult(
                        answer=answer,
                        llm_context=llm_context,
                        raw_output=raw_output,
                    )
                fallback = self._fallback_generate(query, hits, history, source_mode)
                return GenerationResult(
                    answer=fallback,
                    used_fallback=True,
                    error="LLM returned empty content.",
                    llm_context=llm_context,
                    raw_output=raw_output,
                )
            except Exception as exc:
                fallback = self._fallback_generate(query, hits, history, source_mode)
                return GenerationResult(
                    answer=fallback,
                    used_fallback=True,
                    error=self._format_error(exc),
                    llm_context=llm_context,
                    raw_output=None,
                )

        return GenerationResult(
            answer=self._fallback_generate(query, hits, history, source_mode),
            used_fallback=True,
            error="LLM client is not configured.",
            llm_context=llm_context,
            raw_output=None,
        )

    def _build_llm_context(
        self,
        *,
        query: str,
        hits: list[RetrievalHit],
        history: list[str],
        instruction: str | None,
        source_mode: str,
    ) -> dict[str, object]:
        context_blocks = []
        for hit in hits[:8]:
            context_blocks.append(
                f"Chunk ID: {hit.chunk_id}\n"
                f"Source: {hit.chunk.source or 'unknown'}\n"
                f"Content: {hit.chunk.context}"
            )
        source_instruction = {
            "retrieval": "只基于检索到的 chunks 回答；不要使用 chunks 之外的事实。",
            "memory": "只基于对话记忆回答；不要补充对话中没有出现的新事实。",
            "mixed": "优先基于检索到的 chunks，并可结合对话记忆补充上下文。",
        }[source_mode]
        user_prompt = f"""
用户问题：
{query}

对话记忆：
{chr(10).join(history[-6:]) if history else "(empty)"}

答案来源模式：
{source_mode}

来源约束：
{source_instruction}

生成要求：
{instruction or "(none)"}

检索到的 chunks：
{chr(10).join(context_blocks) if context_blocks else "(none)"}
"""
        return {
            "system_prompt": ANSWER_SYSTEM_PROMPT,
            "user_prompt": user_prompt.strip(),
            "chunk_ids": [hit.chunk_id for hit in hits[:8]],
            "source_mode": source_mode,
        }

    def _llm_generate(self, llm_context: dict[str, object]) -> str:
        return self.llm.generate_text(
            system_prompt=str(llm_context["system_prompt"]),
            user_prompt=str(llm_context["user_prompt"]),
            max_tokens=1400,
        )

    def _fallback_generate(
        self,
        query: str,
        hits: list[RetrievalHit],
        history: list[str],
        source_mode: str,
    ) -> str:
        if not hits:
            if source_mode in {"memory", "mixed"} and history:
                recent_history = " ".join(history[-4:])
                return (
                    "基于当前对话记忆，最近内容是："
                    f"{truncate_text(recent_history, max_chars=600)}"
                )
            return f"未检索到足够证据回答：{query}"

        sentences = ["基于当前检索证据，可参考以下内容："]
        for hit in hits[:3]:
            sentences.append(f"[{hit.chunk_id}] {truncate_text(hit.chunk.context, max_chars=180)}")
        return " ".join(sentences)

    def _format_error(self, exc: Exception) -> str:
        message = str(exc).strip()
        if len(message) > 600:
            message = f"{message[:600]}..."
        return f"{exc.__class__.__name__}: {message}"

    @staticmethod
    def _normalize_source_mode(source_mode: str) -> str:
        normalized = str(source_mode or "retrieval").strip().lower()
        if normalized not in {"retrieval", "memory", "mixed"}:
            return "retrieval"
        return normalized
