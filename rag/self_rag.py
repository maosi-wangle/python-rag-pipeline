from __future__ import annotations

from .config import RAGConfig
from .orchestrator import ModularRAGOrchestrator
from .schemas import StructuredRAGResponse
from .state import RAGSessionState


class SelfRAGPipeline:
    def __init__(
        self,
        orchestrator: ModularRAGOrchestrator | None = None,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.orchestrator = orchestrator or ModularRAGOrchestrator(self.config)

    def run(
        self,
        query: str,
        *,
        history: list[str] | None = None,
        topk: int | None = None,
        max_rounds: int | None = None,
    ) -> StructuredRAGResponse:
        session = RAGSessionState(
            user_query=query,
            history=list(history or []),
            max_rounds=max_rounds or self.config.max_rounds,
        )
        current_query = query
        final_round = None

        for round_index in range(1, session.max_rounds + 1):
            round_state = self.orchestrator.run_round(
                current_query,
                history=self._history_with_memory(session),
                topk=topk or self.config.default_topk,
                round_index=round_index,
                max_rounds=session.max_rounds,
            )
            session.rounds.append(round_state)
            final_round = round_state

            if round_state.completeness == "yes":
                break
            if not round_state.need_followup:
                break
            if not round_state.next_query or round_state.next_query == current_query:
                break

            current_query = round_state.next_query

        if final_round is None:
            final_round = self.orchestrator.run_round(query)

        return StructuredRAGResponse(
            response=final_round.answer_draft,
            query=final_round.active_query,
            retrieved_chunk_ids=[hit.chunk_id for hit in final_round.fused_hits],
            completeness=final_round.completeness,
            relevance_score=final_round.relevance_score,
            support_score=final_round.support_score,
            need_followup=final_round.need_followup,
            next_query=final_round.next_query,
            round=final_round.round_index,
            traces=[round_state.to_trace() for round_state in session.rounds],
        )

    def _history_with_memory(self, session: RAGSessionState) -> list[str]:
        memory = list(session.history)
        for round_state in session.rounds:
            memory.append(round_state.active_query)
            memory.append(round_state.answer_draft)
        return memory
