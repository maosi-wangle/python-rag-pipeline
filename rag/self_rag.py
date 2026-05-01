from __future__ import annotations

from .agent import ToolCallingRAGAgent
from .config import RAGConfig
from .orchestrator import ModularRAGOrchestrator
from .schemas import StructuredRAGResponse


class SelfRAGPipeline:
    """Compatibility wrapper around the new tool-calling agent."""

    def __init__(
        self,
        orchestrator: ModularRAGOrchestrator | None = None,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.orchestrator = orchestrator or ModularRAGOrchestrator(self.config)
        self.agent = ToolCallingRAGAgent(self.orchestrator, self.config)

    def run(
        self,
        query: str,
        *,
        history: list[str] | None = None,
        topk: int | None = None,
        max_rounds: int | None = None,
    ) -> StructuredRAGResponse:
        return self.agent.run(
            query,
            history=history,
            topk=topk,
            max_rounds=max_rounds,
        )
