from __future__ import annotations

import json
from typing import Any

from rag import ModularRAGOrchestrator, RAGConfig, ToolCallingRAGAgent
from rag.input import MultimodalInputParser


class FaceAiSystem:
    def __init__(
        self,
        dataPath: str = "./knowledgeBase.json",
        index_path: str = "./knowledge.index",
        embeddings_path: str = "./knowledge_embeddings.npy",
        inverted_index_path: str = "./inverted_index.json",
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig(
            data_path=dataPath,
            index_path=index_path,
            embeddings_path=embeddings_path,
            inverted_index_path=inverted_index_path,
        )
        self.orchestrator = ModularRAGOrchestrator(self.config)
        self.agent = ToolCallingRAGAgent(self.orchestrator, self.config)
        self.initialized = self.orchestrator.initialized
        self.input_parser = MultimodalInputParser()

    def retrieve_for_ragas(self, query: str, topk: int = 5) -> dict[str, Any]:
        return self.orchestrator.build_ragas_payload(query, topk=topk)

    def batch_retrieve_for_ragas(self, queries: list[str], topk: int = 5) -> list[dict[str, Any]]:
        return [self.retrieve_for_ragas(query, topk=topk) for query in queries]

    def retrieve_top_contexts(self, query: str, topk: int = 5) -> list[str]:
        payload = self.retrieve_for_ragas(query, topk=topk)
        return payload["retrieved_contexts"]

    def run_agentic_query(
        self,
        query: str,
        *,
        history: list[str] | None = None,
        topk: int = 5,
        max_rounds: int | None = None,
        input_file: str | None = None,
    ) -> dict[str, Any]:
        parsed_message = None
        if input_file:
            parsed_message = self.input_parser.parse_file(input_file, persist=False)
            query = self.input_parser.build_query_context(query, parsed_message)
        return self.agent.run(
            query,
            history=history,
            topk=topk,
            max_rounds=max_rounds,
        ).to_dict()


def main() -> None:
    system = FaceAiSystem()
    while True:
        query = input("User query: ").strip()
        if not query:
            continue
        result = system.run_agentic_query(query, topk=5)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
