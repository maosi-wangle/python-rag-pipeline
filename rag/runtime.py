from __future__ import annotations

from typing import Any


class RuntimeStore:
    """Lightweight per-run state inspired by RAGFlow canvas variables."""

    def __init__(self) -> None:
        self.globals: dict[str, Any] = {}
        self.retrievals: dict[str, dict[str, Any]] = {}
        self.answers: dict[str, dict[str, Any]] = {}

    def reset(self, query: str = "", history: list[str] | None = None) -> None:
        self.globals = {
            "sys.query": query,
            "sys.history": list(history or []),
            "sys.last_retrieval_id": None,
            "sys.last_answer_id": None,
            "sys.last_valid_answer_id": None,
        }
        self.retrievals = {}
        self.answers = {}

    def put_retrieval(self, retrieval: dict[str, Any]) -> str:
        retrieval_id = f"retrieval_{len(self.retrievals) + 1:03d}"
        retrieval["retrieval_id"] = retrieval_id
        self.retrievals[retrieval_id] = retrieval
        self.globals["sys.last_retrieval_id"] = retrieval_id
        return retrieval_id

    def get_retrieval(self, retrieval_id: str | None = None) -> dict[str, Any] | None:
        active_id = retrieval_id or self.globals.get("sys.last_retrieval_id")
        if not active_id:
            return None
        return self.retrievals.get(str(active_id))

    def put_answer(self, answer: dict[str, Any], *, mark_latest: bool = True) -> str:
        answer_id = f"answer_{len(self.answers) + 1:03d}"
        self.answers[answer_id] = answer
        if mark_latest:
            self.globals["sys.last_answer_id"] = answer_id
            if answer.get("is_valid"):
                self.globals["sys.last_valid_answer_id"] = answer_id
        return answer_id

    def get_answer(self, answer_id: str | None = None) -> dict[str, Any] | None:
        active_id = answer_id or self.globals.get("sys.last_answer_id")
        if not active_id:
            return None
        return self.answers.get(str(active_id))

    def get_last_valid_answer(self) -> dict[str, Any] | None:
        answer_id = self.globals.get("sys.last_valid_answer_id")
        if answer_id:
            return self.answers.get(str(answer_id))
        for answer in reversed(list(self.answers.values())):
            if answer.get("is_valid"):
                return answer
        return None

    def last_valid_answer_content(self) -> str:
        answer = self.get_last_valid_answer()
        if not answer:
            return ""
        return str(answer.get("content") or "")

    def last_answer_content(self) -> str:
        answer = self.get_answer()
        if not answer:
            return ""
        return str(answer.get("content") or "")

    def get_retrievals(self, retrieval_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if retrieval_ids is None:
            return list(self.retrievals.values())
        return [
            self.retrievals[item]
            for item in retrieval_ids
            if item in self.retrievals
        ]
