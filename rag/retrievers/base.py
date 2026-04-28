from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import RetrievalResult


class BaseRetriever(ABC):
    name = "base"

    @abstractmethod
    def retrieve(self, query: str, topk: int) -> RetrievalResult:
        raise NotImplementedError
