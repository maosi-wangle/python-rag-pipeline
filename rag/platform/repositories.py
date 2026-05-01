from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from .schemas import (
    ChatProfile,
    ConversationSession,
    KnowledgeBaseProfile,
    LLMProfile,
    TraceRecord,
    UserProfile,
)

T = TypeVar("T")


class JsonRepository(Generic[T]):
    def __init__(
        self,
        path: Path,
        *,
        id_field: str,
        factory: Callable[[dict[str, Any]], T],
        serializer: Callable[[T], dict[str, Any]],
        defaults: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.id_field = id_field
        self.factory = factory
        self.serializer = serializer
        self.defaults = defaults or []

    def all(self) -> list[T]:
        return [self.factory(item) for item in self._read_items()]

    def get(self, item_id: str) -> T | None:
        for item in self._read_items():
            if str(item.get(self.id_field)) == item_id:
                return self.factory(item)
        return None

    def upsert(self, item: T) -> None:
        payload = self.serializer(item)
        item_id = str(payload[self.id_field])
        items = self._read_items()
        for index, existing in enumerate(items):
            if str(existing.get(self.id_field)) == item_id:
                items[index] = payload
                break
        else:
            items.append(payload)
        self._write_items(items)

    def _read_items(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return [dict(item) for item in self.defaults]
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        if not isinstance(payload, list):
            raise ValueError(f"{self.path} must contain a JSON list.")
        return [dict(item) for item in payload]

    def _write_items(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class TraceRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, trace: TraceRecord) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f"{trace.trace_id}.json"
        target.write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


class PlatformRepository:
    def __init__(self, root: str | Path = "data/platform") -> None:
        self.root = Path(root)
        self.users = JsonRepository(
            self.root / "users.json",
            id_field="user_id",
            factory=UserProfile.from_dict,
            serializer=lambda item: item.to_dict(),
            defaults=[
                {
                    "user_id": "default",
                    "display_name": "Default User",
                    "default_chat_profile_id": "default",
                }
            ],
        )
        self.llm_profiles = JsonRepository(
            self.root / "llm_profiles.json",
            id_field="profile_id",
            factory=LLMProfile.from_dict,
            serializer=lambda item: item.to_dict(),
            defaults=[
                {
                    "profile_id": "default_chat",
                    "provider": "openai-compatible",
                    "model_type": "chat",
                    "model_name": "",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url_env": "OPENAI_BASE_URL",
                }
            ],
        )
        self.knowledge_bases = JsonRepository(
            self.root / "knowledge_bases.json",
            id_field="kb_id",
            factory=KnowledgeBaseProfile.from_dict,
            serializer=lambda item: item.to_dict(),
            defaults=[
                {
                    "kb_id": "default",
                    "name": "Default Knowledge Base",
                    "chunk_store_path": "./knowledgeBase.json",
                    "index_path": "./knowledge.index",
                    "embeddings_path": "./knowledge_embeddings.npy",
                    "inverted_index_path": "./inverted_index.json",
                    "retrieval_modes": ["semantic", "keyword"],
                }
            ],
        )
        self.chat_profiles = JsonRepository(
            self.root / "chat_profiles.json",
            id_field="profile_id",
            factory=ChatProfile.from_dict,
            serializer=lambda item: item.to_dict(),
            defaults=[
                {
                    "profile_id": "default",
                    "name": "Default Chat",
                    "chat_llm_profile_id": "default_chat",
                    "default_kb_ids": ["default"],
                }
            ],
        )
        self.conversations = JsonRepository(
            self.root / "conversations.json",
            id_field="conversation_id",
            factory=ConversationSession.from_dict,
            serializer=lambda item: item.to_dict(),
            defaults=[],
        )
        self.traces = TraceRepository(self.root / "traces")

    def ensure_defaults(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for repo in (
            self.users,
            self.llm_profiles,
            self.knowledge_bases,
            self.chat_profiles,
            self.conversations,
        ):
            if not repo.path.exists():
                repo._write_items(repo._read_items())
