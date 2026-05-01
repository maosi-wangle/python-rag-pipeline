from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class UserProfile:
    user_id: str
    display_name: str = ""
    default_chat_profile_id: str | None = None
    default_embedding_profile_id: str | None = None
    default_rerank_profile_id: str | None = None
    permissions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=str(payload["user_id"]),
            display_name=str(payload.get("display_name") or payload["user_id"]),
            default_chat_profile_id=payload.get("default_chat_profile_id"),
            default_embedding_profile_id=payload.get("default_embedding_profile_id"),
            default_rerank_profile_id=payload.get("default_rerank_profile_id"),
            permissions=_list_of_str(payload.get("permissions")),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LLMProfile:
    profile_id: str
    provider: str = "openai-compatible"
    model_type: str = "chat"
    model_name: str = ""
    owner_user_id: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    base_url_env: str | None = "OPENAI_BASE_URL"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    timeout: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMProfile":
        return cls(
            profile_id=str(payload["profile_id"]),
            provider=str(payload.get("provider") or "openai-compatible"),
            model_type=str(payload.get("model_type") or "chat"),
            model_name=str(payload.get("model_name") or ""),
            owner_user_id=payload.get("owner_user_id"),
            api_key_env=payload.get("api_key_env", "OPENAI_API_KEY"),
            base_url_env=payload.get("base_url_env", "OPENAI_BASE_URL"),
            api_key=payload.get("api_key"),
            base_url=payload.get("base_url"),
            temperature=payload.get("temperature"),
            timeout=payload.get("timeout"),
            max_tokens=payload.get("max_tokens"),
            extra=_dict(payload.get("extra")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KnowledgeBaseProfile:
    kb_id: str
    name: str
    description: str = ""
    owner_user_id: str | None = None
    language: str = "zh"
    chunk_store_path: str = "./knowledgeBase.json"
    index_path: str = "./knowledge.index"
    embeddings_path: str = "./knowledge_embeddings.npy"
    inverted_index_path: str = "./inverted_index.json"
    embedding_model_name: str | None = None
    embedding_profile_id: str | None = None
    rerank_profile_id: str | None = None
    retrieval_modes: list[str] = field(default_factory=lambda: ["semantic", "keyword"])
    default_topk: int | None = None
    similarity_threshold: float | None = None
    vector_similarity_weight: float | None = None
    parser_id: str = "text"
    parser_config: dict[str, Any] = field(default_factory=dict)
    chunking_config: dict[str, Any] = field(default_factory=dict)
    index_config: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeBaseProfile":
        return cls(
            kb_id=str(payload["kb_id"]),
            name=str(payload.get("name") or payload["kb_id"]),
            description=str(payload.get("description") or ""),
            owner_user_id=payload.get("owner_user_id"),
            language=str(payload.get("language") or "zh"),
            chunk_store_path=str(payload.get("chunk_store_path") or "./knowledgeBase.json"),
            index_path=str(payload.get("index_path") or "./knowledge.index"),
            embeddings_path=str(payload.get("embeddings_path") or "./knowledge_embeddings.npy"),
            inverted_index_path=str(payload.get("inverted_index_path") or "./inverted_index.json"),
            embedding_model_name=payload.get("embedding_model_name"),
            embedding_profile_id=payload.get("embedding_profile_id"),
            rerank_profile_id=payload.get("rerank_profile_id"),
            retrieval_modes=_list_of_str(payload.get("retrieval_modes")) or ["semantic", "keyword"],
            default_topk=payload.get("default_topk"),
            similarity_threshold=payload.get("similarity_threshold"),
            vector_similarity_weight=payload.get("vector_similarity_weight"),
            parser_id=str(payload.get("parser_id") or "text"),
            parser_config=_dict(payload.get("parser_config")),
            chunking_config=_dict(payload.get("chunking_config")),
            index_config=_dict(payload.get("index_config")),
            status=str(payload.get("status") or "ready"),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatProfile:
    profile_id: str
    name: str = ""
    chat_llm_profile_id: str | None = None
    default_kb_ids: list[str] = field(default_factory=list)
    default_topk: int | None = None
    max_rounds: int | None = None
    prompt_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatProfile":
        return cls(
            profile_id=str(payload["profile_id"]),
            name=str(payload.get("name") or payload["profile_id"]),
            chat_llm_profile_id=payload.get("chat_llm_profile_id"),
            default_kb_ids=_list_of_str(payload.get("default_kb_ids")),
            default_topk=payload.get("default_topk"),
            max_rounds=payload.get("max_rounds"),
            prompt_config=_dict(payload.get("prompt_config")),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConversationSession:
    conversation_id: str
    user_id: str
    active_kb_ids: list[str] = field(default_factory=list)
    chat_profile_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    short_memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationSession":
        return cls(
            conversation_id=str(payload["conversation_id"]),
            user_id=str(payload.get("user_id") or "default"),
            active_kb_ids=_list_of_str(payload.get("active_kb_ids")),
            chat_profile_id=payload.get("chat_profile_id"),
            messages=list(payload.get("messages") or []),
            summary=str(payload.get("summary") or ""),
            short_memory=_dict(payload.get("short_memory")),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    user_id: str | None
    conversation_id: str | None
    kb_ids: list[str]
    query: str
    response: str | None = None
    status: str = "running"
    started_at: str | None = None
    ended_at: str | None = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceRecord":
        return cls(
            trace_id=str(payload["trace_id"]),
            user_id=payload.get("user_id"),
            conversation_id=payload.get("conversation_id"),
            kb_ids=_list_of_str(payload.get("kb_ids")),
            query=str(payload.get("query") or ""),
            response=payload.get("response"),
            status=str(payload.get("status") or "running"),
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            spans=list(payload.get("spans") or []),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
