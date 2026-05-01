from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
from typing import Any
from uuid import uuid4

from ..agent import ToolCallingRAGAgent
from ..config import RAGConfig
from ..orchestrator import ModularRAGOrchestrator
from .repositories import PlatformRepository
from .kb_catalog import KnowledgeBaseCatalogItem
from .schemas import (
    ChatProfile,
    ConversationSession,
    KnowledgeBaseProfile,
    LLMProfile,
    TraceRecord,
    UserProfile,
)


class AgenticRAGService:
    def __init__(
        self,
        *,
        repository: PlatformRepository | None = None,
        base_config: RAGConfig | None = None,
    ) -> None:
        self.repository = repository or PlatformRepository()
        self.repository.ensure_defaults()
        self.base_config = base_config or RAGConfig()

    def answer(
        self,
        query: str,
        *,
        user_id: str = "default",
        kb_ids: list[str] | None = None,
        conversation_id: str | None = None,
        chat_profile_id: str | None = None,
        history: list[str] | None = None,
        topk: int | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        started_at = self._now()
        trace_id = f"trace_{uuid4().hex[:12]}"
        user = self._get_user(user_id)
        conversation = self._get_or_create_conversation(
            conversation_id=conversation_id,
            user=user,
        )
        chat_profile = self._get_chat_profile(chat_profile_id, user, conversation)
        active_kb_ids = self._resolve_kb_ids(kb_ids, conversation, chat_profile)
        kb_profiles = self._get_available_kbs(active_kb_ids, user, chat_profile)
        kb_profile = kb_profiles[0]
        llm_profile = self._get_llm_profile(chat_profile)
        config = self._build_config(
            kb_profile=kb_profile,
            llm_profile=llm_profile,
            chat_profile=chat_profile,
            topk=topk,
            max_rounds=max_rounds,
        )

        orchestrators_by_kb = {
            profile.kb_id: ModularRAGOrchestrator(
                self._build_config(
                    kb_profile=profile,
                    llm_profile=llm_profile,
                    chat_profile=chat_profile,
                    topk=topk,
                    max_rounds=max_rounds,
                )
            )
            for profile in kb_profiles
        }
        catalog_items = [
            KnowledgeBaseCatalogItem.from_profile(profile)
            for profile in kb_profiles
        ]
        orchestrator = orchestrators_by_kb[kb_profile.kb_id]
        agent = ToolCallingRAGAgent(
            orchestrator,
            config,
            orchestrators_by_kb=orchestrators_by_kb,
            available_kbs=[item.to_dict() for item in catalog_items],
            default_kb_ids=[profile.kb_id for profile in kb_profiles],
        )
        combined_history = self._conversation_history(conversation) + list(history or [])
        response = agent.run(
            query,
            history=combined_history,
            topk=topk or chat_profile.default_topk or kb_profile.default_topk,
            max_rounds=max_rounds or chat_profile.max_rounds,
        ).to_dict()

        self._append_conversation(
            conversation=conversation,
            query=query,
            response=str(response.get("response") or ""),
            kb_ids=active_kb_ids,
            chat_profile_id=chat_profile.profile_id,
            trace_id=trace_id,
        )
        if conversation_id:
            self.repository.conversations.upsert(conversation)

        trace = TraceRecord(
            trace_id=trace_id,
            user_id=user.user_id,
            conversation_id=conversation.conversation_id if conversation_id else None,
            kb_ids=active_kb_ids,
            query=query,
            response=str(response.get("response") or ""),
            status="completed",
            started_at=started_at,
            ended_at=self._now(),
            spans=list(response.get("traces") or []),
            metadata={
                "chat_profile_id": chat_profile.profile_id,
                "llm_profile_id": llm_profile.profile_id if llm_profile else None,
                "primary_kb_id": kb_profile.kb_id,
                "available_kbs": [item.to_dict() for item in catalog_items],
                "result": {
                    key: value
                    for key, value in response.items()
                    if key != "traces"
                },
            },
        )
        trace_path = self.repository.traces.save(trace)

        response["platform"] = {
            "trace_id": trace_id,
            "trace_path": str(trace_path),
            "user_id": user.user_id,
            "conversation_id": conversation.conversation_id if conversation_id else None,
            "kb_ids": active_kb_ids,
            "available_kb_ids": [profile.kb_id for profile in kb_profiles],
            "chat_profile_id": chat_profile.profile_id,
            "llm_profile_id": llm_profile.profile_id if llm_profile else None,
        }
        return response

    def _get_user(self, user_id: str) -> UserProfile:
        user = self.repository.users.get(user_id)
        if not user:
            raise ValueError(f"Unknown user profile: {user_id}")
        return user

    def _get_or_create_conversation(
        self,
        *,
        conversation_id: str | None,
        user: UserProfile,
    ) -> ConversationSession:
        if not conversation_id:
            return ConversationSession(
                conversation_id=f"ephemeral_{uuid4().hex[:8]}",
                user_id=user.user_id,
            )
        conversation = self.repository.conversations.get(conversation_id)
        if conversation:
            return conversation
        return ConversationSession(conversation_id=conversation_id, user_id=user.user_id)

    def _get_chat_profile(
        self,
        chat_profile_id: str | None,
        user: UserProfile,
        conversation: ConversationSession,
    ) -> ChatProfile:
        active_id = (
            chat_profile_id
            or conversation.chat_profile_id
            or user.default_chat_profile_id
            or "default"
        )
        profile = self.repository.chat_profiles.get(active_id)
        if not profile:
            raise ValueError(f"Unknown chat profile: {active_id}")
        return profile

    def _resolve_kb_ids(
        self,
        kb_ids: list[str] | None,
        conversation: ConversationSession,
        chat_profile: ChatProfile,
    ) -> list[str]:
        resolved = (
            list(kb_ids or [])
            or list(conversation.active_kb_ids)
            or list(chat_profile.default_kb_ids)
            or ["default"]
        )
        return [str(item) for item in resolved]

    def _get_primary_kb(self, kb_ids: list[str]) -> KnowledgeBaseProfile:
        kb_id = kb_ids[0] if kb_ids else "default"
        profile = self.repository.knowledge_bases.get(kb_id)
        if not profile:
            raise ValueError(f"Unknown knowledge base: {kb_id}")
        if profile.status != "ready":
            raise ValueError(f"Knowledge base is not ready: {kb_id}")
        return profile

    def _get_available_kbs(
        self,
        candidate_kb_ids: list[str],
        user: UserProfile,
        chat_profile: ChatProfile,
    ) -> list[KnowledgeBaseProfile]:
        allowed = self._allowed_kb_ids(user, chat_profile)
        requested = candidate_kb_ids or list(chat_profile.default_kb_ids) or ["default"]
        effective_ids = [
            kb_id for kb_id in requested
            if not allowed or kb_id in allowed
        ]
        if not effective_ids and allowed:
            effective_ids = list(allowed)
        if not effective_ids:
            effective_ids = ["default"]

        profiles: list[KnowledgeBaseProfile] = []
        for kb_id in effective_ids:
            profile = self.repository.knowledge_bases.get(kb_id)
            if not profile:
                raise ValueError(f"Unknown knowledge base: {kb_id}")
            if profile.status != "ready":
                raise ValueError(f"Knowledge base is not ready: {kb_id}")
            profiles.append(profile)
        return profiles

    @staticmethod
    def _allowed_kb_ids(user: UserProfile, chat_profile: ChatProfile) -> set[str]:
        allowed: set[str] = set()
        user_allowed = user.metadata.get("allowed_kb_ids") if user.metadata else None
        chat_allowed = chat_profile.metadata.get("allowed_kb_ids") if chat_profile.metadata else None
        for value in (user_allowed, chat_allowed):
            if isinstance(value, list):
                allowed.update(str(item) for item in value)
            elif value:
                allowed.add(str(value))
        return allowed

    def _get_llm_profile(self, chat_profile: ChatProfile) -> LLMProfile | None:
        if not chat_profile.chat_llm_profile_id:
            return None
        profile = self.repository.llm_profiles.get(chat_profile.chat_llm_profile_id)
        if not profile:
            raise ValueError(f"Unknown LLM profile: {chat_profile.chat_llm_profile_id}")
        return profile

    def _build_config(
        self,
        *,
        kb_profile: KnowledgeBaseProfile,
        llm_profile: LLMProfile | None,
        chat_profile: ChatProfile,
        topk: int | None,
        max_rounds: int | None,
    ) -> RAGConfig:
        config = replace(
            self.base_config,
            data_path=kb_profile.chunk_store_path,
            index_path=kb_profile.index_path,
            embeddings_path=kb_profile.embeddings_path,
            inverted_index_path=kb_profile.inverted_index_path,
        )
        if kb_profile.embedding_model_name:
            config = replace(config, embedding_model_name=kb_profile.embedding_model_name)
        if topk or chat_profile.default_topk or kb_profile.default_topk:
            config = replace(
                config,
                default_topk=int(topk or chat_profile.default_topk or kb_profile.default_topk or config.default_topk),
            )
        if max_rounds or chat_profile.max_rounds:
            config = replace(config, max_rounds=int(max_rounds or chat_profile.max_rounds or config.max_rounds))
        if llm_profile:
            config = self._apply_llm_profile(config, llm_profile)
        return config

    def _apply_llm_profile(self, config: RAGConfig, profile: LLMProfile) -> RAGConfig:
        if profile.model_type != "chat":
            return config
        api_key = profile.api_key or self._env(profile.api_key_env) or config.openai_api_key
        base_url = profile.base_url or self._env(profile.base_url_env) or config.openai_base_url
        model = profile.model_name or config.openai_model or os.getenv("OPENAI_MODEL")
        return replace(
            config,
            openai_api_key=api_key,
            openai_base_url=base_url,
            openai_model=model,
            openai_temperature=(
                float(profile.temperature)
                if profile.temperature is not None
                else config.openai_temperature
            ),
            openai_timeout=(
                float(profile.timeout)
                if profile.timeout is not None
                else config.openai_timeout
            ),
        )

    @staticmethod
    def _env(name: str | None) -> str | None:
        return os.getenv(name) if name else None

    @staticmethod
    def _conversation_history(conversation: ConversationSession) -> list[str]:
        history: list[str] = []
        for message in conversation.messages[-12:]:
            role = str(message.get("role") or "message")
            content = str(message.get("content") or "")
            if content:
                history.append(f"{role}: {content}")
        return history

    def _append_conversation(
        self,
        *,
        conversation: ConversationSession,
        query: str,
        response: str,
        kb_ids: list[str],
        chat_profile_id: str,
        trace_id: str,
    ) -> None:
        now = self._now()
        conversation.active_kb_ids = list(kb_ids)
        conversation.chat_profile_id = chat_profile_id
        conversation.messages.append(
            {
                "role": "user",
                "content": query,
                "created_at": now,
                "trace_id": trace_id,
            }
        )
        conversation.messages.append(
            {
                "role": "assistant",
                "content": response,
                "created_at": now,
                "trace_id": trace_id,
            }
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
