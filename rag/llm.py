from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .config import RAGConfig


class LLMClient:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client: OpenAI | None = None
        if config.openai_api_key and config.openai_model:
            client_kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
            if config.openai_base_url:
                client_kwargs["base_url"] = config.openai_base_url
            self.client = OpenAI(**client_kwargs)

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.config.openai_model)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 1000,
    ) -> str:
        if not self.available or self.client is None or not self.config.openai_model:
            raise RuntimeError("LLM client is not configured.")

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.openai_temperature if temperature is None else temperature,
            max_completion_tokens=max_tokens,
            timeout=self.config.openai_timeout,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 1000,
    ) -> dict[str, Any]:
        if not self.available or self.client is None or not self.config.openai_model:
            raise RuntimeError("LLM client is not configured.")

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.config.openai_temperature if temperature is None else temperature,
            max_completion_tokens=max_tokens,
            timeout=self.config.openai_timeout,
        )
        content = (response.choices[0].message.content or "").strip()
        return self._parse_json(content)

    def _parse_json(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
