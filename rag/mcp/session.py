from __future__ import annotations

import os
from string import Template
from typing import Any

import httpx

from .schemas import MCPServerProfile, MCPToolSpec, mcp_function_name


class MCPSessionError(RuntimeError):
    pass


class BaseMCPSession:
    def __init__(self, server: MCPServerProfile) -> None:
        self.server = server

    def list_tools(self, timeout: float | int = 10) -> list[MCPToolSpec]:
        raise NotImplementedError

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float | int = 20,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return


class UnsupportedMCPSession(BaseMCPSession):
    def list_tools(self, timeout: float | int = 10) -> list[MCPToolSpec]:
        raise MCPSessionError(
            f"Unsupported MCP transport: {self.server.transport}. "
            "Only the firecrawl provider is implemented in this build."
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float | int = 20,
    ) -> dict[str, Any]:
        raise MCPSessionError(
            f"Unsupported MCP transport: {self.server.transport}. "
            "Only the firecrawl provider is implemented in this build."
        )


class FirecrawlSearchSession(BaseMCPSession):
    tool_name = "firecrawl_search"

    def __init__(self, server: MCPServerProfile) -> None:
        super().__init__(server)
        self._client: httpx.Client | None = None

    def list_tools(self, timeout: float | int = 10) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                server_id=self.server.server_id,
                tool_name=self.tool_name,
                function_name=mcp_function_name(self.server.server_id, self.tool_name),
                description=(
                    "Search the live web for up-to-date information. Use this when local "
                    "knowledge base retrieval is empty, weak, unavailable, or the user asks "
                    "for current web information."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to run on the web.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                            "description": "Maximum number of web results.",
                        },
                        "tbs": {
                            "type": "string",
                            "description": "Optional time filter, such as qdr:d, qdr:w, qdr:m, or qdr:y.",
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional search location hint.",
                        },
                    },
                    "required": ["query"],
                },
                enabled=True,
                metadata={"provider": "firecrawl", "kind": "web_search"},
            )
        ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float | int = 20,
    ) -> dict[str, Any]:
        if name != self.tool_name:
            raise MCPSessionError(f"Unknown Firecrawl tool: {name}")

        query = str(arguments.get("query") or "").strip()
        if not query:
            raise MCPSessionError("firecrawl_search requires a non-empty query.")

        api_key = self._api_key()
        if not api_key:
            raise MCPSessionError(
                "FIRECRAWL_API_KEY is not configured. Set it in the environment "
                "or configure variables.api_key / variables.api_key_env."
            )

        payload = self._build_payload(arguments)
        headers = self._headers(api_key)
        client = self._get_client(timeout)
        response = client.post(self._api_url(), json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        results = _normalize_firecrawl_results(body)
        return {
            "provider": "firecrawl",
            "tool_name": name,
            "query": query,
            "results": results,
            "raw": body,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _get_client(self, timeout: float | int) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=float(timeout))
        return self._client

    def _api_url(self) -> str:
        return str(
            self.server.variables.get("api_url")
            or self.server.url
            or "https://api.firecrawl.dev/v2/search"
        )

    def _api_key(self) -> str:
        variables = self.server.variables or {}
        api_key = str(variables.get("api_key") or "").strip()
        if api_key:
            return api_key
        api_key_env = str(variables.get("api_key_env") or "FIRECRAWL_API_KEY")
        return os.getenv(api_key_env, "").strip()

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self._render_headers().items()
            if key and value
        }
        headers.setdefault("Authorization", f"Bearer {api_key}")
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _render_headers(self) -> dict[str, str]:
        variables = dict(self.server.variables or {})
        for key, value in os.environ.items():
            variables.setdefault(key, value)
        headers: dict[str, str] = {}
        for key, value in (self.server.headers or {}).items():
            rendered_key = Template(str(key)).safe_substitute(variables).strip()
            rendered_value = Template(str(value)).safe_substitute(variables).strip()
            if rendered_key and rendered_value:
                headers[rendered_key] = rendered_value
        return headers

    @staticmethod
    def _build_payload(arguments: dict[str, Any]) -> dict[str, Any]:
        limit = int(arguments.get("limit") or 5)
        payload: dict[str, Any] = {
            "query": str(arguments.get("query") or ""),
            "limit": max(1, min(limit, 10)),
            "sources": arguments.get("sources") or ["web"],
            "scrapeOptions": arguments.get("scrapeOptions")
            or {"formats": ["markdown"], "onlyMainContent": True},
        }
        for key in ("tbs", "location", "country", "lang"):
            value = arguments.get(key)
            if value:
                payload[key] = value
        return payload


class MCPSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, BaseMCPSession] = {}

    def get_session(self, server: MCPServerProfile) -> BaseMCPSession:
        session = self._sessions.get(server.server_id)
        if session is not None:
            return session
        session = self._create_session(server)
        self._sessions[server.server_id] = session
        return session

    def close_session(self, server_id: str) -> None:
        session = self._sessions.pop(server_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        for server_id in list(self._sessions):
            self.close_session(server_id)

    @staticmethod
    def _create_session(server: MCPServerProfile) -> BaseMCPSession:
        transport = str(server.transport or "").strip().lower()
        if transport in {"firecrawl", "firecrawl_api", "web_search"}:
            return FirecrawlSearchSession(server)
        return UnsupportedMCPSession(server)


def _normalize_firecrawl_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    data = body.get("data", body)
    if isinstance(data, dict):
        candidates = data.get("web") or data.get("results") or data.get("data") or []
    else:
        candidates = data if isinstance(data, list) else []
    results: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        markdown = str(item.get("markdown") or item.get("content") or "").strip()
        description = str(
            item.get("description")
            or item.get("snippet")
            or item.get("summary")
            or ""
        ).strip()
        results.append(
            {
                "title": str(item.get("title") or item.get("url") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "description": description,
                "content": markdown or description,
                "score": float(item.get("score") or item.get("rank") or 1.0),
            }
        )
    return results
