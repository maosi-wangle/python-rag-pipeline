from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_tool_name(value: str, *, max_len: int = 64) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "tool"
    if name[0].isdigit():
        name = f"tool_{name}"
    return name[:max_len]


def mcp_function_name(server_id: str, tool_name: str) -> str:
    server = safe_tool_name(server_id, max_len=24)
    tool = safe_tool_name(tool_name, max_len=36)
    return safe_tool_name(f"mcp_{server}_{tool}")


@dataclass(slots=True)
class MCPToolSpec:
    server_id: str
    tool_name: str
    function_name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    source: str = "mcp"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cached_tool(
        cls,
        server: "MCPServerProfile",
        payload: dict[str, Any],
    ) -> "MCPToolSpec":
        raw_name = str(payload.get("name") or payload.get("tool_name") or "")
        return cls(
            server_id=server.server_id,
            tool_name=raw_name,
            function_name=str(
                payload.get("function_name")
                or mcp_function_name(server.server_id, raw_name)
            ),
            description=str(payload.get("description") or ""),
            input_schema=_dict(payload.get("inputSchema") or payload.get("input_schema")),
            enabled=_bool(payload.get("enabled"), True),
            metadata=_dict(payload.get("metadata")),
        )

    def to_cached_tool(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "function_name": self.function_name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MCPServerProfile:
    server_id: str
    name: str = ""
    transport: str = "firecrawl"
    url: str = ""
    description: str = ""
    enabled: bool = True
    headers: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    tool_cache: list[dict[str, Any]] = field(default_factory=list)
    cache_updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MCPServerProfile":
        variables = _dict(payload.get("variables"))
        raw_tool_cache = payload.get("tool_cache")
        if raw_tool_cache is None:
            raw_tool_cache = variables.get("tools")
        tool_cache = _coerce_tool_cache(raw_tool_cache)
        return cls(
            server_id=str(payload.get("server_id") or payload.get("id") or payload["name"]),
            name=str(payload.get("name") or payload.get("server_id") or payload.get("id") or ""),
            transport=str(payload.get("transport") or payload.get("server_type") or "firecrawl"),
            url=str(payload.get("url") or ""),
            description=str(payload.get("description") or ""),
            enabled=_bool(payload.get("enabled"), True),
            headers=_dict(payload.get("headers")),
            variables=variables,
            tool_cache=tool_cache,
            cache_updated_at=payload.get("cache_updated_at"),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def tool_specs(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec.from_cached_tool(self, item)
            for item in self.tool_cache
            if isinstance(item, dict) and (item.get("name") or item.get("tool_name"))
        ]


def _coerce_tool_cache(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        items = []
        for name, tool in value.items():
            payload = dict(tool) if isinstance(tool, dict) else {}
            payload.setdefault("name", name)
            items.append(payload)
        return items
    return [dict(item) for item in _list(value) if isinstance(item, dict)]
