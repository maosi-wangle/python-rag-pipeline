from __future__ import annotations

from dataclasses import replace
from typing import Any

from .schemas import MCPServerProfile, MCPToolSpec, utc_now
from .session import MCPSessionManager


class MCPDiscoveryService:
    def __init__(self, session_manager: MCPSessionManager | None = None) -> None:
        self.session_manager = session_manager or MCPSessionManager()

    def refresh_server(
        self,
        server: MCPServerProfile,
        *,
        timeout: float | int = 10,
    ) -> MCPServerProfile:
        previous = {
            str(tool.get("name") or tool.get("tool_name")): bool(tool.get("enabled", True))
            for tool in server.tool_cache
            if isinstance(tool, dict)
        }
        session = self.session_manager.get_session(server)
        tools = session.list_tools(timeout=timeout)
        tool_cache = []
        for tool in tools:
            payload = tool.to_cached_tool()
            payload["enabled"] = previous.get(tool.tool_name, tool.enabled)
            tool_cache.append(payload)
        return replace(
            server,
            tool_cache=tool_cache,
            cache_updated_at=utc_now(),
        )

    def tool_specs(
        self,
        servers: list[MCPServerProfile],
        *,
        enabled_server_ids: list[str] | None = None,
        enabled_tools: dict[str, list[str]] | None = None,
        refresh_empty: bool = True,
    ) -> list[MCPToolSpec]:
        enabled_set = {str(item) for item in enabled_server_ids or []}
        enabled_map = {
            str(server_id): {str(tool_name) for tool_name in names}
            for server_id, names in (enabled_tools or {}).items()
            if isinstance(names, list)
        }
        specs: list[MCPToolSpec] = []
        for server in servers:
            if not server.enabled:
                continue
            if enabled_set and server.server_id not in enabled_set:
                continue
            active_server = server
            if refresh_empty and not active_server.tool_cache:
                active_server = self.refresh_server(active_server)
            selected_names = enabled_map.get(active_server.server_id)
            for spec in active_server.tool_specs():
                if not spec.enabled:
                    continue
                if selected_names and spec.tool_name not in selected_names and spec.function_name not in selected_names:
                    continue
                specs.append(spec)
        return specs


def parse_enabled_tools(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, list[str]] = {}
    for server_id, names in value.items():
        if isinstance(names, dict):
            parsed[str(server_id)] = [str(name) for name, enabled in names.items() if enabled]
        elif isinstance(names, list):
            parsed[str(server_id)] = [str(name) for name in names]
        elif names:
            parsed[str(server_id)] = [str(names)]
    return parsed
