from __future__ import annotations

import hashlib
from typing import Any

from ..schemas import ChunkRecord, RetrievalHit
from .schemas import MCPServerProfile, MCPToolSpec
from .session import MCPSessionManager


class MCPRuntime:
    def __init__(
        self,
        *,
        servers: list[MCPServerProfile] | None = None,
        tools: list[MCPToolSpec] | None = None,
        session_manager: MCPSessionManager | None = None,
    ) -> None:
        self.servers = {server.server_id: server for server in servers or []}
        self.tools = {tool.function_name: tool for tool in tools or []}
        self.session_manager = session_manager or MCPSessionManager()

    def available_tools(self) -> list[MCPToolSpec]:
        return list(self.tools.values())

    def has_tool(self, function_name: str) -> bool:
        return function_name in self.tools

    def call_tool(
        self,
        function_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | int = 30,
    ) -> dict[str, Any]:
        tool = self.tools.get(function_name)
        if tool is None:
            return {"error": f"Unknown MCP tool: {function_name}"}
        server = self.servers.get(tool.server_id)
        if server is None:
            return {"error": f"Unknown MCP server: {tool.server_id}"}
        session = self.session_manager.get_session(server)
        try:
            result = session.call_tool(tool.tool_name, arguments, timeout=timeout)
        except Exception as exc:
            return {
                "error": f"{exc.__class__.__name__}: {exc}",
                "mcp_tool": function_name,
                "server_id": tool.server_id,
                "arguments": arguments,
            }

        payload = {
            "mcp_tool": function_name,
            "server_id": tool.server_id,
            "tool_name": tool.tool_name,
            "arguments": arguments,
            "result": result,
        }
        retrieval = self._to_retrieval(tool, arguments, result)
        if retrieval is not None:
            payload["retrieval"] = retrieval
            payload["chunks"] = [
                {
                    "chunk_id": hit.chunk_id,
                    "source": hit.chunk.source,
                    "title": hit.chunk.title,
                    "text": hit.text[:1200],
                    "score": hit.score,
                }
                for hit in retrieval["fused_hits"]
            ]
        return payload

    @staticmethod
    def _to_retrieval(
        tool: MCPToolSpec,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        results = result.get("results")
        if not isinstance(results, list):
            return None
        query = str(result.get("query") or arguments.get("query") or "")
        hits: list[RetrievalHit] = []
        for rank, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or item.get("description") or "").strip()
            if not content:
                continue
            title = str(item.get("title") or item.get("url") or f"Web result {rank}")
            url = str(item.get("url") or "")
            chunk_id = _stable_chunk_id(tool.server_id, url or title, rank)
            chunk = ChunkRecord(
                chunk_id=chunk_id,
                context=content,
                keywords=[],
                document_id=url or chunk_id,
                title=title,
                section="web_search",
                source=url or tool.server_id,
                metadata={
                    "mcp_server_id": tool.server_id,
                    "mcp_tool_name": tool.tool_name,
                    "url": url,
                    "description": item.get("description"),
                },
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(item.get("score") or 1.0),
                    retriever=f"mcp:{tool.server_id}:{tool.tool_name}",
                    query=query,
                    rank=rank,
                    metadata={
                        "source": "mcp",
                        "server_id": tool.server_id,
                        "tool_name": tool.tool_name,
                    },
                )
            )
        return {
            "query": query,
            "used_queries": [query] if query else [],
            "retrieval_results": [
                {
                    "kb_id": None,
                    "result": {
                        "retriever": f"mcp:{tool.server_id}:{tool.tool_name}",
                        "query": query,
                        "chunk_ids": [hit.chunk_id for hit in hits],
                    },
                }
            ],
            "fused_hits": hits,
            "retrieval_budget": {
                "source": "mcp",
                "server_id": tool.server_id,
                "tool_name": tool.tool_name,
                "result_count": len(hits),
            },
        }

    def close(self) -> None:
        self.session_manager.close_all()


def _stable_chunk_id(server_id: str, value: str, rank: int) -> str:
    digest = hashlib.sha1(f"{server_id}:{value}:{rank}".encode("utf-8")).hexdigest()[:12]
    return f"mcp_{digest}"
