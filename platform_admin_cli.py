from __future__ import annotations

import argparse
import json
from typing import Any

from rag.mcp.discovery import MCPDiscoveryService
from rag.mcp.schemas import MCPServerProfile
from rag.mcp.session import MCPSessionManager
from rag.platform.repositories import PlatformRepository
from rag.platform.schemas import ChatProfile, KnowledgeBaseProfile, LLMProfile, UserProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage AstraRAG platform profiles.")
    parser.add_argument("--platform-root", default="data/platform", help="Directory containing platform JSON config files.")
    subparsers = parser.add_subparsers(dest="resource", required=True)

    users = subparsers.add_parser("users", help="Manage user profiles.")
    user_actions = users.add_subparsers(dest="action", required=True)
    user_actions.add_parser("list", help="List user profiles.")
    users_create = user_actions.add_parser("create", help="Create or update a user profile.")
    users_create.add_argument("--user", required=True, help="User profile id.")
    users_create.add_argument("--display-name", help="Display name.")
    users_create.add_argument("--default-chat-profile", default="default", help="Default chat profile id.")

    llms = subparsers.add_parser("llms", help="Manage LLM profiles.")
    llm_actions = llms.add_subparsers(dest="action", required=True)
    llm_actions.add_parser("list", help="List LLM profiles.")
    llms_create = llm_actions.add_parser("create", help="Create or update an LLM profile.")
    llms_create.add_argument("--profile", required=True, help="LLM profile id.")
    llms_create.add_argument("--model", required=True, help="Model name.")
    llms_create.add_argument("--provider", default="openai-compatible", help="Model provider.")
    llms_create.add_argument("--model-type", default="chat", help="Model type, e.g. chat, embedding, rerank.")
    llms_create.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    llms_create.add_argument("--base-url", help="OpenAI-compatible base URL.")
    llms_create.add_argument("--base-url-env", default="OPENAI_BASE_URL", help="Environment variable containing the base URL.")
    llms_create.add_argument("--temperature", type=float, help="Default temperature.")
    llms_create.add_argument("--timeout", type=float, help="Request timeout in seconds.")
    llms_create.add_argument("--max-tokens", type=int, help="Optional max token hint.")

    chats = subparsers.add_parser("chats", help="Manage chat profiles.")
    chat_actions = chats.add_subparsers(dest="action", required=True)
    chat_actions.add_parser("list", help="List chat profiles.")
    chats_create = chat_actions.add_parser("create", help="Create or update a chat profile.")
    chats_create.add_argument("--profile", required=True, help="Chat profile id.")
    chats_create.add_argument("--name", help="Display name.")
    chats_create.add_argument("--llm", required=True, help="Chat LLM profile id.")
    chats_create.add_argument("--kb", action="append", required=True, help="Default KB id. Can be repeated or comma-separated.")
    chats_create.add_argument("--topk", type=int, help="Default retrieval topk.")
    chats_create.add_argument("--max-rounds", type=int, help="Default agent max rounds.")

    kbs = subparsers.add_parser("kbs", help="Manage knowledge base profiles.")
    kb_actions = kbs.add_subparsers(dest="action", required=True)
    kb_actions.add_parser("list", help="List knowledge base profiles.")
    kbs_create = kb_actions.add_parser("create", help="Create or update a knowledge base profile.")
    kbs_create.add_argument("--kb", required=True, help="Knowledge base id.")
    kbs_create.add_argument("--name", help="Display name.")
    kbs_create.add_argument("--description", default="", help="Description.")
    kbs_create.add_argument("--language", default="zh", help="Knowledge base language.")
    kbs_create.add_argument("--chunk-store-path", default="./knowledgeBase.json")
    kbs_create.add_argument("--index-path", default="./knowledge.index")
    kbs_create.add_argument("--embeddings-path", default="./knowledge_embeddings.npy")
    kbs_create.add_argument("--inverted-index-path", default="./inverted_index.json")
    kbs_create.add_argument("--retrieval-mode", action="append", help="Retrieval mode. Can be repeated or comma-separated.")
    kbs_create.add_argument("--parser-id", default="legacy_json")
    kbs_create.add_argument("--status", default="ready")

    mcps = subparsers.add_parser("mcps", help="Manage MCP server profiles.")
    mcp_actions = mcps.add_subparsers(dest="action", required=True)
    mcp_actions.add_parser("list", help="List MCP servers.")
    mcp_tools = mcp_actions.add_parser("tools", help="List cached tools for an MCP server.")
    mcp_tools.add_argument("--server", help="MCP server id. Omit to show all.")
    mcp_create = mcp_actions.add_parser("create", help="Create or update an MCP server.")
    mcp_create.add_argument("--server", required=True, help="MCP server id.")
    mcp_create.add_argument("--name", help="Display name.")
    mcp_create.add_argument("--transport", default="firecrawl", help="MCP transport/provider. First supported provider: firecrawl.")
    mcp_create.add_argument("--url", default="https://api.firecrawl.dev/v2/search", help="Server URL or provider API URL.")
    mcp_create.add_argument("--description", default="", help="Description.")
    mcp_create.add_argument("--api-key-env", default="FIRECRAWL_API_KEY", help="Environment variable for the provider API key.")
    mcp_create.add_argument("--disabled", action="store_true", help="Create the server disabled.")
    mcp_refresh = mcp_actions.add_parser("refresh", help="Refresh MCP tool cache.")
    mcp_refresh.add_argument("--server", required=True, help="MCP server id.")
    mcp_refresh.add_argument("--timeout", type=float, default=10)
    return parser.parse_args()


def split_values(values: list[str] | None, default: list[str] | None = None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items or list(default or [])


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def repo_from_args(args: argparse.Namespace) -> PlatformRepository:
    repository = PlatformRepository(args.platform_root)
    repository.ensure_defaults()
    return repository


def handle_users(args: argparse.Namespace, repository: PlatformRepository) -> None:
    if args.action == "list":
        print_json(
            {
                "users": [user.to_dict() for user in repository.users.all()],
                "path": str(repository.users.path),
            }
        )
        return

    user = UserProfile(
        user_id=args.user,
        display_name=args.display_name or args.user,
        default_chat_profile_id=args.default_chat_profile,
        permissions=["rag:query"],
    )
    repository.users.upsert(user)
    print_json(
        {
            "status": "ok",
            "action": "users.create",
            "user": user.to_dict(),
            "path": str(repository.users.path),
        }
    )


def handle_llms(args: argparse.Namespace, repository: PlatformRepository) -> None:
    if args.action == "list":
        print_json(
            {
                "llm_profiles": [profile.to_dict() for profile in repository.llm_profiles.all()],
                "path": str(repository.llm_profiles.path),
            }
        )
        return

    profile = LLMProfile(
        profile_id=args.profile,
        provider=args.provider,
        model_type=args.model_type,
        model_name=args.model,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    repository.llm_profiles.upsert(profile)
    print_json(
        {
            "status": "ok",
            "action": "llms.create",
            "llm_profile": profile.to_dict(),
            "path": str(repository.llm_profiles.path),
        }
    )


def handle_chats(args: argparse.Namespace, repository: PlatformRepository) -> None:
    if args.action == "list":
        print_json(
            {
                "chat_profiles": [profile.to_dict() for profile in repository.chat_profiles.all()],
                "path": str(repository.chat_profiles.path),
            }
        )
        return

    profile = ChatProfile(
        profile_id=args.profile,
        name=args.name or args.profile,
        chat_llm_profile_id=args.llm,
        default_kb_ids=split_values(args.kb),
        default_topk=args.topk,
        max_rounds=args.max_rounds,
    )
    repository.chat_profiles.upsert(profile)
    print_json(
        {
            "status": "ok",
            "action": "chats.create",
            "chat_profile": profile.to_dict(),
            "path": str(repository.chat_profiles.path),
        }
    )


def handle_kbs(args: argparse.Namespace, repository: PlatformRepository) -> None:
    if args.action == "list":
        print_json(
            {
                "knowledge_bases": [profile.to_dict() for profile in repository.knowledge_bases.all()],
                "path": str(repository.knowledge_bases.path),
            }
        )
        return

    profile = KnowledgeBaseProfile(
        kb_id=args.kb,
        name=args.name or args.kb,
        description=args.description,
        language=args.language,
        chunk_store_path=args.chunk_store_path,
        index_path=args.index_path,
        embeddings_path=args.embeddings_path,
        inverted_index_path=args.inverted_index_path,
        retrieval_modes=split_values(args.retrieval_mode, ["semantic", "keyword"]),
        parser_id=args.parser_id,
        status=args.status,
    )
    repository.knowledge_bases.upsert(profile)
    print_json(
        {
            "status": "ok",
            "action": "kbs.create",
            "knowledge_base": profile.to_dict(),
            "path": str(repository.knowledge_bases.path),
        }
    )


def handle_mcps(args: argparse.Namespace, repository: PlatformRepository) -> None:
    if args.action == "list":
        print_json(
            {
                "mcp_servers": [server.to_dict() for server in repository.mcp_servers.all()],
                "path": str(repository.mcp_servers.path),
            }
        )
        return

    if args.action == "tools":
        servers = repository.mcp_servers.all()
        if args.server:
            servers = [server for server in servers if server.server_id == args.server]
        print_json(
            {
                "tools": {
                    server.server_id: [tool.to_dict() for tool in server.tool_specs()]
                    for server in servers
                },
                "path": str(repository.mcp_servers.path),
            }
        )
        return

    if args.action == "create":
        server = MCPServerProfile(
            server_id=args.server,
            name=args.name or args.server,
            transport=args.transport,
            url=args.url,
            description=args.description,
            enabled=not args.disabled,
            headers={"Authorization": f"Bearer ${{{args.api_key_env}}}"},
            variables={
                "api_key_env": args.api_key_env,
                "api_url": args.url,
            },
            metadata={"provider": args.transport},
        )
        repository.mcp_servers.upsert(server)
        print_json(
            {
                "status": "ok",
                "action": "mcps.create",
                "mcp_server": server.to_dict(),
                "path": str(repository.mcp_servers.path),
            }
        )
        return

    if args.action == "refresh":
        server = repository.mcp_servers.get(args.server)
        if not server:
            raise ValueError(f"Unknown MCP server: {args.server}")
        discovery = MCPDiscoveryService(MCPSessionManager())
        updated = discovery.refresh_server(server, timeout=args.timeout)
        repository.mcp_servers.upsert(updated)
        print_json(
            {
                "status": "ok",
                "action": "mcps.refresh",
                "mcp_server": updated.to_dict(),
                "tools": [tool.to_dict() for tool in updated.tool_specs()],
                "path": str(repository.mcp_servers.path),
            }
        )
        return

    raise ValueError(f"Unsupported mcps action: {args.action}")


def main() -> None:
    args = parse_args()
    repository = repo_from_args(args)
    handlers = {
        "users": handle_users,
        "llms": handle_llms,
        "chats": handle_chats,
        "kbs": handle_kbs,
        "mcps": handle_mcps,
    }
    handlers[args.resource](args, repository)


if __name__ == "__main__":
    main()
