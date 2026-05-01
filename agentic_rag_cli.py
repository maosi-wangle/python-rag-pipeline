from __future__ import annotations

import argparse
import json
from pathlib import Path

from faceaiRAG import FaceAiSystem
from rag.platform.repositories import PlatformRepository
from rag.platform.service import AgenticRAGService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tool-calling agentic modular RAG pipeline.")
    parser.add_argument("--query", help="Single query to run.")
    parser.add_argument(
        "--history-file",
        help="Optional JSON file containing a list of previous turns.",
    )
    parser.add_argument("--data-path", default="./knowledgeBase.json")
    parser.add_argument("--index-path", default="./knowledge.index")
    parser.add_argument("--embeddings-path", default="./knowledge_embeddings.npy")
    parser.add_argument("--inverted-index-path", default="./inverted_index.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--print-traces", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--platform-root", help="Directory containing platform JSON config files.")
    parser.add_argument("--user", help="Platform user profile id.")
    parser.add_argument("--kb", action="append", help="Platform knowledge base id. Can be repeated or comma-separated.")
    parser.add_argument("--conversation", help="Platform conversation id to load and persist.")
    parser.add_argument("--chat-profile", help="Platform chat profile id.")
    return parser.parse_args()


def load_history(path: str | None) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("History file must be a JSON list.")
    return [str(item) for item in payload]


def platform_mode(args: argparse.Namespace) -> bool:
    return any(
        [
            args.platform_root,
            args.user,
            args.kb,
            args.conversation,
            args.chat_profile,
        ]
    )


def parse_kb_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    kb_ids: list[str] = []
    for value in values:
        kb_ids.extend(item.strip() for item in value.split(",") if item.strip())
    return kb_ids or None


def print_result(result: dict, args: argparse.Namespace) -> None:
    if not args.print_traces:
        result = dict(result)
        result.pop("traces", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_once(system: FaceAiSystem, query: str, history: list[str], args: argparse.Namespace) -> None:
    result = system.run_agentic_query(
        query,
        history=history,
        topk=args.topk,
        max_rounds=args.max_rounds,
    )
    print_result(result, args)


def run_once_platform(
    service: AgenticRAGService,
    query: str,
    history: list[str],
    args: argparse.Namespace,
) -> None:
    result = service.answer(
        query,
        user_id=args.user or "default",
        kb_ids=parse_kb_ids(args.kb),
        conversation_id=args.conversation,
        chat_profile_id=args.chat_profile,
        history=history,
        topk=args.topk,
        max_rounds=args.max_rounds,
    )
    print_result(result, args)


def main() -> None:
    args = parse_args()
    history = load_history(args.history_file)
    if platform_mode(args):
        repository = PlatformRepository(args.platform_root or "data/platform")
        service = AgenticRAGService(repository=repository)

        if args.interactive:
            live_history = list(history)
            while True:
                query = input("Query: ").strip()
                if not query:
                    continue
                if query.lower() in {"exit", "quit"}:
                    break
                run_once_platform(service, query, live_history, args)
                live_history.append(query)
            return

        if not args.query:
            raise ValueError("Either --query or --interactive is required.")

        run_once_platform(service, args.query, history, args)
        return

    system = FaceAiSystem(
        dataPath=args.data_path,
        index_path=args.index_path,
        embeddings_path=args.embeddings_path,
        inverted_index_path=args.inverted_index_path,
    )

    if args.interactive:
        live_history = list(history)
        while True:
            query = input("Query: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            run_once(system, query, live_history, args)
            live_history.append(query)
        return

    if not args.query:
        raise ValueError("Either --query or --interactive is required.")

    run_once(system, args.query, history, args)


if __name__ == "__main__":
    main()
