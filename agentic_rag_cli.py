from __future__ import annotations

import argparse
import json
from pathlib import Path

from faceaiRAG import FaceAiSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic modular self-RAG pipeline.")
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
    return parser.parse_args()


def load_history(path: str | None) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("History file must be a JSON list.")
    return [str(item) for item in payload]


def run_once(system: FaceAiSystem, query: str, history: list[str], args: argparse.Namespace) -> None:
    result = system.run_agentic_query(
        query,
        history=history,
        topk=args.topk,
        max_rounds=args.max_rounds,
    )
    if not args.print_traces:
        result = dict(result)
        result.pop("traces", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    history = load_history(args.history_file)
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
