from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from faceaiRAG import FaceAiSystem


REQUIRED_FIELDS = {
    "response",
    "query",
    "grounded",
    "retrieved_chunk_ids",
    "completeness",
    "rationale",
    "next_focus",
    "if_multi_turn",
    "need_followup",
    "round",
    "tool_rounds",
    "used_queries",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 0 agentic RAG smoke check.")
    parser.add_argument("--query", default="防晒需要注意什么")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--data-path", default="./knowledgeBase.json")
    parser.add_argument("--index-path", default="./knowledge.index")
    parser.add_argument("--embeddings-path", default="./knowledge_embeddings.npy")
    parser.add_argument("--inverted-index-path", default="./inverted_index.json")
    return parser.parse_args()


def validate_result(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(result))
    if missing:
        errors.append(f"Missing required fields: {missing}")

    if not isinstance(result.get("response"), str) or not result.get("response", "").strip():
        errors.append("response must be a non-empty string")
    if not isinstance(result.get("query"), str) or not result.get("query", "").strip():
        errors.append("query must be a non-empty string")
    if result.get("completeness") not in {"yes", "no"}:
        errors.append('completeness must be "yes" or "no"')
    if not isinstance(result.get("grounded"), bool):
        errors.append("grounded must be boolean")
    if not isinstance(result.get("if_multi_turn"), bool):
        errors.append("if_multi_turn must be boolean")
    if not isinstance(result.get("retrieved_chunk_ids"), list):
        errors.append("retrieved_chunk_ids must be a list")
    if not isinstance(result.get("used_queries"), list):
        errors.append("used_queries must be a list")
    if not isinstance(result.get("tool_rounds"), int) or result.get("tool_rounds", 0) < 1:
        errors.append("tool_rounds must be a positive integer")

    return errors


def main() -> int:
    args = parse_args()
    system = FaceAiSystem(
        dataPath=args.data_path,
        index_path=args.index_path,
        embeddings_path=args.embeddings_path,
        inverted_index_path=args.inverted_index_path,
    )
    result = system.run_agentic_query(
        args.query,
        topk=args.topk,
        max_rounds=args.max_rounds,
    )
    errors = validate_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if errors:
        print("\nPhase 0 smoke check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nPhase 0 smoke check passed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
