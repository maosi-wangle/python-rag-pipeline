from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from faceaiRAG import FaceAiSystem


RAGAS_ALLOWED_FIELDS = {
    "user_input",
    "retrieved_contexts",
    "retrieved_context_ids",
    "reference_contexts",
    "reference_context_ids",
    "reference",
    "response",
    "rubric",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-focused RAGAS evaluation for the current RAG pipeline."
    )
    parser.add_argument(
        "--dataset",
        default="ragas_eval_dataset.sample.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k contexts to retrieve for each sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="ragas_outputs",
        help="Directory for enriched retrieval outputs and optional CSV results.",
    )
    parser.add_argument(
        "--skip-nonllm",
        action="store_true",
        help="Skip non-LLM retrieval metrics based on reference contexts.",
    )
    parser.add_argument(
        "--skip-id-metrics",
        action="store_true",
        help="Skip ID-based retrieval metrics even if *_context_ids are available.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def validate_dataset(samples: list[dict]) -> None:
    if not isinstance(samples, list) or not samples:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {idx} is not an object.")
        if not sample.get("user_input"):
            raise ValueError(f"Sample {idx} is missing user_input.")


def enrich_samples(system: FaceAiSystem, samples: list[dict], topk: int) -> list[dict]:
    enriched_samples: list[dict] = []
    for sample in samples:
        retrieval = system.retrieve_for_ragas(sample["user_input"], topk=topk)
        merged = dict(sample)
        merged.update(retrieval)
        enriched_samples.append(merged)
    return enriched_samples


def to_ragas_rows(samples: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        row = {}
        for key, value in sample.items():
            if key not in RAGAS_ALLOWED_FIELDS:
                continue
            if value is None:
                continue
            row[key] = value
        rows.append(row)
    return rows


def build_ragas_dataset(rows: list[dict]):
    try:
        from ragas import EvaluationDataset
    except ImportError:
        from ragas.dataset_schema import EvaluationDataset

    if hasattr(EvaluationDataset, "from_list"):
        return EvaluationDataset.from_list(rows)

    try:
        from ragas import SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import SingleTurnSample

    samples = [SingleTurnSample(**row) for row in rows]
    return EvaluationDataset(samples=samples)


def resolve_metric(metric_name: str):
    metric_cls = None

    try:
        from ragas.metrics import collections as ragas_metric_collections

        metric_cls = getattr(ragas_metric_collections, metric_name, None)
    except ImportError:
        metric_cls = None

    if metric_cls is None:
        from ragas import metrics as ragas_metrics

        metric_cls = getattr(ragas_metrics, metric_name, None)

    if metric_cls is None:
        return None

    return metric_cls()


def has_field(samples: list[dict], field_name: str) -> bool:
    return all(sample.get(field_name) for sample in samples)


def collect_metrics(samples: list[dict], args: argparse.Namespace):
    metrics = []
    skipped = []

    if not args.skip_nonllm:
        if has_field(samples, "reference_contexts"):
            for metric_name in (
                "NonLLMContextPrecisionWithReference",
                "NonLLMContextRecall",
            ):
                metric = resolve_metric(metric_name)
                if metric is None:
                    skipped.append(f"{metric_name} (not available in this ragas version)")
                    continue
                metrics.append(metric)
        else:
            skipped.append("Non-LLM context metrics (missing reference_contexts)")

    if not args.skip_id_metrics:
        if has_field(samples, "reference_context_ids") and has_field(
            samples, "retrieved_context_ids"
        ):
            for metric_name in ("IDBasedContextPrecision", "IDBasedContextRecall"):
                metric = resolve_metric(metric_name)
                if metric is None:
                    skipped.append(f"{metric_name} (not available in this ragas version)")
                    continue
                metrics.append(metric)
        else:
            skipped.append("ID-based metrics (missing reference_context_ids)")

    if not metrics:
        skipped_text = "; ".join(skipped) if skipped else "No metrics available."
        raise RuntimeError(f"Could not resolve any runnable RAGAS metrics: {skipped_text}")

    return metrics, skipped


def maybe_export_result_csv(result: Any, output_path: Path) -> None:
    if not hasattr(result, "to_pandas"):
        return

    try:
        df = result.to_pandas()
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"Skipping CSV export: {exc}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        from ragas import evaluate  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Run `python -m pip install -r requirements-ragas.txt` first."
        ) from exc

    raw_samples = load_json(dataset_path)
    validate_dataset(raw_samples)

    print("Initializing retrieval system...")
    system = FaceAiSystem()
    if not system.initialized:
        raise RuntimeError("FaceAiSystem failed to initialize.")

    print("Generating structured retrieval payloads for RAGAS...")
    enriched_samples = enrich_samples(system, raw_samples, topk=args.topk)
    enriched_path = output_dir / "retrieval_enriched.json"
    dump_json(enriched_path, enriched_samples)
    print(f"Saved enriched retrieval payloads to: {enriched_path}")

    metrics, skipped = collect_metrics(enriched_samples, args)
    print("Metrics to run:")
    for metric in metrics:
        print(f"  - {metric.name}")
    if skipped:
        print("Skipped metrics:")
        for item in skipped:
            print(f"  - {item}")

    ragas_rows = to_ragas_rows(enriched_samples)
    ragas_dataset = build_ragas_dataset(ragas_rows)

    from ragas import evaluate

    print("Running RAGAS evaluation...")
    result = evaluate(dataset=ragas_dataset, metrics=metrics)
    print("\nRAGAS summary:")
    print(result)

    maybe_export_result_csv(result, output_dir / "ragas_result.csv")


if __name__ == "__main__":
    main()
