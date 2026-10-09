"""Tune NMF summarization variants on a labeled validation JSONL split."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from app.core.evaluate import rouge_f1
from app.core.sentences import split_sentences
from app.core.summarize import (
    DEFAULT_PAIRWISE_LAMBDA,
    DEFAULT_QUERY_WEIGHT,
    METHOD_LOCAL_KL,
    METHOD_NMFTS,
    METHOD_SNMF,
    summarize,
)


METRIC_FIELDS = ("rouge1_f1", "rouge2_f1", "rougeL_f1")
TUNABLE_METHODS = ("local_kl_mmr", "local_kl_prf", METHOD_NMFTS, METHOD_SNMF)


def _load_articles(input_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    articles: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or not all(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("id", "article", "reference_summary")
            ):
                raise ValueError("missing id, article or reference_summary")
            if not split_sentences(item["article"]):
                raise ValueError("article has no sentences")
            articles.append(item)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"line": line_number, "reason": str(exc)})
    return articles, errors


def _configurations(
    methods: list[str],
    k_values: list[int],
    sentence_values: list[int],
    alpha_w_values: list[float],
    alpha_h_values: list[float],
    l1_ratio_values: list[float],
    mmr_values: list[float],
    pairwise_values: list[float],
    query_weight_values: list[float],
):
    for method in methods:
        for k, sentence_count in product(k_values, sentence_values):
            common = {
                "method": method,
                "k": None if k == 0 else k,
                "summary_sentences": sentence_count,
            }
            if method in {"local_kl_mmr", "local_kl_prf"}:
                weights = query_weight_values if method == "local_kl_prf" else [DEFAULT_QUERY_WEIGHT]
                for alpha_w, alpha_h, l1_ratio, mmr_lambda, query_weight in product(
                    alpha_w_values,
                    alpha_h_values,
                    l1_ratio_values,
                    mmr_values,
                    weights,
                ):
                    yield {
                        **common,
                        "alpha_w": alpha_w,
                        "alpha_h": alpha_h,
                        "l1_ratio": l1_ratio,
                        "mmr_lambda": mmr_lambda,
                        "query_weight": query_weight,
                        "pairwise_lambda": DEFAULT_PAIRWISE_LAMBDA,
                    }
            elif method == METHOD_NMFTS:
                for pairwise_lambda, mmr_lambda in product(pairwise_values, mmr_values):
                    yield {
                        **common,
                        "pairwise_lambda": pairwise_lambda,
                        "mmr_lambda": mmr_lambda,
                        "alpha_w": 0.0,
                        "alpha_h": 0.0,
                        "l1_ratio": 1.0,
                        "query_weight": DEFAULT_QUERY_WEIGHT,
                    }
            else:
                yield {
                    **common,
                    "pairwise_lambda": DEFAULT_PAIRWISE_LAMBDA,
                    "mmr_lambda": 0.7,
                    "alpha_w": 0.0,
                    "alpha_h": 0.0,
                    "l1_ratio": 1.0,
                    "query_weight": DEFAULT_QUERY_WEIGHT,
                }


def _summarize_config(item: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    method = config["method"]
    kwargs = {
        "text": item["article"],
        "k": config["k"],
        "summary_sentences": config["summary_sentences"],
        "seed": seed,
        "method": METHOD_NMFTS if method == METHOD_NMFTS else METHOD_SNMF if method == METHOD_SNMF else METHOD_LOCAL_KL,
        "pairwise_lambda": config["pairwise_lambda"],
        "mmr_lambda": config["mmr_lambda"],
        "alpha_w": config["alpha_w"],
        "alpha_h": config["alpha_h"],
        "l1_ratio": config["l1_ratio"],
    }
    if method == "local_kl_prf":
        kwargs["query"] = item["query"]
        kwargs["query_weight"] = config["query_weight"]
    return summarize(**kwargs)


def tune_params(
    input_path: Path,
    output_dir: Path,
    *,
    methods: list[str],
    k_values: list[int],
    summary_sentence_values: list[int],
    alpha_w_values: list[float],
    alpha_h_values: list[float],
    l1_ratio_values: list[float],
    mmr_values: list[float],
    pairwise_values: list[float],
    query_weight_values: list[float],
    seed: int = 42,
    rank_metric: str = "rougeL_f1",
) -> dict[str, Any]:
    if rank_metric not in METRIC_FIELDS:
        raise ValueError(f"rank_metric must be one of {', '.join(METRIC_FIELDS)}")
    if not input_path.is_file():
        raise ValueError(f"Validation dataset not found: {input_path}")
    unknown_methods = set(methods) - set(TUNABLE_METHODS)
    if unknown_methods or not methods:
        raise ValueError(f"methods must be selected from: {', '.join(TUNABLE_METHODS)}")
    if not k_values or any(value < 0 for value in k_values):
        raise ValueError("k candidates must be nonnegative; use 0 for automatic k")
    if not summary_sentence_values or any(value < 1 for value in summary_sentence_values):
        raise ValueError("summary sentence candidates must be positive")
    for name, values, upper in (
        ("alpha_w", alpha_w_values, 1.0),
        ("alpha_h", alpha_h_values, 1.0),
        ("l1_ratio", l1_ratio_values, 1.0),
        ("mmr_lambda", mmr_values, 1.0),
        ("pairwise_lambda", pairwise_values, 10.0),
        ("query_weight", query_weight_values, 1.0),
    ):
        if not values or any(value < 0 or value > upper for value in values):
            raise ValueError(f"{name} candidates must be between 0 and {upper}")

    articles, errors = _load_articles(input_path)
    if not articles:
        raise ValueError("No valid validation articles found; no tuning scores were produced")
    if "local_kl_prf" in methods and not all(
        isinstance(item.get("query"), str) and item["query"].strip() for item in articles
    ):
        raise ValueError("local_kl_prf requires a non-empty query for every validation article")

    results: list[dict[str, Any]] = []
    for config in _configurations(
        methods,
        k_values,
        summary_sentence_values,
        alpha_w_values,
        alpha_h_values,
        l1_ratio_values,
        mmr_values,
        pairwise_values,
        query_weight_values,
    ):
        start = perf_counter()
        scores: list[dict[str, float]] = []
        fallback_count = 0
        for item in articles:
            try:
                result = _summarize_config(item, config, seed)
                fallback_count += int(bool(result["fallback_reason"]))
                scores.append(rouge_f1(item["reference_summary"], result["summary"]))
            except (ValueError, TypeError) as exc:
                errors.append({"id": item["id"], "config": config, "reason": str(exc)})

        # Only rank complete runs so every configuration is compared on the same examples.
        if len(scores) != len(articles):
            continue
        results.append({
            **config,
            "k": "auto" if config["k"] is None else config["k"],
            "valid_articles": len(scores),
            "fallback_articles": fallback_count,
            "runtime_seconds": perf_counter() - start,
            **{field: fmean(score[field] for score in scores) for field in METRIC_FIELDS},
        })

    if not results:
        raise ValueError("No complete tuning configurations succeeded; see validation errors")
    results.sort(key=lambda row: (
        row[rank_metric],
        row["rouge1_f1"],
        -row["fallback_articles"],
        -row["runtime_seconds"],
    ), reverse=True)
    best_config = {
        key: results[0][key]
        for key in (
            "method", "k", "summary_sentences", "alpha_w", "alpha_h", "l1_ratio",
            "mmr_lambda", "pairwise_lambda", "query_weight",
        )
    }
    report = {
        "input": str(input_path),
        "split_policy": "validation only; do not tune on test data",
        "seed": seed,
        "rank_metric": rank_metric,
        "best_config": best_config,
        "best_scores": {field: results[0][field] for field in METRIC_FIELDS},
        "valid_articles": len(articles),
        "errors": errors,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tuning.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "tuning.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Labeled validation JSONL; never use the test split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=TUNABLE_METHODS, default=["local_kl_mmr", METHOD_NMFTS, METHOD_SNMF])
    parser.add_argument("--k", type=int, nargs="+", default=[0, 3], help="Use 0 for automatic masked-KL k")
    parser.add_argument("--summary-sentences", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--alpha-w", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--alpha-h", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--l1-ratio", type=float, nargs="+", default=[1.0])
    parser.add_argument("--mmr-lambda", type=float, nargs="+", default=[0.65, 0.8])
    parser.add_argument("--pairwise-lambda", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--query-weight", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-metric", choices=METRIC_FIELDS, default="rougeL_f1")
    args = parser.parse_args()
    report = tune_params(
        args.input,
        args.output,
        methods=args.methods,
        k_values=args.k,
        summary_sentence_values=args.summary_sentences,
        alpha_w_values=args.alpha_w,
        alpha_h_values=args.alpha_h,
        l1_ratio_values=args.l1_ratio,
        mmr_values=args.mmr_lambda,
        pairwise_values=args.pairwise_lambda,
        query_weight_values=args.query_weight,
        seed=args.seed,
        rank_metric=args.rank_metric,
    )
    print(json.dumps({
        "best_config": report["best_config"],
        "best_scores": report["best_scores"],
        "valid_articles": report["valid_articles"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
