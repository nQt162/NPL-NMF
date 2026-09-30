"""Grid-search NMF summarization parameters on a validation split."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from app.core.evaluate import rouge_f1
from app.core.sentences import split_sentences
from app.core.summarize import summarize


METRIC_FIELDS = ("rouge1_f1", "rouge2_f1", "rougeL_f1")


def _load_articles(input_path: Path) -> tuple[list[dict], list[dict]]:
    articles: list[dict] = []
    errors: list[dict] = []
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


def tune_params(
    input_path: Path,
    output_dir: Path,
    *,
    k_values: list[int],
    summary_sentence_values: list[int],
    alpha_values: list[float],
    beta_values: list[float],
    gamma_values: list[float],
    position_weight_values: list[float],
    length_weight_values: list[float],
    seed: int = 42,
    rank_metric: str = "rougeL_f1",
) -> dict:
    if rank_metric not in METRIC_FIELDS:
        raise ValueError(f"rank_metric must be one of {', '.join(METRIC_FIELDS)}")
    if not input_path.is_file():
        raise ValueError(f"Dataset not found: {input_path}")

    articles, errors = _load_articles(input_path)
    if not articles:
        raise ValueError("No valid validation articles found")

    results: list[dict] = []
    for k, summary_sentences, alpha, beta, gamma, position_weight, length_weight in product(
        k_values,
        summary_sentence_values,
        alpha_values,
        beta_values,
        gamma_values,
        position_weight_values,
        length_weight_values,
    ):
        start = perf_counter()
        scores: list[dict] = []
        fallback_count = 0
        failed_count = 0
        for item in articles:
            try:
                result = summarize(
                    item["article"],
                    k=k,
                    summary_sentences=summary_sentences,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    position_weight=position_weight,
                    length_weight=length_weight,
                    seed=seed,
                )
                fallback_count += int(bool(result["fallback_reason"]))
                scores.append(rouge_f1(item["reference_summary"], result["summary"]))
            except (ValueError, TypeError) as exc:
                failed_count += 1
                errors.append({
                    "id": item.get("id", ""),
                    "reason": str(exc),
                    "config": {
                        "k": k,
                        "summary_sentences": summary_sentences,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "position_weight": position_weight,
                        "length_weight": length_weight,
                    },
                })
        if not scores:
            continue
        results.append({
            "k": k,
            "summary_sentences": summary_sentences,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "position_weight": position_weight,
            "length_weight": length_weight,
            "valid_articles": len(scores),
            "failed_articles": failed_count,
            "fallback_articles": fallback_count,
            "runtime_seconds": perf_counter() - start,
            **{field: fmean(score[field] for score in scores) for field in METRIC_FIELDS},
        })

    if not results:
        raise ValueError("No valid tuning results produced")

    results.sort(key=lambda row: (
        row[rank_metric],
        row["rouge1_f1"],
        -row["fallback_articles"],
        -row["runtime_seconds"],
    ), reverse=True)
    report = {
        "input": str(input_path),
        "seed": seed,
        "rank_metric": rank_metric,
        "best_config": {
            key: results[0][key]
            for key in (
                "k",
                "summary_sentences",
                "alpha",
                "beta",
                "gamma",
                "position_weight",
                "length_weight",
            )
        },
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--summary-sentences", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0])
    parser.add_argument("--beta", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--position-weight", type=float, nargs="+", default=[0.0, 0.1, 0.15])
    parser.add_argument("--length-weight", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-metric", choices=METRIC_FIELDS, default="rougeL_f1")
    args = parser.parse_args()
    report = tune_params(
        args.input,
        args.output,
        k_values=args.k,
        summary_sentence_values=args.summary_sentences,
        alpha_values=args.alpha,
        beta_values=args.beta,
        gamma_values=args.gamma,
        position_weight_values=args.position_weight,
        length_weight_values=args.length_weight,
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
