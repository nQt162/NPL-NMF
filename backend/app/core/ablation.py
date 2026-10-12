"""Paired L1 ablation for Local KL-NMF + MMR."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any, Callable

from .evaluate import rouge_f1
from .preprocess import preprocess_sentences
from .sentences import split_sentences
from .summarize import (
    METHOD_LOCAL_KL,
    _select_or_resolve_k,
    summarize,
)
from .vectorize import (
    LOCAL_TFIDF_MAX_FEATURES,
    LOCAL_TFIDF_NGRAM_RANGE,
    fit_sentence_vectorizer,
)


FACTOR_ZERO_THRESHOLD = 1e-8
ROUGE_FIELDS = ("rouge1_f1", "rouge2_f1", "rougeL_f1")
METRIC_FIELDS = (
    "id",
    "seed",
    "configuration",
    "split",
    "k",
    "k_selection_method",
    "k_selection_seconds",
    "alpha_w",
    "alpha_h",
    "l1_ratio",
    *ROUGE_FIELDS,
    "runtime_seconds",
    "sentence_count",
    "w_zero_fraction",
    "h_zero_fraction",
    "active_topics_per_sentence",
    "active_terms_per_topic",
    "fallback_reason",
)


def parse_articles_jsonl(content: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Parse JSONL, retaining valid rows and reporting malformed/duplicate rows."""
    articles: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("Each JSONL row must be an object")
            values = {key: item.get(key) for key in ("id", "article", "reference_summary")}
            if not all(isinstance(value, str) and value.strip() for value in values.values()):
                raise ValueError("Required non-empty string fields: id, article, reference_summary")
            item_id = values["id"].strip()
            if item_id in seen_ids:
                raise ValueError(f"Duplicate id: {item_id}")
            if not split_sentences(values["article"]):
                raise ValueError("Article contains no sentence")
            seen_ids.add(item_id)
            articles.append({
                "id": item_id,
                "article": values["article"],
                "reference_summary": values["reference_summary"],
            })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"line": line_number, "reason": str(exc)})
    if not articles:
        raise ValueError("JSONL contains no valid articles with id, article, and reference_summary")
    return articles, errors


def _resolve_shared_k(text: str, requested_k: int | None, seed: int) -> tuple[int, str, float]:
    started = perf_counter()
    sentences = split_sentences(text)
    processed = preprocess_sentences(sentences)
    V, terms, _ = fit_sentence_vectorizer(
        processed,
        ngram_range=LOCAL_TFIDF_NGRAM_RANGE,
        max_features=LOCAL_TFIDF_MAX_FEATURES,
    )
    if len(terms) == 0:
        k_used = min(requested_k or max(1, len(sentences) // 2), max(1, len(sentences)))
        method = "fallback_no_vocabulary"
    else:
        # Select k once from the unregularized KL baseline, then hold it fixed.
        k_used, method, _ = _select_or_resolve_k(V, requested_k, seed)
    return k_used, method, perf_counter() - started


def _configuration_params(alpha: float) -> dict[str, tuple[float, float, float]]:
    return {
        "kl_no_l1": (0.0, 0.0, 1.0),
        "kl_l1_w_only": (alpha, 0.0, 1.0),
        "kl_l1_h_only": (0.0, alpha, 1.0),
        "kl_l1_w_h": (alpha, alpha, 1.0),
    }


def _mean_std(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "mean": fmean(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
    }


def run_l1_ablation(
    articles: list[dict[str, str]],
    output_dir: Path,
    *,
    split: str,
    dataset_name: str | None = None,
    k: int | None = None,
    summary_sentences: int = 3,
    mmr_lambda: float = 0.7,
    alpha: float = 0.001,
    seeds: list[int] | None = None,
    input_errors: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run all four conditions on matched article/seed pairs and write artifacts."""
    seeds = list(seeds or [42])
    if not articles:
        raise ValueError("At least one valid article is required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")
    if k is not None and k < 1:
        raise ValueError("k must be positive when specified")
    if summary_sentences < 1:
        raise ValueError("summary_sentences must be positive")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be between 0 and 1")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = _configuration_params(alpha)
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    errors = list(input_errors or [])
    total_units = len(articles) * len(seeds)
    completed_units = 0

    for article in articles:
        text = article["article"]
        for seed in seeds:
            shared_k, k_method, k_seconds = _resolve_shared_k(text, k, seed)
            for name, (alpha_w, alpha_h, l1_ratio) in configurations.items():
                started = perf_counter()
                try:
                    result = summarize(
                        text=text,
                        k=shared_k,
                        summary_sentences=summary_sentences,
                        mmr_lambda=mmr_lambda,
                        seed=seed,
                        method=METHOD_LOCAL_KL,
                        query=None,
                        alpha_w=alpha_w,
                        alpha_h=alpha_h,
                        l1_ratio=l1_ratio,
                        include_factor_stats=True,
                        factor_zero_threshold=FACTOR_ZERO_THRESHOLD,
                    )
                    runtime = perf_counter() - started
                    summary = result["summary"]
                    factor_stats = result.get("factor_sparsity") or {}
                    row = {
                        "id": article["id"],
                        "seed": seed,
                        "configuration": name,
                        "split": split,
                        "k": result["k"],
                        "k_selection_method": k_method,
                        "k_selection_seconds": k_seconds,
                        "alpha_w": alpha_w,
                        "alpha_h": alpha_h,
                        "l1_ratio": l1_ratio,
                        **rouge_f1(article["reference_summary"], summary),
                        "runtime_seconds": runtime,
                        "sentence_count": len(result["selected_indices"]),
                        "w_zero_fraction": factor_stats.get("w_zero_fraction"),
                        "h_zero_fraction": factor_stats.get("h_zero_fraction"),
                        "active_topics_per_sentence": factor_stats.get("active_topics_per_sentence"),
                        "active_terms_per_topic": factor_stats.get("active_terms_per_topic"),
                        "fallback_reason": result.get("fallback_reason") or "",
                    }
                    rows.append(row)
                    predictions.append({
                        **row,
                        "summary": summary,
                        "selected_indices": result["selected_indices"],
                    })
                except Exception as exc:
                    errors.append({
                        "id": article["id"],
                        "seed": seed,
                        "configuration": name,
                        "reason": str(exc),
                    })
            completed_units += 1
            if progress_callback:
                progress_callback(completed_units, total_units)

    successful = {
        name: [
            row for row in rows
            if row["configuration"] == name and not row["fallback_reason"]
        ]
        for name in configurations
    }
    summaries: dict[str, Any] = {}
    baseline_by_pair = {
        (row["id"], row["seed"]): row
        for row in successful["kl_no_l1"]
    }
    paired_deltas: dict[str, Any] = {}
    for name in configurations:
        group = successful[name]
        summaries[name] = {
            "n_success": len(group),
            "n_fallback": sum(
                row["configuration"] == name and bool(row["fallback_reason"])
                for row in rows
            ),
            "n_errors": sum(
                error.get("configuration") == name and "id" in error
                for error in errors
            ),
            "rouge": {field: _mean_std(group, field) for field in ROUGE_FIELDS},
            "runtime_seconds": _mean_std(group, "runtime_seconds"),
            "w_zero_fraction": _mean_std(group, "w_zero_fraction"),
            "h_zero_fraction": _mean_std(group, "h_zero_fraction"),
            "active_topics_per_sentence": _mean_std(group, "active_topics_per_sentence"),
            "active_terms_per_topic": _mean_std(group, "active_terms_per_topic"),
        }
        deltas: dict[str, Any] = {}
        for field in ROUGE_FIELDS:
            differences = [
                row[field] - baseline_by_pair[(row["id"], row["seed"])][field]
                for row in group
                if (row["id"], row["seed"]) in baseline_by_pair
            ]
            deltas[field] = {
                "mean": fmean(differences) if differences else None,
                "std": pstdev(differences) if len(differences) > 1 else (0.0 if differences else None),
            }
        paired_deltas[name] = {
            "paired_count": sum(
                (row["id"], row["seed"]) in baseline_by_pair for row in group
            ),
            "rouge": deltas,
        }

    with (output_dir / "ablation_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "ablation_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "config": {
            "split": split,
            "dataset_name": dataset_name,
            "articles": len(articles),
            "seeds": seeds,
            "k": k,
            "k_policy": "one k per article/seed, selected with unregularized KL imputation and reused across configurations",
            "summary_sentences": summary_sentences,
            "mmr_lambda": mmr_lambda,
            "alpha": alpha,
            "l1_ratio": 1.0,
            "query_prf": "disabled for every configuration",
            "factor_zero_threshold": FACTOR_ZERO_THRESHOLD,
            "factor_zero_threshold_definition": "absolute W/H value <= 1e-8",
            "configurations": {
                name: {"alpha_w": values[0], "alpha_h": values[1], "l1_ratio": values[2]}
                for name, values in configurations.items()
            },
            "runtime_definition": "summarize() wall time with shared k; k-selection time reported separately",
        },
        "valid_articles": len(articles),
        "seeds_per_article": len(seeds),
        "expected_runs_per_configuration": len(articles) * len(seeds),
        "input_error_count": sum("line" in error for error in errors),
        "run_error_count": sum("id" in error for error in errors),
        "errors": errors,
        "methods": summaries,
        "paired_deltas_vs_kl_no_l1": paired_deltas,
        "interpretation": (
            "Use paired ROUGE differences and W/H sparsity statistics. Objective values are not compared "
            "across configurations because their regularization terms differ."
        ),
    }
    (output_dir / "ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
