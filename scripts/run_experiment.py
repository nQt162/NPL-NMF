"""Benchmark extractive NMF variants against Lead-N on one fixed JSONL split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from app.core.baselines import lead_n
from app.core.compare import summarize_global_text
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


def _read_articles(input_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not input_path.is_file():
        raise ValueError(f"Không tìm thấy tập dữ liệu: {input_path}")
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
                raise ValueError("thiếu id, article hoặc reference_summary")
            if not split_sentences(item["article"]):
                raise ValueError("bài không có câu")
            articles.append(item)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"line": line_number, "reason": str(exc)})
    if not articles:
        raise ValueError("Tập dữ liệu không có bài hợp lệ; chưa chạy benchmark")
    return articles, errors


def _run_method(
    method: str,
    article: dict[str, Any],
    *,
    k: int | None,
    summary_sentences: int,
    pairwise_lambda: float,
    mmr_lambda: float,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
    query_weight: float,
    seed: int,
) -> dict[str, Any]:
    text = article["article"]
    sentences = split_sentences(text)
    budget = min(summary_sentences, len(sentences))
    if method == "Lead-N":
        summary, indices = lead_n(sentences, budget)
        return {"summary": summary, "selected_indices": indices, "fallback_reason": None}
    if method == "Global NMF":
        return summarize_global_text(
            text,
            summary_sentences=budget,
            topic_count=k or 3,
        )
    if method == "Query PRF" and not str(article.get("query", "")).strip():
        raise ValueError("bài thiếu trường query; bỏ qua phương pháp Query PRF")

    request: dict[str, Any] = {
        "text": text,
        "k": k,
        "summary_sentences": budget,
        "seed": seed,
        "method": METHOD_LOCAL_KL,
        "mmr_lambda": mmr_lambda,
        "alpha_w": alpha_w,
        "alpha_h": alpha_h,
        "l1_ratio": l1_ratio,
    }
    if method == "NMFTS pairwise":
        request.update(method=METHOD_NMFTS, pairwise_lambda=pairwise_lambda)
    elif method == "SNMF graph":
        request.update(method=METHOD_SNMF)
    elif method == "Query PRF":
        request.update(query=article["query"], query_weight=query_weight)
    return summarize(**request)


def run_experiment(
    input_path: Path,
    output_dir: Path,
    *,
    k: int | None = None,
    summary_sentences: int = 3,
    pairwise_lambda: float = DEFAULT_PAIRWISE_LAMBDA,
    mmr_lambda: float = 0.7,
    alpha_w: float = 0.1,
    alpha_h: float = 0.1,
    l1_ratio: float = 1.0,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    seed: int = 42,
) -> dict[str, Any]:
    articles, errors = _read_articles(input_path)
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    methods = ["Lead-N", "Global NMF", "Local KL+L1+MMR", "NMFTS pairwise", "SNMF graph"]
    has_query_for_all = bool(articles) and all(
        str(item.get("query", "")).strip() for item in articles
    )
    if has_query_for_all:
        methods.append("Query PRF")
    elif any(str(item.get("query", "")).strip() for item in articles):
        errors.append({
            "reason": "Query PRF omitted: every article in the comparison split must include a query",
        })

    for article in articles:
        for method in methods:
            start = perf_counter()
            try:
                result = _run_method(
                    method,
                    article,
                    k=k,
                    summary_sentences=summary_sentences,
                    pairwise_lambda=pairwise_lambda,
                    mmr_lambda=mmr_lambda,
                    alpha_w=alpha_w,
                    alpha_h=alpha_h,
                    l1_ratio=l1_ratio,
                    query_weight=query_weight,
                    seed=seed,
                )
                runtime = perf_counter() - start
                summary = result["summary"]
                indices = result["selected_indices"]
                fallback = result.get("fallback_reason") or ""
                rows.append({
                    "id": article["id"],
                    "method": method,
                    **rouge_f1(article["reference_summary"], summary),
                    "runtime_seconds": runtime,
                    "sentence_count": len(indices),
                    "fallback_reason": fallback,
                })
                predictions.append({
                    "id": article["id"],
                    "method": method,
                    "summary": summary,
                    "selected_indices": indices,
                    "fallback_reason": fallback,
                })
            except Exception as exc:
                errors.append({"id": article["id"], "method": method, "reason": str(exc)})

    successful_methods = [
        method for method in methods
        if any(row["method"] == method for row in rows)
    ]
    if not successful_methods:
        raise ValueError("Không có phương pháp nào chạy thành công")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "id", "method", "rouge1_f1", "rouge2_f1", "rougeL_f1",
            "runtime_seconds", "sentence_count", "fallback_reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )

    aggregate = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        aggregate[method] = {
            key: fmean(row[key] for row in method_rows)
            for key in ("rouge1_f1", "rouge2_f1", "rougeL_f1", "runtime_seconds", "sentence_count")
        }
        aggregate[method]["count"] = len(method_rows)
    report = {
        "config": {
            "input": str(input_path),
            "k": k,
            "summary_sentences": summary_sentences,
            "pairwise_lambda": pairwise_lambda,
            "mmr_lambda": mmr_lambda,
            "alpha_w": alpha_w,
            "alpha_h": alpha_h,
            "l1_ratio": l1_ratio,
            "query_weight": query_weight,
            "seed": seed,
            "methods": methods,
            "k_selection": "masked_kl_imputation shared by local KL, NMFTS and SNMF when k is omitted",
            "query_feedback": "included only when every article in this split has a query",
        },
        "valid_articles": len(articles),
        "errors": errors,
        "methods": aggregate,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL split containing article and reference_summary")
    parser.add_argument("--output", type=Path, default=Path("../results"))
    parser.add_argument("--k", type=int, default=None, help="Fixed topic count; omitted uses masked KL imputation")
    parser.add_argument("--summary-sentences", type=int, default=3)
    parser.add_argument("--pairwise-lambda", type=float, default=DEFAULT_PAIRWISE_LAMBDA)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--alpha-w", type=float, default=0.1)
    parser.add_argument("--alpha-h", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=1.0)
    parser.add_argument("--query-weight", type=float, default=DEFAULT_QUERY_WEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_experiment(
        args.input,
        args.output,
        k=args.k,
        summary_sentences=args.summary_sentences,
        pairwise_lambda=args.pairwise_lambda,
        mmr_lambda=args.mmr_lambda,
        alpha_w=args.alpha_w,
        alpha_h=args.alpha_h,
        l1_ratio=args.l1_ratio,
        query_weight=args.query_weight,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
