"""Compare NMF and Lead-N on the same article-level validation or test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from app.core.baselines import lead_n
from app.core.evaluate import rouge_f1
from app.core.sentences import split_sentences
from app.core.summarize import summarize


def run_experiment(
    input_path: Path,
    output_dir: Path,
    *,
    k: int = 2,
    summary_sentences: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    seed: int = 42,
) -> dict:
    if not input_path.is_file():
        raise ValueError(f"Không tìm thấy tập dữ liệu: {input_path}")
    config = dict(k=k, summary_sentences=summary_sentences, alpha=alpha,
                  beta=beta, gamma=gamma, seed=seed, input=str(input_path))
    rows: list[dict] = []
    predictions: list[dict] = []
    errors: list[dict] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key].strip()
                       for key in ("id", "article", "reference_summary")):
                raise ValueError("thiếu id, article hoặc reference_summary")
            sentences = split_sentences(item["article"])
            if not sentences:
                raise ValueError("bài không có câu")
            budget = min(summary_sentences, len(sentences))

            start = perf_counter()
            baseline_summary, baseline_indices = lead_n(sentences, budget)
            baseline_runtime = perf_counter() - start
            start = perf_counter()
            nmf_result = summarize(item["article"], k=k,
                                   summary_sentences=summary_sentences,
                                   alpha=alpha, beta=beta, gamma=gamma, seed=seed)
            nmf_runtime = perf_counter() - start
            methods = [
                ("Lead-N", baseline_summary, baseline_indices, baseline_runtime, None),
                ("NMF", nmf_result["summary"], nmf_result["selected_indices"],
                 nmf_runtime, nmf_result["fallback_reason"]),
            ]
            for method, summary, indices, runtime, fallback in methods:
                rows.append({
                    "id": item["id"], "method": method,
                    **rouge_f1(item["reference_summary"], summary),
                    "runtime_seconds": runtime, "sentence_count": len(indices),
                    "fallback_reason": fallback or "",
                })
                predictions.append({
                    "id": item["id"], "method": method, "summary": summary,
                    "selected_indices": indices, "fallback_reason": fallback,
                })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"line": line_number, "reason": str(exc)})
    if not rows:
        raise ValueError("Không có bài đánh giá hợp lệ; chưa ghi kết quả")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    report = {
        "config": config,
        "valid_articles": len(rows) // 2,
        "errors": errors,
        "fallback_articles": sum(bool(row["fallback_reason"]) for row in rows if row["method"] == "NMF"),
        "methods": {
            method: {
                key: fmean(row[key] for row in rows if row["method"] == method)
                for key in ("rouge1_f1", "rouge2_f1", "rougeL_f1", "runtime_seconds")
            }
            for method in ("Lead-N", "NMF")
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--summary-sentences", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_experiment(
        args.input, args.output, k=args.k, summary_sentences=args.summary_sentences,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, seed=args.seed,
    ), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
