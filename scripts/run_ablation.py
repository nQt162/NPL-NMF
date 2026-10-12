"""Run the paired L1 ablation for Local KL-NMF + MMR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.ablation import parse_articles_jsonl, run_l1_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id, article, reference_summary")
    parser.add_argument("--output", type=Path, default=Path("../results/ablation"))
    parser.add_argument("--split", choices=("validation", "test", "pilot"), default="validation")
    parser.add_argument("--k", type=int, default=None, help="Fixed topic count; omitted selects once per article/seed")
    parser.add_argument("--summary-sentences", type=int, default=3)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.001, help="L1 strength for penalized W/H conditions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="One or more shared random seeds")
    args = parser.parse_args()

    if args.k is not None and args.k < 1:
        parser.error("--k must be positive")
    if args.summary_sentences < 1:
        parser.error("--summary-sentences must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("--seeds must be between 0 and 4294967295")
    if not 0 <= args.mmr_lambda <= 1:
        parser.error("--mmr-lambda must be between 0 and 1")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input JSONL not found: {input_path}")
    articles, input_errors = parse_articles_jsonl(input_path.read_text(encoding="utf-8-sig"))
    report = run_l1_ablation(
        articles,
        args.output.expanduser().resolve(),
        split=args.split,
        dataset_name=str(input_path),
        k=args.k,
        summary_sentences=args.summary_sentences,
        mmr_lambda=args.mmr_lambda,
        alpha=args.alpha,
        seeds=args.seeds,
        input_errors=input_errors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
