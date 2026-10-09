"""Read local batch experiment output for the evaluation screen."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).resolve().parents[3] / "results" / "metrics.csv"
NUMERIC_FIELDS = (
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "runtime_seconds",
    "sentence_count",
)


def load_batch_metrics(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("error"):
                continue
            for field in NUMERIC_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    return rows
