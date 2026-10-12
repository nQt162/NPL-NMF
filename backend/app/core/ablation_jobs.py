"""Small in-process job registry for web-triggered L1 ablations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Lock
from uuid import uuid4
from typing import Any

from .ablation import run_l1_ablation


ABLATION_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "results" / "ablation"
ABLATION_ARTIFACTS = {
    "metrics": "ablation_metrics.csv",
    "predictions": "ablation_predictions.jsonl",
    "report": "ablation_report.json",
}

_LOCK = Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create_ablation_job(dataset_name: str, total: int) -> tuple[str, Path]:
    run_id = uuid4().hex
    output_dir = ABLATION_OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    with _LOCK:
        _JOBS[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "dataset_name": dataset_name,
            "completed": 0,
            "total": total,
            "report": None,
            "error": None,
        }
    return run_id, output_dir


def get_ablation_job(run_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(run_id)
        return deepcopy(job) if job is not None else None


def execute_ablation_job(
    run_id: str,
    output_dir: Path,
    articles: list[dict[str, str]],
    input_errors: list[dict[str, Any]],
    options: dict[str, Any],
) -> None:
    with _LOCK:
        job = _JOBS.get(run_id)
        if job is None:
            return
        job["status"] = "running"

    def update_progress(completed: int, total: int) -> None:
        with _LOCK:
            job = _JOBS.get(run_id)
            if job is not None:
                job["completed"] = completed
                job["total"] = total

    try:
        report = run_l1_ablation(
            articles,
            output_dir,
            input_errors=input_errors,
            progress_callback=update_progress,
            **options,
        )
    except Exception as exc:
        with _LOCK:
            job = _JOBS.get(run_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = str(exc)
        return

    with _LOCK:
        job = _JOBS.get(run_id)
        if job is not None:
            job["status"] = "complete"
            job["completed"] = job["total"]
            job["report"] = report
