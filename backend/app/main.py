"""HTTP API for the shared TopicSum backend pipeline."""

from __future__ import annotations

import re

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .core.ablation_jobs import (
    ABLATION_ARTIFACTS,
    ABLATION_OUTPUT_ROOT,
    create_ablation_job,
    execute_ablation_job,
    get_ablation_job,
)
from .core.evaluate import rouge_f1
from .core.experiment import load_batch_metrics
from .core.nmf import (
    SPARSE_KL_ALPHA_H,
    SPARSE_KL_ALPHA_W,
    SPARSE_KL_L1_RATIO,
    SPARSE_KL_LOSS_NAME,
    SPARSE_KL_SOLVER,
    fit_nmf,
    select_k_by_imputation,
)
from .core.preprocess import preprocess_sentences
from .core.sentences import split_sentences
from .core.summarize import METHOD_FROBENIUS, METHOD_LOCAL_KL, summarize
from .core.vectorize import vectorize_sentences
from .schemas import (
    AblationRequest,
    EvaluateRequest,
    EvaluateResponse,
    SimulateRequest,
    SimulateResponse,
    SummarizeRequest,
    SummarizeResponse,
)


app = FastAPI(title="TopicSum API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> dict:
    sentences = split_sentences(request.text)
    V, terms = vectorize_sentences(preprocess_sentences(sentences))
    if len(sentences) < 2 or len(terms) < 2:
        raise HTTPException(422, "Cần ít nhất 2 câu và 2 từ hữu ích để mô phỏng NMF")
    is_kl = request.method == METHOD_LOCAL_KL
    loss_name = SPARSE_KL_LOSS_NAME if is_kl else "frobenius"
    # The visualizer compares data-fit losses under the same L1 regularization.
    alpha_w = SPARSE_KL_ALPHA_W
    alpha_h = SPARSE_KL_ALPHA_H
    l1_ratio = SPARSE_KL_L1_RATIO
    try:
        if request.k is None:
            k_to_use, k_candidates, k_selection_method = select_k_by_imputation(
                V,
                seed=request.seed,
            )
        else:
            k_to_use = min(request.k, V.shape[0], V.shape[1])
            k_candidates = [{"k": k_to_use, "score": None}]
            k_selection_method = "requested"
        result = fit_nmf(
            V,
            k_to_use,
            request.seed,
            request.iterations,
            trace=True,
            loss=loss_name,
            alpha_w=alpha_w,
            alpha_h=alpha_h,
            l1_ratio=l1_ratio,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "k": k_to_use,
        "method": request.method,
        "loss_name": loss_name,
        "solver": SPARSE_KL_SOLVER,
        "alpha_W": alpha_w,
        "alpha_H": alpha_h,
        "l1_ratio": l1_ratio,
        "implementation": "educational_custom_mu",
        "k_selection_method": k_selection_method,
        "k_candidates": k_candidates,
        "sentences": sentences,
        "terms": terms,
        "V": V.tolist(),
        "snapshots": [
            {
                "iteration": item.iteration,
                "W": item.W.tolist(),
                "H": item.H.tolist(),
                "WH": item.WH.tolist(),
                "loss": item.loss,
                "data_loss": item.data_loss,
                "regularization_loss": item.regularization_loss,
            }
            for item in result.snapshots
        ],
    }


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize_api(request: SummarizeRequest) -> dict:
    try:
        return summarize(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate_api(request: EvaluateRequest) -> dict:
    options = request.model_dump(exclude={"reference_summary"})
    try:
        kl_result = summarize(**options, method=METHOD_LOCAL_KL)
        frobenius_result = summarize(
            **{**options, "k": kl_result["k"]},
            method=METHOD_FROBENIUS,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return {
        "k": kl_result["k"],
        "k_selection_method": kl_result["k_selection_method"],
        "kl_l1_mmr": {
            "summary": kl_result["summary"],
            **rouge_f1(request.reference_summary, kl_result["summary"]),
        },
        "frobenius": {
            "summary": frobenius_result["summary"],
            **rouge_f1(request.reference_summary, frobenius_result["summary"]),
        },
    }


@app.post("/api/ablation", status_code=202)
def start_l1_ablation(request: AblationRequest, background_tasks: BackgroundTasks) -> dict:
    if not split_sentences(request.article):
        raise HTTPException(422, "Article must contain at least one sentence.")
    article_id = request.article_id.strip()
    articles = [{
        "id": article_id,
        "article": request.article,
        "reference_summary": request.reference_summary,
    }]
    options = request.model_dump(exclude={"article_id", "article", "reference_summary"})
    options.update(split="pilot", dataset_name=article_id)
    total = len(request.seeds)
    run_id, output_dir = create_ablation_job(article_id, total)
    background_tasks.add_task(
        execute_ablation_job,
        run_id,
        output_dir,
        articles,
        [],
        options,
    )
    return {"run_id": run_id, "status": "queued", "total": total}


@app.get("/api/ablation/{run_id}")
def ablation_job_status(run_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise HTTPException(404, "Ablation run not found")
    job = get_ablation_job(run_id)
    if job is None:
        raise HTTPException(404, "Ablation run not found in this backend process")
    return job


@app.get("/api/ablation/{run_id}/files/{artifact}")
def download_ablation_artifact(run_id: str, artifact: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id) or artifact not in ABLATION_ARTIFACTS:
        raise HTTPException(404, "Ablation artifact not found")
    path = ABLATION_OUTPUT_ROOT / run_id / ABLATION_ARTIFACTS[artifact]
    if not path.is_file():
        raise HTTPException(404, "Ablation artifact is not ready")
    return FileResponse(path, filename=path.name)


@app.get("/api/experiment/metrics")
def experiment_metrics_api() -> list[dict]:
    return load_batch_metrics()
