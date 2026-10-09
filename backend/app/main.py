"""HTTP API for the shared TopicSum backend pipeline."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core.compare import compare_global_local
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
from .core.summarize import summarize
from .core.vectorize import vectorize_sentences
from .schemas import (
    CompareRequest,
    CompareResponse,
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
    try:
        if request.k is None:
            k_to_use, k_candidates = select_k_by_imputation(
                V,
                seed=request.seed,
                loss=SPARSE_KL_LOSS_NAME,
                alpha_w=SPARSE_KL_ALPHA_W,
                alpha_h=SPARSE_KL_ALPHA_H,
                l1_ratio=SPARSE_KL_L1_RATIO,
            )
            k_selection_method = "masked_kl_imputation"
        else:
            k_to_use = min(request.k, V.shape[0], V.shape[1])
            k_candidates = [{"k": k_to_use, "score": 0.0}]
            k_selection_method = "requested"
        result = fit_nmf(
            V,
            k_to_use,
            request.seed,
            request.iterations,
            trace=True,
            loss=SPARSE_KL_LOSS_NAME,
            alpha_w=SPARSE_KL_ALPHA_W,
            alpha_h=SPARSE_KL_ALPHA_H,
            l1_ratio=SPARSE_KL_L1_RATIO,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "k": k_to_use,
        "loss_name": SPARSE_KL_LOSS_NAME,
        "solver": SPARSE_KL_SOLVER,
        "alpha_W": SPARSE_KL_ALPHA_W,
        "alpha_H": SPARSE_KL_ALPHA_H,
        "l1_ratio": SPARSE_KL_L1_RATIO,
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


@app.post("/api/compare-global-local", response_model=CompareResponse)
def compare_global_local_api(request: CompareRequest) -> dict:
    try:
        return compare_global_local(**request.model_dump())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate_api(request: EvaluateRequest) -> dict:
    try:
        result = summarize(**request.model_dump(exclude={"reference_summary"}))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "summary": result["summary"],
        "method": result["method"],
        "objective": result["objective"],
        "k": result["k"],
        "selection_method": result["selection_method"],
        "query_feedback_applied": result["query_feedback_applied"],
        "query_feedback_reason": result["query_feedback_reason"],
        "loss_value": result["loss_value"],
        "data_loss": result["data_loss"],
        "regularization_loss": result["regularization_loss"],
        "fallback_reason": result["fallback_reason"],
        **rouge_f1(request.reference_summary, result["summary"]),
    }


@app.get("/api/experiment/metrics")
def experiment_metrics_api() -> list[dict]:
    return load_batch_metrics()
