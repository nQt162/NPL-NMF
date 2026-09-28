"""HTTP API for the shared TopicSum backend pipeline."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core.evaluate import rouge_f1
from .core.nmf import fit_nmf
from .core.preprocess import preprocess_sentences
from .core.sentences import split_sentences
from .core.summarize import summarize
from .core.vectorize import vectorize_sentences
from .schemas import (
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
    if len(sentences) > 10:
        raise HTTPException(422, "Mô phỏng chỉ hỗ trợ tối đa 10 câu")
    V, terms = vectorize_sentences(preprocess_sentences(sentences))
    if len(terms) > 50:
        raise HTTPException(422, "Mô phỏng chỉ hỗ trợ tối đa 50 từ vựng")
    if len(sentences) < 2 or len(terms) < 2:
        raise HTTPException(422, "Cần ít nhất 2 câu và 2 từ hữu ích để mô phỏng NMF")
    try:
        result = fit_nmf(V, request.k, request.seed, request.iterations, trace=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
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
    try:
        result = summarize(request.text, summary_sentences=request.summary_sentences)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "summary": result["summary"],
        "fallback_reason": result["fallback_reason"],
        **rouge_f1(request.reference_summary, result["summary"]),
    }
