"""Validated API contracts for simulation, summarization and evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Văn bản không được để trống")
        return value


class SimulateRequest(TextRequest):
    k: int | None = Field(default=None, ge=1)
    iterations: int = Field(default=10, ge=1, le=20)
    seed: int = 42


class SnapshotResponse(BaseModel):
    iteration: int
    W: list[list[float]]
    H: list[list[float]]
    WH: list[list[float]]
    loss: float
    data_loss: float
    regularization_loss: float


class KCandidateResponse(BaseModel):
    k: int
    score: float


class SimulateResponse(BaseModel):
    k: int
    loss_name: str
    solver: str
    alpha_W: float
    alpha_H: float
    l1_ratio: float
    k_selection_method: str
    k_candidates: list[KCandidateResponse]
    sentences: list[str]
    terms: list[str]
    V: list[list[float]]
    snapshots: list[SnapshotResponse]


class SummarizeRequest(TextRequest):
    k: int | None = Field(default=None, ge=1)
    summary_sentences: int = Field(default=3, ge=1)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)


class TopicResponse(BaseModel):
    top_terms: list[str]


class SentenceAnalysisResponse(BaseModel):
    index: int
    text: str
    selected: bool
    relevance: float
    redundancy: float
    score: float
    dominant_topic: int | None


class SummarizeResponse(BaseModel):
    k: int
    loss_name: str
    solver: str
    alpha_W: float
    alpha_H: float
    l1_ratio: float
    k_selection_method: str
    k_candidates: list[KCandidateResponse]
    selection_method: str
    mmr_lambda: float
    summary: str
    selected_indices: list[int]
    topics: list[TopicResponse]
    sentence_analysis: list[SentenceAnalysisResponse]
    fallback_reason: str | None


class EvaluateRequest(TextRequest):
    reference_summary: str = Field(min_length=1)
    summary_sentences: int = Field(default=3, ge=1)

    @field_validator("reference_summary")
    @classmethod
    def nonblank_reference(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Bản tóm tắt tham chiếu không được để trống")
        return value


class EvaluateResponse(BaseModel):
    summary: str
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    fallback_reason: str | None
