"""Validated API contracts for simulation, summarization and evaluation."""

from __future__ import annotations

from typing import Literal

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
    method: Literal["local_kl_mmr", "local_frobenius_mmr"] = "local_kl_mmr"


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
    score: float | None


class SimulateResponse(BaseModel):
    k: int
    method: str
    loss_name: str
    solver: str
    alpha_W: float | None
    alpha_H: float | None
    l1_ratio: float | None
    implementation: str
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
    method: Literal["local_kl_mmr", "local_frobenius_mmr"] = "local_kl_mmr"
    query: str | None = Field(default=None, max_length=2000)
    query_weight: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    alpha_w: float = Field(default=0.001, ge=0, le=1, allow_inf_nan=False)
    alpha_h: float = Field(default=0.001, ge=0, le=1, allow_inf_nan=False)
    l1_ratio: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)


class TopicResponse(BaseModel):
    top_terms: list[str]
    representative_sentences: list[str] = Field(default_factory=list)


class SentenceAnalysisResponse(BaseModel):
    index: int
    text: str
    selected: bool
    relevance: float
    redundancy: float
    score: float
    score_round: int | None = None
    dominant_topic: int | None
    query_relevance: float = 0.0


class MMRCandidateResponse(BaseModel):
    index: int
    relevance: float
    redundancy: float
    score: float


class MMRRoundResponse(BaseModel):
    round: int
    selected_index: int
    candidates: list[MMRCandidateResponse]


class SummarizeResponse(BaseModel):
    k: int
    method: str
    objective: str
    loss_name: str
    solver: str
    alpha_W: float | None
    alpha_H: float | None
    l1_ratio: float | None
    k_selection_method: str
    k_candidates: list[KCandidateResponse]
    selection_method: str
    mmr_lambda: float
    query_feedback_applied: bool
    query_feedback_reason: str | None = None
    pseudo_relevant_indices: list[int]
    query_weight: float
    loss_value: float | None = None
    data_loss: float | None = None
    regularization_loss: float | None = None
    summary: str
    selected_indices: list[int]
    topics: list[TopicResponse]
    sentence_analysis: list[SentenceAnalysisResponse]
    selection_trace: list[MMRRoundResponse]
    fallback_reason: str | None


class EvaluateRequest(TextRequest):
    reference_summary: str = Field(min_length=1)
    summary_sentences: int = Field(default=3, ge=1, le=20)
    k: int | None = Field(default=None, ge=1)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    alpha_w: float = Field(default=0.001, ge=0, le=1, allow_inf_nan=False)
    alpha_h: float = Field(default=0.001, ge=0, le=1, allow_inf_nan=False)
    l1_ratio: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    seed: int = 42

    @field_validator("reference_summary")
    @classmethod
    def nonblank_reference(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Bản tóm tắt tham chiếu không được để trống")
        return value


class EvaluatedSummaryResponse(BaseModel):
    summary: str
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float


class EvaluateResponse(BaseModel):
    k: int
    k_selection_method: str
    kl_l1_mmr: EvaluatedSummaryResponse
    frobenius: EvaluatedSummaryResponse


class AblationRequest(BaseModel):
    article_id: str = Field(default="manual-article", min_length=1, max_length=200)
    article: str = Field(min_length=1, max_length=100_000)
    reference_summary: str = Field(min_length=1, max_length=20_000)
    k: int | None = Field(default=None, ge=1)
    summary_sentences: int = Field(default=3, ge=1, le=20)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    alpha: float = Field(default=0.001, ge=0, le=1, allow_inf_nan=False)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=20)

    @field_validator("article_id")
    @classmethod
    def default_ablation_id(cls, value: str) -> str:
        return value.strip() or "manual-article"

    @field_validator("article", "reference_summary")
    @classmethod
    def nonblank_ablation_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Article and reference_summary must not be blank")
        return value

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values: list[int]) -> list[int]:
        if any(seed < 0 or seed > 2**32 - 1 for seed in values):
            raise ValueError("Seeds must be between 0 and 4294967295")
        if len(set(values)) != len(values):
            raise ValueError("Seeds must be unique")
        return values
