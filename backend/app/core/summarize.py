"""Extract sentences with local, on-the-fly KL-NMF for one input text."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import NMF as SklearnNMF

from .baselines import lead_n
from .nmf import (
    EPS,
    SPARSE_KL_ALPHA_H,
    SPARSE_KL_ALPHA_W,
    SPARSE_KL_L1_RATIO,
    SPARSE_KL_LOSS_NAME,
    SPARSE_KL_MAX_ITER,
    SPARSE_KL_SOLVER,
    select_k_by_imputation,
)
from .preprocess import preprocess_sentences
from .sentences import split_sentences
from .vectorize import vectorize_sentences


SUMMARY_NGRAM_RANGE = (1, 2)
SUMMARY_MAX_FEATURES = 800
K_SELECTION_METHOD_AUTO = "masked_kl_imputation"
K_SELECTION_METHOD_REQUESTED = "requested"


def _position_prior(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 / (1.0 + index)


def _length_quality(processed_sentence: str, min_good: int = 8, max_good: int = 35) -> float:
    token_count = len(processed_sentence.split())
    if token_count <= 0:
        return 0.0
    if token_count < min_good:
        return token_count / min_good
    if token_count <= max_good:
        return 1.0
    return max(0.25, max_good / token_count)


def _method_metadata() -> dict:
    return {
        "loss_name": SPARSE_KL_LOSS_NAME,
        "solver": SPARSE_KL_SOLVER,
        "alpha_W": SPARSE_KL_ALPHA_W,
        "alpha_H": SPARSE_KL_ALPHA_H,
        "l1_ratio": SPARSE_KL_L1_RATIO,
    }


def _fallback(sentences: list[str], budget: int, reason: str, k: int = 0) -> dict:
    summary, indices = lead_n(sentences, budget)
    return {
        "k": k,
        **_method_metadata(),
        "k_selection_method": "fallback",
        "k_candidates": [],
        "summary": summary,
        "selected_indices": indices,
        "topics": [],
        "sentence_analysis": [
            {
                "index": i,
                "text": sentence,
                "selected": i in indices,
                "relevance": 0.0,
                "coverage_gain": 0.0,
                "redundancy": 0.0,
                "position_prior": 0.0,
                "length_quality": 0.0,
                "score": 0.0,
                "dominant_topic": None,
            }
            for i, sentence in enumerate(sentences)
        ],
        "fallback_reason": reason,
    }


def _select_or_resolve_k(
    V: np.ndarray,
    requested_k: int | None,
    seed: int,
) -> tuple[int, str, list[dict]]:
    upper = min(V.shape[0], V.shape[1])
    if requested_k is not None:
        k_used = min(requested_k, upper)
        return k_used, K_SELECTION_METHOD_REQUESTED, [{"k": k_used, "score": 0.0}]
    k_used, candidates = select_k_by_imputation(
        V,
        seed=seed,
        loss=SPARSE_KL_LOSS_NAME,
        alpha_w=SPARSE_KL_ALPHA_W,
        alpha_h=SPARSE_KL_ALPHA_H,
        l1_ratio=SPARSE_KL_L1_RATIO,
    )
    return min(k_used, upper), K_SELECTION_METHOD_AUTO, candidates


def _fit_local_kl_nmf(V: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    nmf = SklearnNMF(
        n_components=k,
        init="nndsvda",
        solver=SPARSE_KL_SOLVER,
        beta_loss=SPARSE_KL_LOSS_NAME,
        alpha_W=SPARSE_KL_ALPHA_W,
        alpha_H=SPARSE_KL_ALPHA_H,
        l1_ratio=SPARSE_KL_L1_RATIO,
        max_iter=SPARSE_KL_MAX_ITER,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in scalar divide",
            category=RuntimeWarning,
            module=r"sklearn\.decomposition\._nmf",
        )
        W = nmf.fit_transform(V)
    return W, nmf.components_


def _topic_terms(H: np.ndarray, terms: list[str], top_n: int = 5) -> list[dict]:
    return [
        {"top_terms": [terms[int(i)] for i in np.argsort(-row, kind="stable")[:top_n]]}
        for row in H
    ]


def _select_sentences(
    sentences: list[str],
    processed: list[str],
    V: np.ndarray,
    W: np.ndarray,
    budget: int,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    position_weight: float,
    length_weight: float,
) -> tuple[list[int], list[dict]]:
    salience = W.sum(axis=0)
    salience = salience / (salience.sum() + EPS)
    W_normalized = W / (W.sum(axis=1, keepdims=True) + EPS)
    selected: list[int] = []
    coverage = np.zeros(W.shape[1], dtype=float)
    analysis: list[dict | None] = [None] * len(sentences)
    norms = np.linalg.norm(V, axis=1)
    position_priors = [_position_prior(i, len(sentences)) for i in range(len(sentences))]
    length_qualities = [_length_quality(sentence) for sentence in processed]

    for _ in range(budget):
        candidates: list[tuple[float, int]] = []
        current_scores: dict[int, dict] = {}
        for i, sentence in enumerate(sentences):
            if i in selected:
                continue
            relevance = float(np.dot(W_normalized[i], salience))
            coverage_gain = float(np.dot(np.maximum(W_normalized[i] - coverage, 0), salience))
            redundancy = 0.0
            if selected and norms[i] > 0:
                redundancy = max(
                    (
                        float(np.dot(V[i], V[j]) / (norms[i] * norms[j]))
                        for j in selected
                        if norms[j] > 0
                    ),
                    default=0.0,
                )
            score = (
                alpha * relevance
                + beta * coverage_gain
                + position_weight * position_priors[i]
                + length_weight * length_qualities[i]
                - gamma * redundancy
            )
            current = {
                "index": i,
                "text": sentence,
                "selected": False,
                "relevance": relevance,
                "coverage_gain": coverage_gain,
                "redundancy": float(redundancy),
                "position_prior": float(position_priors[i]),
                "length_quality": float(length_qualities[i]),
                "score": float(score),
                "dominant_topic": int(np.argmax(W_normalized[i])),
            }
            if analysis[i] is None:
                analysis[i] = current
            candidates.append((score, i))
            current_scores[i] = current
        chosen = min(candidates, key=lambda item: (-item[0], item[1]))[1]
        analysis[chosen] = {**current_scores[chosen], "selected": True}
        selected.append(chosen)
        coverage = np.maximum(coverage, W_normalized[chosen])

    ordered = sorted(selected)
    selected_set = set(ordered)
    final_analysis = [
        {**item, "selected": item["index"] in selected_set}
        for item in analysis
        if item is not None
    ]
    return ordered, final_analysis


def summarize(
    text: str,
    k: int | None = None,
    summary_sentences: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    position_weight: float = 0.15,
    length_weight: float = 0.1,
    seed: int = 42,
) -> dict:
    """Summarize one text by fitting KL-NMF only on the input sentences."""
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k phải là số nguyên dương")
    if not isinstance(summary_sentences, int) or summary_sentences < 1:
        raise ValueError("summary_sentences phải là số nguyên dương")
    if not all(
        np.isfinite(value) and value >= 0
        for value in (alpha, beta, gamma, position_weight, length_weight)
    ):
        raise ValueError("Scoring weights must be finite and nonnegative")

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Văn bản phải có ít nhất một câu")

    budget = min(summary_sentences, len(sentences))
    processed = preprocess_sentences(sentences)
    V, terms = vectorize_sentences(
        processed,
        ngram_range=SUMMARY_NGRAM_RANGE,
        max_features=SUMMARY_MAX_FEATURES,
    )
    if len(terms) == 0 or not np.any(V):
        fallback_k = max(1, len(sentences) // 2) if k is None else k
        return _fallback(sentences, budget, "TF-IDF không có từ hữu ích", fallback_k)

    k_used, k_selection_method, k_candidates = _select_or_resolve_k(V, k, seed)
    if k_used < 1:
        return _fallback(sentences, budget, "Không đủ câu hoặc từ để chạy NMF cục bộ", 0)

    W, H = _fit_local_kl_nmf(V, k_used, seed)
    ordered, analysis = _select_sentences(
        sentences,
        processed,
        V,
        W,
        budget,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        position_weight=position_weight,
        length_weight=length_weight,
    )

    return {
        "k": k_used,
        **_method_metadata(),
        "k_selection_method": k_selection_method,
        "k_candidates": k_candidates,
        "summary": " ".join(sentences[i] for i in ordered),
        "selected_indices": ordered,
        "topics": _topic_terms(H, terms),
        "sentence_analysis": analysis,
        "fallback_reason": None,
    }
