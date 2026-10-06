"""Extract sentences with local, on-the-fly KL-NMF and standard MMR."""

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
DEFAULT_MMR_LAMBDA = 0.7
SELECTION_METHOD_MMR = "maximum_marginal_relevance"
SELECTION_METHOD_FALLBACK = "lead_n_fallback"


def _method_metadata() -> dict:
    return {
        "loss_name": SPARSE_KL_LOSS_NAME,
        "solver": SPARSE_KL_SOLVER,
        "alpha_W": SPARSE_KL_ALPHA_W,
        "alpha_H": SPARSE_KL_ALPHA_H,
        "l1_ratio": SPARSE_KL_L1_RATIO,
    }


def _fallback(
    sentences: list[str],
    budget: int,
    reason: str,
    k: int = 0,
    *,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> dict:
    summary, indices = lead_n(sentences, budget)
    return {
        "k": k,
        **_method_metadata(),
        "k_selection_method": "fallback",
        "k_candidates": [],
        "selection_method": SELECTION_METHOD_FALLBACK,
        "mmr_lambda": mmr_lambda,
        "summary": summary,
        "selected_indices": indices,
        "topics": [],
        "sentence_analysis": [
            {
                "index": i,
                "text": sentence,
                "selected": i in indices,
                "relevance": 0.0,
                "redundancy": 0.0,
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


def _topic_relevance(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    salience = W.sum(axis=0)
    salience = salience / (salience.sum() + EPS)
    W_normalized = W / (W.sum(axis=1, keepdims=True) + EPS)
    relevance = W_normalized @ salience
    max_relevance = float(np.max(relevance)) if len(relevance) else 0.0
    if max_relevance > EPS:
        return relevance / max_relevance

    norms = np.linalg.norm(V, axis=1)
    max_norm = float(np.max(norms)) if len(norms) else 0.0
    if max_norm > EPS:
        return norms / max_norm
    return np.zeros(V.shape[0], dtype=float)


def _max_similarity_to_selected(
    V: np.ndarray,
    norms: np.ndarray,
    index: int,
    selected: list[int],
) -> float:
    if not selected or norms[index] <= 0:
        return 0.0
    return max(
        (
            float(np.dot(V[index], V[j]) / (norms[index] * norms[j]))
            for j in selected
            if norms[j] > 0
        ),
        default=0.0,
    )


def _select_sentences(
    sentences: list[str],
    V: np.ndarray,
    W: np.ndarray,
    budget: int,
    *,
    mmr_lambda: float,
) -> tuple[list[int], list[dict]]:
    """Select sentences with standard Maximum Marginal Relevance."""
    W_normalized = W / (W.sum(axis=1, keepdims=True) + EPS)
    relevance_scores = _topic_relevance(W, V)
    selected: list[int] = []
    analysis: list[dict | None] = [None] * len(sentences)
    norms = np.linalg.norm(V, axis=1)

    for _ in range(budget):
        candidates: list[tuple[float, float, int]] = []
        current_scores: dict[int, dict] = {}
        for i, sentence in enumerate(sentences):
            if i in selected:
                continue
            relevance = float(relevance_scores[i])
            redundancy = _max_similarity_to_selected(V, norms, i, selected)
            score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            current = {
                "index": i,
                "text": sentence,
                "selected": False,
                "relevance": relevance,
                "redundancy": float(redundancy),
                "score": float(score),
                "dominant_topic": int(np.argmax(W_normalized[i])),
            }
            analysis[i] = current
            candidates.append((score, relevance, i))
            current_scores[i] = current

        chosen = min(candidates, key=lambda item: (-item[0], -item[1], item[2]))[2]
        analysis[chosen] = {**current_scores[chosen], "selected": True}
        selected.append(chosen)

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
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    seed: int = 42,
) -> dict:
    """Summarize one text by fitting KL-NMF only on the input sentences."""
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k must be a positive integer")
    if not isinstance(summary_sentences, int) or summary_sentences < 1:
        raise ValueError("summary_sentences must be a positive integer")
    if not np.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be finite and between 0 and 1")

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("text must contain at least one sentence")

    budget = min(summary_sentences, len(sentences))
    processed = preprocess_sentences(sentences)
    V, terms = vectorize_sentences(
        processed,
        ngram_range=SUMMARY_NGRAM_RANGE,
        max_features=SUMMARY_MAX_FEATURES,
    )
    if len(terms) == 0 or not np.any(V):
        fallback_k = max(1, len(sentences) // 2) if k is None else k
        return _fallback(
            sentences,
            budget,
            "TF-IDF has no useful terms",
            fallback_k,
            mmr_lambda=mmr_lambda,
        )

    k_used, k_selection_method, k_candidates = _select_or_resolve_k(V, k, seed)
    if k_used < 1:
        return _fallback(
            sentences,
            budget,
            "Not enough sentences or terms for local NMF",
            0,
            mmr_lambda=mmr_lambda,
        )

    W, H = _fit_local_kl_nmf(V, k_used, seed)
    ordered, analysis = _select_sentences(
        sentences,
        V,
        W,
        budget,
        mmr_lambda=mmr_lambda,
    )

    return {
        "k": k_used,
        **_method_metadata(),
        "k_selection_method": k_selection_method,
        "k_candidates": k_candidates,
        "selection_method": SELECTION_METHOD_MMR,
        "mmr_lambda": mmr_lambda,
        "summary": " ".join(sentences[i] for i in ordered),
        "selected_indices": ordered,
        "topics": _topic_terms(H, terms),
        "sentence_analysis": analysis,
        "fallback_reason": None,
    }
