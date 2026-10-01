"""Extract sentences using NMF topic coverage and TF-IDF diversity."""

from __future__ import annotations

import numpy as np

from .baselines import lead_n
from .corpus_model import load_corpus_nmf_model
from .nmf import EPS, fit_nmf, auto_select_k
from .preprocess import preprocess_sentences
from .sentences import split_sentences
from .vectorize import vectorize_sentences


SUMMARY_NGRAM_RANGE = (1, 2)
SUMMARY_MAX_FEATURES = 800


def _position_prior(index: int, total: int) -> float:
    """Favor lead sentences mildly without turning NMF into Lead-N."""
    if total <= 1:
        return 1.0
    return 1.0 / (1.0 + index)


def _length_quality(processed_sentence: str, min_good: int = 8, max_good: int = 35) -> float:
    """Prefer contentful sentences and downweight very short or very long ones."""
    token_count = len(processed_sentence.split())
    if token_count <= 0:
        return 0.0
    if token_count < min_good:
        return token_count / min_good
    if token_count <= max_good:
        return 1.0
    return max(0.25, max_good / token_count)


def _fallback(sentences: list[str], budget: int, reason: str) -> dict:
    summary, indices = lead_n(sentences, budget)
    return {
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
    """Summarize one article; scores are from selection time for chosen rows.

    For rows never chosen, sentence_analysis records their first-round score.
    Selected sentences are restored to source order in the final summary.
    """
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k phải là số nguyên dương")
    if not isinstance(summary_sentences, int) or summary_sentences < 1:
        raise ValueError("summary_sentences phải là số nguyên dương")
    if not all(np.isfinite(value) and value >= 0
               for value in (alpha, beta, gamma, position_weight, length_weight)):
        raise ValueError("Scoring weights must be finite and nonnegative")

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Văn bản phải có ít nhất một câu")
    max_budget = max(1, (len(sentences) - 1) // 2)
    budget = min(summary_sentences, max_budget)
    if len(sentences) < 2:
        return _fallback(sentences, budget, "Văn bản có ít hơn 2 câu hữu ích")

    processed = preprocess_sentences(sentences)
    corpus_model = load_corpus_nmf_model()
    
    # Needs V to select k if not provided
    if corpus_model is not None:
        V, terms, W_temp, H_temp = corpus_model.project(processed, k if k is not None else 2)
        if k is None:
            k = auto_select_k(V, seed=seed)
            V, terms, W, H = corpus_model.project(processed, k)
        else:
            W, H = W_temp, H_temp
    else:
        V, terms = vectorize_sentences(
            processed,
            ngram_range=SUMMARY_NGRAM_RANGE,
            max_features=SUMMARY_MAX_FEATURES,
        )
        if len(terms) < 2 or np.count_nonzero(V.any(axis=1)) < 2:
            return _fallback(sentences, budget, "TF-IDF có ít hơn 2 câu/từ hữu ích")
        if k is None:
            k = auto_select_k(V, seed=seed)
        nmf = fit_nmf(V, k, seed=seed)
        W, H = nmf.W, nmf.H
    if len(terms) < 2 or np.count_nonzero(V.any(axis=1)) < 2:
        return _fallback(sentences, budget, "TF-IDF có ít hơn 2 câu/từ hữu ích")
    salience = W.sum(axis=0)
    salience = salience / (salience.sum() + EPS)
    W_normalized = W / (W.sum(axis=1, keepdims=True) + EPS)
    topics = [
        {"top_terms": [terms[int(i)] for i in np.argsort(-row, kind="stable")[:5]]}
        for row in H
    ]

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
            gain = float(np.dot(np.maximum(W_normalized[i] - coverage, 0), salience))
            position = position_priors[i]
            length = length_qualities[i]
            redundancy = 0.0
            if selected and norms[i] > 0:
                redundancy = max(
                    (float(np.dot(V[i], V[j]) / (norms[i] * norms[j]))
                     for j in selected if norms[j] > 0),
                    default=0.0,
                )
            score = (
                alpha * relevance
                + beta * gain
                + position_weight * position
                + length_weight * length
                - gamma * redundancy
            )
            current = {
                "index": i,
                "text": sentence,
                "selected": False,
                "relevance": relevance,
                "coverage_gain": gain,
                "redundancy": redundancy,
                "position_prior": float(position),
                "length_quality": float(length),
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
    return {
        "summary": " ".join(sentences[i] for i in ordered),
        "selected_indices": ordered,
        "topics": topics,
        "sentence_analysis": analysis,
        "fallback_reason": None,
    }
