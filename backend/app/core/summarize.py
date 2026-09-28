"""Extract sentences using NMF topic coverage and TF-IDF diversity."""

from __future__ import annotations

import numpy as np

from .baselines import lead_n
from .nmf import EPS, fit_nmf
from .preprocess import preprocess_sentences
from .sentences import split_sentences
from .vectorize import vectorize_sentences


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
                "score": 0.0,
                "dominant_topic": None,
            }
            for i, sentence in enumerate(sentences)
        ],
        "fallback_reason": reason,
    }


def summarize(
    text: str,
    k: int = 2,
    summary_sentences: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    seed: int = 42,
) -> dict:
    """Summarize one article; scores are from selection time for chosen rows.

    For rows never chosen, sentence_analysis records their first-round score.
    Selected sentences are restored to source order in the final summary.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k phải là số nguyên dương")
    if not isinstance(summary_sentences, int) or not 1 <= summary_sentences <= 5:
        raise ValueError("summary_sentences phải nằm trong 1..5")
    if not all(np.isfinite(value) and value >= 0 for value in (alpha, beta, gamma)):
        raise ValueError("alpha, beta, gamma phải hữu hạn và không âm")

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Văn bản phải có ít nhất một câu")
    budget = min(summary_sentences, len(sentences))
    if len(sentences) < 2:
        return _fallback(sentences, budget, "Văn bản có ít hơn 2 câu hữu ích")

    processed = preprocess_sentences(sentences)
    V, terms = vectorize_sentences(processed)
    if len(terms) < 2 or np.count_nonzero(V.any(axis=1)) < 2:
        return _fallback(sentences, budget, "TF-IDF có ít hơn 2 câu/từ hữu ích")

    nmf = fit_nmf(V, k, seed=seed)
    W, H = nmf.W, nmf.H
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
    for _ in range(budget):
        candidates: list[tuple[float, int]] = []
        current_scores: dict[int, dict] = {}
        for i, sentence in enumerate(sentences):
            if i in selected:
                continue
            relevance = float(np.dot(W_normalized[i], salience))
            gain = float(np.dot(np.maximum(W_normalized[i] - coverage, 0), salience))
            redundancy = 0.0
            if selected and norms[i] > 0:
                redundancy = max(
                    (float(np.dot(V[i], V[j]) / (norms[i] * norms[j]))
                     for j in selected if norms[j] > 0),
                    default=0.0,
                )
            score = alpha * relevance + beta * gain - gamma * redundancy
            current = {
                "index": i,
                "text": sentence,
                "selected": False,
                "relevance": relevance,
                "coverage_gain": gain,
                "redundancy": redundancy,
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
