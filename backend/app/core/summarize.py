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
    K_SELECTION_METHOD_MASKED,
    objective_breakdown,
    select_k_by_imputation,
)
from .preprocess import preprocess_sentences
from .sentences import split_sentences
from .vectorize import (
    LOCAL_TFIDF_MAX_FEATURES,
    LOCAL_TFIDF_NGRAM_RANGE,
    fit_sentence_vectorizer,
)


K_SELECTION_METHOD_REQUESTED = "requested"
DEFAULT_MMR_LAMBDA = 0.7
SELECTION_METHOD_MMR = "maximum_marginal_relevance"
SELECTION_METHOD_FALLBACK = "lead_n_fallback"
METHOD_LOCAL_KL = "local_kl_mmr"
METHOD_FROBENIUS = "local_frobenius_mmr"
SUMMARY_METHODS = {METHOD_LOCAL_KL, METHOD_FROBENIUS}
DEFAULT_QUERY_WEIGHT = 0.5


def _method_metadata(alpha_w: float, alpha_h: float, l1_ratio: float) -> dict:
    return {
        "loss_name": SPARSE_KL_LOSS_NAME,
        "solver": SPARSE_KL_SOLVER,
        "alpha_W": alpha_w,
        "alpha_H": alpha_h,
        "l1_ratio": l1_ratio,
    }


def _metadata_for_method(
    method: str,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
) -> dict:
    if method == METHOD_FROBENIUS:
        return {
            "loss_name": "frobenius",
            "solver": SPARSE_KL_SOLVER,
            "alpha_W": alpha_w,
            "alpha_H": alpha_h,
            "l1_ratio": l1_ratio,
            "objective": "local_frobenius_plus_l1",
        }
    return {
        **_method_metadata(alpha_w, alpha_h, l1_ratio),
        "objective": "local_kl_plus_l1",
    }


def _fallback(
    sentences: list[str],
    budget: int,
    reason: str,
    k: int = 0,
    *,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    method: str = METHOD_LOCAL_KL,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    alpha_w: float = SPARSE_KL_ALPHA_W,
    alpha_h: float = SPARSE_KL_ALPHA_H,
    l1_ratio: float = SPARSE_KL_L1_RATIO,
) -> dict:
    summary, indices = lead_n(sentences, budget)
    return {
        "k": k,
        "method": method,
        **_metadata_for_method(method, alpha_w, alpha_h, l1_ratio),
        "k_selection_method": "fallback",
        "k_candidates": [],
        "selection_method": SELECTION_METHOD_FALLBACK,
        "mmr_lambda": mmr_lambda,
        "query_feedback_applied": False,
        "query_feedback_reason": None,
        "pseudo_relevant_indices": [],
        "query_weight": query_weight,
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
                "score_round": None,
                "dominant_topic": None,
                "query_relevance": 0.0,
            }
            for i, sentence in enumerate(sentences)
        ],
        "selection_trace": [],
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
        return k_used, K_SELECTION_METHOD_REQUESTED, [{"k": k_used, "score": None}]
    k_used, candidates, selection_method = select_k_by_imputation(
        V,
        seed=seed,
    )
    return min(k_used, upper), selection_method, candidates


def _fit_local_nmf(
    V: np.ndarray,
    k: int,
    seed: int,
    loss_name: str,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    nmf = SklearnNMF(
        n_components=k,
        init="nndsvdar",
        solver=SPARSE_KL_SOLVER,
        beta_loss=loss_name,
        alpha_W=alpha_w,
        alpha_H=alpha_h,
        l1_ratio=l1_ratio,
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


def _factor_sparsity_stats(W: np.ndarray, H: np.ndarray, threshold: float) -> dict:
    return {
        "zero_threshold": threshold,
        "w_zero_fraction": float(np.mean(np.abs(W) <= threshold)),
        "h_zero_fraction": float(np.mean(np.abs(H) <= threshold)),
        "active_topics_per_sentence": float(np.mean(np.sum(np.abs(W) > threshold, axis=1))),
        "active_terms_per_topic": float(np.mean(np.sum(np.abs(H) > threshold, axis=1))),
    }


def _topic_terms(
    H: np.ndarray,
    terms: list[str],
    W: np.ndarray,
    sentences: list[str],
    top_n: int = 5,
    example_n: int = 2,
) -> list[dict]:
    topic_distributions = H / (H.sum(axis=1, keepdims=True) + EPS)
    term_exclusivity = H / (H.sum(axis=0, keepdims=True) + EPS)
    distinctive_scores = topic_distributions * term_exclusivity
    sentence_topic_weights = W / (W.sum(axis=1, keepdims=True) + EPS)
    topics = []
    for topic_index, row in enumerate(distinctive_scores):
        term_indices = np.argsort(-row, kind="stable")[:top_n]
        example_indices = sorted(
            range(len(sentences)),
            key=lambda index: (-float(sentence_topic_weights[index, topic_index]), index),
        )[:example_n]
        topics.append({
            "top_terms": [terms[int(i)] for i in term_indices],
            "representative_sentences": [sentences[i] for i in example_indices],
        })
    return topics


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
    relevance_scores: np.ndarray | None = None,
    query_scores: np.ndarray | None = None,
) -> tuple[list[int], list[dict], list[dict]]:
    """Select sentences with standard Maximum Marginal Relevance."""
    W_normalized = W / (W.sum(axis=1, keepdims=True) + EPS)
    relevance_scores = (
        _topic_relevance(W, V)
        if relevance_scores is None
        else np.asarray(relevance_scores, dtype=float)
    )
    selected: list[int] = []
    analysis: list[dict | None] = [None] * len(sentences)
    selection_trace: list[dict] = []
    norms = np.linalg.norm(V, axis=1)

    for round_number in range(1, budget + 1):
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
                "score_round": round_number,
                "dominant_topic": int(np.argmax(W_normalized[i])),
                "query_relevance": float(query_scores[i]) if query_scores is not None else 0.0,
            }
            analysis[i] = current
            candidates.append((score, relevance, i))
            current_scores[i] = current

        chosen = min(candidates, key=lambda item: (-item[0], -item[1], item[2]))[2]
        analysis[chosen] = {**current_scores[chosen], "selected": True}
        selection_trace.append({
            "round": round_number,
            "selected_index": chosen,
            "candidates": [
                {
                    "index": item["index"],
                    "relevance": item["relevance"],
                    "redundancy": item["redundancy"],
                    "score": item["score"],
                }
                for item in sorted(
                    current_scores.values(),
                    key=lambda candidate: (
                        -candidate["score"], -candidate["relevance"], candidate["index"]
                    ),
                )
            ],
        })
        selected.append(chosen)

    ordered = sorted(selected)
    selected_set = set(ordered)
    final_analysis = [
        {**item, "selected": item["index"] in selected_set}
        for item in analysis
        if item is not None
    ]
    return ordered, final_analysis, selection_trace


def _query_feedback(
    query: str | None,
    vectorizer,
    V: np.ndarray,
    budget: int,
) -> tuple[np.ndarray | None, list[int], str | None]:
    if not query:
        return None, [], None
    if vectorizer is None:
        return None, [], "Không có vectorizer cục bộ"
    processed_query = preprocess_sentences([query])[0]
    if not processed_query:
        return None, [], "Truy vấn không còn từ hữu ích sau tiền xử lý"
    query_vector = vectorizer.transform([processed_query]).toarray()[0]
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= EPS:
        return None, [], "Không có từ nào trong truy vấn thuộc vocabulary của văn bản"
    query_vector /= query_norm

    initial_scores = V @ query_vector
    eligible = np.flatnonzero(initial_scores > 0)
    if eligible.size == 0:
        return None, [], "Không tìm thấy câu liên quan truy vấn trong văn bản"
    feedback_count = min(3, max(1, budget), eligible.size)
    pseudo_relevant = sorted(
        eligible.tolist(), key=lambda index: (-float(initial_scores[index]), index)
    )[:feedback_count]
    feedback_centroid = V[pseudo_relevant].mean(axis=0)
    expanded_query = 0.5 * query_vector + 0.5 * feedback_centroid
    expanded_norm = float(np.linalg.norm(expanded_query))
    if expanded_norm > EPS:
        expanded_query /= expanded_norm
    query_scores = V @ expanded_query
    max_score = float(np.max(query_scores)) if query_scores.size else 0.0
    if max_score > EPS:
        query_scores = query_scores / max_score
    return query_scores, pseudo_relevant, None


def summarize(
    text: str,
    k: int | None = None,
    summary_sentences: int = 3,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    seed: int = 42,
    method: str = METHOD_LOCAL_KL,
    query: str | None = None,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    alpha_w: float = SPARSE_KL_ALPHA_W,
    alpha_h: float = SPARSE_KL_ALPHA_H,
    l1_ratio: float = SPARSE_KL_L1_RATIO,
    include_factor_stats: bool = False,
    factor_zero_threshold: float = 1e-8,
) -> dict:
    """Summarize a text using local KL-NMF + L1 or local Frobenius-NMF."""
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k must be a positive integer")
    if not isinstance(summary_sentences, int) or summary_sentences < 1:
        raise ValueError("summary_sentences must be a positive integer")
    if not np.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be finite and between 0 and 1")
    if method not in SUMMARY_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(SUMMARY_METHODS))}")
    if not np.isfinite(query_weight) or not 0 <= query_weight <= 1:
        raise ValueError("query_weight must be finite and between 0 and 1")
    if not all(np.isfinite(value) and value >= 0 for value in (alpha_w, alpha_h)):
        raise ValueError("alpha_w and alpha_h must be finite and nonnegative")
    if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be finite and between 0 and 1")
    if not np.isfinite(factor_zero_threshold) or factor_zero_threshold < 0:
        raise ValueError("factor_zero_threshold must be finite and nonnegative")
    if query and method != METHOD_LOCAL_KL:
        raise ValueError("Query PRF hiện chỉ hỗ trợ với phương pháp Local KL-NMF + MMR")

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("text must contain at least one sentence")

    budget = min(summary_sentences, len(sentences))
    processed = preprocess_sentences(sentences)
    V, terms, vectorizer = fit_sentence_vectorizer(
        processed,
        ngram_range=LOCAL_TFIDF_NGRAM_RANGE,
        max_features=LOCAL_TFIDF_MAX_FEATURES,
    )
    if len(terms) == 0 or not np.any(V):
        fallback_k = max(1, len(sentences) // 2) if k is None else k
        return _fallback(
            sentences,
            budget,
            "TF-IDF has no useful terms",
            fallback_k,
            mmr_lambda=mmr_lambda,
            method=method,
            query_weight=query_weight,
            alpha_w=alpha_w,
            alpha_h=alpha_h,
            l1_ratio=l1_ratio,
        )

    k_used, k_selection_method, k_candidates = _select_or_resolve_k(V, k, seed)
    if k_used < 1:
        return _fallback(
            sentences,
            budget,
            "Not enough sentences or terms for local NMF",
            0,
            mmr_lambda=mmr_lambda,
            method=method,
            query_weight=query_weight,
            alpha_w=alpha_w,
            alpha_h=alpha_h,
            l1_ratio=l1_ratio,
        )

    query_scores: np.ndarray | None = None
    pseudo_relevant: list[int] = []
    query_feedback_reason: str | None = None
    loss_name = SPARSE_KL_LOSS_NAME if method == METHOD_LOCAL_KL else "frobenius"
    W, H = _fit_local_nmf(
        V, k_used, seed, loss_name, alpha_w, alpha_h, l1_ratio
    )
    if not np.any(W > EPS) or not np.any(H > EPS):
        return _fallback(
            sentences,
            budget,
            "NMF factors collapsed to zero; reduce alpha_W/alpha_H or L1 ratio",
            k_used,
            mmr_lambda=mmr_lambda,
            method=method,
            query_weight=query_weight,
            alpha_w=alpha_w,
            alpha_h=alpha_h,
            l1_ratio=l1_ratio,
        )
    if method == METHOD_LOCAL_KL:
        query_scores, pseudo_relevant, query_feedback_reason = _query_feedback(
            query, vectorizer, V, budget
        )
    local_loss = objective_breakdown(
        V,
        W,
        H,
        loss=loss_name,
        alpha_w=alpha_w * V.shape[1],
        alpha_h=alpha_h * V.shape[0],
        l1_ratio=l1_ratio,
    )
    relevance_scores = _topic_relevance(W, V)
    if query_scores is not None:
        relevance_scores = (
            (1.0 - query_weight) * relevance_scores
            + query_weight * query_scores
        )
    ordered, analysis, selection_trace = _select_sentences(
        sentences,
        V,
        W,
        budget,
        mmr_lambda=mmr_lambda,
        relevance_scores=relevance_scores,
        query_scores=query_scores,
    )
    topics = _topic_terms(H, terms, W, sentences)
    factor_stats = (
        _factor_sparsity_stats(W, H, factor_zero_threshold)
        if include_factor_stats
        else None
    )
    selection_method = (
        "query_prf_mmr" if query_scores is not None else SELECTION_METHOD_MMR
    )
    method_extra = {
        "query_feedback_method": (
            "pseudo_relevance_feedback" if query_scores is not None else None
        )
    }

    return {
        "k": k_used,
        "method": method,
        **_metadata_for_method(method, alpha_w, alpha_h, l1_ratio),
        "k_selection_method": k_selection_method,
        "k_candidates": k_candidates,
        "selection_method": selection_method,
        "mmr_lambda": mmr_lambda,
        "query_feedback_applied": query_scores is not None,
        "query_feedback_reason": query_feedback_reason,
        "pseudo_relevant_indices": pseudo_relevant,
        "query_weight": query_weight,
        "loss_value": local_loss.total,
        "data_loss": local_loss.data,
        "regularization_loss": local_loss.regularization,
        **method_extra,
        "summary": " ".join(sentences[i] for i in ordered),
        "selected_indices": ordered,
        "topics": topics,
        "sentence_analysis": analysis,
        "selection_trace": selection_trace,
        "fallback_reason": None,
        **({"factor_sparsity": factor_stats} if include_factor_stats else {}),
    }
