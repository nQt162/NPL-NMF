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
    objective_breakdown,
    select_k_by_imputation,
)
from .paper_methods import fit_nmfts, fit_symmetric_nmf, sentence_similarity_graph
from .preprocess import preprocess_sentences
from .sentences import split_sentences
from .vectorize import fit_sentence_vectorizer


SUMMARY_NGRAM_RANGE = (1, 2)
SUMMARY_MAX_FEATURES = 800
K_SELECTION_METHOD_AUTO = "masked_kl_imputation"
K_SELECTION_METHOD_REQUESTED = "requested"
DEFAULT_MMR_LAMBDA = 0.7
SELECTION_METHOD_MMR = "maximum_marginal_relevance"
SELECTION_METHOD_FALLBACK = "lead_n_fallback"
METHOD_LOCAL_KL = "local_kl_mmr"
METHOD_NMFTS = "nmfts_pairwise"
METHOD_SNMF = "snmf"
SUMMARY_METHODS = {METHOD_LOCAL_KL, METHOD_NMFTS, METHOD_SNMF}
DEFAULT_PAIRWISE_LAMBDA = 0.1
DEFAULT_QUERY_WEIGHT = 0.5
MAX_PAIRWISE_SENTENCES = 500


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
    pairwise_lambda: float,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
) -> dict:
    if method == METHOD_NMFTS:
        return {
            "loss_name": "frobenius_plus_pairwise_symmetric_kl",
            "solver": "custom_multiplicative_updates",
            "alpha_W": None,
            "alpha_H": None,
            "l1_ratio": None,
            "objective": "nmfts_frobenius_pairwise_symmetric_kl",
            "pairwise_lambda": pairwise_lambda,
        }
    if method == METHOD_SNMF:
        return {
            "loss_name": "symmetric_frobenius",
            "solver": "symmetric_multiplicative_updates",
            "alpha_W": None,
            "alpha_H": None,
            "l1_ratio": None,
            "objective": "symmetric_similarity_reconstruction",
            "pairwise_lambda": None,
        }
    return {
        **_method_metadata(alpha_w, alpha_h, l1_ratio),
        "objective": "local_kl_plus_l1",
        "pairwise_lambda": None,
    }


def _fallback(
    sentences: list[str],
    budget: int,
    reason: str,
    k: int = 0,
    *,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    method: str = METHOD_LOCAL_KL,
    pairwise_lambda: float = DEFAULT_PAIRWISE_LAMBDA,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    alpha_w: float = SPARSE_KL_ALPHA_W,
    alpha_h: float = SPARSE_KL_ALPHA_H,
    l1_ratio: float = SPARSE_KL_L1_RATIO,
) -> dict:
    summary, indices = lead_n(sentences, budget)
    return {
        "k": k,
        "method": method,
        **_metadata_for_method(method, pairwise_lambda, alpha_w, alpha_h, l1_ratio),
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
                "dominant_topic": None,
                "query_relevance": 0.0,
            }
            for i, sentence in enumerate(sentences)
        ],
        "fallback_reason": reason,
    }


def _select_or_resolve_k(
    V: np.ndarray,
    requested_k: int | None,
    seed: int,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
) -> tuple[int, str, list[dict]]:
    upper = min(V.shape[0], V.shape[1])
    if requested_k is not None:
        k_used = min(requested_k, upper)
        return k_used, K_SELECTION_METHOD_REQUESTED, [{"k": k_used, "score": 0.0}]
    k_used, candidates = select_k_by_imputation(
        V,
        seed=seed,
        loss=SPARSE_KL_LOSS_NAME,
        alpha_w=alpha_w,
        alpha_h=alpha_h,
        l1_ratio=l1_ratio,
    )
    return min(k_used, upper), K_SELECTION_METHOD_AUTO, candidates


def _fit_local_kl_nmf(
    V: np.ndarray,
    k: int,
    seed: int,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    nmf = SklearnNMF(
        n_components=k,
        init="nndsvda",
        solver=SPARSE_KL_SOLVER,
        beta_loss=SPARSE_KL_LOSS_NAME,
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
    relevance_scores: np.ndarray | None = None,
    query_scores: np.ndarray | None = None,
) -> tuple[list[int], list[dict]]:
    """Select sentences with standard Maximum Marginal Relevance."""
    W_normalized = W / (W.sum(axis=1, keepdims=True) + EPS)
    relevance_scores = (
        _topic_relevance(W, V)
        if relevance_scores is None
        else np.asarray(relevance_scores, dtype=float)
    )
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
                "query_relevance": float(query_scores[i]) if query_scores is not None else 0.0,
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


def _snmf_sentence_selection(
    sentences: list[str],
    V: np.ndarray,
    terms: list[str],
    similarity: np.ndarray,
    budget: int,
    k: int,
    seed: int,
) -> tuple[list[int], list[dict], list[dict], float]:
    G, losses = fit_symmetric_nmf(similarity, k, seed=seed)
    memberships = G / (G.sum(axis=1, keepdims=True) + EPS)
    labels = np.argmax(memberships, axis=1)
    global_centrality = similarity.sum(axis=1)
    max_global = float(global_centrality.max()) if global_centrality.size else 0.0
    if max_global > EPS:
        global_centrality /= max_global

    intra_cluster = np.zeros(len(sentences), dtype=float)
    cluster_salience = memberships.sum(axis=0)
    for index, label in enumerate(labels):
        members = np.flatnonzero(labels == label)
        other_members = members[members != index]
        if other_members.size:
            intra_cluster[index] = float(similarity[index, other_members].mean())
    max_intra = float(intra_cluster.max()) if intra_cluster.size else 0.0
    if max_intra > EPS:
        intra_cluster /= max_intra
    relevance = 0.7 * intra_cluster + 0.3 * global_centrality

    cluster_order = np.argsort(-cluster_salience, kind="stable").tolist()
    selected: list[int] = []
    for cluster in cluster_order:
        candidates = [i for i in range(len(sentences)) if labels[i] == cluster and i not in selected]
        if candidates and len(selected) < budget:
            selected.append(min(candidates, key=lambda i: (-relevance[i], i)))
    if len(selected) < budget:
        remaining = [i for i in range(len(sentences)) if i not in selected]
        selected.extend(sorted(remaining, key=lambda i: (-relevance[i], i))[: budget - len(selected)])
    ordered = sorted(selected)
    analysis = []
    for index, sentence in enumerate(sentences):
        redundancy = max(
            (float(similarity[index, chosen]) for chosen in selected if chosen != index),
            default=0.0,
        )
        analysis.append({
            "index": index,
            "text": sentence,
            "selected": index in selected,
            "relevance": float(relevance[index]),
            "redundancy": redundancy,
            "score": float(relevance[index]),
            "dominant_topic": int(labels[index]),
            "query_relevance": 0.0,
        })

    topics = []
    for cluster in range(G.shape[1]):
        member_indices = np.flatnonzero(labels == cluster)
        if member_indices.size == 0:
            continue
        term_weights = V[member_indices].sum(axis=0)
        top_terms = [terms[int(i)] for i in np.argsort(-term_weights, kind="stable")[:5]]
        topics.append({"top_terms": top_terms})
    return ordered, analysis, topics, losses[-1] if losses else 0.0


def summarize(
    text: str,
    k: int | None = None,
    summary_sentences: int = 3,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    seed: int = 42,
    method: str = METHOD_LOCAL_KL,
    pairwise_lambda: float = DEFAULT_PAIRWISE_LAMBDA,
    query: str | None = None,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    alpha_w: float = SPARSE_KL_ALPHA_W,
    alpha_h: float = SPARSE_KL_ALPHA_H,
    l1_ratio: float = SPARSE_KL_L1_RATIO,
) -> dict:
    """Summarize a text with local KL-NMF, paper-inspired NMFTS, or SNMF."""
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k must be a positive integer")
    if not isinstance(summary_sentences, int) or summary_sentences < 1:
        raise ValueError("summary_sentences must be a positive integer")
    if not np.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be finite and between 0 and 1")
    if method not in SUMMARY_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(SUMMARY_METHODS))}")
    if not np.isfinite(pairwise_lambda) or pairwise_lambda < 0:
        raise ValueError("pairwise_lambda must be finite and nonnegative")
    if not np.isfinite(query_weight) or not 0 <= query_weight <= 1:
        raise ValueError("query_weight must be finite and between 0 and 1")
    if not all(np.isfinite(value) and value >= 0 for value in (alpha_w, alpha_h)):
        raise ValueError("alpha_w and alpha_h must be finite and nonnegative")
    if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be finite and between 0 and 1")
    if query and method != METHOD_LOCAL_KL:
        raise ValueError("Query PRF hiện chỉ hỗ trợ với phương pháp Local KL-NMF + MMR")

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("text must contain at least one sentence")

    budget = min(summary_sentences, len(sentences))
    processed = preprocess_sentences(sentences)
    V, terms, vectorizer = fit_sentence_vectorizer(
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
            method=method,
            pairwise_lambda=pairwise_lambda,
            query_weight=query_weight,
            alpha_w=alpha_w,
            alpha_h=alpha_h,
            l1_ratio=l1_ratio,
        )

    k_used, k_selection_method, k_candidates = _select_or_resolve_k(
        V, k, seed, alpha_w, alpha_h, l1_ratio
    )
    if k_used < 1:
        return _fallback(
            sentences,
            budget,
            "Not enough sentences or terms for local NMF",
            0,
            mmr_lambda=mmr_lambda,
            method=method,
            pairwise_lambda=pairwise_lambda,
            query_weight=query_weight,
            alpha_w=alpha_w,
            alpha_h=alpha_h,
            l1_ratio=l1_ratio,
        )

    query_scores: np.ndarray | None = None
    pseudo_relevant: list[int] = []
    loss_value: float | None = None
    data_loss: float | None = None
    regularization_loss: float | None = None
    query_feedback_reason: str | None = None
    if method == METHOD_NMFTS:
        if len(sentences) > MAX_PAIRWISE_SENTENCES:
            raise ValueError(f"NMFTS xử lý tối đa {MAX_PAIRWISE_SENTENCES} câu mỗi lần")
        similarity = sentence_similarity_graph(V)
        W, H, losses = fit_nmfts(
            V,
            k_used,
            similarity,
            pairwise_lambda=pairwise_lambda,
            seed=seed,
        )
        loss_value = losses[-1]["loss"] if losses else None
        data_loss = losses[-1]["reconstruction_loss"] if losses else None
        regularization_loss = losses[-1]["pairwise_loss"] if losses else None
        ordered, analysis = _select_sentences(
            sentences, V, W, budget, mmr_lambda=mmr_lambda
        )
        topics = _topic_terms(H, terms)
        selection_method = SELECTION_METHOD_MMR
        method_extra = {"loss_trace": losses, "sentence_similarity": "tfidf_cosine_knn"}
    elif method == METHOD_SNMF:
        if len(sentences) > MAX_PAIRWISE_SENTENCES:
            raise ValueError(f"SNMF xử lý tối đa {MAX_PAIRWISE_SENTENCES} câu mỗi lần")
        similarity = sentence_similarity_graph(V)
        ordered, analysis, topics, loss_value = _snmf_sentence_selection(
            sentences, V, terms, similarity, budget, k_used, seed
        )
        data_loss = loss_value
        selection_method = "symmetric_nmf_cluster_ranking"
        method_extra = {"sentence_similarity": "tfidf_cosine_knn", "cluster_weight": 0.7}
    else:
        W, H = _fit_local_kl_nmf(V, k_used, seed, alpha_w, alpha_h, l1_ratio)
        query_scores, pseudo_relevant, query_feedback_reason = _query_feedback(
            query, vectorizer, V, budget
        )
        local_loss = objective_breakdown(
            V,
            W,
            H,
            loss=SPARSE_KL_LOSS_NAME,
            alpha_w=alpha_w * V.shape[1],
            alpha_h=alpha_h * V.shape[0],
            l1_ratio=l1_ratio,
        )
        loss_value = local_loss.total
        data_loss = local_loss.data
        regularization_loss = local_loss.regularization
        relevance_scores = _topic_relevance(W, V)
        if query_scores is not None:
            relevance_scores = (
                (1.0 - query_weight) * relevance_scores
                + query_weight * query_scores
            )
        ordered, analysis = _select_sentences(
            sentences,
            V,
            W,
            budget,
            mmr_lambda=mmr_lambda,
            relevance_scores=relevance_scores,
            query_scores=query_scores,
        )
        topics = _topic_terms(H, terms)
        selection_method = "query_prf_mmr" if query_scores is not None else SELECTION_METHOD_MMR
        method_extra = {"query_feedback_method": "pseudo_relevance_feedback" if query_scores is not None else None}

    return {
        "k": k_used,
        "method": method,
        **_metadata_for_method(method, pairwise_lambda, alpha_w, alpha_h, l1_ratio),
        "k_selection_method": k_selection_method,
        "k_candidates": k_candidates,
        "selection_method": selection_method,
        "mmr_lambda": mmr_lambda,
        "query_feedback_applied": query_scores is not None,
        "query_feedback_reason": query_feedback_reason,
        "pseudo_relevant_indices": pseudo_relevant,
        "query_weight": query_weight,
        "loss_value": loss_value,
        "data_loss": data_loss,
        "regularization_loss": regularization_loss,
        **method_extra,
        "summary": " ".join(sentences[i] for i in ordered),
        "selected_indices": ordered,
        "topics": topics,
        "sentence_analysis": analysis,
        "fallback_reason": None,
    }
