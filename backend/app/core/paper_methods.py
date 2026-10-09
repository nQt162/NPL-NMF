"""NMF variants inspired by sentence-relation summarization papers."""

from __future__ import annotations

import numpy as np

from .nmf import EPS


def sentence_similarity_graph(V: np.ndarray, neighbors: int = 5) -> np.ndarray:
    """Build a symmetric, sparse TF-IDF cosine graph over sentences."""
    V = np.asarray(V, dtype=float)
    norms = np.linalg.norm(V, axis=1)
    normalized = V / np.maximum(norms[:, None], EPS)
    similarities = np.clip(normalized @ normalized.T, 0.0, 1.0)
    np.fill_diagonal(similarities, 0.0)

    n_sentences = V.shape[0]
    graph = np.zeros_like(similarities)
    neighbor_count = min(max(1, neighbors), max(1, n_sentences - 1))
    for index in range(n_sentences):
        candidates = np.flatnonzero(similarities[index] > 0)
        if candidates.size == 0:
            continue
        ranked = candidates[np.argsort(-similarities[index, candidates], kind="stable")]
        chosen = ranked[:neighbor_count]
        graph[index, chosen] = similarities[index, chosen]
    return np.maximum(graph, graph.T)


def nmfts_objective(
    V: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    similarity: np.ndarray,
    pairwise_lambda: float,
) -> tuple[float, float, float]:
    """Return reconstruction, pairwise symmetric-KL, and total objectives."""
    residual = V - W @ H
    reconstruction_loss = float(np.sum(residual * residual))
    left = np.maximum(W[:, None, :], EPS)
    right = np.maximum(W[None, :, :], EPS)
    symmetric_kl = np.sum((left - right) * np.log(left / right), axis=2)
    pairwise_loss = float(pairwise_lambda * np.sum(similarity * symmetric_kl) / max(1, len(V)))
    return reconstruction_loss, pairwise_loss, reconstruction_loss + pairwise_loss


def fit_nmfts(
    V: np.ndarray,
    k: int,
    similarity: np.ndarray,
    *,
    pairwise_lambda: float = 0.1,
    seed: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit Frobenius NMF with the paper's pairwise symmetric-KL penalty."""
    V = np.asarray(V, dtype=float)
    similarity = np.asarray(similarity, dtype=float)
    if V.ndim != 2 or not all(V.shape) or np.any(V < 0) or not np.all(np.isfinite(V)):
        raise ValueError("V phải là ma trận hữu hạn, không âm và không rỗng")
    if similarity.shape != (V.shape[0], V.shape[0]):
        raise ValueError("Ma trận tương đồng phải có kích thước số câu × số câu")
    if np.any(similarity < 0) or not np.all(np.isfinite(similarity)):
        raise ValueError("Ma trận tương đồng phải hữu hạn và không âm")
    if not np.isfinite(pairwise_lambda) or pairwise_lambda < 0:
        raise ValueError("pairwise_lambda phải hữu hạn và không âm")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter phải là số nguyên dương")

    rank = min(k, *V.shape)
    rng = np.random.default_rng(seed)
    W = rng.random((V.shape[0], rank)) + EPS
    H = rng.random((rank, V.shape[1])) + EPS
    graph_degree = similarity.sum(axis=1, keepdims=True)
    losses: list[dict[str, float]] = []

    for iteration in range(1, max_iter + 1):
        neighbor_mass = similarity @ W
        pairwise_numerator = neighbor_mass / np.maximum(W, EPS)
        numerator = V @ H.T + pairwise_lambda * pairwise_numerator
        denominator = W @ H @ H.T + pairwise_lambda * graph_degree + EPS
        W *= numerator / denominator
        H *= (W.T @ V) / (W.T @ W @ H + EPS)

        reconstruction, pairwise, total = nmfts_objective(
            V, W, H, similarity, pairwise_lambda
        )
        losses.append({
            "iteration": float(iteration),
            "reconstruction_loss": reconstruction,
            "pairwise_loss": pairwise,
            "loss": total,
        })
        if iteration >= 5:
            previous = losses[-2]["loss"]
            relative_improvement = (previous - total) / max(previous, EPS)
            if 0 <= relative_improvement < 1e-5:
                break
    return W, H, losses


def fit_symmetric_nmf(
    similarity: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, list[float]]:
    """Factor a sentence graph as G @ G.T using symmetric NMF updates."""
    similarity = np.asarray(similarity, dtype=float)
    if (
        similarity.ndim != 2
        or similarity.shape[0] != similarity.shape[1]
        or not all(similarity.shape)
        or np.any(similarity < 0)
        or not np.all(np.isfinite(similarity))
    ):
        raise ValueError("Ma trận tương đồng phải vuông, hữu hạn và không âm")
    rank = min(k, similarity.shape[0])
    rng = np.random.default_rng(seed)
    G = rng.random((similarity.shape[0], rank)) + EPS
    losses: list[float] = []
    for iteration in range(1, max_iter + 1):
        G *= (similarity @ G) / (G @ (G.T @ G) + EPS)
        residual = similarity - G @ G.T
        loss = float(np.sum(residual * residual))
        losses.append(loss)
        if iteration >= 5:
            relative_improvement = (losses[-2] - loss) / max(losses[-2], EPS)
            if 0 <= relative_improvement < 1e-5:
                break
    return G, losses
