"""Shared, from-scratch NMF with multiplicative updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class Snapshot:
    iteration: int
    W: np.ndarray
    H: np.ndarray
    WH: np.ndarray
    loss: float


@dataclass(frozen=True)
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    losses: list[float]
    snapshots: list[Snapshot]


def _snapshot(iteration: int, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Snapshot:
    reconstruction = W @ H
    loss = 0.5 * float(np.sum((V - reconstruction) ** 2))
    return Snapshot(iteration, W.copy(), H.copy(), reconstruction.copy(), loss)


def fit_nmf(
    V: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
    trace: bool = False,
) -> NMFResult:
    """Factor V ≈ W @ H; record snapshots only for a small visual trace.

    `losses` contains post-update losses. Snapshot zero records the initial
    matrices and their loss before any update.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or not all(V.shape) or not np.all(np.isfinite(V)) or np.any(V < 0):
        raise ValueError("V phải là ma trận 2 chiều không rỗng, hữu hạn và không âm")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k phải là số nguyên dương")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter phải là số nguyên dương")
    if trace and (V.shape[0] > 10 or V.shape[1] > 50 or max_iter > 20):
        raise ValueError("trace chỉ hỗ trợ tối đa 10 câu, 50 từ và 20 vòng")

    rank = min(k, *V.shape)
    rng = np.random.default_rng(seed)
    W = rng.random((V.shape[0], rank)) + EPS
    H = rng.random((rank, V.shape[1])) + EPS
    snapshots = [_snapshot(0, V, W, H)] if trace else []
    losses: list[float] = []

    for iteration in range(1, max_iter + 1):
        H *= (W.T @ V) / (W.T @ W @ H + EPS)
        W *= (V @ H.T) / (W @ H @ H.T + EPS)
        if trace:
            current = _snapshot(iteration, V, W, H)
            loss = current.loss
        else:
            loss = 0.5 * float(np.sum((V - W @ H) ** 2))
        losses.append(loss)
        if trace:
            snapshots.append(current)
        if iteration >= 5:
            previous = losses[-2]
            relative_improvement = (previous - loss) / max(previous, EPS)
            if 0 <= relative_improvement < 1e-5:
                break
    return NMFResult(W, H, losses, snapshots)
