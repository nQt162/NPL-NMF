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
    mask: np.ndarray | None = None,
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

    rank = min(k, *V.shape)
    rng = np.random.default_rng(seed)
    W = rng.random((V.shape[0], rank)) + EPS
    H = rng.random((rank, V.shape[1])) + EPS
    snapshots = [_snapshot(0, V, W, H)] if trace else []
    losses: list[float] = []

    for iteration in range(1, max_iter + 1):
        if mask is not None:
            V_masked = mask * V
            WH_masked = mask * (W @ H)
            H *= (W.T @ V_masked) / (W.T @ WH_masked + EPS)
            # Recompute WH_masked after H is updated
            WH_masked = mask * (W @ H)
            W *= (V_masked @ H.T) / (WH_masked @ H.T + EPS)
            if trace:
                current = _snapshot(iteration, V, W, H)
                loss = current.loss
            else:
                loss = 0.5 * float(np.sum((mask * (V - W @ H)) ** 2))
        else:
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


def auto_select_k(V: np.ndarray, max_k: int = 10, seed: int = 42) -> int:
    """Determine the optimal k using missing value imputation over multiple trials."""
    V = np.asarray(V, dtype=float)
    n_sentences = V.shape[0]
    limit = min(max_k, n_sentences - 1)
    if limit < 2:
        return max(1, limit)
    
    k_values = list(range(2, limit + 1))
    if len(k_values) == 1:
        return k_values[0]

    n_trials = 5
    rng = np.random.default_rng(seed)
    selected_ks = []

    for trial in range(n_trials):
        # Create a mask where ~30% of entries are selected as missing (0)
        mask = (rng.random(V.shape) > 0.30).astype(float)
        missing_mask = (mask == 0.0)
    
        val_errors = []
        for k in k_values:
            # Impute missing entries with NMF. Use different seed for each trial
            res = fit_nmf(V, k=k, seed=seed + trial * 100, max_iter=50, mask=mask)
            reconstruction = res.W @ res.H
            
            # Compare imputed entries to their observed values
            if np.any(missing_mask):
                mse = float(np.mean((V[missing_mask] - reconstruction[missing_mask]) ** 2))
            else:
                mse = res.losses[-1] if res.losses else 0.0
                
            val_errors.append(mse)
    
        # The k that gives the smallest imputation error is selected
        best_idx = int(np.argmin(val_errors))
        selected_ks.append(k_values[best_idx])

    # Select the k that appears most frequently (mode)
    from collections import Counter
    most_common_k = Counter(selected_ks).most_common(1)[0][0]
    return most_common_k
