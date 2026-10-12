"""Shared NMF utilities for KL/Frobenius optimization and k selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


EPS = 1e-9
LossName = Literal["frobenius", "kullback-leibler"]

SPARSE_KL_LOSS_NAME: LossName = "kullback-leibler"
SPARSE_KL_SOLVER = "mu"
SPARSE_KL_ALPHA_W = 0.001
SPARSE_KL_ALPHA_H = 0.001
SPARSE_KL_L1_RATIO = 1.0
SPARSE_KL_MAX_ITER = 500
K_SELECTION_MAX_K = 10
K_SELECTION_TRIALS = 10
K_SELECTION_VALIDATION_FRACTION = 0.2
K_SELECTION_MAX_ITER = 120
K_SELECTION_METHOD_MASKED = "positive_entry_masked_kl_cv"
K_SELECTION_METHOD_HEURISTIC = "heuristic_low_observations"
K_SELECTION_METHOD_SINGLE = "single_feasible_k"


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    data: float
    regularization: float


@dataclass(frozen=True)
class Snapshot:
    iteration: int
    W: np.ndarray
    H: np.ndarray
    WH: np.ndarray
    loss: float
    data_loss: float
    regularization_loss: float


@dataclass(frozen=True)
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    losses: list[float]
    snapshots: list[Snapshot]


def _regularization(
    W: np.ndarray,
    H: np.ndarray,
    *,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
) -> float:
    l1 = l1_ratio * (alpha_w * float(W.sum()) + alpha_h * float(H.sum()))
    l2_weight = 1.0 - l1_ratio
    l2 = 0.5 * l2_weight * (
        alpha_w * float(np.sum(W ** 2)) + alpha_h * float(np.sum(H ** 2))
    )
    return l1 + l2


def kl_divergence(V: np.ndarray, reconstruction: np.ndarray) -> float:
    """Return elementwise KL divergence sum for nonnegative matrices."""
    V = np.asarray(V, dtype=float)
    safe_reconstruction = np.maximum(np.asarray(reconstruction, dtype=float), EPS)
    element_loss = np.where(
        V > 0,
        V * np.log((V + EPS) / safe_reconstruction) - V + safe_reconstruction,
        safe_reconstruction,
    )
    return float(np.sum(element_loss))


def objective_breakdown(
    V: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    loss: LossName,
    alpha_w: float = 0.0,
    alpha_h: float = 0.0,
    l1_ratio: float = 1.0,
    mask: np.ndarray | None = None,
) -> LossBreakdown:
    reconstruction = W @ H
    if loss == "frobenius":
        residual = V - reconstruction
        if mask is not None:
            residual = mask * residual
        data_loss = 0.5 * float(np.sum(residual ** 2))
    elif loss == "kullback-leibler":
        safe_reconstruction = np.maximum(reconstruction, EPS)
        element_loss = np.where(
            V > 0,
            V * np.log((V + EPS) / safe_reconstruction) - V + safe_reconstruction,
            safe_reconstruction,
        )
        if mask is not None:
            element_loss = mask * element_loss
        data_loss = float(np.sum(element_loss))
    else:
        raise ValueError("loss phải là 'frobenius' hoặc 'kullback-leibler'")
    regularization = _regularization(
        W,
        H,
        alpha_w=alpha_w,
        alpha_h=alpha_h,
        l1_ratio=l1_ratio,
    )
    return LossBreakdown(
        total=data_loss + regularization,
        data=data_loss,
        regularization=regularization,
    )


def _snapshot(
    iteration: int,
    V: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    loss: LossName,
    alpha_w: float = 0.0,
    alpha_h: float = 0.0,
    l1_ratio: float = 1.0,
    mask: np.ndarray | None = None,
) -> Snapshot:
    reconstruction = W @ H
    breakdown = objective_breakdown(
        V,
        W,
        H,
        loss=loss,
        alpha_w=alpha_w,
        alpha_h=alpha_h,
        l1_ratio=l1_ratio,
        mask=mask,
    )
    return Snapshot(
        iteration=iteration,
        W=W.copy(),
        H=H.copy(),
        WH=reconstruction.copy(),
        loss=breakdown.total,
        data_loss=breakdown.data,
        regularization_loss=breakdown.regularization,
    )


def _validate_factorization_inputs(
    V: np.ndarray,
    k: int,
    max_iter: int,
    loss: LossName,
    alpha_w: float,
    alpha_h: float,
    l1_ratio: float,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    if V.ndim != 2 or not all(V.shape) or not np.all(np.isfinite(V)) or np.any(V < 0):
        raise ValueError("V phải là ma trận 2 chiều không rỗng, hữu hạn và không âm")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k phải là số nguyên dương")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter phải là số nguyên dương")
    if loss not in ("frobenius", "kullback-leibler"):
        raise ValueError("loss phải là 'frobenius' hoặc 'kullback-leibler'")
    if not all(np.isfinite(value) and value >= 0 for value in (alpha_w, alpha_h)):
        raise ValueError("alpha_w và alpha_h phải hữu hạn và không âm")
    if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio phải nằm trong khoảng [0, 1]")
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=float)
    if mask.shape != V.shape or np.any(mask < 0) or not np.all(np.isfinite(mask)):
        raise ValueError("mask phải cùng kích thước với V, hữu hạn và không âm")
    return mask


def fit_nmf(
    V: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
    trace: bool = False,
    mask: np.ndarray | None = None,
    *,
    loss: LossName = "frobenius",
    alpha_w: float = 0.0,
    alpha_h: float = 0.0,
    l1_ratio: float = 1.0,
) -> NMFResult:
    """Factor V approximately as W @ H; supports KL + sparsity for tracing."""
    V = np.asarray(V, dtype=float)
    mask = _validate_factorization_inputs(
        V,
        k,
        max_iter,
        loss,
        alpha_w,
        alpha_h,
        l1_ratio,
        mask,
    )

    rank = min(k, *V.shape)
    scaled_alpha_w = alpha_w * V.shape[1]
    scaled_alpha_h = alpha_h * V.shape[0]
    rng = np.random.default_rng(seed)
    W = rng.random((V.shape[0], rank)) + EPS
    H = rng.random((rank, V.shape[1])) + EPS
    snapshots = [
        _snapshot(
            0,
            V,
            W,
            H,
            loss=loss,
            alpha_w=scaled_alpha_w,
            alpha_h=scaled_alpha_h,
            l1_ratio=l1_ratio,
            mask=mask,
        )
    ] if trace else []
    losses: list[float] = []

    for iteration in range(1, max_iter + 1):
        if loss == "frobenius" and mask is not None:
            V_masked = mask * V
            WH_masked = mask * (W @ H)
            h_penalty = scaled_alpha_h * (
                l1_ratio + (1.0 - l1_ratio) * H
            )
            H *= (W.T @ V_masked) / (W.T @ WH_masked + h_penalty + EPS)
            WH_masked = mask * (W @ H)
            w_penalty = scaled_alpha_w * (
                l1_ratio + (1.0 - l1_ratio) * W
            )
            W *= (V_masked @ H.T) / (WH_masked @ H.T + w_penalty + EPS)
        elif loss == "frobenius":
            h_penalty = scaled_alpha_h * (
                l1_ratio + (1.0 - l1_ratio) * H
            )
            H *= (W.T @ V) / (W.T @ W @ H + h_penalty + EPS)
            w_penalty = scaled_alpha_w * (
                l1_ratio + (1.0 - l1_ratio) * W
            )
            W *= (V @ H.T) / (W @ H @ H.T + w_penalty + EPS)
        else:
            WH = np.maximum(W @ H, EPS)
            ratio = V / WH
            if mask is not None:
                ratio = mask * ratio
                h_denominator = W.T @ mask
            else:
                h_denominator = W.sum(axis=0)[:, np.newaxis]
            h_denominator = (
                h_denominator
                + scaled_alpha_h * l1_ratio
                + scaled_alpha_h * (1.0 - l1_ratio) * H
            )
            H *= (W.T @ ratio) / (h_denominator + EPS)

            WH = np.maximum(W @ H, EPS)
            ratio = V / WH
            if mask is not None:
                ratio = mask * ratio
                w_denominator = mask @ H.T
            else:
                w_denominator = H.sum(axis=1)[np.newaxis, :]
            w_denominator = (
                w_denominator
                + scaled_alpha_w * l1_ratio
                + scaled_alpha_w * (1.0 - l1_ratio) * W
            )
            W *= (ratio @ H.T) / (w_denominator + EPS)

        if trace:
            current = _snapshot(
                iteration,
                V,
                W,
                H,
                loss=loss,
                alpha_w=scaled_alpha_w,
                alpha_h=scaled_alpha_h,
                l1_ratio=l1_ratio,
                mask=mask,
            )
            current_loss = current.loss
            snapshots.append(current)
        else:
            current_loss = objective_breakdown(
                V,
                W,
                H,
                loss=loss,
                alpha_w=scaled_alpha_w,
                alpha_h=scaled_alpha_h,
                l1_ratio=l1_ratio,
                mask=mask,
            ).total
        losses.append(current_loss)
        if iteration >= 5:
            previous = losses[-2]
            relative_improvement = (previous - current_loss) / max(previous, EPS)
            if 0 <= relative_improvement < 1e-5:
                break
    return NMFResult(W, H, losses, snapshots)


def select_k_by_imputation(
    V: np.ndarray,
    *,
    seed: int = 42,
    max_k: int = K_SELECTION_MAX_K,
    n_trials: int = K_SELECTION_TRIALS,
    validation_fraction: float = K_SELECTION_VALIDATION_FRACTION,
    max_iter: int = K_SELECTION_MAX_ITER,
) -> tuple[int, list[dict], str]:
    """Choose k by positive-entry masked KL CV, or disclose a low-data heuristic."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or not all(V.shape) or np.any(V < 0) or not np.all(np.isfinite(V)):
        raise ValueError("V phải là ma trận 2 chiều không rỗng, hữu hạn và không âm")
    if max_k < 1 or n_trials < 1 or not 0 < validation_fraction < 1:
        raise ValueError("max_k and n_trials must be positive; validation_fraction must be in (0, 1)")
    upper = min(max_k, V.shape[0], V.shape[1])
    if upper <= 1:
        return 1, [{"k": 1, "score": None}], K_SELECTION_METHOD_SINGLE

    positive_positions = np.argwhere(V > 0)
    if len(positive_positions) < 4:
        heuristic = min(upper, max(1, V.shape[0] // 2))
        return heuristic, [{"k": heuristic, "score": None}], K_SELECTION_METHOD_HEURISTIC

    rng = np.random.default_rng(seed)
    candidates = list(range(1, upper + 1))
    scores: dict[int, list[float]] = {candidate: [] for candidate in candidates}
    validation_count = max(1, int(round(len(positive_positions) * validation_fraction)))

    for trial in range(n_trials):
        chosen = rng.choice(len(positive_positions), size=validation_count, replace=False)
        heldout = positive_positions[chosen]
        rows = heldout[:, 0]
        cols = heldout[:, 1]
        train_mask = np.ones_like(V, dtype=float)
        train_mask[rows, cols] = 0.0

        for candidate in candidates:
            result = fit_nmf(
                V,
                candidate,
                seed=seed + trial * 997 + candidate,
                max_iter=max_iter,
                mask=train_mask,
                loss=SPARSE_KL_LOSS_NAME,
                alpha_w=0.0,
                alpha_h=0.0,
                l1_ratio=1.0,
            )
            reconstruction = np.maximum(result.W @ result.H, EPS)
            observed = V[rows, cols]
            predicted = reconstruction[rows, cols]
            kl_errors = observed * np.log((observed + EPS) / predicted) - observed + predicted
            scores[candidate].append(float(np.mean(kl_errors)))

    ranked = [
        {"k": candidate, "score": float(np.mean(values))}
        for candidate, values in scores.items()
    ]
    ranked.sort(key=lambda item: (item["score"], item["k"]))
    return int(ranked[0]["k"]), ranked, K_SELECTION_METHOD_MASKED


def auto_select_k(V: np.ndarray, max_k: int = K_SELECTION_MAX_K, seed: int = 42) -> int:
    """Backward-compatible k selector using masked KL imputation."""
    selected, _, _ = select_k_by_imputation(V, seed=seed, max_k=max_k)
    return selected
