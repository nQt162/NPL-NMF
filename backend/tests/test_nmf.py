import numpy as np
import pytest

from app.core.nmf import (
    SPARSE_KL_ALPHA_H,
    SPARSE_KL_ALPHA_W,
    SPARSE_KL_L1_RATIO,
    SPARSE_KL_LOSS_NAME,
    fit_nmf,
)


def test_nmf_shapes_reproducibility_and_loss() -> None:
    V = np.array([[1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.0, 0.3, 1.0]])
    first = fit_nmf(V, k=2, seed=42, max_iter=20, trace=True)
    second = fit_nmf(V, k=2, seed=42, max_iter=20, trace=True)

    assert first.W.shape == (3, 2)
    assert first.H.shape == (2, 3)
    assert np.all(first.W >= 0) and np.all(first.H >= 0)
    np.testing.assert_allclose(first.W, second.W)
    np.testing.assert_allclose(first.H, second.H)
    assert first.snapshots[0].iteration == 0
    assert len(first.snapshots) == len(first.losses) + 1
    for snapshot in first.snapshots:
        np.testing.assert_allclose(snapshot.W @ snapshot.H, snapshot.WH)
        assert snapshot.loss == pytest.approx(0.5 * np.sum((V - snapshot.WH) ** 2))
        assert snapshot.regularization_loss == pytest.approx(0.0)
        assert snapshot.data_loss == pytest.approx(snapshot.loss)
    assert all(after <= before + 1e-7 for before, after in zip(
        [first.snapshots[0].loss, *first.losses[:-1]], first.losses
    ))
    assert first.losses[-1] < first.snapshots[0].loss


def test_nmf_caps_rank_and_does_not_trace_batch() -> None:
    V = np.ones((2, 3))
    result = fit_nmf(V, k=10, max_iter=8)
    assert result.W.shape == (2, 2)
    assert result.snapshots == []


def test_nmf_can_trace_sparse_kl_loss() -> None:
    V = np.array([[1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.0, 0.3, 1.0]])
    result = fit_nmf(
        V,
        k=2,
        max_iter=8,
        trace=True,
        loss=SPARSE_KL_LOSS_NAME,
        alpha_w=SPARSE_KL_ALPHA_W,
        alpha_h=SPARSE_KL_ALPHA_H,
        l1_ratio=SPARSE_KL_L1_RATIO,
    )

    assert len(result.snapshots) == len(result.losses) + 1
    assert result.losses[-1] < result.snapshots[0].loss
    assert result.snapshots[-1].regularization_loss > 0
    assert result.snapshots[-1].loss == pytest.approx(
        result.snapshots[-1].data_loss + result.snapshots[-1].regularization_loss
    )
    assert result.snapshots[-1].loss != pytest.approx(
        0.5 * np.sum((V - result.snapshots[-1].WH) ** 2)
    )


@pytest.mark.parametrize("matrix", [
    np.array([[-1.0, 2.0]]),
    np.array([[float("nan")]]),
    np.empty((0, 3)),
    np.array([1.0, 2.0]),
])
def test_nmf_rejects_invalid_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_nmf(matrix, 2)
