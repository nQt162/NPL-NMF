import numpy as np
import pytest

from app.core.nmf import fit_nmf


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
    assert all(after <= before + 1e-7 for before, after in zip(
        [first.snapshots[0].loss, *first.losses[:-1]], first.losses
    ))
    assert first.losses[-1] < first.snapshots[0].loss


def test_nmf_caps_rank_and_does_not_trace_batch() -> None:
    V = np.ones((2, 3))
    result = fit_nmf(V, k=10, max_iter=8)
    assert result.W.shape == (2, 2)
    assert result.snapshots == []


@pytest.mark.parametrize("matrix", [
    np.array([[-1.0, 2.0]]),
    np.array([[float("nan")]]),
    np.empty((0, 3)),
    np.array([1.0, 2.0]),
])
def test_nmf_rejects_invalid_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_nmf(matrix, 2)
