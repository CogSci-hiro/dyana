import numpy as np
import pytest

from dyana.core.resample import downsample, resample, to_canonical_grid, upsample_hold_last


def test_upsample_repeat_vector_and_matrix() -> None:
    vec = np.array([1.0, 2.0, 3.0])
    out_vec = upsample_hold_last(vec, 0.02, 0.01)
    assert out_vec.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    mat = np.array([[1.0, 10.0], [2.0, 20.0]])
    out_mat = upsample_hold_last(mat, 0.02, 0.01)
    assert out_mat.tolist() == [
        [1.0, 10.0],
        [1.0, 10.0],
        [2.0, 20.0],
        [2.0, 20.0],
    ]


def test_downsample_mean_and_max() -> None:
    vec = np.array([1.0, 3.0, 5.0, 7.0])
    out_mean = downsample(vec, 0.01, 0.02, agg="mean")
    assert np.allclose(out_mean, np.array([2.0, 6.0]))

    mat = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 0.0], [1.0, 5.0]])
    out_max = downsample(mat, 0.01, 0.02, agg="max")
    assert out_max.tolist() == [[2.0, 3.0], [4.0, 5.0]]


def test_downsample_requires_divisibility() -> None:
    vec = np.ones(5)
    with pytest.raises(ValueError):
        _ = downsample(vec, 0.01, 0.03, agg="mean")


def test_invalid_factor_raises() -> None:
    vec = np.ones(3)
    with pytest.raises(ValueError):
        _ = upsample_hold_last(vec, 0.015, 0.01)

    with pytest.raises(ValueError):
        _ = resample(vec, 0.015, 0.01)


def test_to_canonical_routes_correctly() -> None:
    coarse = np.array([0.1, 0.2, 0.3])
    fine = to_canonical_grid(coarse, 0.02, semantics="probability")
    assert fine.tolist() == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]

    fine_src = np.array([1.0, 2.0, 3.0, 4.0])
    coarse_ds = to_canonical_grid(fine_src, 0.005, semantics="score", downsample_agg="max")
    assert coarse_ds.tolist() == [2.0, 4.0]

    with pytest.raises(ValueError):
        _ = to_canonical_grid(fine_src, 0.005, semantics="score")


def test_resample_downsample_requires_agg_and_mean_vs_max() -> None:
    values = np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32)

    with pytest.raises(ValueError):
        _ = resample(values, 0.01, 0.02)

    mean_ds = resample(values, 0.01, 0.02, agg="mean")
    assert np.allclose(mean_ds, np.array([2.0, 3.0]))

    max_ds = resample(values, 0.01, 0.02, agg="max")
    assert np.allclose(max_ds, np.array([3.0, 4.0]))
