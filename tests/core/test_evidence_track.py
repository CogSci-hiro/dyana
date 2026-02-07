import numpy as np
import pytest

from dyana.core.timebase import TimeBase
from dyana.core.evidence import EvidenceTrack

def test_accepts_vector_values() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.random.rand(100).astype(np.float32)

    tr = EvidenceTrack(
        name="vad",
        timebase=tb,
        values=values,
        semantics="probability",
    )
    assert tr.T == 100
    assert tr.K == 1

def test_accepts_matrix_values() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.random.randn(100, 5).astype(np.float32)

    tr = EvidenceTrack(
        name="state_logits",
        timebase=tb,
        values=values,
        semantics="logit",
    )
    assert tr.T == 100
    assert tr.K == 5

def test_rejects_wrong_ndim() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        _ = EvidenceTrack(name="bad", timebase=tb, values=values, semantics="score")

def test_rejects_non_float_dtype() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.zeros((10,), dtype=np.int32)

    with pytest.raises(TypeError):
        _ = EvidenceTrack(name="bad", timebase=tb, values=values, semantics="score")

def test_rejects_nan_inf() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.array([0.0, np.nan, 1.0], dtype=np.float32)

    with pytest.raises(ValueError):
        _ = EvidenceTrack(name="bad", timebase=tb, values=values, semantics="score")

def test_confidence_shape_must_match_T() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.random.rand(100).astype(np.float32)
    conf = np.random.rand(99).astype(np.float32)

    with pytest.raises(ValueError):
        _ = EvidenceTrack(
            name="vad",
            timebase=tb,
            values=values,
            semantics="probability",
            confidence=conf,
        )

def test_probability_bounds_enforced() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.array([0.0, 1.2, 0.5], dtype=np.float32)

    with pytest.raises(ValueError):
        _ = EvidenceTrack(name="vad", timebase=tb, values=values, semantics="probability")
