import numpy as np
import pytest

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceBundle, EvidenceTrack

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


def test_confidence_k_mismatch_raises() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.random.rand(8, 3).astype(np.float32)
    confidence = np.random.rand(8, 2).astype(np.float32)

    with pytest.raises(ValueError):
        _ = EvidenceTrack(
            name="vad",
            timebase=tb,
            values=values,
            semantics="score",
            confidence=confidence,
        )


def test_confidence_non_float_raises() -> None:
    tb = TimeBase(hop_s=0.01)
    values = np.random.rand(5).astype(np.float32)
    confidence = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    with pytest.raises(TypeError):
        _ = EvidenceTrack(
            name="vad",
            timebase=tb,
            values=values,
            semantics="score",
            confidence=confidence,
        )


def test_bundle_add_get_iter_and_hop_validation() -> None:
    tb = TimeBase.canonical()
    track = EvidenceTrack(
        name="vad",
        timebase=tb,
        values=np.random.rand(4).astype(np.float32),
        semantics="probability",
    )
    bundle = EvidenceBundle(timebase=tb, require_canonical=True)
    bundle.add(track)

    assert bundle.get("vad") is track
    assert list(bundle) == [track]

    non_canonical_tb = TimeBase(hop_s=0.02)
    with pytest.raises(ValueError):
        _ = EvidenceBundle(timebase=non_canonical_tb, require_canonical=True)

    wrong_hop_track = EvidenceTrack(
        name="bad",
        timebase=non_canonical_tb,
        values=np.random.rand(4).astype(np.float32),
        semantics="score",
    )
    with pytest.raises(ValueError):
        bundle.add(wrong_hop_track)
