import numpy as np
import pytest

from pathlib import Path

from dyana.core.timebase import CANONICAL_HOP_S, TimeBase
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


def test_to_canonical_requires_agg_on_downsample() -> None:
    tb = TimeBase(hop_s=0.005)
    values = np.ones(4, dtype=np.float32)

    track = EvidenceTrack(name="prob", timebase=tb, values=values, semantics="probability")
    with pytest.raises(ValueError):
        _ = track.to_canonical()

    ds = track.to_canonical(downsample_agg="mean")
    assert ds.timebase.hop_s == CANONICAL_HOP_S
    assert ds.values.shape[0] == 2


def test_bundle_merge_and_missing_tracks() -> None:
    tb = TimeBase.canonical()
    t1 = EvidenceTrack(
        name="a",
        timebase=tb,
        values=np.ones(2, dtype=np.float32),
        semantics="score",
    )
    t2 = EvidenceTrack(
        name="b",
        timebase=tb,
        values=np.zeros(2, dtype=np.float32),
        semantics="score",
    )
    b1 = EvidenceBundle(timebase=tb)
    b1.add(t1)
    b2 = EvidenceBundle(timebase=tb)
    b2.add(t2)

    merged = b1.merge(b2)
    assert merged.get("a") is t1
    assert merged.get("b") is t2
    assert merged.get("missing") is None


def test_serialization_roundtrip(tmp_path: Path) -> None:
    tb = TimeBase.canonical()
    track = EvidenceTrack(
        name="vad",
        timebase=tb,
        values=np.array([0.1, 0.2], dtype=np.float32),
        semantics="probability",
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata={"source": "unit"},
    )
    bundle = EvidenceBundle(timebase=tb)
    bundle.add(track)

    out_dir = tmp_path / "bundle"
    bundle.to_directory(out_dir)

    loaded = EvidenceBundle.from_directory(out_dir)
    loaded_track = loaded.get("vad")
    assert loaded_track is not None
    assert np.allclose(loaded_track.values, track.values)
    assert np.allclose(loaded_track.confidence, track.confidence)
    assert loaded_track.metadata == track.metadata
    assert loaded_track.timebase.hop_s == track.timebase.hop_s


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
