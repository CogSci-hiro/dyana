import numpy as np
import pytest

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle


def test_add_and_get_order_independent() -> None:
    tb = TimeBase.canonical()
    t1 = EvidenceTrack(name="a", timebase=tb, values=np.ones(2, dtype=np.float32), semantics="score")
    t2 = EvidenceTrack(name="b", timebase=tb, values=np.zeros(2, dtype=np.float32), semantics="score")

    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("b", t2)
    bundle.add_track("a", t1)

    assert set(bundle.keys()) == {"a", "b"}
    assert bundle.get("a") is t1
    assert bundle.get("b") is t2
    assert bundle.get("missing") is None


def test_rejects_non_canonical_when_required() -> None:
    tb_canon = TimeBase.canonical()
    tb_other = TimeBase.from_hop_seconds(0.02)
    track_other = EvidenceTrack(name="x", timebase=tb_other, values=np.ones(1, dtype=np.float32), semantics="score")

    bundle = EvidenceBundle(timebase=tb_canon, require_canonical=True)
    with pytest.raises(ValueError):
        bundle.add_track("x", track_other)

def test_mixed_hops_rejected() -> None:
    tb_canon = TimeBase.canonical()
    bundle = EvidenceBundle(timebase=tb_canon, require_canonical=False)
    bundle.add_track("a", EvidenceTrack(name="a", timebase=tb_canon, values=np.ones(2, dtype=np.float32), semantics="score"))

    tb_other = TimeBase.from_hop_seconds(0.02, n_frames=2)
    track_other = EvidenceTrack(name="b", timebase=tb_other, values=np.ones(2, dtype=np.float32), semantics="score")
    with pytest.raises(ValueError):
        bundle.add_track("b", track_other)

def test_resample_all_to_requires_agg_for_downsample() -> None:
    src_tb = TimeBase.from_hop_seconds(0.01)
    tgt_tb = TimeBase.from_hop_seconds(0.02)
    track = EvidenceTrack(name="a", timebase=src_tb, values=np.array([1.0, 3.0], dtype=np.float32), semantics="score")
    bundle = EvidenceBundle(timebase=src_tb, require_canonical=False, tracks={"a": track})

    with pytest.raises(ValueError):
        _ = bundle.resample_all_to(tgt_tb)

    out = bundle.resample_all_to(tgt_tb, default_downsample_agg="mean")
    assert out.get("a") is not None
    assert out.get("a").values.tolist() == [2.0]


def test_merge_behavior() -> None:
    tb = TimeBase.canonical()
    t1 = EvidenceTrack(name="a", timebase=tb, values=np.array([1.0], dtype=np.float32), semantics="score")
    t2_old = EvidenceTrack(name="b", timebase=tb, values=np.array([2.0], dtype=np.float32), semantics="score")
    t2_new = EvidenceTrack(name="b", timebase=tb, values=np.array([3.0], dtype=np.float32), semantics="score")

    b1 = EvidenceBundle(timebase=tb)
    b1.add_track("a", t1)
    b1.add_track("b", t2_old)

    b2 = EvidenceBundle(timebase=tb)
    b2.add_track("b", t2_new)

    merged = b1.merge(b2)
    assert merged.get("a") is t1
    assert merged.get("b") is t2_new


def test_serialize_roundtrip(tmp_path) -> None:
    tb = TimeBase.canonical()
    t1 = EvidenceTrack(
        name="vad",
        timebase=tb,
        values=np.array([0.1, 0.2], dtype=np.float32),
        semantics="probability",
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata={"source": "unit"},
    )
    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("vad", t1)

    out_dir = tmp_path / "bundle"
    bundle.to_directory(out_dir)

    loaded = EvidenceBundle.from_directory(out_dir)
    loaded_track = loaded.get("vad")
    assert loaded_track is not None
    assert np.allclose(loaded_track.values, t1.values)
    assert np.allclose(loaded_track.confidence, t1.confidence)
    assert loaded_track.metadata == t1.metadata
    assert loaded_track.semantics == t1.semantics
