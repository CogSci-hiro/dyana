import numpy as np
import pytest

from dyana.decode import fusion, decoder
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence import synthetic
from dyana.core.timebase import TimeBase


def _center_index(start: int, end: int) -> int:
    return start + (end - start) // 2


def test_synthetic_pipeline_scripted_blocks_decodes_expected_sequence() -> None:
    # Durations in frames (10 ms hop)
    hops = [100, 150, 50, 100, 80, 70]  # SIL, A, SIL, B, OVL, SIL
    regions = synthetic.cumulative_regions(hops)
    total_frames = sum(hops)
    tb = synthetic.make_timebase(total_frames)

    speech_regions = [regions[1], regions[3], regions[4]]
    vad = synthetic.make_vad_track(tb, speech_regions)

    diar_a = synthetic.make_diar_track(tb, "diar_a", [regions[1], regions[4]])
    diar_b = synthetic.make_diar_track(tb, "diar_b", [regions[3], regions[4]])
    leak = synthetic.make_leak_track(tb, [])

    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("vad", vad)
    bundle.add_track("diar_a", diar_a)
    bundle.add_track("diar_b", diar_b)
    bundle.add_track("leak", leak)

    scores = fusion.fuse_bundle_to_scores(bundle)
    path = decoder.decode_with_constraints(scores)

    expected = ["SIL", "A", "SIL", "B", "OVL", "SIL"]
    for (start, end), label in zip(regions, expected):
        idx = _center_index(start, end)
        assert path[idx] == label

    # No jitter: runs for A/B are at least 2 frames
    run_label = path[0]
    run_len = 1
    for p in path[1:]:
        if p == run_label:
            run_len += 1
            continue
        if run_label in ("A", "B"):
            assert run_len >= 2
        run_label = p
        run_len = 1


def test_missing_tracks_is_ok() -> None:
    hops = [80, 120, 80]  # SIL, speech, SIL
    regions = synthetic.cumulative_regions(hops)
    tb = synthetic.make_timebase(sum(hops))
    vad = synthetic.make_vad_track(tb, [regions[1]])

    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("vad", vad)

    scores = fusion.fuse_bundle_to_scores(bundle)
    path = decoder.decode_with_constraints(scores)

    assert len(path) == sum(hops)
    assert path[_center_index(*regions[0])] == "SIL"
    # With only VAD, tie-break picks A for speech
    assert path[_center_index(*regions[1])] == "A"


def test_timebase_mismatch_fails_loudly() -> None:
    tb = TimeBase.canonical(n_frames=30)
    vad = synthetic.make_vad_track(tb, [(0, 30)])

    tb_short = TimeBase.canonical(n_frames=20)
    diar_a = synthetic.make_diar_track(tb_short, "diar_a", [(0, 20)])

    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("vad", vad)
    with pytest.raises(ValueError):
        bundle.add_track("diar_a", diar_a)


def test_leak_cannot_initiate_even_if_evidence_favors() -> None:
    tb = TimeBase.canonical(n_frames=40)
    speech_region = [(0, 20)]
    vad = synthetic.make_vad_track(tb, speech_region, p_speech=0.9, p_sil=0.1)
    diar_a = synthetic.make_diar_track(tb, "diar_a", speech_region, p_on=0.6, p_off=0.4)
    leak = synthetic.make_leak_track(tb, [(0, 5)], p_on=0.99, p_off=0.01)

    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("vad", vad)
    bundle.add_track("diar_a", diar_a)
    bundle.add_track("leak", leak)

    path = decoder.decode_with_constraints(fusion.fuse_bundle_to_scores(bundle))
    assert "LEAK" not in path[:5]
