import numpy as np

from dyana.core.timebase import TimeBase
from dyana.decode import decoder, fusion, state_space
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.leakage import LEAKAGE_TRACK_NAME


def _bundle_with_tracks(
    n_frames: int,
    vad_values: np.ndarray | None = None,
    leak_values: np.ndarray | None = None,
    diar_a_values: np.ndarray | None = None,
    diar_b_values: np.ndarray | None = None,
) -> EvidenceBundle:
    tb = TimeBase.canonical(n_frames=n_frames)
    bundle = EvidenceBundle(timebase=tb)
    if vad_values is not None:
        bundle.add_track(
            "vad",
            EvidenceTrack(name="vad", timebase=tb, values=vad_values.astype(np.float32), semantics="probability"),
        )
    if leak_values is not None:
        bundle.add_track(
            LEAKAGE_TRACK_NAME,
            EvidenceTrack(
                name=LEAKAGE_TRACK_NAME,
                timebase=tb,
                values=leak_values.astype(np.float32),
                semantics="probability",
            ),
        )
    if diar_a_values is not None:
        bundle.add_track(
            "diar_a",
            EvidenceTrack(name="diar_a", timebase=tb, values=diar_a_values.astype(np.float32), semantics="probability"),
        )
    if diar_b_values is not None:
        bundle.add_track(
            "diar_b",
            EvidenceTrack(name="diar_b", timebase=tb, values=diar_b_values.astype(np.float32), semantics="probability"),
        )
    return bundle


def test_leak_scoring_increases_leak_state_when_evidence_high() -> None:
    n_frames = 20
    vad = np.full(n_frames, 0.6, dtype=np.float32)
    leak = np.full(n_frames, 0.05, dtype=np.float32)
    leak[8:12] = 0.95
    bundle = _bundle_with_tracks(n_frames=n_frames, vad_values=vad, leak_values=leak)
    scores = fusion.fuse_bundle_to_scores(bundle)
    idx_leak = state_space.state_index("LEAK")
    high_region = float(np.mean(scores[8:12, idx_leak]))
    low_region = float(np.mean(np.concatenate([scores[:8, idx_leak], scores[12:, idx_leak]])))
    assert high_region > low_region


def test_leak_cannot_initiate_ipu_even_if_evidence_high() -> None:
    n_frames = 30
    vad = np.full(n_frames, 0.8, dtype=np.float32)
    leak = np.zeros(n_frames, dtype=np.float32)
    leak[:10] = 0.99
    diar_a = np.full(n_frames, 0.55, dtype=np.float32)
    bundle = _bundle_with_tracks(n_frames=n_frames, vad_values=vad, leak_values=leak, diar_a_values=diar_a)

    path = decoder.decode_with_constraints(fusion.fuse_bundle_to_scores(bundle))
    # Start and first non-SIL cannot be LEAK.
    first_non_sil = next((s for s in path if s != "SIL"), "SIL")
    assert path[0] != "LEAK"
    assert first_non_sil != "LEAK"
    for i in range(1, len(path)):
        assert not (path[i - 1] == "SIL" and path[i] == "LEAK")


def test_leak_to_ab_initiation_penalized() -> None:
    n_frames = 40
    vad = np.full(n_frames, 0.75, dtype=np.float32)
    leak = np.zeros(n_frames, dtype=np.float32)
    leak[10:22] = 0.98
    diar_a = np.full(n_frames, 0.5, dtype=np.float32)
    diar_a[22:] = 0.7
    bundle = _bundle_with_tracks(n_frames=n_frames, vad_values=vad, leak_values=leak, diar_a_values=diar_a)

    path = decoder.decode_with_constraints(fusion.fuse_bundle_to_scores(bundle))
    for i in range(1, len(path)):
        assert not (path[i - 1] == "LEAK" and path[i] in ("A", "B"))


def test_missing_leakage_track_does_not_crash() -> None:
    n_frames = 20
    vad = np.full(n_frames, 0.9, dtype=np.float32)
    diar_a = np.full(n_frames, 0.6, dtype=np.float32)
    bundle = _bundle_with_tracks(n_frames=n_frames, vad_values=vad, diar_a_values=diar_a)
    scores = fusion.fuse_bundle_to_scores(bundle)
    path = decoder.decode_with_constraints(scores)
    assert len(path) == n_frames


def test_diagnostic_counter_counts_starts_after_leak() -> None:
    states = ["SIL", "LEAK", "LEAK", "A", "A", "SIL", "LEAK", "OVL", "SIL"]
    diagnostics = decoder.decode_diagnostics(states)
    assert diagnostics["ipu_start_after_leak_count"] == 2
