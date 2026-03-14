import numpy as np

from dyana.core.timebase import TimeBase
from dyana.decode import decoder, fusion
from dyana.decode.ipu import Segment, extract_ipus, merge_ipus_across_short_silence
from dyana.decode.params import DecodeTuningParams
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle


def _bundle_with_vad_and_diar(vad_values: np.ndarray, diar_values: np.ndarray | None = None) -> EvidenceBundle:
    tb = TimeBase.canonical(n_frames=int(vad_values.shape[0]))
    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track(
        "vad",
        EvidenceTrack(
            name="vad",
            timebase=tb,
            values=vad_values.astype(np.float32),
            semantics="probability",
        ),
    )
    if diar_values is not None:
        bundle.add_track(
            "diar_a",
            EvidenceTrack(
                name="diar_a",
                timebase=tb,
                values=diar_values.astype(np.float32),
                semantics="probability",
            ),
        )
    return bundle


def _count_runs(states: list[str], label: str, start: int, end: int) -> int:
    count = 0
    in_run = False
    for state in states[start:end]:
        if state == label:
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return count


def test_high_recall_mode_produces_fewer_silence_segments() -> None:
    vad_values = np.full(80, 0.05, dtype=np.float32)
    vad_values[10:70] = 0.70
    vad_values[30:33] = 0.20
    vad_values[50:53] = 0.20
    bundle = _bundle_with_vad_and_diar(vad_values)

    balanced_states = decoder.decode_with_constraints(
        fusion.fuse_bundle_to_scores(bundle, tuning_params=DecodeTuningParams())
    )
    high_recall_params = DecodeTuningParams(ipu_detection_mode="high_recall")
    high_recall_states = decoder.decode_with_constraints(
        fusion.fuse_bundle_to_scores(bundle, tuning_params=high_recall_params),
        tuning_params=high_recall_params,
    )

    assert _count_runs(high_recall_states, "SIL", 10, 70) < _count_runs(balanced_states, "SIL", 10, 70)


def test_ipu_count_decreases_after_gap_merging() -> None:
    segments = [
        Segment(start_time=0.0, end_time=0.5, label="A"),
        Segment(start_time=0.7, end_time=1.1, label="A"),
        Segment(start_time=1.7, end_time=2.0, label="A"),
    ]

    merged = merge_ipus_across_short_silence(segments, max_gap_s=0.4)

    assert len(merged) == 2
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 1.1


def test_quiet_speech_dips_remain_one_ipu_in_high_recall_mode() -> None:
    tb = TimeBase.canonical(n_frames=90)
    vad_values = np.full(90, 0.05, dtype=np.float32)
    vad_values[10:80] = 0.52
    vad_values[35:37] = 0.38
    vad_values[55:57] = 0.38
    diar_values = np.full(90, 0.10, dtype=np.float32)
    diar_values[10:80] = 0.92

    bundle = _bundle_with_vad_and_diar(vad_values, diar_values)
    high_recall_params = DecodeTuningParams(ipu_detection_mode="high_recall", silence_bias=-0.25)
    high_recall_states = decoder.decode_with_constraints(
        fusion.fuse_bundle_to_scores(bundle, tuning_params=high_recall_params),
        tuning_params=high_recall_params,
    )

    ipus = extract_ipus(high_recall_states, tb, "A", min_duration_s=0.2)
    merged = merge_ipus_across_short_silence(ipus, max_gap_s=high_recall_params.merge_silence_gap_ms / 1000.0)

    assert len(merged) == 1
    assert merged[0].start_time <= tb.frame_to_time(10)
    assert merged[0].end_time >= tb.frame_to_time(80)
