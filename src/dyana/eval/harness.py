from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from dyana.core.timebase import CANONICAL_HOP_SECONDS
from dyana.decode.params import DecodeTuningParams
from dyana.decode.ipu import Segment
from dyana.eval.synthetic_cases import materialize_synthetic_case
from dyana.eval.metrics import (
    boundary_f1,
    framewise_iou,
    micro_ipus_per_min,
    rapid_alternations,
    speaker_switches_per_min,
)
from dyana.pipeline.run_pipeline import run_pipeline
from dyana.io.praat_textgrid import parse_textgrid


def segments_to_states(segments: Dict[str, List[Segment]], n_frames: int, hop_s: float) -> List[str]:
    states = ["SIL"] * n_frames
    for seg in segments.get("Leak", []):
        start = int(seg.start_time / hop_s)
        end = int(seg.end_time / hop_s)
        for i in range(start, min(end, n_frames)):
            states[i] = "LEAK"
    for seg in segments.get("Overlap", []):
        start = int(seg.start_time / hop_s)
        end = int(seg.end_time / hop_s)
        for i in range(start, min(end, n_frames)):
            states[i] = "OVL"
    for seg in segments.get("SpeakerA", []):
        start = int(seg.start_time / hop_s)
        end = int(seg.end_time / hop_s)
        for i in range(start, min(end, n_frames)):
            if states[i] == "SIL":
                states[i] = "A"
    for seg in segments.get("SpeakerB", []):
        start = int(seg.start_time / hop_s)
        end = int(seg.end_time / hop_s)
        for i in range(start, min(end, n_frames)):
            if states[i] == "SIL":
                states[i] = "B"
    return states


def load_reference_states(ref_path: Path, n_frames: int, hop_s: float) -> List[str]:
    suffix = ref_path.suffix.lower()
    if suffix == ".npy":
        return list(np.load(ref_path, allow_pickle=True))
    if suffix == ".json":
        return json.loads(ref_path.read_text())
    if suffix == ".textgrid":
        segments = parse_textgrid(ref_path)
        return segments_to_states(segments, n_frames, hop_s)
    raise ValueError(f"Unsupported reference format: {ref_path}")


def state_boundaries(states: Sequence[str], hop_s: float) -> np.ndarray:
    boundaries = []
    prev = states[0] if states else None
    for i, s in enumerate(states[1:], start=1):
        if s != prev:
            boundaries.append(i * hop_s)
            prev = s
    return np.array(boundaries, dtype=float)


def evaluate_item(
    item: Dict[str, Any],
    out_dir: Path,
    cache_dir: Path | None = None,
    tuning_params: DecodeTuningParams | None = None,
) -> Dict[str, Any]:
    if item.get("tier") == "synthetic" and item.get("audio_path") is None:
        item = materialize_synthetic_case(item, out_dir / "_synthetic")
    audio_path = Path(item["audio_path"])
    run_pipeline(audio_path, out_dir=out_dir, cache_dir=cache_dir, tuning_params=tuning_params)

    hyp_states = list(np.load(out_dir / "decode" / f"{audio_path.stem}_states.npy", allow_pickle=True))
    hop_s = CANONICAL_HOP_SECONDS
    n_frames = len(hyp_states)

    ref_path = item.get("ref_path")
    if ref_path is None:
        ref_states = hyp_states
    else:
        ref_states = load_reference_states(Path(ref_path), n_frames, hop_s)

    if len(ref_states) != len(hyp_states):
        min_len = min(len(ref_states), len(hyp_states))
        ref_states = ref_states[:min_len]
        hyp_states = hyp_states[:min_len]
        n_frames = min_len

    # Metrics
    ref_bound = state_boundaries(ref_states, hop_s)
    hyp_bound = state_boundaries(hyp_states, hop_s)
    b20 = boundary_f1(ref_bound, hyp_bound, tol_s=0.02)
    b50 = boundary_f1(ref_bound, hyp_bound, tol_s=0.05)

    def mask_for(label_set):
        return np.array([s in label_set for s in ref_states]), np.array([s in label_set for s in hyp_states])

    ref_a, hyp_a = mask_for({"A", "OVL"})
    ref_b, hyp_b = mask_for({"B", "OVL"})
    ref_any, hyp_any = mask_for({"A", "B", "OVL", "LEAK"})
    iou_a = framewise_iou(ref_a, hyp_a)
    iou_b = framewise_iou(ref_b, hyp_b)
    iou_any = framewise_iou(ref_any, hyp_any)

    # structural
    total_duration_s = n_frames * hop_s
    # use decoded IPUs
    decode_ipus = json.loads((out_dir / "decode" / f"{audio_path.stem}_ipus.json").read_text())
    ipu_objs = [Segment(**seg) for seg in decode_ipus]
    micro = micro_ipus_per_min(ipu_objs, total_duration_s)
    switches = speaker_switches_per_min(hyp_states, hop_s)
    rapid = rapid_alternations(hyp_states, hop_s)

    return {
        "id": item["id"],
        "tier": item.get("tier", "unknown"),
        "boundary_f1_20": b20["f1"],
        "boundary_f1_50": b50["f1"],
        "boundary_f1_20ms": b20["f1"],
        "boundary_f1_50ms": b50["f1"],
        "iou_a": iou_a,
        "iou_b": iou_b,
        "iou_any": iou_any,
        "micro_ipus_per_min": micro,
        "speaker_switches_per_min": switches,
        "switches_per_min": switches,
        "rapid_alternations": rapid,
        "rapid_alternations_per_min": rapid / max((n_frames * hop_s) / 60.0, 1e-9),
    }


def evaluate_manifest(
    manifest_path: Path,
    out_dir: Path,
    cache_dir: Path | None = None,
    tuning_params: DecodeTuningParams | None = None,
) -> List[Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    sorted_manifest = sorted(manifest, key=lambda entry: (str(entry.get("tier", "")), str(entry.get("id", ""))))
    for item in sorted_manifest:
        res = evaluate_item(item, out_dir / item["id"], cache_dir=cache_dir, tuning_params=tuning_params)
        results.append(res)
    return results
