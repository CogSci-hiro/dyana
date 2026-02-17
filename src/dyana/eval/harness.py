from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from dyana.core.timebase import CANONICAL_HOP_SECONDS
from dyana.decode.params import DecodeTuningParams
from dyana.decode.ipu import Segment
from dyana.errors import ErrorHandlingConfig, ErrorReporter, Pipeline, configure_logging
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

METRIC_NAMES: tuple[str, ...] = (
    "boundary_f1_20ms",
    "boundary_f1_50ms",
    "micro_ipus_per_min",
    "switches_per_min",
)


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
    reporter: ErrorReporter | None = None,
) -> Dict[str, Any]:
    run_reporter = reporter
    if run_reporter is None:
        cfg = ErrorHandlingConfig(mode="run", log_dir=out_dir / "logs", run_id=f"eval_{item.get('id', 'item')}")
        logger, event_logger = configure_logging(cfg=cfg)
        run_reporter = ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)

    state: Dict[str, Any] = {}
    item_id = str(item.get("id", "item"))
    step_prefix = f"{item_id}"

    def _load_data() -> Dict[str, Any]:
        resolved = dict(item)
        if resolved.get("tier") == "synthetic" and resolved.get("audio_path") is None:
            resolved = materialize_synthetic_case(resolved, out_dir / "_synthetic")
        audio_path = Path(str(resolved["audio_path"]))
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for item '{item_id}': {audio_path}")
        ref_path = resolved.get("ref_path")
        if ref_path is not None and not Path(str(ref_path)).exists():
            raise FileNotFoundError(f"Reference file not found for item '{item_id}': {ref_path}")
        state["item"] = resolved
        state["audio_path"] = audio_path
        return resolved

    def _run_inference() -> Dict[str, Any]:
        audio_path: Path = state["audio_path"]
        summary = run_pipeline(audio_path, out_dir=out_dir, cache_dir=cache_dir, tuning_params=tuning_params)
        state["summary"] = summary
        return summary

    def _export_ipus() -> Dict[str, Any]:
        audio_path: Path = state["audio_path"]
        hyp_states = list(np.load(out_dir / "decode" / f"{audio_path.stem}_states.npy", allow_pickle=True))
        decode_ipus = json.loads((out_dir / "decode" / f"{audio_path.stem}_ipus.json").read_text())
        state["hyp_states"] = hyp_states
        state["decode_ipus"] = decode_ipus
        return {"n_states": len(hyp_states), "n_ipus": len(decode_ipus)}

    def _compute_metrics() -> Dict[str, Any]:
        resolved_item: Dict[str, Any] = state["item"]
        audio_path: Path = state["audio_path"]
        hyp_states: List[str] = state["hyp_states"]

        hop_s = CANONICAL_HOP_SECONDS
        n_frames = len(hyp_states)
        ref_path = resolved_item.get("ref_path")
        if ref_path is None:
            ref_states = hyp_states
        else:
            ref_states = load_reference_states(Path(str(ref_path)), n_frames, hop_s)

        if len(ref_states) != len(hyp_states):
            min_len = min(len(ref_states), len(hyp_states))
            ref_states = ref_states[:min_len]
            hyp_states = hyp_states[:min_len]
            n_frames = min_len

        ref_bound = state_boundaries(ref_states, hop_s)
        hyp_bound = state_boundaries(hyp_states, hop_s)
        b20 = boundary_f1(ref_bound, hyp_bound, tol_s=0.02)
        b50 = boundary_f1(ref_bound, hyp_bound, tol_s=0.05)

        def mask_for(label_set: set[str]) -> tuple[np.ndarray, np.ndarray]:
            return np.array([s in label_set for s in ref_states]), np.array([s in label_set for s in hyp_states])

        ref_a, hyp_a = mask_for({"A", "OVL"})
        ref_b, hyp_b = mask_for({"B", "OVL"})
        ref_any, hyp_any = mask_for({"A", "B", "OVL", "LEAK"})
        iou_a = framewise_iou(ref_a, hyp_a)
        iou_b = framewise_iou(ref_b, hyp_b)
        iou_any = framewise_iou(ref_any, hyp_any)

        total_duration_s = n_frames * hop_s
        ipu_objs = [Segment(**seg) for seg in state["decode_ipus"]]
        micro = micro_ipus_per_min(ipu_objs, total_duration_s)
        switches = speaker_switches_per_min(hyp_states, hop_s)
        rapid = rapid_alternations(hyp_states, hop_s)

        row = {
            "id": resolved_item["id"],
            "tier": resolved_item.get("tier", "unknown"),
            "status": "ok",
            "audio_path": str(audio_path),
            "boundary_f1_20ms": b20["f1"],
            "boundary_f1_50ms": b50["f1"],
            "boundary_f1_20": b20["f1"],
            "boundary_f1_50": b50["f1"],
            "iou_a": iou_a,
            "iou_b": iou_b,
            "iou_any": iou_any,
            "micro_ipus_per_min": micro,
            "switches_per_min": switches,
            "speaker_switches_per_min": switches,
            "rapid_alternations": rapid,
            "rapid_alternations_per_min": rapid / max((n_frames * hop_s) / 60.0, 1e-9),
        }
        state["row"] = row
        return row

    def _write_scorecard() -> Dict[str, Any]:
        row: Dict[str, Any] = state["row"]
        (out_dir / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True))
        return row

    pipeline = Pipeline(run_reporter)
    pipeline.add(f"{step_prefix}.load_data", _load_data, context={"tier": item.get("tier", "unknown"), "id": item_id})
    pipeline.add(f"{step_prefix}.run_inference", _run_inference, deps=[f"{step_prefix}.load_data"])
    pipeline.add(f"{step_prefix}.export_ipus", _export_ipus, deps=[f"{step_prefix}.run_inference"])
    pipeline.add(f"{step_prefix}.compute_metrics", _compute_metrics, deps=[f"{step_prefix}.export_ipus"])
    pipeline.add(f"{step_prefix}.write_scorecard", _write_scorecard, deps=[f"{step_prefix}.compute_metrics"])
    results = pipeline.run()

    row = results.get(f"{step_prefix}.write_scorecard")
    if isinstance(row, dict):
        return row

    failed_row = {
        "id": item_id,
        "tier": item.get("tier", "unknown"),
        "status": "failed",
        "audio_path": str(item.get("audio_path")),
        "boundary_f1_20ms": 0.0,
        "boundary_f1_50ms": 0.0,
        "micro_ipus_per_min": 0.0,
        "switches_per_min": 0.0,
        "rapid_alternations_per_min": 0.0,
    }
    return failed_row


def evaluate_manifest(
    manifest_path: Path,
    out_dir: Path,
    cache_dir: Path | None = None,
    tuning_params: DecodeTuningParams | None = None,
) -> List[Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ErrorHandlingConfig(mode="run", log_dir=out_dir / "logs", run_id=f"eval_{manifest_path.stem}")
    logger, event_logger = configure_logging(cfg=cfg)
    reporter = ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)
    results = []
    sorted_manifest = sorted(manifest, key=lambda entry: (str(entry.get("tier", "")), str(entry.get("id", ""))))
    for item in sorted_manifest:
        res = evaluate_item(
            item,
            out_dir / item["id"],
            cache_dir=cache_dir,
            tuning_params=tuning_params,
            reporter=reporter,
        )
        results.append(res)
    (out_dir / "pipeline_summary.txt").write_text(reporter.render_summary())
    return results
