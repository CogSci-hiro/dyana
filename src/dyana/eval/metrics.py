"""Evaluation metrics for DYANA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def boundary_f1(ref_boundaries_s: np.ndarray, hyp_boundaries_s: np.ndarray, *, tol_s: float) -> Dict[str, float]:
    ref = np.sort(np.asarray(ref_boundaries_s, dtype=float))
    hyp = np.sort(np.asarray(hyp_boundaries_s, dtype=float))
    ref_used = np.zeros(len(ref), dtype=bool)
    tp = 0
    for h in hyp:
        candidates = np.where(np.abs(ref - h) <= tol_s)[0]
        candidates = [i for i in candidates if not ref_used[i]]
        if not candidates:
            continue
        # choose closest
        best = min(candidates, key=lambda i: abs(ref[i] - h))
        ref_used[best] = True
        tp += 1
    fp = len(hyp) - tp
    fn = len(ref) - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def framewise_iou(ref_mask: np.ndarray, hyp_mask: np.ndarray) -> float:
    ref_b = np.asarray(ref_mask, dtype=bool)
    hyp_b = np.asarray(hyp_mask, dtype=bool)
    inter = np.logical_and(ref_b, hyp_b).sum()
    union = np.logical_or(ref_b, hyp_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def micro_ipus_per_min(ipus: Sequence, total_duration_s: float, *, max_duration_s: float = 0.2) -> float:
    count = sum(1 for seg in ipus if (seg.end_time - seg.start_time) < max_duration_s)
    return count / (total_duration_s / 60.0) if total_duration_s > 0 else 0.0


def speaker_switches_per_min(states: Sequence[str], hop_s: float) -> float:
    last = None
    switches = 0
    for s in states:
        if s not in ("A", "B"):
            continue
        if last is None:
            last = s
            continue
        if s != last:
            switches += 1
            last = s
    total_duration_s = len(states) * hop_s
    return switches / (total_duration_s / 60.0) if total_duration_s > 0 else 0.0


def rapid_alternations(states: Sequence[str], hop_s: float, window_s: float = 1.0) -> int:
    window_frames = int(window_s / hop_s) if hop_s > 0 else 0
    count = 0
    for i in range(len(states) - 2):
        trip = states[i : i + 3]
        if trip[0] in ("A", "B") and trip[1] in ("A", "B") and trip[2] in ("A", "B"):
            if trip[0] != trip[1] and trip[0] == trip[2] and (2 <= window_frames or True):
                count += 1
    return count
