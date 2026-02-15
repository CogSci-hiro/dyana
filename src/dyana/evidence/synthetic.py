# src/dyana/evidence/synthetic.py

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack


def frames_for_seconds(seconds: float, hop_s: float = 0.01) -> int:
    return int(round(seconds / hop_s))


def make_timebase(n_frames: int) -> TimeBase:
    return TimeBase.canonical(n_frames=n_frames)


def _piecewise_constant(n_frames: int, regions: Iterable[Tuple[int, int]], on_value: float, off_value: float) -> np.ndarray:
    arr = np.full(n_frames, off_value, dtype=np.float32)
    for start, end in regions:
        arr[start:end] = on_value
    return arr


def make_vad_track(tb: TimeBase, speech_regions: Sequence[Tuple[int, int]], p_speech: float = 0.95, p_sil: float = 0.05) -> EvidenceTrack:
    values = _piecewise_constant(tb.n_frames or 0, speech_regions, on_value=p_speech, off_value=p_sil)
    return EvidenceTrack(name="vad", timebase=tb, values=values, semantics="probability")


def make_diar_track(tb: TimeBase, name: str, regions: Sequence[Tuple[int, int]], p_on: float = 0.9, p_off: float = 0.1) -> EvidenceTrack:
    values = _piecewise_constant(tb.n_frames or 0, regions, on_value=p_on, off_value=p_off)
    return EvidenceTrack(name=name, timebase=tb, values=values, semantics="probability")


def make_leak_track(tb: TimeBase, regions: Sequence[Tuple[int, int]], p_on: float = 0.7, p_off: float = 0.05) -> EvidenceTrack:
    values = _piecewise_constant(tb.n_frames or 0, regions, on_value=p_on, off_value=p_off)
    return EvidenceTrack(name="leak", timebase=tb, values=values, semantics="probability")


def cumulative_regions(lengths: Sequence[int]) -> List[Tuple[int, int]]:
    regions = []
    start = 0
    for L in lengths:
        end = start + L
        regions.append((start, end))
        start = end
    return regions
