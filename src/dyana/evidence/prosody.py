"""Prosodic cue extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase
from dyana.core.cache import cache_get, cache_put, make_cache_key
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.energy import compute_energy_smooth_track, compute_energy_slope_track
from dyana.evidence.vad import compute_webrtc_vad_soft_track


def compute_voiced_soft_track(
    audio_path: Path,
    *,
    hop_s: float = CANONICAL_HOP_SECONDS,
    vad_mode: int = 2,
    subframe_ms: int = 5,
    cache_dir: Path | None = None,
) -> EvidenceTrack:
    key = make_cache_key(audio_path, "voiced_soft", {"hop_s": hop_s, "vad_mode": vad_mode, "sub_ms": subframe_ms})
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            values = npz["values"]
        tb = TimeBase.canonical(n_frames=len(values))
        return EvidenceTrack(name="voiced_soft", timebase=tb, values=values, semantics="probability")

    vad_soft = compute_webrtc_vad_soft_track(audio_path, hop_s=hop_s, vad_mode=vad_mode, subframe_ms=subframe_ms, cache_dir=cache_dir)
    values = np.asarray(vad_soft.values, dtype=np.float32)
    cache_put(cache_dir, key, {"values": values})
    return EvidenceTrack(name="voiced_soft", timebase=vad_soft.timebase, values=values, semantics="probability")


def compute_energy_slope_prosody_track(
    audio_path: Path,
    *,
    hop_s: float = CANONICAL_HOP_SECONDS,
    smooth_ms: float = 80.0,
    cache_dir: Path | None = None,
) -> EvidenceTrack:
    # wrapper to expose slope as prosody
    return compute_energy_slope_track(audio_path, hop_s=hop_s, smooth_ms=smooth_ms, cache_dir=cache_dir)
