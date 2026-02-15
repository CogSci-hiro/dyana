# src/dyana/evidence/energy.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from dyana.core.cache import cache_get, cache_put, make_cache_key
from dyana.core.timebase import TimeBase, CANONICAL_HOP_SECONDS
from dyana.evidence.base import EvidenceTrack
from dyana.io.audio import load_audio_mono

SMOOTH_MS_DEFAULT: float = 80.0


def _frame_audio(samples: np.ndarray, sr: int, hop_s: float) -> Tuple[np.ndarray, int]:
    hop = int(round(sr * hop_s))
    if hop <= 0:
        raise ValueError("hop_s too small for given sample rate.")
    n_frames = int(len(samples) // hop)
    samples = samples[: n_frames * hop]
    framed = samples.reshape(n_frames, hop)
    return framed, hop


def compute_energy_rms_track(audio_path: Path, *, hop_s: float = CANONICAL_HOP_SECONDS, cache_dir: Path | None = None) -> EvidenceTrack:
    key = make_cache_key(audio_path, "energy_rms", {"hop_s": hop_s})
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            values = npz["values"]
        tb = TimeBase.canonical(n_frames=len(values))
        return EvidenceTrack(name="energy_rms", timebase=tb, values=values, semantics="score")

    samples, sr = load_audio_mono(audio_path)
    frames, _ = _frame_audio(samples, sr, hop_s)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    tb = TimeBase.canonical(n_frames=len(rms))

    cache_put(cache_dir, key, {"values": rms})
    return EvidenceTrack(name="energy_rms", timebase=tb, values=rms, semantics="score")


def _smooth(values: np.ndarray, smooth_ms: float, hop_s: float) -> np.ndarray:
    win = max(1, int(round(smooth_ms / (hop_s * 1000.0))))
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(values, kernel, mode="same")


def compute_energy_smooth_track(audio_path: Path, *, hop_s: float = CANONICAL_HOP_SECONDS, smooth_ms: float = SMOOTH_MS_DEFAULT, cache_dir: Path | None = None) -> EvidenceTrack:
    key = make_cache_key(audio_path, "energy_smooth", {"hop_s": hop_s, "smooth_ms": smooth_ms})
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            values = npz["values"]
        tb = TimeBase.canonical(n_frames=len(values))
        return EvidenceTrack(name="energy_smooth", timebase=tb, values=values, semantics="score")

    base = compute_energy_rms_track(audio_path, hop_s=hop_s, cache_dir=cache_dir)
    smoothed = _smooth(np.asarray(base.values), smooth_ms, hop_s)
    tb = TimeBase.canonical(n_frames=len(smoothed))
    cache_put(cache_dir, key, {"values": smoothed})
    return EvidenceTrack(name="energy_smooth", timebase=tb, values=smoothed, semantics="score")


def compute_energy_slope_track(audio_path: Path, *, hop_s: float = CANONICAL_HOP_SECONDS, smooth_ms: float = SMOOTH_MS_DEFAULT, cache_dir: Path | None = None) -> EvidenceTrack:
    key = make_cache_key(audio_path, "energy_slope", {"hop_s": hop_s, "smooth_ms": smooth_ms})
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            values = npz["values"]
        tb = TimeBase.canonical(n_frames=len(values))
        return EvidenceTrack(name="energy_slope", timebase=tb, values=values, semantics="score")

    smooth_track = compute_energy_smooth_track(audio_path, hop_s=hop_s, smooth_ms=smooth_ms, cache_dir=cache_dir)
    smooth_vals = np.asarray(smooth_track.values, dtype=float)
    slope = np.empty_like(smooth_vals)
    slope[0] = 0.0
    slope[1:] = np.diff(smooth_vals) / hop_s  # approximate derivative per second
    # Light smoothing to reduce tail spikes from padding
    slope = _smooth(slope, 20.0, hop_s)
    tail_win = max(1, int(round(20.0 / (hop_s * 1000.0))))
    if tail_win < len(slope):
        slope[-tail_win:] = 0.0
    tb = TimeBase.canonical(n_frames=len(slope))
    cache_put(cache_dir, key, {"values": slope})
    return EvidenceTrack(name="energy_slope", timebase=tb, values=slope, semantics="score")
