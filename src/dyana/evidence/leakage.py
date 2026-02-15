"""Stereo leakage likelihood evidence (Recipe B1 PoC)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from dyana.core.cache import cache_get, cache_put, make_cache_key
from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase
from dyana.evidence.base import EvidenceTrack


LEAKAGE_TRACK_NAME: str = "leakage_likelihood"
DEFAULT_SPEC_BINS: int = 64
DEFAULT_WIN_MS: float = 25.0
EPS: float = 1e-8


def _load_stereo_first_two(audio_path: Path) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(audio_path, always_2d=True)
    if data.shape[1] == 1:
        raise ValueError("compute_leakage_likelihood requires stereo input (2 channels), got mono.")
    stereo = data[:, :2].astype(np.float32, copy=False)
    return stereo, int(sample_rate)


def _frame_count(num_samples: int, hop_samples: int) -> int:
    return num_samples // hop_samples


def _rms_per_frame(channel: np.ndarray, hop_samples: int, n_frames: int) -> np.ndarray:
    trimmed = channel[: n_frames * hop_samples]
    frames = trimmed.reshape(n_frames, hop_samples)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))


def _centered_window(channel: np.ndarray, center_sample: int, win_samples: int) -> np.ndarray:
    half = win_samples // 2
    start = center_sample - half
    end = start + win_samples
    if start >= 0 and end <= channel.shape[0]:
        return channel[start:end]
    # Zero-pad for deterministic boundary handling.
    out = np.zeros(win_samples, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(channel.shape[0], end)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    if src_end > src_start:
        out[dst_start:dst_end] = channel[src_start:src_end]
    return out


def _pooled_log_spectrum(frame: np.ndarray, spec_bins: int) -> np.ndarray:
    windowed = frame * np.hanning(frame.shape[0]).astype(np.float32)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    if power.shape[0] < spec_bins:
        padded = np.zeros(spec_bins, dtype=np.float32)
        padded[: power.shape[0]] = power.astype(np.float32)
        pooled = padded
    else:
        chunks = np.array_split(power, spec_bins)
        pooled = np.array([float(np.mean(chunk)) for chunk in chunks], dtype=np.float32)
    return np.log1p(pooled)


def compute_leakage_likelihood(
    audio_path: Path,
    *,
    hop_s: float = CANONICAL_HOP_SECONDS,
    spec_bins: int = DEFAULT_SPEC_BINS,
    win_ms: float = DEFAULT_WIN_MS,
    cache_dir: Path | None = None,
) -> EvidenceTrack:
    """
    Compute stereo leakage likelihood on canonical 10 ms grid.

    Notes
    -----
    - Output semantics are ``probability`` on ``[0, 1]``.
    - For inputs with >2 channels, only the first two channels are used.
    - Mono input raises ``ValueError``.
    """

    if abs(hop_s - CANONICAL_HOP_SECONDS) > 1e-12:
        raise ValueError(f"Leakage evidence must run on canonical hop {CANONICAL_HOP_SECONDS}, got {hop_s}.")
    if spec_bins <= 0:
        raise ValueError(f"spec_bins must be positive, got {spec_bins}.")
    if win_ms <= 0:
        raise ValueError(f"win_ms must be positive, got {win_ms}.")

    key = make_cache_key(
        audio_path,
        "leakage_likelihood",
        {"hop_s": hop_s, "spec_bins": spec_bins, "win_ms": win_ms},
    )
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            values = npz["values"].astype(np.float32)
        tb = TimeBase.canonical(n_frames=values.shape[0])
        return EvidenceTrack(
            name=LEAKAGE_TRACK_NAME,
            timebase=tb,
            values=values,
            semantics="probability",
            metadata={"source": "cache", "recipe": "B1"},
        )

    stereo, sample_rate = _load_stereo_first_two(audio_path)
    hop_samples = int(round(sample_rate * hop_s))
    if hop_samples <= 0:
        raise ValueError("hop_s too small for sample rate.")

    n_frames = _frame_count(stereo.shape[0], hop_samples)
    if n_frames <= 0:
        raise ValueError("Audio too short for one 10 ms frame.")

    left = stereo[:, 0]
    right = stereo[:, 1]
    energy_left = _rms_per_frame(left, hop_samples, n_frames)
    energy_right = _rms_per_frame(right, hop_samples, n_frames)

    dominance = (energy_left - energy_right) / (energy_left + energy_right + EPS)
    dom_strength = np.clip(np.abs(dominance), 0.0, 1.0)

    win_samples = max(1, int(round(sample_rate * (win_ms / 1000.0))))
    cos_sim = np.zeros(n_frames, dtype=np.float32)
    for frame_index in range(n_frames):
        center = frame_index * hop_samples + hop_samples // 2
        frame_left = _centered_window(left, center, win_samples)
        frame_right = _centered_window(right, center, win_samples)
        spec_left = _pooled_log_spectrum(frame_left, spec_bins)
        spec_right = _pooled_log_spectrum(frame_right, spec_bins)
        denom = float(np.linalg.norm(spec_left) * np.linalg.norm(spec_right) + EPS)
        cos_value = float(np.dot(spec_left, spec_right) / denom)
        cos_sim[frame_index] = float(np.clip(cos_value, 0.0, 1.0))

    total_energy = energy_left + energy_right
    ref_energy = float(np.percentile(total_energy, 90))
    gate = np.clip(total_energy / (ref_energy + EPS), 0.0, 1.0)

    leak_raw = dom_strength * cos_sim
    leakage = np.clip(leak_raw * gate, 0.0, 1.0).astype(np.float32)

    tb = TimeBase.canonical(n_frames=n_frames)
    cache_put(
        cache_dir,
        key,
        {
            "values": leakage,
            "metadata": np.array(
                [
                    f'{{"name":"{LEAKAGE_TRACK_NAME}","semantics":"probability","recipe":"B1","hop_s":{hop_s}}}'
                ],
                dtype=object,
            ),
        },
    )
    return EvidenceTrack(
        name=LEAKAGE_TRACK_NAME,
        timebase=tb,
        values=leakage,
        semantics="probability",
        metadata={"source": "computed", "recipe": "B1"},
    )
