"""Lightweight stereo diarization evidence extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dyana.core.cache import cache_get, cache_put, make_cache_key
from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.io.audio import detect_channel_similarity, load_audio_stereo


def _frame_rms(samples: np.ndarray, hop_samples: int, n_frames: int) -> np.ndarray:
    trimmed = samples[: n_frames * hop_samples]
    frames = trimmed.reshape(n_frames, hop_samples)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))


def compute_stereo_diarization_tracks(
    audio_path: Path,
    *,
    hop_s: float = CANONICAL_HOP_SECONDS,
    similarity_threshold: float = 0.98,
    max_similarity_samples: int = 1_000_000,
    cache_dir: Path | None = None,
) -> tuple[EvidenceTrack, EvidenceTrack] | None:
    """
    Build speaker-A/B soft tracks from stereo channel dominance.

    Returns ``None`` when the file is dual-mono or not exactly stereo.
    """

    key = make_cache_key(
        audio_path,
        "stereo_diarization",
        {
            "hop_s": hop_s,
            "similarity_threshold": similarity_threshold,
            "max_similarity_samples": max_similarity_samples,
        },
    )
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            diar_a = npz["diar_a"].astype(np.float32)
            diar_b = npz["diar_b"].astype(np.float32)
            correlation = float(npz["correlation"][0])
            label = str(npz["label"][0])
        if label != "different":
            return None
        tb = TimeBase.canonical(n_frames=diar_a.shape[0])
        return (
            EvidenceTrack(
                name="diar_a",
                timebase=tb,
                values=diar_a,
                semantics="probability",
                metadata={"source": "cache", "channel_correlation": correlation},
            ),
            EvidenceTrack(
                name="diar_b",
                timebase=tb,
                values=diar_b,
                semantics="probability",
                metadata={"source": "cache", "channel_correlation": correlation},
            ),
        )

    try:
        label, correlation = detect_channel_similarity(
            str(audio_path), threshold=similarity_threshold, max_samples=max_similarity_samples
        )
        stereo, sample_rate = load_audio_stereo(audio_path)
    except ValueError:
        return None

    if label != "different":
        cache_put(
            cache_dir,
            key,
            {"diar_a": np.zeros(1, dtype=np.float32), "diar_b": np.zeros(1, dtype=np.float32), "correlation": np.array([correlation]), "label": np.array([label], dtype=object)},
        )
        return None

    hop_samples = int(round(sample_rate * hop_s))
    if hop_samples <= 0:
        raise ValueError("hop_s too small for sample rate.")

    n_frames = stereo.shape[0] // hop_samples
    if n_frames <= 0:
        raise ValueError("Audio too short for one frame.")

    left_rms = _frame_rms(stereo[:, 0], hop_samples, n_frames)
    right_rms = _frame_rms(stereo[:, 1], hop_samples, n_frames)
    total = left_rms + right_rms + 1e-8
    dominance = np.clip((left_rms - right_rms) / total, -1.0, 1.0)
    diar_a = np.clip(0.5 + 0.49 * dominance, 0.01, 0.99).astype(np.float32)
    diar_b = np.clip(0.5 - 0.49 * dominance, 0.01, 0.99).astype(np.float32)

    tb = TimeBase.canonical(n_frames=n_frames)
    cache_put(
        cache_dir,
        key,
        {
            "diar_a": diar_a,
            "diar_b": diar_b,
            "correlation": np.array([correlation]),
            "label": np.array([label], dtype=object),
        },
    )
    return (
        EvidenceTrack(
            name="diar_a",
            timebase=tb,
            values=diar_a,
            semantics="probability",
            metadata={"source": "stereo_dominance", "channel_correlation": correlation},
        ),
        EvidenceTrack(
            name="diar_b",
            timebase=tb,
            values=diar_b,
            semantics="probability",
            metadata={"source": "stereo_dominance", "channel_correlation": correlation},
        ),
    )
