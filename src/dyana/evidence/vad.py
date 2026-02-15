"""Speech activity evidence extraction via WebRTC VAD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for tests
    webrtcvad = None

from dyana.core.cache import cache_get, cache_put, make_cache_key
from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.io.audio import load_audio_mono


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    t_in = np.linspace(0, len(x) / sr_in, num=len(x), endpoint=False)
    t_out = np.linspace(0, len(x) / sr_in, num=int(round(len(x) * sr_out / sr_in)), endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def _bytes_16k(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    return int16.tobytes()


def compute_webrtc_vad_soft_track(
    audio_path: Path,
    *,
    hop_s: float = CANONICAL_HOP_SECONDS,
    vad_mode: int = 2,
    subframe_ms: int = 5,
    cache_dir: Path | None = None,
) -> EvidenceTrack:
    if webrtcvad is None:
        raise ImportError("webrtcvad package not installed; install webrtcvad to use VAD features.")
    key = make_cache_key(audio_path, "webrtc_vad_soft", {"hop_s": hop_s, "vad_mode": vad_mode, "sub_ms": subframe_ms})
    cached = cache_get(cache_dir, key)
    if cached is not None:
        with np.load(cached) as npz:
            values = npz["values"]
        tb = TimeBase.canonical(n_frames=len(values))
        return EvidenceTrack(name="webrtc_vad_soft", timebase=tb, values=values, semantics="probability")

    samples, sr = load_audio_mono(audio_path)
    sr_vad = 16000
    samples_16k = _resample_linear(samples, sr, sr_vad)

    vad = webrtcvad.Vad(vad_mode)

    hop_samples = int(round(sr_vad * hop_s))
    n_frames = int(np.ceil(len(samples_16k) / hop_samples))
    pad = n_frames * hop_samples - len(samples_16k)
    if pad > 0:
        samples_16k = np.concatenate([samples_16k, np.zeros(pad, dtype=np.float32)])

    sub_hop = int(round(sr_vad * subframe_ms / 1000.0))
    sub_hop = max(sub_hop, 80)  # at least 5 ms
    window = int(round(sr_vad * 0.01))  # 10ms window required by VAD
    values = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + hop_samples
        slice_samples = samples_16k[start:end]
        if len(slice_samples) < window:
            slice_samples = np.pad(slice_samples, (0, window - len(slice_samples)))
        voiced_count = 0
        trials = 0
        for sub_start in range(0, len(slice_samples) - window + 1, sub_hop):
            frame = slice_samples[sub_start : sub_start + window]
            if len(frame) < window:
                frame = np.pad(frame, (0, window - len(frame)))
            voiced = vad.is_speech(_bytes_16k(frame), sample_rate=sr_vad)
            voiced_count += int(voiced)
            trials += 1
        if trials == 0:
            trials = 1
        values[i] = voiced_count / trials

    # soften hard 0/1 outputs by temporal blur to encourage fractional values
    if len(values) > 1:
        kernel = np.ones(9, dtype=np.float32) / 9.0
        padded = np.pad(values, (4, 4), mode="constant")
        values = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    tb = TimeBase.canonical(n_frames=n_frames)
    cache_put(cache_dir, key, {"values": values})
    return EvidenceTrack(name="webrtc_vad_soft", timebase=tb, values=values, semantics="probability")
