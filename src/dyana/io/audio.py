"""Audio loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def _soundfile():
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DYANA requires the 'soundfile' package for audio I/O. "
            "Install runtime dependencies with `pip install -e .` or test dependencies with "
            "`pip install -e '.[test]'`."
        ) from exc

    return sf


def load_audio_stereo(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio and require exactly two channels."""

    sf = _soundfile()
    data, sr = sf.read(path, always_2d=True)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected stereo audio with shape (n_samples, 2), got {data.shape}.")
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, int(sr)


def load_audio_mono(path: Path, *, channel: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32 PCM.

    Parameters
    ----------
    path
        Path to wav/flac/etc supported by soundfile.

    Returns
    -------
    samples : np.ndarray
        Shape (N,), float32, range roughly [-1, 1].
    sample_rate : int
    """

    sf = _soundfile()
    data, sr = sf.read(path, always_2d=True)
    if channel is not None:
        if channel < 0 or channel >= data.shape[1]:
            raise ValueError(f"Requested channel {channel} but file has {data.shape[1]} channels.")
        data = data[:, channel]
    else:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, int(sr)


def detect_channel_similarity(
    wav_path: str,
    threshold: float = 0.98,
    max_samples: int = 1_000_000,
) -> tuple[str, float]:
    """
    Detect whether stereo channels contain the same signal.

    Returns
    -------
    label : str
        ``"same"`` or ``"different"``.
    correlation : float
        Pearson correlation between channels.
    """

    stereo, sample_rate = load_audio_stereo(Path(wav_path))
    max_window = min(stereo.shape[0], int(sample_rate * 30))
    window = stereo[:max_window]
    if window.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(window.shape[0], size=max_samples, replace=False))
        window = window[indices]

    left = np.asarray(window[:, 0], dtype=np.float64)
    right = np.asarray(window[:, 1], dtype=np.float64)
    left -= left.mean()
    right -= right.mean()
    left_std = float(left.std())
    right_std = float(right.std())

    if left_std <= 1e-12 or right_std <= 1e-12:
        corr = 1.0 if np.allclose(left, right, atol=1e-9, rtol=1e-6) else 0.0
    else:
        left /= left_std
        right /= right_std
        corr = float(np.mean(left * right))
        corr = float(np.clip(corr, -1.0, 1.0))

    label = "same" if corr > threshold else "different"
    return label, corr
