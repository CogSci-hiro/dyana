"""Audio loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
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

    data, sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, int(sr)
