"""Audio loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf


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
