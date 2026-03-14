"""Audio input contract for internal pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt
import numpy as np


WaveformArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class AudioInput:
    """
    Normalized audio input passed between internal pipeline stages.

    Parameters
    ----------
    path
        Original audio path when file-backed input is available.
    waveform
        In-memory waveform when audio has already been loaded.
    sample_rate
        Sampling rate in Hz.
    channels
        Number of audio channels.
    duration
        Audio duration in seconds.
    """

    path: Path | None
    waveform: WaveformArray | None
    sample_rate: int
    channels: int
    duration: float

    def __post_init__(self) -> None:
        has_path = self.path is not None
        has_waveform = self.waveform is not None
        if has_path == has_waveform:
            raise ValueError("Exactly one of 'path' or 'waveform' must be provided.")
