from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from dyana.core.contracts.audio import AudioInput


def test_audio_input_accepts_path_backed_input() -> None:
    audio = AudioInput(path=Path("sample.wav"), waveform=None, sample_rate=16000, channels=1, duration=1.0)

    assert audio.path == Path("sample.wav")
    assert audio.waveform is None


def test_audio_input_accepts_waveform_backed_input() -> None:
    waveform = np.zeros(160, dtype=np.float32)
    audio = AudioInput(path=None, waveform=waveform, sample_rate=16000, channels=1, duration=0.01)

    assert audio.path is None
    assert audio.waveform is waveform


@pytest.mark.parametrize(
    ("path", "waveform"),
    [
        (None, None),
        (Path("sample.wav"), np.zeros(160, dtype=np.float32)),
    ],
)
def test_audio_input_requires_exactly_one_source(path: Path | None, waveform: np.ndarray | None) -> None:
    with pytest.raises(ValueError, match="Exactly one of 'path' or 'waveform' must be provided."):
        AudioInput(path=path, waveform=waveform, sample_rate=16000, channels=1, duration=1.0)


def test_audio_input_is_frozen() -> None:
    audio = AudioInput(path=Path("sample.wav"), waveform=None, sample_rate=16000, channels=1, duration=1.0)

    with pytest.raises(FrozenInstanceError):
        audio.sample_rate = 8000  # type: ignore[misc]
