from __future__ import annotations

import numpy as np

from dyana.asr.whisper_backend import _convert_whisper_segments, _resample_linear


def test_convert_whisper_segments_skips_reversed_intervals() -> None:
    whisper_segments = [
        {
            "start": 2.1,
            "end": 0.0,
            "text": "bad segment",
            "words": [
                {
                    "word": "bad",
                    "start": 3.0,
                    "end": 0.0,
                    "probability": 0.2,
                }
            ],
        },
        {
            "start": 0.1,
            "end": 0.4,
            "text": "good segment",
            "words": [
                {
                    "word": "good",
                    "start": 0.1,
                    "end": 0.2,
                    "probability": 0.9,
                }
            ],
        },
    ]

    segments = _convert_whisper_segments(
        whisper_segments,
        chunk_start_time=49.465,
        chunk_end_time=49.965,
    )

    assert len(segments) == 1
    assert segments[0].text == "good segment"
    assert len(segments[0].words) == 1


def test_resample_linear_converts_audio_to_whisper_rate() -> None:
    waveform = np.linspace(-1.0, 1.0, num=44100, dtype=np.float32)

    resampled = _resample_linear(waveform, 44100, 16000)

    assert resampled.dtype == np.float32
    assert abs(resampled.shape[0] - 16000) <= 1
