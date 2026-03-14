"""Public annotation entrypoint."""

from __future__ import annotations

from pathlib import Path

from dyana.core.contracts.audio import AudioInput
from dyana.pipeline.run import run_annotation_pipeline

from .types import AnnotationResult


def annotate(
    audio: str | Path,
    *,
    language: str | None = None,
    diarize: bool = True,
    align: bool = True,
) -> AnnotationResult:
    """
    Run the end-to-end annotation pipeline for an audio input.

    Parameters
    ----------
    audio
        Path to the audio source.
    language
        Optional language hint for the pipeline.
    diarize
        Whether speaker-attributed outputs should be requested.
    align
        Whether time-aligned outputs should be requested.

    Returns
    -------
    AnnotationResult
        Structured annotation outputs for the input audio.
    """
    del language, diarize, align

    audio_input = AudioInput(
        path=Path(audio),
        waveform=None,
        sample_rate=0,
        channels=0,
        duration=0.0,
    )
    return run_annotation_pipeline(audio_input)
