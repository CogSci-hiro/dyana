"""Public alignment entrypoint."""

from __future__ import annotations

from pathlib import Path

from .types import Alignment, Transcript


def align(
    audio: str | Path,
    transcript: Transcript | str | Path,
    *,
    language: str | None = None,
) -> Alignment:
    """
    Align a transcript to an audio input.

    Parameters
    ----------
    audio
        Path to the audio source.
    transcript
        Transcript object or a path-like transcript reference.
    language
        Optional language hint for alignment.

    Returns
    -------
    Alignment
        Word- and phoneme-level timing annotations.
    """

    raise NotImplementedError("Pipeline not implemented yet")
