"""Public conversational-structure decoding entrypoint."""

from __future__ import annotations

from pathlib import Path

from .types import ConversationalState, IPU


def decode_structure(
    audio: str | Path,
    *,
    speakers: tuple[str, str] = ("A", "B"),
) -> tuple[list[IPU], list[ConversationalState]]:
    """
    Decode speaker turns and conversational state from an audio input.

    Parameters
    ----------
    audio
        Path to the audio source.
    speakers
        Ordered speaker labels to use in the decoded outputs.

    Returns
    -------
    tuple[list[IPU], list[ConversationalState]]
        Inter-pausal units and conversational state intervals.
    """

    raise NotImplementedError("Pipeline not implemented yet")
