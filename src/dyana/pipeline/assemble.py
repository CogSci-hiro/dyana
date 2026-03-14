"""Assembly helpers for the public annotation result."""

from __future__ import annotations

from dyana.api.types import Alignment, AnnotationResult, ConversationalState, Diagnostics, IPU, Phoneme, Transcript, Word
from dyana.core.contracts.alignment import AlignmentBundle
from dyana.core.contracts.decoder import DecodeResult
from dyana.core.contracts.diagnostics import DiagnosticsBundle
from dyana.core.contracts.transcript import TranscriptBundle


def assemble_annotation(
    transcript: object,
    alignment: object,
    decode: object,
    diagnostics: object,
) -> AnnotationResult:
    """
    Assemble the final public annotation result from pipeline contracts.

    Parameters
    ----------
    transcript
        Transcript contract object.
    alignment
        Alignment contract object.
    decode
        Decode contract object.
    diagnostics
        Diagnostics contract object.
    """
    public_transcript = _assemble_transcript(transcript)
    public_alignment = _assemble_alignment(alignment)
    public_ipus, public_states = _assemble_decode(decode)
    public_diagnostics = _assemble_diagnostics(diagnostics)
    return AnnotationResult(
        transcript=public_transcript,
        alignment=public_alignment,
        ipus=public_ipus,
        states=public_states,
        diagnostics=public_diagnostics,
    )


def _assemble_transcript(transcript: object) -> Transcript:
    if isinstance(transcript, Transcript):
        return transcript
    if isinstance(transcript, TranscriptBundle):
        return Transcript(
            words=[
                Word(text=token.text, start=0.0, end=0.0, speaker=token.speaker, confidence=None)
                for token in transcript.tokens
            ],
            language=transcript.language,
        )
    raise TypeError(f"Unsupported transcript contract: {type(transcript).__name__}")


def _assemble_alignment(alignment: object) -> Alignment:
    if isinstance(alignment, Alignment):
        return alignment
    if isinstance(alignment, AlignmentBundle):
        return Alignment(
            words=[
                Word(text=word.text, start=word.start, end=word.end, speaker=word.speaker, confidence=None)
                for word in alignment.words
            ],
            phonemes=[
                Phoneme(symbol=phoneme.symbol, start=phoneme.start, end=phoneme.end)
                for phoneme in (alignment.phonemes or [])
            ]
            or None,
        )
    raise TypeError(f"Unsupported alignment contract: {type(alignment).__name__}")


def _assemble_decode(decode: object) -> tuple[list[IPU], list[ConversationalState]]:
    if isinstance(decode, DecodeResult):
        ipus = [IPU(speaker=ipu.speaker, start=ipu.start, end=ipu.end) for ipu in decode.ipus]
        states = [ConversationalState(label=state.label, start=state.start, end=state.end) for state in decode.states]
        return ipus, states
    raise TypeError(f"Unsupported decode contract: {type(decode).__name__}")


def _assemble_diagnostics(diagnostics: object) -> Diagnostics:
    if isinstance(diagnostics, Diagnostics):
        return diagnostics
    if isinstance(diagnostics, DiagnosticsBundle):
        return Diagnostics(metrics=dict(diagnostics.metrics), flags=list(diagnostics.flags) if diagnostics.flags is not None else None)
    raise TypeError(f"Unsupported diagnostics contract: {type(diagnostics).__name__}")
