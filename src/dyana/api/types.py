"""Public dataclasses exposed by the stable DYANA API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """Word-level transcript item."""

    text: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Transcript:
    """Transcript container."""

    words: list[Word]
    language: str | None = None


@dataclass(frozen=True)
class Phoneme:
    """Phoneme-level alignment item."""

    symbol: str
    start: float
    end: float


@dataclass(frozen=True)
class Alignment:
    """Alignment outputs for a transcript."""

    words: list[Word]
    phonemes: list[Phoneme] | None = None


@dataclass(frozen=True)
class IPU:
    """Inter-pausal unit interval."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class ConversationalState:
    """Conversation state interval."""

    label: str
    start: float
    end: float


@dataclass(frozen=True)
class Diagnostics:
    """Backend-independent diagnostics surfaced by the pipeline."""

    metrics: dict[str, float]
    flags: list[str] | None = None


@dataclass(frozen=True)
class AnnotationResult:
    """Combined annotation outputs returned by :func:`annotate`."""

    transcript: Transcript
    alignment: Alignment
    ipus: list[IPU]
    states: list[ConversationalState]
    diagnostics: Diagnostics
