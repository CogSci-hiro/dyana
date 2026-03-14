"""Alignment contracts used between internal pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordInterval:
    """Word-level timing interval."""

    text: str
    start: float
    end: float
    speaker: str | None


@dataclass(frozen=True)
class PhonemeInterval:
    """Phoneme-level timing interval."""

    symbol: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignmentBundle:
    """Alignment payload shared by internal stages."""

    words: list[WordInterval]
    phonemes: list[PhonemeInterval] | None
