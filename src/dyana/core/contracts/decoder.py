"""Decoder contracts used between internal pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass

from .alignment import AlignmentBundle
from .evidence import EvidenceBundle


@dataclass(frozen=True)
class DecoderInput:
    """Inputs required by a conversational structure decoder."""

    alignment: AlignmentBundle | None
    evidence: EvidenceBundle
    speakers: tuple[str, ...]


@dataclass(frozen=True)
class StateInterval:
    """Decoded conversational state interval."""

    label: str
    start: float
    end: float


@dataclass(frozen=True)
class IPUInterval:
    """Decoded inter-pausal unit interval."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class DecodeResult:
    """Decoded conversational structure outputs."""

    states: list[StateInterval]
    ipus: list[IPUInterval]
