"""Transcript contracts used between internal pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Token:
    """Token-level transcript item."""

    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptBundle:
    """Transcript payload shared by internal stages."""

    tokens: list[Token]
    language: str | None
    metadata: dict[str, Any] | None = None
