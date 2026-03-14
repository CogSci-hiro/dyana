"""Diagnostics contracts used between internal pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsBundle:
    """Structured diagnostics produced by pipeline stages."""

    metrics: dict[str, float]
    flags: list[str] | None = None
