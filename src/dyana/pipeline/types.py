"""Types for dependency-safe pipeline orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PipelineStep:
    """
    Declarative pipeline step definition.

    Parameters
    ----------
    name
        Unique step name in the registry.
    inputs
        State keys required before the step may execute.
    outputs
        State keys produced by the step.
    run
        Callable that accepts the required state slice and returns produced outputs.
    parameters
        Stable configuration captured in the cache key and provenance metadata.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    run: Callable[..., object]
    parameters: Mapping[str, Any] = field(default_factory=dict)
