from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ArtifactMetadata:
    step: str
    inputs: dict[str, str]
    parameters: dict[str, Any]
    timestamp: float
    code_version: str | None
