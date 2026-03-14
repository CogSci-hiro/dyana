from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    hash: str
    metadata: dict[str, Any]
