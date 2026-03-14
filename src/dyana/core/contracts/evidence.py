"""Evidence contracts used between internal pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


EvidenceArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class EvidenceTrack:
    """
    Time series evidence track on a fixed frame grid.

    Parameters
    ----------
    name
        Stable track name used by downstream modules.
    values
        Frame-aligned evidence values.
    frame_hop
        Frame hop in seconds.
    start_time
        Absolute start time of the first frame in seconds.
    """

    name: str
    values: EvidenceArray
    frame_hop: float
    start_time: float


@dataclass(frozen=True)
class EvidenceBundle:
    """Collection of named evidence tracks for a single recording."""

    tracks: dict[str, EvidenceTrack]
    duration: float
