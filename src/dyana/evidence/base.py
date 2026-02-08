# src/dyana/evidence/base.py

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from dyana.core.timebase import CANONICAL_HOP_S, TimeBase


Semantics = Literal["probability", "logit", "score"]


@dataclass(frozen=True)
class EvidenceTrack:
    """
    Time-aligned soft evidence on a shared TimeBase.

    Usage example
    -------------
        tb = TimeBase.canonical()
        T = tb.num_frames(3.0)

        vad = EvidenceTrack(
            name="vad",
            timebase=tb,
            values=np.random.rand(T).astype(np.float32),
            semantics="probability",
            metadata={"module": "webrtcvad"},
        )
    """

    name: str
    timebase: TimeBase
    values: NDArray[np.floating]
    semantics: Semantics
    confidence: Optional[NDArray[np.floating]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)

        if values.ndim not in (1, 2):
            raise ValueError(
                f"EvidenceTrack.values must have ndim 1 or 2, got {values.ndim} for '{self.name}'."
            )
        if values.shape[0] <= 0:
            raise ValueError(f"EvidenceTrack.values must have T>0 for '{self.name}'.")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"EvidenceTrack.values must be a floating dtype for '{self.name}'.")
        if not np.isfinite(values).all():
            raise ValueError(f"EvidenceTrack.values contains NaN/Inf for '{self.name}'.")

        object.__setattr__(self, "values", values)

        if self.confidence is not None:
            confidence = np.asarray(self.confidence)

            if confidence.ndim not in (1, 2):
                raise ValueError(
                    f"EvidenceTrack.confidence must have ndim 1 or 2, got {confidence.ndim} for '{self.name}'."
                )
            if confidence.shape[0] != values.shape[0]:
                raise ValueError(
                    f"EvidenceTrack.confidence T mismatch for '{self.name}': "
                    f"values has T={values.shape[0]}, confidence has T={confidence.shape[0]}."
                )
            if values.ndim == 2 and confidence.ndim == 2 and confidence.shape[1] != values.shape[1]:
                raise ValueError(
                    f"EvidenceTrack.confidence K mismatch for '{self.name}': "
                    f"values has K={values.shape[1]}, confidence has K={confidence.shape[1]}."
                )
            if not np.issubdtype(confidence.dtype, np.floating):
                raise TypeError(f"EvidenceTrack.confidence must be a floating dtype for '{self.name}'.")
            if not np.isfinite(confidence).all():
                raise ValueError(f"EvidenceTrack.confidence contains NaN/Inf for '{self.name}'.")

            object.__setattr__(self, "confidence", confidence)

        if self.semantics == "probability":
            if (values < -1e-3).any() or (values > 1.0 + 1e-3).any():
                raise ValueError(
                    f"EvidenceTrack '{self.name}' semantics='probability' but values fall outside ~[0,1]."
                )

    @property
    def T(self) -> int:
        """Number of frames."""
        return int(self.values.shape[0])

    @property
    def K(self) -> int:
        """Evidence dimensionality (1 for (T,), else K for (T, K))."""
        return 1 if self.values.ndim == 1 else int(self.values.shape[1])


@dataclass
class EvidenceBundle:
    """
    Named collection of EvidenceTracks sharing a single timebase.

    Parameters
    ----------
    timebase
        Bundle timebase. All tracks must match this hop.
    require_canonical
        If True, enforce that bundle timebase hop equals canonical 10 ms hop.
        This is the Week 1 default and satisfies the checklist.

    Usage example
    -------------
        tb = TimeBase.canonical()
        bundle = EvidenceBundle(timebase=tb, require_canonical=True)
        bundle.add(vad_track)
    """

    timebase: TimeBase
    require_canonical: bool = True
    tracks: Dict[str, EvidenceTrack] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.require_canonical and abs(self.timebase.hop_s - CANONICAL_HOP_S) > 1e-12:
            raise ValueError(
                f"EvidenceBundle requires canonical hop {CANONICAL_HOP_S}, got {self.timebase.hop_s}."
            )

    def add(self, track: EvidenceTrack) -> None:
        """
        Add a track (replaces any existing track with the same name).
        """
        if abs(track.timebase.hop_s - self.timebase.hop_s) > 1e-12:
            raise ValueError(
                f"Track '{track.name}' has hop {track.timebase.hop_s}, "
                f"bundle hop is {self.timebase.hop_s}."
            )
        self.tracks[track.name] = track

    def get(self, name: str) -> Optional[EvidenceTrack]:
        """Return a track by name, or None if missing."""
        return self.tracks.get(name)

    def __iter__(self) -> Iterable[EvidenceTrack]:
        return iter(self.tracks.values())
