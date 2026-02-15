# src/dyana/core/timebase.py

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


CANONICAL_HOP_S: float = 0.01
_HOP_TOL: float = 1e-12


@dataclass(frozen=True)
class TimeBase:
    """
    Canonical global timebase for DYANA.

    Notes
    -----
    - TimeBase is immutable.
    - The *canonical* DYANA timebase is 10 ms hop.
    - Internal computations may use other hops, but boundaries must resample.

    Usage example
    -------------
        tb = TimeBase.canonical()

        t = tb.frame_to_time(42)        # 0.42
        i = tb.time_to_frame(0.421)     # 42
        n = tb.num_frames(3.7)          # 370
    """

    hop_s: float = CANONICAL_HOP_S
    n_frames: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hop_s <= 0:
            raise ValueError("hop_s must be positive")
        if self.n_frames is not None:
            if not isinstance(self.n_frames, int):
                raise TypeError("n_frames must be an int when provided")
            if self.n_frames < 0:
                raise ValueError("n_frames must be non-negative when provided")

        # Normalize exactly to canonical hop when within tolerance to avoid
        # floating drift in downstream equality checks.
        if math.isclose(self.hop_s, CANONICAL_HOP_S, rel_tol=0.0, abs_tol=_HOP_TOL):
            object.__setattr__(self, "hop_s", CANONICAL_HOP_S)

    @classmethod
    def canonical(cls, n_frames: Optional[int] = None) -> "TimeBase":
        """
        Construct the canonical 10 ms timebase.

        Parameters
        ----------
        n_frames
            Optional number of frames to bind to this timebase. Validation
            will enforce matching length when present.

        Usage example
        -------------
            tb = TimeBase.canonical()
            tb_frames = TimeBase.canonical(n_frames=500)
        """

        return cls(hop_s=CANONICAL_HOP_S, n_frames=n_frames)

    @classmethod
    def from_hop_seconds(cls, hop_seconds: float, n_frames: Optional[int] = None) -> "TimeBase":
        """
        Construct a timebase from an explicit hop size in seconds.

        Notes
        -----
        - This helper is explicit about non-canonical hops. Use
          :meth:`canonical` for the global 10 ms grid.

        Parameters
        ----------
        hop_seconds
            Frame hop in seconds. Must be positive.
        n_frames
            Optional frame count bound to this timebase.

        Usage example
        -------------
            tb = TimeBase.from_hop_seconds(0.02)
        """

        return cls(hop_s=hop_seconds, n_frames=n_frames)

    @property
    def is_canonical(self) -> bool:
        """Whether this timebase matches the canonical 10 ms hop."""

        return math.isclose(self.hop_s, CANONICAL_HOP_S, rel_tol=0.0, abs_tol=_HOP_TOL)

    def require_canonical(self) -> None:
        """Raise a ValueError if the timebase is not canonical."""

        if not self.is_canonical:
            raise ValueError(
                f"TimeBase is not canonical: hop_s={self.hop_s} (expected {CANONICAL_HOP_S})."
            )

    @property
    def hop_ms(self) -> float:
        """Frame hop in milliseconds."""
        return self.hop_s * 1000.0

    def frame_to_time(self, frame_index: int) -> float:
        """
        Convert frame index to time in seconds.
        """
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        return frame_index * self.hop_s

    def time_to_frame(self, time_s: float) -> int:
        """
        Convert time in seconds to frame index (floor semantics).

        Formula
        -------
            i = floor(t / hop_s)

        Usage example
        -------------
            tb = TimeBase.canonical()
            tb.time_to_frame(0.421)  # 42
        """
        if time_s < 0:
            raise ValueError("time_s must be non-negative")
        return int(math.floor(time_s / self.hop_s))

    def num_frames(self, duration_s: float) -> int:
        """
        Number of frames needed to cover a duration.

        Formula
        -------
            N = ceil(duration_s / hop_s)

        Usage example
        -------------
            tb = TimeBase.canonical()
            tb.num_frames(3.7)  # 370
        """
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        return int(math.ceil(duration_s / self.hop_s))

    def frame_times(self, n_frames: int) -> np.ndarray:
        """
        Array of frame times (seconds), shape (n_frames,).

        Usage example
        -------------
            tb = TimeBase.canonical()
            times = tb.frame_times(5)  # [0.0, 0.01, 0.02, 0.03, 0.04]
        """
        if n_frames < 0:
            raise ValueError("n_frames must be non-negative")
        return np.arange(n_frames, dtype=float) * self.hop_s
