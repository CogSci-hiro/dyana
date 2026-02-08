# src/dyana/core/timebase.py

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


CANONICAL_HOP_S: float = 0.01


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

    def __post_init__(self) -> None:
        if self.hop_s <= 0:
            raise ValueError("hop_s must be positive")

    @classmethod
    def canonical(cls) -> "TimeBase":
        """
        Construct the canonical 10 ms timebase.
        """
        return cls(hop_s=CANONICAL_HOP_S)

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
