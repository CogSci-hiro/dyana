# dyana/core/timebase.py

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TimeBase:
    """
    Canonical global timebase for DYANA.

    This class defines a fixed mapping between frame indices and time in seconds.
    All EvidenceTracks and decoders must operate on a shared TimeBase.

    Notes
    -----
    - TimeBase is immutable.
    - TimeBase is independent of audio sample rate.
    - The canonical DYANA v0 timebase uses a 10 ms hop.

    Usage example
    -------------
        tb = TimeBase(hop_s=0.01)

        t = tb.frame_to_time(42)        # 0.42
        i = tb.time_to_frame(0.421)     # 42
        n = tb.num_frames(3.7)          # number of frames covering 3.7 s
    """

    hop_s: float = 0.01

    def __post_init__(self) -> None:
        if self.hop_s <= 0:
            raise ValueError("hop_s must be positive")

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
        Convert time in seconds to nearest frame index.

        Uses floor semantics to ensure deterministic alignment.
        """
        if time_s < 0:
            raise ValueError("time_s must be non-negative")
        return int(math.floor(time_s / self.hop_s))

    def num_frames(self, duration_s: float) -> int:
        """
        Number of frames needed to cover a duration.

        This guarantees:
            frame_to_time(num_frames - 1) < duration_s
        """
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        return int(math.ceil(duration_s / self.hop_s))

    def frame_times(self, n_frames: int) -> np.ndarray:
        """
        Return an array of frame center times.

        Useful for plotting and diagnostics.
        """
        if n_frames < 0:
            raise ValueError("n_frames must be non-negative")
        return np.arange(n_frames, dtype=float) * self.hop_s
