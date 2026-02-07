# src/dyana/core/evidence/base.py

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from dyana.core.timebase import TimeBase


Semantics = Literal["probability", "logit", "score"]


@dataclass(frozen=True)
class EvidenceTrack:
    """
    Time-aligned soft evidence on a shared TimeBase.

    Parameters
    ----------
    name
        Semantic identifier (e.g., "vad", "energy", "speaker_logits", "overlap").
    timebase
        The TimeBase used to interpret frame indices.
    values
        Evidence array of shape (T,) or (T, K).
    semantics
        Declares what `values` represent:
        - "probability": typically in [0, 1]
        - "logit": unbounded; may be converted to probability downstream
        - "score": arbitrary real-valued score (higher = more evidence)
    confidence
        Optional reliability estimates. Shape must be (T,) or match values (T, K).
        Values should usually be in [0, 1], where 1 = fully reliable.
    metadata
        Provenance and configuration: module name, parameters, versions, etc.

    Notes
    -----
    - EvidenceTrack is immutable (frozen) to prevent accidental drift.
    - This class validates shapes and basic numeric sanity to avoid silent bugs.
    - Axis 2 should tolerate missing tracks; EvidenceTrack itself is strict.

    Usage example
    -------------
        tb = TimeBase(hop_s=0.01)
        T = tb.num_frames(3.0)

        # Example: VAD speech probability
        vad = EvidenceTrack(
            name="vad",
            timebase=tb,
            values=np.random.rand(T).astype(np.float32),
            semantics="probability",
            metadata={"module": "webrtcvad", "mode": 2},
        )

        # Example: per-state scores (T, K)
        logits = EvidenceTrack(
            name="state_logits",
            timebase=tb,
            values=np.random.randn(T, 5).astype(np.float32),
            semantics="logit",
            metadata={"module": "baseline", "states": ["SIL", "A", "B", "OVL", "LEAK"]},
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
            conf = np.asarray(self.confidence)
            if conf.ndim not in (1, 2):
                raise ValueError(
                    f"EvidenceTrack.confidence must have ndim 1 or 2, got {conf.ndim} for '{self.name}'."
                )
            if conf.shape[0] != values.shape[0]:
                raise ValueError(
                    f"EvidenceTrack.confidence T mismatch for '{self.name}': "
                    f"values has T={values.shape[0]}, confidence has T={conf.shape[0]}."
                )
            if values.ndim == 2 and conf.ndim == 2 and conf.shape[1] != values.shape[1]:
                raise ValueError(
                    f"EvidenceTrack.confidence K mismatch for '{self.name}': "
                    f"values has K={values.shape[1]}, confidence has K={conf.shape[1]}."
                )
            if not np.issubdtype(conf.dtype, np.floating):
                raise TypeError(f"EvidenceTrack.confidence must be a floating dtype for '{self.name}'.")
            if not np.isfinite(conf).all():
                raise ValueError(f"EvidenceTrack.confidence contains NaN/Inf for '{self.name}'.")

            object.__setattr__(self, "confidence", conf)

        # Light-touch semantic sanity checks (don’t over-police Week 1)
        if self.semantics == "probability":
            # allow tiny numerical slop, but catch totally wrong scales
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
        """Evidence dimensionality (1 for (T,), otherwise number of columns in (T, K))."""
        return 1 if self.values.ndim == 1 else int(self.values.shape[1])

    def with_values(self, values: NDArray[np.floating]) -> "EvidenceTrack":
        """
        Return a new EvidenceTrack with the same metadata/timebase but new values.

        Useful after resampling or normalization.
        """
        return EvidenceTrack(
            name=self.name,
            timebase=self.timebase,
            values=values,
            semantics=self.semantics,
            confidence=self.confidence,
            metadata=self.metadata,
        )
