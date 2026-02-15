"""Base evidence track types and serialization utilities."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from dyana.core.resample import AggKind, downsample, resample, to_canonical_grid, upsample_hold_last
from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase


Semantics = Literal["probability", "logit", "score"]


@dataclass(frozen=True)
class EvidenceTrack:
    """
    Time-aligned soft evidence on a shared TimeBase.

    Parameters
    ----------
    name
        Track name (e.g., ``"vad"``).
    timebase
        TimeBase describing hop size and optional frame count.
    values
        Array of shape (T,) or (T, K). Must be floating, finite.
    semantics
        One of ``"probability"``, ``"score"``, ``"logit"``.
    confidence
        Optional confidence array aligned with ``values``.
    metadata
        Free-form provenance metadata.

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

        if self.semantics not in ("probability", "score", "logit"):
            raise ValueError(
                f"EvidenceTrack '{self.name}' semantics must be one of probability|score|logit, got '{self.semantics}'."
            )

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
            if values.ndim == 2:
                if confidence.ndim != 2 or confidence.shape[1] != values.shape[1]:
                    raise ValueError(
                        f"EvidenceTrack.confidence K mismatch for '{self.name}': "
                        f"values has shape {values.shape}, confidence has shape {confidence.shape}."
                    )
            else:
                if confidence.ndim != 1:
                    raise ValueError(
                        f"EvidenceTrack.confidence for 1-D values must be 1-D, got shape {confidence.shape}."
                    )
            if not np.issubdtype(confidence.dtype, np.floating):
                raise TypeError(f"EvidenceTrack.confidence must be a floating dtype for '{self.name}'.")
            if not np.isfinite(confidence).all():
                raise ValueError(f"EvidenceTrack.confidence contains NaN/Inf for '{self.name}'.")
            if (confidence < -1e-3).any() or (confidence > 1.0 + 1e-3).any():
                raise ValueError(
                    f"EvidenceTrack.confidence must be in [0,1] for '{self.name}', found values outside tolerance."
                )

            object.__setattr__(self, "confidence", confidence)

        if self.semantics == "probability":
            if (values < -1e-3).any() or (values > 1.0 + 1e-3).any():
                raise ValueError(
                    f"EvidenceTrack '{self.name}' semantics='probability' but values fall outside ~[0,1]."
                )

        if self.timebase.n_frames is not None and values.shape[0] != self.timebase.n_frames:
            raise ValueError(
                f"EvidenceTrack '{self.name}' length {values.shape[0]} does not match timebase.n_frames={self.timebase.n_frames}."
            )

    @property
    def T(self) -> int:
        """Number of frames."""
        return int(self.values.shape[0])

    @property
    def K(self) -> int:
        """Evidence dimensionality (1 for (T,), else K for (T, K))."""
        return 1 if self.values.ndim == 1 else int(self.values.shape[1])

    def to_canonical(
        self,
        target_timebase: Optional[TimeBase] = None,
        *,
        downsample_agg: Optional[AggKind] = None,
    ) -> "EvidenceTrack":
        """
        Return an EvidenceTrack on the canonical 10 ms grid.

        Parameters
        ----------
        target_timebase
            Target timebase (must be canonical). If None, uses global canonical.
        downsample_agg
            Aggregation to use if downsampling is required (``"mean"`` or ``"max"``).

        Returns
        -------
        EvidenceTrack
            Resampled track.

        Usage example
        -------------
            canonical = track.to_canonical(downsample_agg="mean")
        """

        target = target_timebase or TimeBase.canonical()
        target.require_canonical()

        if self.timebase.hop_s == target.hop_s:
            if target.n_frames is not None and target.n_frames != self.T:
                raise ValueError(
                    f"Target timebase expects n_frames={target.n_frames} but track has {self.T}."
                )
            return self

        values_resamp = to_canonical_grid(
            self.values,
            src_hop_s=self.timebase.hop_s,
            semantics=self.semantics,
            downsample_agg=downsample_agg,
        )

        confidence_resamp = None
        if self.confidence is not None:
            if self.timebase.hop_s > target.hop_s:
                confidence_resamp = upsample_hold_last(self.confidence, self.timebase.hop_s, target.hop_s)
            else:
                if downsample_agg is None:
                    raise ValueError("downsample_agg required to resample confidence")
                confidence_resamp = downsample(self.confidence, self.timebase.hop_s, target.hop_s, agg=downsample_agg)

        n_frames = target.n_frames if target.n_frames is not None else values_resamp.shape[0]
        if n_frames != values_resamp.shape[0]:
            raise ValueError(
                f"Resampled length {values_resamp.shape[0]} does not match target n_frames={target.n_frames}."
            )

        canonical_tb = TimeBase.canonical(n_frames=n_frames)

        return EvidenceTrack(
            name=self.name,
            timebase=canonical_tb,
            values=values_resamp,
            semantics=self.semantics,
            confidence=confidence_resamp,
            metadata=self.metadata,
        )

    def resample_to(
        self,
        target_timebase: TimeBase,
        *,
        agg: AggKind | None = None,
    ) -> "EvidenceTrack":
        """
        Resample this track onto ``target_timebase``.

        Parameters
        ----------
        target_timebase
            Desired timebase. If identical to the current one (hop and optional
            ``n_frames``), returns self.
        agg
            Aggregation for downsampling (``\"mean\"`` or ``\"max\"``). Required
            when ``target_timebase.hop_s`` is coarser than ``self.timebase.hop_s``.
        """

        if (
            self.timebase.hop_s == target_timebase.hop_s
            and (target_timebase.n_frames is None or target_timebase.n_frames == self.T)
        ):
            return self

        values_rs = resample(
            self.values,
            src_hop_s=self.timebase.hop_s,
            target_hop_s=target_timebase.hop_s,
            agg=agg,
        )

        confidence_rs = None
        if self.confidence is not None:
            confidence_rs = resample(
                self.confidence,
                src_hop_s=self.timebase.hop_s,
                target_hop_s=target_timebase.hop_s,
                agg=agg,
            )

        if target_timebase.n_frames is not None and values_rs.shape[0] != target_timebase.n_frames:
            raise ValueError(
                f"Resampled track length {values_rs.shape[0]} does not match target n_frames={target_timebase.n_frames}."
            )

        tb = TimeBase.from_hop_seconds(target_timebase.hop_s, n_frames=values_rs.shape[0] if target_timebase.n_frames is None else target_timebase.n_frames)

        return EvidenceTrack(
            name=self.name,
            timebase=tb,
            values=values_rs,
            semantics=self.semantics,
            confidence=confidence_rs,
            metadata=self.metadata,
        )

    def to_npz(self, path: Path) -> None:
        """
        Serialize the track arrays to NPZ.

        Parameters
        ----------
        path
            Destination file path. Parent directory must exist.

        Usage example
        -------------
            track.to_npz(Path("/tmp/vad.npz"))
        """

        arrays = {"values": np.asarray(self.values)}
        if self.confidence is not None:
            arrays["confidence"] = np.asarray(self.confidence)

        meta = {
            "name": self.name,
            "semantics": self.semantics,
            "timebase": {"hop_s": self.timebase.hop_s, "n_frames": self.timebase.n_frames},
            "metadata": dict(self.metadata),
            "has_confidence": self.confidence is not None,
        }
        arrays["metadata_json"] = np.array(json.dumps(meta))

        np.savez_compressed(path, **arrays)

    def to_manifest(self) -> Dict[str, Any]:
        """Return a JSON-serializable manifest entry for this track."""

        return {
            "name": self.name,
            "semantics": self.semantics,
            "timebase": {"hop_s": self.timebase.hop_s, "n_frames": self.timebase.n_frames},
            "metadata": dict(self.metadata),
            "has_confidence": self.confidence is not None,
        }

    @staticmethod
    def from_npz(path: Path) -> "EvidenceTrack":
        """Load an EvidenceTrack serialized with :meth:`to_npz`."""

        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data:
                raise ValueError(f"NPZ at {path} missing metadata_json for EvidenceTrack.")
            meta = json.loads(str(data["metadata_json"]))
            values = data["values"]
            confidence = data["confidence"] if "confidence" in data.files else None

        tb_info = meta["timebase"]
        timebase = TimeBase.from_hop_seconds(tb_info["hop_s"], n_frames=tb_info.get("n_frames"))

        return EvidenceTrack(
            name=meta["name"],
            timebase=timebase,
            values=values,
            semantics=meta["semantics"],
            confidence=confidence,
            metadata=meta.get("metadata", {}),
        )

    @staticmethod
    def from_files(name: str, npz_path: Path, manifest_entry: Mapping[str, Any]) -> "EvidenceTrack":
        """Load an EvidenceTrack from NPZ + manifest entry."""

        with np.load(npz_path, allow_pickle=False) as data:
            values = data["values"]
            confidence = data["confidence"] if "confidence" in data else None

        tb_info = manifest_entry["timebase"]
        timebase = TimeBase.from_hop_seconds(tb_info["hop_s"], n_frames=tb_info.get("n_frames"))

        return EvidenceTrack(
            name=name,
            timebase=timebase,
            values=values,
            semantics=manifest_entry["semantics"],
            confidence=confidence,
            metadata=manifest_entry.get("metadata", {}),
        )
