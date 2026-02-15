# src/dyana/evidence/base.py

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dyana.core.resample import AggKind, downsample, to_canonical_grid, upsample_hold_last
from dyana.core.timebase import CANONICAL_HOP_S, TimeBase


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
            if values.ndim == 2 and confidence.ndim == 2 and confidence.shape[1] != values.shape[1]:
                raise ValueError(
                    f"EvidenceTrack.confidence K mismatch for '{self.name}': "
                    f"values has K={values.shape[1]}, confidence has K={confidence.shape[1]}."
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

    def to_canonical(self, *, downsample_agg: Optional[AggKind] = None) -> "EvidenceTrack":
        """
        Return an EvidenceTrack on the canonical 10 ms grid.

        Parameters
        ----------
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

        if self.timebase.is_canonical:
            return self

        values_resamp = to_canonical_grid(
            self.values,
            src_hop_s=self.timebase.hop_s,
            semantics=self.semantics,
            downsample_agg=downsample_agg,
        )

        confidence_resamp = None
        if self.confidence is not None:
            if self.timebase.hop_s > CANONICAL_HOP_S:
                confidence_resamp = upsample_hold_last(self.confidence, self.timebase.hop_s, CANONICAL_HOP_S)
            else:
                if downsample_agg is None:
                    raise ValueError("downsample_agg required to resample confidence")
                confidence_resamp = downsample(self.confidence, self.timebase.hop_s, CANONICAL_HOP_S, agg=downsample_agg)

        canonical_tb = TimeBase.canonical(n_frames=values_resamp.shape[0])

        return EvidenceTrack(
            name=self.name,
            timebase=canonical_tb,
            values=values_resamp,
            semantics=self.semantics,
            confidence=confidence_resamp,
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
    def from_files(name: str, npz_path: Path, manifest_entry: Mapping[str, Any]) -> "EvidenceTrack":
        """Load an EvidenceTrack from NPZ + manifest entry."""

        with np.load(npz_path) as data:
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

    def merge(self, other: "EvidenceBundle") -> "EvidenceBundle":
        """
        Merge another bundle; other tracks override duplicates.
        """

        if abs(other.timebase.hop_s - self.timebase.hop_s) > 1e-12:
            raise ValueError(
                f"Cannot merge bundles with different hops: {self.timebase.hop_s} vs {other.timebase.hop_s}."
            )

        merged = dict(self.tracks)
        merged.update(other.tracks)
        return EvidenceBundle(timebase=self.timebase, require_canonical=self.require_canonical, tracks=merged)

    def to_directory(self, path: Path) -> None:
        """
        Serialize bundle to a directory with manifest and per-track NPZ files.

        Structure
        ---------
        - manifest.json: lists track metadata and filenames
        - <name>.npz: arrays for each track
        """

        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "timebase": {"hop_s": self.timebase.hop_s, "n_frames": self.timebase.n_frames},
            "tracks": {},
        }

        for name, track in self.tracks.items():
            track_path = path / f"{name}.npz"
            track.to_npz(track_path)
            manifest["tracks"][name] = track.to_manifest()
            manifest["tracks"][name]["file"] = track_path.name

        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def from_directory(path: Path) -> "EvidenceBundle":
        """Load a bundle from directory created by :meth:`to_directory`."""

        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        tb_info = manifest["timebase"]
        timebase = TimeBase.from_hop_seconds(tb_info["hop_s"], n_frames=tb_info.get("n_frames"))

        tracks: Dict[str, EvidenceTrack] = {}
        for name, entry in manifest["tracks"].items():
            npz_path = path / entry["file"]
            tracks[name] = EvidenceTrack.from_files(name, npz_path, entry)

        return EvidenceBundle(timebase=timebase, require_canonical=False, tracks=tracks)
