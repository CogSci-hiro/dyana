# src/dyana/evidence/bundle.py

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from dyana.core.resample import AggKind
from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase
from dyana.evidence.base import EvidenceTrack


# ##########  EvidenceBundle  ##########


@dataclass
class EvidenceBundle:
    """
    Order-independent collection of EvidenceTracks on a shared timebase.

    Parameters
    ----------
    timebase
        Bundle timebase. All tracks must match this hop.
    require_canonical
        If True, enforce that the bundle uses the canonical 10 ms grid.
    tracks
        Optional initial mapping of tracks (by name).

    Usage example
    -------------
        tb = TimeBase.canonical()
        bundle = EvidenceBundle(timebase=tb)
        bundle.add_track("vad", vad_track)
    """

    timebase: TimeBase
    require_canonical: bool = True
    tracks: Dict[str, EvidenceTrack] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.require_canonical and abs(self.timebase.hop_s - CANONICAL_HOP_SECONDS) > 1e-12:
            raise ValueError(
                f"EvidenceBundle requires canonical hop {CANONICAL_HOP_SECONDS}, got {self.timebase.hop_s}."
            )

        for name, track in list(self.tracks.items()):
            self._validate_track(name, track)

    def _validate_track(self, name: str, track: EvidenceTrack) -> None:
        if abs(track.timebase.hop_s - self.timebase.hop_s) > 1e-12:
            raise ValueError(
                f"Track '{name}' has hop {track.timebase.hop_s}, bundle hop is {self.timebase.hop_s}."
            )
        if self.timebase.n_frames is not None and track.timebase.n_frames not in (None, self.timebase.n_frames):
            raise ValueError(
                f"Track '{name}' has n_frames={track.timebase.n_frames}, bundle requires {self.timebase.n_frames}."
            )
        if self.require_canonical and not track.timebase.is_canonical:
            raise ValueError(f"Track '{name}' is not on canonical timebase; convert before adding.")

    # Public API
    def add_track(self, name: str, track: EvidenceTrack) -> None:
        """Add or replace a track by name (mutates bundle)."""

        self._validate_track(name, track)
        self.tracks[name] = track

    def get(self, name: str, default: Optional[EvidenceTrack] = None) -> Optional[EvidenceTrack]:
        """Return a track by name, or default if missing."""

        return self.tracks.get(name, default)

    def keys(self):
        return self.tracks.keys()

    def items(self):
        return self.tracks.items()

    def __iter__(self) -> Iterable[EvidenceTrack]:
        return iter(self.tracks.values())

    def merge(self, other: "EvidenceBundle") -> "EvidenceBundle":
        """
        Merge two bundles; tracks in ``other`` override duplicates.

        Raises
        ------
        ValueError
            If timebases are incompatible.

        Usage example
        -------------
            merged = b1.merge(b2)
        """

        if abs(other.timebase.hop_s - self.timebase.hop_s) > 1e-12:
            raise ValueError(
                f"Cannot merge bundles with different hops: {self.timebase.hop_s} vs {other.timebase.hop_s}."
            )

        merged_tracks = dict(self.tracks)
        merged_tracks.update(other.tracks)
        return EvidenceBundle(
            timebase=self.timebase,
            require_canonical=self.require_canonical,
            tracks=merged_tracks,
        )

    def resample_all_to(
        self,
        timebase: TimeBase,
        *,
        agg_map: dict[str, AggKind] | None = None,
        default_downsample_agg: AggKind | None = None,
    ) -> "EvidenceBundle":
        """
        Resample all tracks to ``timebase``. Returns a new bundle.

        Parameters
        ----------
        timebase
            Target timebase for all tracks.
        agg_map
            Optional per-track aggregation for downsampling.
        default_downsample_agg
            Fallback aggregation if a track needs downsampling and is not in ``agg_map``.
        """

        agg_map = agg_map or {}
        new_tracks: Dict[str, EvidenceTrack] = {}
        for name, track in self.tracks.items():
            needs_downsample = track.timebase.hop_s < timebase.hop_s
            agg = agg_map.get(name, default_downsample_agg)
            if needs_downsample and agg is None:
                raise ValueError(f"Downsampling track '{name}' requires agg (mean|max).")
            new_tracks[name] = track.resample_to(timebase, agg=agg)

        return EvidenceBundle(timebase=timebase, require_canonical=self.require_canonical, tracks=new_tracks)

    # ---------- Serialization ----------
    def to_directory(self, path: Path) -> None:
        """
        Serialize bundle to a directory with manifest and per-track NPZ files.

        Structure
        ---------
        - manifest.json: lists track metadata and filenames
        - <name>.npz: arrays for each track

        Usage example
        -------------
            bundle.to_directory(Path("/tmp/bundle"))
        """

        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "timebase": {"hop_s": self.timebase.hop_s, "n_frames": self.timebase.n_frames},
            "tracks": {},
        }

        for name, track in self.tracks.items():
            track_path = path / f"{name}.npz"
            track.to_npz(track_path)
            manifest_entry = track.to_manifest()
            manifest_entry["file"] = track_path.name
            manifest["tracks"][name] = manifest_entry

        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def from_directory(path: Path) -> "EvidenceBundle":
        """Load a bundle previously written with :meth:`to_directory`."""

        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        tb_info = manifest["timebase"]
        timebase = TimeBase.from_hop_seconds(tb_info["hop_s"], n_frames=tb_info.get("n_frames"))

        tracks: Dict[str, EvidenceTrack] = {}
        for name, entry in manifest["tracks"].items():
            npz_path = path / entry["file"]
            tracks[name] = EvidenceTrack.from_files(name, npz_path, entry)

        return EvidenceBundle(timebase=timebase, require_canonical=False, tracks=tracks)
