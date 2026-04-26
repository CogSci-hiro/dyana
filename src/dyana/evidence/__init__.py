"""Evidence extraction modules for DYANA."""

from __future__ import annotations

import importlib

from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle

__all__ = [
    "EvidenceTrack",
    "EvidenceBundle",
    "PyannoteEvidenceConfig",
    "PyannoteEvidenceResult",
    "compute_pyannote_evidence",
    "synthetic",
    "compute_leakage_likelihood",
    "compute_overlap_proxy_tracker",
    "compute_stereo_evidence",
]


def __getattr__(name: str):
    if name == "synthetic":
        return importlib.import_module("dyana.evidence.synthetic")
    if name == "compute_leakage_likelihood":
        from dyana.evidence.leakage import compute_leakage_likelihood

        return compute_leakage_likelihood
    if name == "compute_overlap_proxy_tracker":
        from dyana.evidence.overlap import compute_overlap_proxy_tracker

        return compute_overlap_proxy_tracker
    if name == "compute_stereo_evidence":
        from dyana.evidence.stereo import compute_stereo_evidence

        return compute_stereo_evidence
    if name in {"PyannoteEvidenceConfig", "PyannoteEvidenceResult", "compute_pyannote_evidence"}:
        from dyana.evidence.pyannote import (
            PyannoteEvidenceConfig,
            PyannoteEvidenceResult,
            compute_pyannote_evidence,
        )

        mapping = {
            "PyannoteEvidenceConfig": PyannoteEvidenceConfig,
            "PyannoteEvidenceResult": PyannoteEvidenceResult,
            "compute_pyannote_evidence": compute_pyannote_evidence,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
