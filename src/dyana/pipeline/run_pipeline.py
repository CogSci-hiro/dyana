from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from dyana.core.timebase import TimeBase
from dyana.decode import decoder, fusion
from dyana.decode.ipu import extract_ipus
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.energy import (
    compute_energy_rms_track,
    compute_energy_smooth_track,
    compute_energy_slope_track,
)
from dyana.evidence.prosody import compute_voiced_soft_track
from dyana.evidence.vad import compute_webrtc_vad_soft_track
from dyana.io import artifacts, praat_textgrid


def run_pipeline(
    audio_path: Path,
    *,
    out_dir: Path,
    cache_dir: Path | None = None,
    vad_mode: int = 2,
    smooth_ms: float = 80.0,
    min_ipu_s: float = 0.2,
    min_sil_s: float = 0.1,
    seed: int = 0,
) -> Dict[str, Any]:
    del seed  # deterministic; seed unused currently

    energy_rms = compute_energy_rms_track(audio_path, cache_dir=cache_dir)
    energy_smooth = compute_energy_smooth_track(audio_path, smooth_ms=smooth_ms, cache_dir=cache_dir)
    energy_slope = compute_energy_slope_track(audio_path, smooth_ms=smooth_ms, cache_dir=cache_dir)
    vad_soft = compute_webrtc_vad_soft_track(audio_path, vad_mode=vad_mode, cache_dir=cache_dir)
    voiced_soft = compute_voiced_soft_track(audio_path, vad_mode=vad_mode, cache_dir=cache_dir)

    tb: TimeBase = energy_rms.timebase
    bundle = EvidenceBundle(timebase=tb)
    for tr in [energy_rms, energy_smooth, energy_slope, vad_soft, voiced_soft]:
        bundle.add_track(tr.name, tr)

    scores = fusion.fuse_bundle_to_scores(bundle)
    states = decoder.decode_with_constraints(scores)

    ipus_a = extract_ipus(states, tb, "A", min_duration_s=min_ipu_s)
    ipus_b = extract_ipus(states, tb, "B", min_duration_s=min_ipu_s)
    ipus_ovl = extract_ipus(states, tb, "OVL", min_duration_s=min_ipu_s)
    ipus_leak = extract_ipus(states, tb, "LEAK", min_duration_s=min_ipu_s)

    stem = audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "evidence"
    decode_dir = out_dir / "decode"

    artifacts.save_evidence_track(energy_rms, evidence_dir / f"{stem}_energy_rms.npz")
    artifacts.save_evidence_track(energy_smooth, evidence_dir / f"{stem}_energy_smooth.npz")
    artifacts.save_evidence_track(energy_slope, evidence_dir / f"{stem}_energy_slope.npz")
    artifacts.save_evidence_track(vad_soft, evidence_dir / f"{stem}_vad_soft.npz")
    artifacts.save_evidence_track(voiced_soft, evidence_dir / f"{stem}_voiced_soft.npz")

    artifacts.save_states(states, decode_dir / f"{stem}_states.npy")
    artifacts.save_json([seg.__dict__ for seg in ipus_a + ipus_b + ipus_ovl + ipus_leak], decode_dir / f"{stem}_ipus.json")

    praat_textgrid.write_textgrid(
        out_dir / f"{stem}.TextGrid",
        speaker_a=ipus_a,
        speaker_b=ipus_b,
        overlap=ipus_ovl,
        leak=ipus_leak,
    )

    return {
        "audio": str(audio_path),
        "timebase_hop": tb.hop_s,
        "n_frames": len(states),
        "ipus": {
            "A": len(ipus_a),
            "B": len(ipus_b),
            "OVL": len(ipus_ovl),
            "LEAK": len(ipus_leak),
        },
        "out_dir": str(out_dir),
    }
