from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from dyana.asr import (
    WhisperBackend,
    align_transcript_to_ipus,
    assign_speaker,
    build_asr_chunks,
    merge_transcripts,
    write_textgrid as write_transcript_textgrid,
)
from dyana.core.timebase import TimeBase
from dyana.decode import decoder, fusion
from dyana.decode.ipu import Segment, count_ipu_starts_after_leak, extract_ipus, merge_ipus_across_short_silence
from dyana.decode.params import DecodeTuningParams
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.diarization import compute_stereo_diarization_tracks
from dyana.evidence.energy import (
    compute_energy_rms_track,
    compute_energy_smooth_track,
    compute_energy_slope_track,
)
from dyana.evidence.overlap import compute_overlap_proxy_tracker
from dyana.evidence.pyannote import PyannoteEvidenceConfig, compute_pyannote_evidence
from dyana.evidence.prosody import compute_voiced_soft_track
from dyana.evidence.stereo import compute_stereo_evidence
from dyana.evidence.vad import compute_webrtc_vad_soft_track
from dyana.errors import BackendConfigurationError, OptionalBackendUnavailableError
from dyana.io.audio import load_audio_stereo
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
    ipu_detection_mode: str = "balanced",
    silence_bias: float = 0.0,
    merge_silence_gap_ms: float = 400.0,
    profile: str = "default",
    vad_backend: str = "webrtc",
    pyannote_config: PyannoteEvidenceConfig | None = None,
    error_mode: str = "run",
    seed: int = 0,
    tuning_params: DecodeTuningParams | None = None,
    channel: int | None = None,
    enable_asr: bool = False,
    asr_model: str = "small",
    asr_model_path: Path | None = None,
    asr_model_dir: Path | None = None,
    asr_language: str | None = None,
    ipus_path: Path | None = None,
) -> Dict[str, Any]:
    del seed  # deterministic; seed unused currently
    del min_sil_s  # reserved for future explicit silence post-processing

    effective_tuning_params = tuning_params or DecodeTuningParams.for_profile(profile)
    if tuning_params is None:
        effective_tuning_params = DecodeTuningParams(
            speaker_switch_penalty=effective_tuning_params.speaker_switch_penalty,
            leak_entry_bias=effective_tuning_params.leak_entry_bias,
            ovl_transition_cost=effective_tuning_params.ovl_transition_cost,
            a_to_ovl_cost=effective_tuning_params.a_to_ovl_cost,
            b_to_ovl_cost=effective_tuning_params.b_to_ovl_cost,
            ovl_to_a_cost=effective_tuning_params.ovl_to_a_cost,
            ovl_to_b_cost=effective_tuning_params.ovl_to_b_cost,
            ipu_detection_mode=ipu_detection_mode,
            silence_bias=silence_bias,
            merge_silence_gap_ms=merge_silence_gap_ms,
            speech_weight_vad=effective_tuning_params.speech_weight_vad,
            speech_weight_pyannote=effective_tuning_params.speech_weight_pyannote,
            speech_weight_energy=effective_tuning_params.speech_weight_energy,
            speech_weight_voiced=effective_tuning_params.speech_weight_voiced,
            none_when_speech_penalty=effective_tuning_params.none_when_speech_penalty,
            speech_exists_to_single_speaker_bonus=effective_tuning_params.speech_exists_to_single_speaker_bonus,
            speech_evidence_threshold=effective_tuning_params.speech_evidence_threshold,
            profile=effective_tuning_params.profile,
        )

    energy_rms = compute_energy_rms_track(audio_path, cache_dir=cache_dir, channel=channel)
    energy_smooth = compute_energy_smooth_track(
        audio_path, smooth_ms=smooth_ms, cache_dir=cache_dir, channel=channel
    )
    energy_slope = compute_energy_slope_track(
        audio_path, smooth_ms=smooth_ms, cache_dir=cache_dir, channel=channel
    )
    if vad_backend not in {"none", "webrtc", "pyannote", "all"}:
        raise BackendConfigurationError(
            f"Unsupported VAD backend selection '{vad_backend}'. Expected one of none|webrtc|pyannote|all."
        )

    vad_soft = None
    if vad_backend in {"webrtc", "all"}:
        vad_soft = compute_webrtc_vad_soft_track(
            audio_path, vad_mode=vad_mode, cache_dir=cache_dir, channel=channel
        )
    voiced_soft = compute_voiced_soft_track(
        audio_path, vad_mode=vad_mode, cache_dir=cache_dir, channel=channel
    )

    stereo_bundle = None
    if channel is None:
        try:
            stereo_audio, sample_rate = load_audio_stereo(audio_path)
        except ValueError:
            stereo_audio = None
        if stereo_audio is not None:
            stereo_bundle = compute_stereo_evidence((stereo_audio, sample_rate), energy_rms.timebase)

    diar_tracks = (
        None
        if channel is not None
        else compute_stereo_diarization_tracks(audio_path, cache_dir=cache_dir)
    )

    tb: TimeBase = energy_rms.timebase
    if pyannote_config is None:
        pyannote_config = PyannoteEvidenceConfig(enabled=vad_backend == "pyannote")
    elif vad_backend == "pyannote" and not pyannote_config.enabled:
        pyannote_config = PyannoteEvidenceConfig(
            enabled=True,
            model_name=pyannote_config.model_name,
            hf_token=pyannote_config.hf_token,
            device=pyannote_config.device,
            num_speakers=pyannote_config.num_speakers,
            min_speakers=pyannote_config.min_speakers,
            max_speakers=pyannote_config.max_speakers,
            pad_seconds=pyannote_config.pad_seconds,
            speech_track_name=pyannote_config.speech_track_name,
            speaker_track_prefix=pyannote_config.speaker_track_prefix,
        )
    pyannote_result = compute_pyannote_evidence(
        audio_path,
        tb,
        pyannote_config,
        cache_dir=cache_dir,
        error_mode=error_mode,
    )
    bundle = EvidenceBundle(timebase=tb)
    for tr in [energy_rms, energy_smooth, energy_slope, vad_soft, voiced_soft]:
        if tr is None:
            continue
        bundle.add_track(tr.name, tr)
    if stereo_bundle is not None:
        bundle = bundle.merge(stereo_bundle)
    if diar_tracks is not None:
        diar_a, diar_b = diar_tracks
        bundle.add_track(diar_a.name, diar_a)
        bundle.add_track(diar_b.name, diar_b)
        bundle.add_track("overlap_proxy", compute_overlap_proxy_tracker(diar_a, diar_b))
    if pyannote_result.bundle.tracks:
        bundle = bundle.merge(pyannote_result.bundle)

    scores = fusion.fuse_bundle_to_scores(bundle, tuning_params=effective_tuning_params)
    states = decoder.decode_with_constraints(scores, tuning_params=effective_tuning_params)

    if ipus_path is None:
        ipus_a = extract_ipus(states, tb, "A", min_duration_s=min_ipu_s)
        ipus_b = extract_ipus(states, tb, "B", min_duration_s=min_ipu_s)
        ipus_ovl = extract_ipus(states, tb, "OVL", min_duration_s=min_ipu_s)
        ipus_leak = extract_ipus(states, tb, "LEAK", min_duration_s=min_ipu_s)
        merge_gap_s = effective_tuning_params.merge_silence_gap_ms / 1000.0
        ipus_a = merge_ipus_across_short_silence(ipus_a, max_gap_s=merge_gap_s)
        ipus_b = merge_ipus_across_short_silence(ipus_b, max_gap_s=merge_gap_s)
        ipus_ovl = merge_ipus_across_short_silence(ipus_ovl, max_gap_s=merge_gap_s)
    else:
        imported_ipus = _load_external_ipus(ipus_path)
        ipus_a = [segment for segment in imported_ipus if segment.label == "A"]
        ipus_b = [segment for segment in imported_ipus if segment.label == "B"]
        ipus_ovl = [segment for segment in imported_ipus if segment.label == "OVL"]
        ipus_leak = [segment for segment in imported_ipus if segment.label == "LEAK"]
    total_ipus = len(ipus_a) + len(ipus_b) + len(ipus_ovl)
    pyannote_track = bundle.get("pyannote_speech")
    energy_none_diag = _count_evidence_against_silence(
        _normalize_track_probability(energy_smooth),
        states,
        threshold=effective_tuning_params.speech_evidence_threshold,
        hop_s=tb.hop_s,
    )
    pyannote_none_diag = _count_evidence_against_silence(
        _normalize_track_probability(pyannote_track),
        states,
        threshold=effective_tuning_params.speech_evidence_threshold,
        hop_s=tb.hop_s,
    )
    both_none_diag = _count_joint_evidence_against_silence(
        _normalize_track_probability(pyannote_track),
        _normalize_track_probability(energy_smooth),
        states,
        threshold=effective_tuning_params.speech_evidence_threshold,
        hop_s=tb.hop_s,
    )
    diagnostics = {
        "ipu_starts_after_leak": count_ipu_starts_after_leak(states),
        "total_ipus": total_ipus,
        "total_leak_segments": len(ipus_leak),
        "pyannote_enabled": bool(pyannote_config.enabled),
        "pyannote_status": pyannote_result.status,
        "pyannote_message": pyannote_result.message,
        "pyannote_num_speakers_detected": float(pyannote_result.backend_metadata.get("num_speakers_detected", 0)),
        "pyannote_speech_but_decoded_none_frames": float(pyannote_none_diag["frames"]),
        "pyannote_speech_but_decoded_none_seconds": pyannote_none_diag["seconds"],
        "energy_speech_but_decoded_none_frames": float(energy_none_diag["frames"]),
        "energy_speech_but_decoded_none_seconds": energy_none_diag["seconds"],
        "speech_evidence_but_none_frames": float(both_none_diag["frames"]),
        "speech_evidence_but_none_seconds": both_none_diag["seconds"],
    }
    speaker_durations = pyannote_result.backend_metadata.get("speaker_durations_seconds", {})
    for label, duration in speaker_durations.items():
        diagnostics[f"pyannote_speaker_duration_seconds_{label}"] = float(duration)

    stem = audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "evidence"
    decode_dir = out_dir / "decode"

    artifacts.save_evidence_track(energy_rms, evidence_dir / f"{stem}_energy_rms.npz")
    artifacts.save_evidence_track(energy_smooth, evidence_dir / f"{stem}_energy_smooth.npz")
    artifacts.save_evidence_track(energy_slope, evidence_dir / f"{stem}_energy_slope.npz")
    if vad_soft is not None:
        artifacts.save_evidence_track(vad_soft, evidence_dir / f"{stem}_vad_soft.npz")
    artifacts.save_evidence_track(voiced_soft, evidence_dir / f"{stem}_voiced_soft.npz")
    if stereo_bundle is not None:
        for name, track in stereo_bundle.items():
            artifacts.save_evidence_track(track, evidence_dir / f"{stem}_{name}.npz")
    if diar_tracks is not None:
        diar_a, diar_b = diar_tracks
        artifacts.save_evidence_track(diar_a, evidence_dir / f"{stem}_diar_a.npz")
        artifacts.save_evidence_track(diar_b, evidence_dir / f"{stem}_diar_b.npz")
    if pyannote_result.bundle.tracks:
        for name, track in pyannote_result.bundle.items():
            artifacts.save_evidence_track(track, evidence_dir / f"{stem}_{name}.npz")
        artifacts.save_json(
            [segment.__dict__ for segment in pyannote_result.segments],
            evidence_dir / f"{stem}_pyannote_segments.json",
        )

    artifacts.save_states(states, decode_dir / f"{stem}_states.npy")
    artifacts.save_json(
        [seg.__dict__ for seg in ipus_a + ipus_b + ipus_ovl + ipus_leak],
        decode_dir / f"{stem}_ipus.json",
    )
    artifacts.dump_diagnostics(out_dir, stem, diagnostics)

    asr_enabled = bool(enable_asr)
    transcript_payload: dict[str, Any] | None = None
    asr_chunk_payload: dict[str, list[dict[str, Any]]] | None = None
    if asr_enabled:
        audio_duration_seconds = len(states) * tb.hop_s
        speaker_transcripts = []
        asr_chunk_payload = {}
        speaker_channel_map = (
            {"A": 0, "B": 1, "OVL": None}
            if stereo_bundle is not None and channel is None
            else {"A": channel, "B": channel, "OVL": channel}
        )
        speaker_configs = {
            "A": {
                "chunks": build_asr_chunks(ipus_a, audio_duration_seconds),
                "channel": speaker_channel_map["A"],
                "ipus": ipus_a,
            },
            "B": {
                "chunks": build_asr_chunks(ipus_b, audio_duration_seconds),
                "channel": speaker_channel_map["B"],
                "ipus": ipus_b,
            },
            "OVL": {
                "chunks": build_asr_chunks(ipus_ovl, audio_duration_seconds),
                "channel": speaker_channel_map["OVL"],
                "ipus": ipus_ovl,
            },
        }

        for speaker_label, speaker_config in speaker_configs.items():
            speaker_chunks = speaker_config["chunks"]
            whisper_backend = WhisperBackend(
                model_name=asr_model,
                model_path=asr_model_path,
                model_dir=asr_model_dir,
                language=asr_language,
                audio_channel=speaker_config["channel"],
            )
            speaker_transcript = whisper_backend.transcribe_chunks(audio_path, speaker_chunks)
            aligned_speaker_transcript = align_transcript_to_ipus(
                speaker_transcript,
                speaker_config["ipus"],
                speaker=speaker_label,
            )
            speaker_transcripts.append(assign_speaker(aligned_speaker_transcript, speaker_label))
            asr_chunk_payload[speaker_label] = [
                {
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "ipu_indices": chunk.ipu_indices,
                }
                for chunk in speaker_chunks
            ]

        transcript = merge_transcripts(speaker_transcripts)
        transcript_payload = transcript.to_json()
        artifacts.save_json(asr_chunk_payload, out_dir / "asr_chunks.json")
        artifacts.save_json(transcript_payload, out_dir / "transcript.json")
        write_transcript_textgrid(transcript, out_dir / "transcript.TextGrid")

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
        "diagnostics": diagnostics,
        "stereo_diarization": diar_tracks is not None,
        "ipu_detection_mode": effective_tuning_params.ipu_detection_mode,
        "profile": effective_tuning_params.profile,
        "vad_backend": vad_backend,
        "pyannote_enabled": bool(pyannote_config.enabled),
        "pyannote_status": pyannote_result.status,
        "pyannote_message": pyannote_result.message,
        "pyannote_warnings": list(pyannote_result.warnings),
        "ipus_source": "external" if ipus_path is not None else "decoded",
        "asr_enabled": asr_enabled,
        "asr_model": asr_model if asr_enabled else None,
        "asr_language": asr_language if asr_enabled else None,
        "transcript": transcript_payload,
        "asr_chunks": asr_chunk_payload,
        "out_dir": str(out_dir),
}


def _normalize_track_probability(track: object | None) -> np.ndarray | None:
    if track is None:
        return None
    values = np.asarray(getattr(track, "values"), dtype=float)
    semantics = getattr(track, "semantics")
    if semantics == "probability":
        return values
    if semantics == "logit":
        return 1.0 / (1.0 + np.exp(-values))
    if semantics == "score":
        low = float(np.percentile(values, 10))
        high = float(np.percentile(values, 90))
        if high - low < 1e-6:
            return np.zeros_like(values, dtype=float)
        return np.clip((values - low) / (high - low), 0.0, 1.0)
    raise OptionalBackendUnavailableError(f"Unsupported evidence semantics '{semantics}'.")


def _count_evidence_against_silence(
    probs: np.ndarray | None,
    states: list[str],
    *,
    threshold: float,
    hop_s: float,
) -> dict[str, float]:
    if probs is None:
        return {"frames": 0.0, "seconds": 0.0}
    mask = (probs >= threshold) & (np.asarray(states) == "SIL")
    frames = float(int(mask.sum()))
    return {"frames": frames, "seconds": frames * hop_s}


def _count_joint_evidence_against_silence(
    probs_a: np.ndarray | None,
    probs_b: np.ndarray | None,
    states: list[str],
    *,
    threshold: float,
    hop_s: float,
) -> dict[str, float]:
    if probs_a is None or probs_b is None:
        return {"frames": 0.0, "seconds": 0.0}
    mask = (probs_a >= threshold) & (probs_b >= threshold) & (np.asarray(states) == "SIL")
    frames = float(int(mask.sum()))
    return {"frames": frames, "seconds": frames * hop_s}


def _load_external_ipus(path: Path) -> list[Segment]:
    if path.suffix.lower() == ".textgrid":
        return _load_external_ipus_from_textgrid(path)

    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            "External IPU file must be either a DYANA TextGrid or the current decode JSON format: "
            "[{\"start_time\": ..., \"end_time\": ..., \"label\": ...}, ...]."
        )

    segments = [
        Segment(
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            label=str(item["label"]),
        )
        for item in payload
    ]
    _validate_external_ipus(segments, path=path)
    return segments


def _load_external_ipus_from_textgrid(path: Path) -> list[Segment]:
    parsed_tiers = praat_textgrid.parse_textgrid(path)
    tier_to_label = {
        "SpeakerA": "A",
        "SpeakerB": "B",
        "Overlap": "OVL",
        "Leak": "LEAK",
    }
    segments = [
        Segment(start_time=segment.start_time, end_time=segment.end_time, label=tier_to_label[tier_name])
        for tier_name, label in tier_to_label.items()
        for segment in parsed_tiers.get(tier_name, [])
    ]
    segments.sort(key=lambda segment: (segment.start_time, segment.end_time, segment.label))
    _validate_external_ipus(segments, path=path)
    return segments


def _validate_external_ipus(segments: list[Segment], *, path: Path) -> None:
    allowed_labels = {"A", "B", "OVL", "LEAK"}
    previous_start = -1.0
    for segment in segments:
        if segment.label not in allowed_labels:
            raise ValueError(
                f"Unsupported IPU label '{segment.label}' in {path}. "
                "Expected one of A, B, OVL, LEAK."
            )
        if segment.end_time <= segment.start_time:
            raise ValueError(
                f"Invalid IPU interval in {path}: end_time must be greater than start_time "
                f"({segment.start_time}, {segment.end_time})."
            )
        if segment.start_time < previous_start:
            raise ValueError(f"External IPUs in {path} must be sorted by start_time.")
        previous_start = segment.start_time
