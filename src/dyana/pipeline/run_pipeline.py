from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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
from dyana.decode.ipu import count_ipu_starts_after_leak, extract_ipus, merge_ipus_across_short_silence
from dyana.decode.params import DecodeTuningParams
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.diarization import compute_stereo_diarization_tracks
from dyana.evidence.energy import (
    compute_energy_rms_track,
    compute_energy_smooth_track,
    compute_energy_slope_track,
)
from dyana.evidence.overlap import compute_overlap_proxy_tracker
from dyana.evidence.prosody import compute_voiced_soft_track
from dyana.evidence.stereo import compute_stereo_evidence
from dyana.evidence.vad import compute_webrtc_vad_soft_track
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
    seed: int = 0,
    tuning_params: DecodeTuningParams | None = None,
    channel: int | None = None,
    enable_asr: bool = False,
    asr_model: str = "small",
    asr_model_path: Path | None = None,
    asr_model_dir: Path | None = None,
    asr_language: str | None = None,
) -> Dict[str, Any]:
    del seed  # deterministic; seed unused currently
    del min_sil_s  # reserved for future explicit silence post-processing

    effective_tuning_params = tuning_params or DecodeTuningParams()
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
        )

    energy_rms = compute_energy_rms_track(audio_path, cache_dir=cache_dir, channel=channel)
    energy_smooth = compute_energy_smooth_track(
        audio_path, smooth_ms=smooth_ms, cache_dir=cache_dir, channel=channel
    )
    energy_slope = compute_energy_slope_track(
        audio_path, smooth_ms=smooth_ms, cache_dir=cache_dir, channel=channel
    )
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
    bundle = EvidenceBundle(timebase=tb)
    for tr in [energy_rms, energy_smooth, energy_slope, vad_soft, voiced_soft]:
        bundle.add_track(tr.name, tr)
    if stereo_bundle is not None:
        bundle = bundle.merge(stereo_bundle)
    if diar_tracks is not None:
        diar_a, diar_b = diar_tracks
        bundle.add_track(diar_a.name, diar_a)
        bundle.add_track(diar_b.name, diar_b)
        bundle.add_track("overlap_proxy", compute_overlap_proxy_tracker(diar_a, diar_b))

    scores = fusion.fuse_bundle_to_scores(bundle, tuning_params=effective_tuning_params)
    states = decoder.decode_with_constraints(scores, tuning_params=effective_tuning_params)

    ipus_a = extract_ipus(states, tb, "A", min_duration_s=min_ipu_s)
    ipus_b = extract_ipus(states, tb, "B", min_duration_s=min_ipu_s)
    ipus_ovl = extract_ipus(states, tb, "OVL", min_duration_s=min_ipu_s)
    ipus_leak = extract_ipus(states, tb, "LEAK", min_duration_s=min_ipu_s)
    merge_gap_s = effective_tuning_params.merge_silence_gap_ms / 1000.0
    ipus_a = merge_ipus_across_short_silence(ipus_a, max_gap_s=merge_gap_s)
    ipus_b = merge_ipus_across_short_silence(ipus_b, max_gap_s=merge_gap_s)
    ipus_ovl = merge_ipus_across_short_silence(ipus_ovl, max_gap_s=merge_gap_s)
    total_ipus = len(ipus_a) + len(ipus_b) + len(ipus_ovl)
    diagnostics = {
        "ipu_starts_after_leak": count_ipu_starts_after_leak(states),
        "total_ipus": total_ipus,
        "total_leak_segments": len(ipus_leak),
    }

    stem = audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "evidence"
    decode_dir = out_dir / "decode"

    artifacts.save_evidence_track(energy_rms, evidence_dir / f"{stem}_energy_rms.npz")
    artifacts.save_evidence_track(energy_smooth, evidence_dir / f"{stem}_energy_smooth.npz")
    artifacts.save_evidence_track(energy_slope, evidence_dir / f"{stem}_energy_slope.npz")
    artifacts.save_evidence_track(vad_soft, evidence_dir / f"{stem}_vad_soft.npz")
    artifacts.save_evidence_track(voiced_soft, evidence_dir / f"{stem}_voiced_soft.npz")
    if stereo_bundle is not None:
        for name, track in stereo_bundle.items():
            artifacts.save_evidence_track(track, evidence_dir / f"{stem}_{name}.npz")
    if diar_tracks is not None:
        diar_a, diar_b = diar_tracks
        artifacts.save_evidence_track(diar_a, evidence_dir / f"{stem}_diar_a.npz")
        artifacts.save_evidence_track(diar_b, evidence_dir / f"{stem}_diar_b.npz")

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
        "asr_enabled": asr_enabled,
        "asr_model": asr_model if asr_enabled else None,
        "asr_language": asr_language if asr_enabled else None,
        "transcript": transcript_payload,
        "asr_chunks": asr_chunk_payload,
        "out_dir": str(out_dir),
    }
