from pathlib import Path

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from dyana.asr.transcript import Transcript, TranscriptSegment, WordTimestamp
from dyana.pipeline.run_pipeline import run_pipeline


sf = pytest.importorskip("soundfile")
pytest.importorskip("webrtcvad")


def _make_audio(path: Path) -> None:
    sr = 16000
    t = np.arange(0, sr * 1.0) / sr
    sig = np.concatenate([np.zeros(sr // 4), 0.05 * np.sin(2 * np.pi * 200 * t[: sr // 2]), np.zeros(sr // 4)])
    sf.write(path, sig, sr)


def test_pipeline_runs_and_writes_outputs(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    _make_audio(audio)
    out_dir = tmp_path / "out1"
    summary = run_pipeline(audio, out_dir=out_dir)
    assert (out_dir / "a.TextGrid").exists()
    assert (out_dir / "evidence" / "a_vad_soft.npz").exists()
    assert summary["n_frames"] > 0


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    audio = tmp_path / "b.wav"
    _make_audio(audio)
    out1 = tmp_path / "out_a"
    out2 = tmp_path / "out_b"
    run_pipeline(audio, out_dir=out1)
    run_pipeline(audio, out_dir=out2)
    states1 = np.load(out1 / "decode" / "b_states.npy", allow_pickle=True)
    states2 = np.load(out2 / "decode" / "b_states.npy", allow_pickle=True)
    assert np.array_equal(states1, states2)


def test_pipeline_uses_stereo_channels_for_two_speakers(tmp_path: Path) -> None:
    sr = 16000
    left = np.concatenate([0.06 * np.ones(sr // 2), np.zeros(sr // 2)]).astype(np.float32)
    right = np.concatenate([np.zeros(sr // 2), 0.06 * np.ones(sr // 2)]).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    audio = tmp_path / "stereo.wav"
    sf.write(audio, stereo, sr)

    out_dir = tmp_path / "stereo_out"
    summary = run_pipeline(audio, out_dir=out_dir)
    states = np.load(out_dir / "decode" / "stereo_states.npy", allow_pickle=True)

    assert summary["stereo_diarization"] is True
    assert summary["ipus"]["A"] >= 1
    assert summary["ipus"]["B"] >= 1
    assert "A" in states.tolist()
    assert "B" in states.tolist()
    assert (out_dir / "evidence" / "stereo_stereo_energy_left.npz").exists()
    assert (out_dir / "evidence" / "stereo_stereo_energy_right.npz").exists()
    assert (out_dir / "evidence" / "stereo_stereo_ratio.npz").exists()
    assert (out_dir / "evidence" / "stereo_stereo_corr.npz").exists()
    assert (out_dir / "evidence" / "stereo_diar_a.npz").exists()
    assert (out_dir / "evidence" / "stereo_diar_b.npz").exists()


def test_pipeline_writes_asr_artifacts_when_enabled(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    audio = tmp_path / "asr.wav"
    _make_audio(audio)
    out_dir = tmp_path / "asr_out"

    def _fake_transcribe_chunks(self, audio_path: Path, chunks: list[object]) -> Transcript:
        del self, audio_path, chunks
        return Transcript(
            segments=[
                TranscriptSegment(
                    start_time=0.0,
                    end_time=0.5,
                    text="hello",
                    words=[WordTimestamp(word="hello", start_time=0.0, end_time=0.5, confidence=0.9)],
                )
            ]
        )

    monkeypatch.setattr(
        "dyana.pipeline.run_pipeline.WhisperBackend.transcribe_chunks",
        _fake_transcribe_chunks,
    )

    summary = run_pipeline(audio, out_dir=out_dir, enable_asr=True, asr_model="tiny")

    assert summary["asr_enabled"] is True
    assert summary["asr_model"] == "tiny"
    assert (out_dir / "asr_chunks.json").exists()
    assert (out_dir / "transcript.json").exists()
    assert (out_dir / "transcript.TextGrid").exists()
    assert summary["transcript"] is not None
    assert all("speaker" in segment for segment in summary["transcript"]["segments"])


def test_pipeline_passes_local_whisper_paths(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    audio = tmp_path / "asr_paths.wav"
    _make_audio(audio)
    out_dir = tmp_path / "asr_paths_out"
    captured_inits: list[dict[str, Path | str | None]] = []

    def _fake_init(
        self,
        model_name: str,
        device: str = "auto",
        model_path: Path | None = None,
        model_dir: Path | None = None,
        language: str | None = None,
        audio_channel: int | None = None,
        show_progress: bool = True,
    ) -> None:
        captured_inits.append(
            {
                "model_name": model_name,
                "device": device,
                "model_path": model_path,
                "model_dir": model_dir,
                "language": language,
                "audio_channel": audio_channel,
                "show_progress": show_progress,
            }
        )

    def _fake_transcribe_chunks(self, audio_path: Path, chunks: list[object]) -> Transcript:
        del self, audio_path, chunks
        return Transcript(segments=[])

    monkeypatch.setattr("dyana.pipeline.run_pipeline.WhisperBackend.__init__", _fake_init)
    monkeypatch.setattr(
        "dyana.pipeline.run_pipeline.WhisperBackend.transcribe_chunks",
        _fake_transcribe_chunks,
    )

    run_pipeline(
        audio,
        out_dir=out_dir,
        enable_asr=True,
        asr_model="base",
        asr_model_path=tmp_path / "base.pt",
        asr_model_dir=tmp_path / "cache",
        asr_language="fr",
    )

    assert {capture["model_name"] for capture in captured_inits} == {"base"}
    assert {capture["model_path"] for capture in captured_inits} == {tmp_path / "base.pt"}
    assert {capture["model_dir"] for capture in captured_inits} == {tmp_path / "cache"}
    assert {capture["language"] for capture in captured_inits} == {"fr"}


def test_pipeline_routes_ipu_asr_to_stereo_speaker_channels(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    sr = 16000
    left = np.concatenate([0.06 * np.ones(sr // 2), np.zeros(sr // 2)]).astype(np.float32)
    right = np.concatenate([np.zeros(sr // 2), 0.06 * np.ones(sr // 2)]).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    audio = tmp_path / "stereo_asr.wav"
    sf.write(audio, stereo, sr)

    captured_channels: list[int | None] = []

    def _fake_init(
        self,
        model_name: str,
        device: str = "auto",
        model_path: Path | None = None,
        model_dir: Path | None = None,
        language: str | None = None,
        audio_channel: int | None = None,
        show_progress: bool = True,
    ) -> None:
        del self, model_name, device, model_path, model_dir, language, show_progress
        captured_channels.append(audio_channel)

    def _fake_transcribe_chunks(self, audio_path: Path, chunks: list[object]) -> Transcript:
        del self, audio_path, chunks
        return Transcript(segments=[])

    monkeypatch.setattr("dyana.pipeline.run_pipeline.WhisperBackend.__init__", _fake_init)
    monkeypatch.setattr(
        "dyana.pipeline.run_pipeline.WhisperBackend.transcribe_chunks",
        _fake_transcribe_chunks,
    )

    run_pipeline(audio, out_dir=tmp_path / "stereo_asr_out", enable_asr=True)

    assert 0 in captured_channels
    assert 1 in captured_channels
