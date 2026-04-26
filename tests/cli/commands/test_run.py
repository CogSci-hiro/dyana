import argparse
import importlib
from pathlib import Path

import pytest

from dyana.cli.commands import run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run.add_subparser(subparsers)
    return parser


def test_add_subparser_registers_run_command() -> None:
    parser = _parser()
    args = parser.parse_args(["run", "--audio", "a.wav", "--out-dir", "o"])
    assert args.command == "run"


def test_add_subparser_accepts_positional_audio_path() -> None:
    parser = _parser()
    args = parser.parse_args(["run", "a.wav", "--out-dir", "o"])
    assert args.command == "run"
    assert args.audio_path == "a.wav"


def test_add_subparser_accepts_ipu_mode_and_silence_bias() -> None:
    parser = _parser()
    args = parser.parse_args(
        ["run", "--audio", "a.wav", "--out-dir", "o", "--ipu-mode", "high_recall", "--silence-bias", "-0.5"]
    )
    assert args.ipu_mode == "high_recall"
    assert args.silence_bias == -0.5


def test_add_subparser_accepts_pyannote_and_profile_flags() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "run",
            "--audio",
            "a.wav",
            "--out-dir",
            "o",
            "--pyannote",
            "--pyannote-model",
            "pyannote/speaker-diarization-3.1",
            "--pyannote-token",
            "hf_test",
            "--profile",
            "recall-first",
            "--vad-backend",
            "all",
        ]
    )
    assert args.pyannote is True
    assert args.pyannote_model == "pyannote/speaker-diarization-3.1"
    assert args.pyannote_token == "hf_test"
    assert args.profile == "recall-first"
    assert args.vad_backend == "all"


def test_add_subparser_accepts_asr_flags() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "run",
            "--audio",
            "a.wav",
            "--out-dir",
            "o",
            "--enable-asr",
            "--asr-model",
            "base",
            "--asr-model-path",
            "/tmp/base.pt",
            "--asr-model-dir",
            "/tmp/whisper",
            "--asr-language",
            "fr",
        ]
    )
    assert args.enable_asr is True
    assert args.asr_model == "base"
    assert args.asr_model_path == "/tmp/base.pt"
    assert args.asr_model_dir == "/tmp/whisper"
    assert args.asr_language == "fr"


def test_add_subparser_accepts_ipus_path() -> None:
    parser = _parser()
    args = parser.parse_args(["run", "--audio", "a.wav", "--out-dir", "o", "--ipus-path", "corrected.TextGrid"])
    assert args.ipus_path == "corrected.TextGrid"


def test_add_subparser_accepts_debug_flag() -> None:
    parser = _parser()
    args = parser.parse_args(["run", "--audio", "a.wav", "--out-dir", "o", "--debug"])
    assert args.debug is True


def test_run_handler_is_noop() -> None:
    run.run(
        argparse.Namespace(
            command="run",
            audio=None,
            audio_path=None,
            out_dir=None,
            cache_dir=None,
            ipus_path=None,
            channel=None,
            vad_mode=2,
            vad_backend="webrtc",
            smooth_ms=80.0,
            min_ipu_s=0.2,
            min_sil_s=0.1,
            ipu_mode="balanced",
            profile="default",
            silence_bias=0.0,
            pyannote=False,
            no_pyannote=False,
            pyannote_model="pyannote/speaker-diarization-3.1",
            pyannote_token=None,
            pyannote_device=None,
            pyannote_num_speakers=2,
            pyannote_min_speakers=None,
            pyannote_max_speakers=2,
            pyannote_pad_seconds=0.25,
            enable_asr=False,
            asr_model="small",
            asr_model_path=None,
            asr_model_dir=None,
            asr_language=None,
            debug=False,
        )
    )


def test_run_handler_executes_minimal_pipeline_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")

    def _fake_run_pipeline(audio_path: Path, *, out_dir: Path, **_: object) -> dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = out_dir / "decode" / f"{audio_path.stem}_states.npy"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"states")
        return {
            "n_frames": 12,
            "ipus": {"A": 1, "B": 0, "OVL": 0, "LEAK": 0},
            "pyannote_enabled": False,
            "pyannote_status": "disabled",
            "pyannote_message": None,
            "pyannote_warnings": [],
            "out_dir": str(out_dir),
        }

    run_pipeline_module = importlib.import_module("dyana.pipeline.run_pipeline")
    monkeypatch.setattr(run_pipeline_module, "run_pipeline", _fake_run_pipeline)

    run.run(
        argparse.Namespace(
            command="run",
            audio=str(audio),
            audio_path=None,
            out_dir=str(tmp_path / "out"),
            cache_dir=None,
            ipus_path=None,
            channel=None,
            vad_mode=2,
            vad_backend="webrtc",
            smooth_ms=80.0,
            min_ipu_s=0.2,
            min_sil_s=0.1,
            ipu_mode="balanced",
            profile="default",
            silence_bias=0.0,
            pyannote=False,
            no_pyannote=False,
            pyannote_model="pyannote/speaker-diarization-3.1",
            pyannote_token=None,
            pyannote_device=None,
            pyannote_num_speakers=2,
            pyannote_min_speakers=None,
            pyannote_max_speakers=2,
            pyannote_pad_seconds=0.25,
            enable_asr=False,
            asr_model="small",
            asr_model_path=None,
            asr_model_dir=None,
            asr_language=None,
            debug=False,
        )
    )

    captured = capsys.readouterr()
    assert "sample.wav: frames=12" in captured.out
    assert "pyannote=disabled" in captured.out
    assert (tmp_path / "out" / "sample" / "decode" / "sample_states.npy").exists()


def test_run_handler_warns_when_requested_pyannote_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "pyannote_skip.wav"
    audio.write_bytes(b"RIFF")

    def _fake_run_pipeline(audio_path: Path, *, out_dir: Path, **_: object) -> dict[str, object]:
        del audio_path
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "n_frames": 12,
            "ipus": {"A": 1, "B": 0, "OVL": 0, "LEAK": 0},
            "pyannote_enabled": True,
            "pyannote_status": "failed",
            "pyannote_message": "pyannote.audio is unavailable",
            "pyannote_warnings": [],
            "out_dir": str(out_dir),
        }

    run_pipeline_module = importlib.import_module("dyana.pipeline.run_pipeline")
    monkeypatch.setattr(run_pipeline_module, "run_pipeline", _fake_run_pipeline)

    run.run(
        argparse.Namespace(
            command="run",
            audio=str(audio),
            audio_path=None,
            out_dir=str(tmp_path / "out"),
            cache_dir=None,
            ipus_path=None,
            channel=None,
            vad_mode=2,
            vad_backend="webrtc",
            smooth_ms=80.0,
            min_ipu_s=0.2,
            min_sil_s=0.1,
            ipu_mode="balanced",
            profile="default",
            silence_bias=0.0,
            pyannote=True,
            no_pyannote=False,
            pyannote_model="pyannote/speaker-diarization-3.1",
            pyannote_token=None,
            pyannote_device=None,
            pyannote_num_speakers=2,
            pyannote_min_speakers=None,
            pyannote_max_speakers=2,
            pyannote_pad_seconds=0.25,
            enable_asr=False,
            asr_model="small",
            asr_model_path=None,
            asr_model_dir=None,
            asr_language=None,
            debug=False,
        )
    )

    captured = capsys.readouterr()
    assert "pyannote=failed (pyannote.audio is unavailable)" in captured.out
    assert "WARNING: pyannote was requested but is not being used" in captured.out
    assert "status=failed" in captured.out
    assert "pyannote.audio is unavailable" in captured.out


def test_run_handler_reports_failures_in_run_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "missing-dep.wav"
    audio.write_bytes(b"RIFF")

    def _fake_run_pipeline(audio_path: Path, *, out_dir: Path, **_: object) -> dict[str, object]:
        del audio_path, out_dir
        raise ModuleNotFoundError("No module named 'soundfile'")

    run_pipeline_module = importlib.import_module("dyana.pipeline.run_pipeline")
    monkeypatch.setattr(run_pipeline_module, "run_pipeline", _fake_run_pipeline)

    with pytest.raises(SystemExit, match="1"):
        run.run(
            argparse.Namespace(
                command="run",
                audio=str(audio),
                audio_path=None,
                out_dir=str(tmp_path / "out"),
                cache_dir=None,
                ipus_path=None,
                channel=None,
                vad_mode=2,
                vad_backend="webrtc",
                smooth_ms=80.0,
                min_ipu_s=0.2,
                min_sil_s=0.1,
                ipu_mode="balanced",
                profile="default",
                silence_bias=0.0,
                pyannote=False,
                no_pyannote=False,
                pyannote_model="pyannote/speaker-diarization-3.1",
                pyannote_token=None,
                pyannote_device=None,
                pyannote_num_speakers=2,
                pyannote_min_speakers=None,
                pyannote_max_speakers=2,
                pyannote_pad_seconds=0.25,
                enable_asr=False,
                asr_model="small",
                asr_model_path=None,
                asr_model_dir=None,
                asr_language=None,
                debug=False,
            )
        )

    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "soundfile" in captured.out
