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
            channel=None,
            vad_mode=2,
            smooth_ms=80.0,
            min_ipu_s=0.2,
            min_sil_s=0.1,
            ipu_mode="balanced",
            silence_bias=0.0,
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
    assert (tmp_path / "out" / "sample" / "decode" / "sample_states.npy").exists()


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
                channel=None,
                vad_mode=2,
                smooth_ms=80.0,
                min_ipu_s=0.2,
                min_sil_s=0.1,
                ipu_mode="balanced",
                silence_bias=0.0,
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
