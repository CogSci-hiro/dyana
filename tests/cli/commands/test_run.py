import argparse

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
        )
    )
