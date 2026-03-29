from __future__ import annotations

import argparse
from pathlib import Path

from dyana.cli.commands import asr_setup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    asr_setup.add_subparser(subparsers)
    return parser


def test_add_subparser_accepts_asr_setup_arguments() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "asr-setup",
            "--model",
            "base",
            "--asr-model-path",
            "/tmp/base.pt",
            "--asr-model-dir",
            "/tmp/whisper",
            "--language",
            "fr",
        ]
    )
    assert args.command == "asr-setup"
    assert args.model == "base"
    assert args.asr_model_path == "/tmp/base.pt"
    assert args.asr_model_dir == "/tmp/whisper"
    assert args.language == "fr"


def test_run_reports_expected_checkpoint_path(tmp_path: Path, capsys) -> None:
    checkpoint_dir = tmp_path / "whisper"
    checkpoint_dir.mkdir()

    asr_setup.run(
        argparse.Namespace(
            command="asr-setup",
            model="small",
            asr_model_path=None,
            asr_model_dir=str(checkpoint_dir),
            language="fr",
        )
    )

    output = capsys.readouterr().out
    assert "expected_checkpoint" in output
    assert str(checkpoint_dir / "small.pt") in output
    assert "language: fr" in output
