from __future__ import annotations

from types import SimpleNamespace

import pytest

from dyana.cli import main as cli_main


def test_build_arg_parser_accepts_all_registered_commands() -> None:
    parser = cli_main.build_arg_parser()
    for command in ("run", "decode", "evidence", "iterate"):
        extra = ["--audio", "a.wav", "--out-dir", "o"] if command == "run" else []
        args = parser.parse_args([command] + extra)
        assert args.command == command


def test_build_arg_parser_requires_add_subparser(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_module = SimpleNamespace(run=lambda _args: None)
    monkeypatch.setattr(cli_main, "_COMMANDS", {"bad": bad_module})
    with pytest.raises(RuntimeError, match="missing add_subparser"):
        cli_main.build_arg_parser()


def test_main_dispatches_to_selected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def _add_subparser(subparsers: object) -> None:
        parser = subparsers.add_parser("fake")  # type: ignore[attr-defined]
        parser.set_defaults(command="fake")

    def _run(args: object) -> None:
        called.append(str(args.command))  # type: ignore[attr-defined]

    fake_module = SimpleNamespace(add_subparser=_add_subparser, run=_run)
    monkeypatch.setattr(cli_main, "_COMMANDS", {"fake": fake_module})

    cli_main.main(["fake"])
    assert called == ["fake"]


def test_main_requires_run_function(monkeypatch: pytest.MonkeyPatch) -> None:
    def _add_subparser(subparsers: object) -> None:
        parser = subparsers.add_parser("fake")  # type: ignore[attr-defined]
        parser.set_defaults(command="fake")

    fake_module = SimpleNamespace(add_subparser=_add_subparser)
    monkeypatch.setattr(cli_main, "_COMMANDS", {"fake": fake_module})

    with pytest.raises(RuntimeError, match="missing run"):
        cli_main.main(["fake"])
