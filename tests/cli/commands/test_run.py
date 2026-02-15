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


def test_run_handler_is_noop() -> None:
    run.run(argparse.Namespace(command="run", audio=None, out_dir=None, cache_dir=None))
