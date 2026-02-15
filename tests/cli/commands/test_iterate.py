import argparse

from dyana.cli.commands import iterate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    iterate.add_subparser(subparsers)
    return parser


def test_add_subparser_registers_iterate_command() -> None:
    parser = _parser()
    args = parser.parse_args(["iterate"])
    assert args.command == "iterate"


def test_run_handler_is_noop() -> None:
    iterate.run(argparse.Namespace(command="iterate"))
