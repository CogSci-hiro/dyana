import argparse

from dyana.cli.commands import evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evidence.add_subparser(subparsers)
    return parser


def test_add_subparser_registers_evidence_command() -> None:
    parser = _parser()
    args = parser.parse_args(["evidence"])
    assert args.command == "evidence"


def test_run_handler_is_noop() -> None:
    evidence.run(argparse.Namespace(command="evidence"))
