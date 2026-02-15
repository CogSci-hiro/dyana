import argparse

from dyana.cli.commands import decode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    decode.add_subparser(subparsers)
    return parser


def test_add_subparser_registers_decode_command() -> None:
    parser = _parser()
    args = parser.parse_args(["decode"])
    assert args.command == "decode"


def test_run_handler_is_noop() -> None:
    decode.run(argparse.Namespace(command="decode"))
