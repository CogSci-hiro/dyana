"""DYANA command-line interface entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from types import ModuleType

from dyana.cli.commands import decode, evidence, iterate, run, eval as eval_cmd

CommandModule = ModuleType
CommandRunner = Callable[[argparse.Namespace], None]

# Map CLI subcommands to their implementation modules.
_COMMANDS: dict[str, CommandModule] = {
    "run": run,
    "decode": decode,
    "evidence": evidence,
    "iterate": iterate,
    "eval": eval_cmd,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="dyana",
        description="DYANA - DYadic Annotation of Naturalistic Audio",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name, module in _COMMANDS.items():
        add_subparser = getattr(module, "add_subparser", None)
        if add_subparser is None:
            raise RuntimeError(f"CLI command module '{name}' is missing add_subparser().")
        add_subparser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse args and dispatch to the selected command implementation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_name = str(args.command)

    module = _COMMANDS.get(command_name)
    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    runner: CommandRunner | None = getattr(module, "run", None)
    if runner is None:
        raise RuntimeError(f"CLI command module '{command_name}' is missing run().")
    runner(args)


# Keep console script compatibility with pyproject's entrypoint.
app = main


if __name__ == "__main__":
    main()
