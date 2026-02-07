# =============================================================================
#                                   CLI
# =============================================================================
#
# Entry point for the `dyana` command-line interface.
#
# This module is intentionally thin:
# - parse global + subcommand arguments
# - dispatch to exactly one command module
#
# All real logic must live in dyana.cli.commands.*
#
# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import argparse
from typing import Dict, Sequence

from dyana.cli.cli_types import CliCommand
from dyana.cli.commands import (
    run as cmd_run,
    eval as cmd_eval,
    inspect as cmd_inspect,
)

# =============================================================================
# Command registry
# =============================================================================

_COMMANDS: Dict[str, CliCommand] = {
    "run": cmd_run,
    "eval": cmd_eval,
    "inspect": cmd_inspect,
}

# =============================================================================
# Argument parsing
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the top-level DYANA CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.

    Usage example
    -------------
        dyana run --audio example.wav --out out/
        dyana eval --pred pred.TextGrid --ref ref.TextGrid
    """
    parser = argparse.ArgumentParser(
        prog="dyana",
        description="DYANA — DYadic Annotation of Naturalistic Audio",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="<command>",
    )

    for name, module in _COMMANDS.items():
        if not hasattr(module, "add_subparser"):
            raise RuntimeError(
                f"CLI command module '{name}' is missing add_subparser()."
            )
        module.add_subparser(subparsers)

    return parser

# =============================================================================
# Entry point
# =============================================================================

def main(argv: Sequence[str] | None = None) -> None:
    """
    DYANA CLI entry point.

    Parameters
    ----------
    argv
        Optional argv for testing. If None, reads from sys.argv.

    Returns
    -------
    None

    Usage example
    -------------
        main(["run", "--audio", "example.wav", "--out", "out/"])
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    command_name = str(args.command)
    module = _COMMANDS.get(command_name)

    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    if not hasattr(module, "run"):
        raise RuntimeError(
            f"CLI command module '{command_name}' is missing run()."
        )

    module.run(args)


if __name__ == "__main__":
    main()
