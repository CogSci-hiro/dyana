"""`dyana run` command implementation."""

from __future__ import annotations

import argparse


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `run` command."""
    parser = subparsers.add_parser("run", help="Run DYANA end-to-end.")
    parser.set_defaults(command="run")


def run(args: argparse.Namespace) -> None:
    """Execute the `run` command."""
    del args
