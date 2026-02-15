"""`dyana evidence` command implementation."""

from __future__ import annotations

import argparse


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `evidence` command."""
    parser = subparsers.add_parser("evidence", help="Run evidence-only diagnostics.")
    parser.set_defaults(command="evidence")


def run(args: argparse.Namespace) -> None:
    """Execute the `evidence` command."""
    del args
