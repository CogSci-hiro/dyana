"""`dyana decode` command implementation."""

from __future__ import annotations

import argparse


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `decode` command."""
    parser = subparsers.add_parser("decode", help="Run only decoding.")
    parser.set_defaults(command="decode")


def run(args: argparse.Namespace) -> None:
    """Execute the `decode` command."""
    del args
