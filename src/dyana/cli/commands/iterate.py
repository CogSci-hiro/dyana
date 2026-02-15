"""`dyana iterate` command implementation."""

from __future__ import annotations

import argparse


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `iterate` command."""
    parser = subparsers.add_parser("iterate", help="Run iterative refinement only.")
    parser.set_defaults(command="iterate")


def run(args: argparse.Namespace) -> None:
    """Execute the `iterate` command."""
    del args
