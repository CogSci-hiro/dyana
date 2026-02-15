"""`dyana eval` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import aggregate, write_scorecard


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Run evaluation harness.")
    parser.add_argument("--manifest", required=False, help="Path to evaluation manifest JSON.")
    parser.add_argument("--out-dir", required=False, help="Output directory for scorecard.")
    parser.add_argument("--cache-dir", required=False, help="Cache directory.")
    parser.set_defaults(command="eval")


def run(args: argparse.Namespace) -> None:
    if not getattr(args, "manifest", None) or not getattr(args, "out_dir", None):
        return
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None
    results = evaluate_manifest(manifest_path, out_dir=out_dir, cache_dir=cache_dir)
    write_scorecard(out_dir / "scorecard.json", out_dir / "scorecard.csv", results, aggregate(results))
