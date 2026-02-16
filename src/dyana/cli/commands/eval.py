"""`dyana eval` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dyana.errors import ConfigError
from dyana.errors.config import load_config, resolve_out_dir
from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import aggregate, write_scorecard


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Run evaluation harness.")
    parser.add_argument("--manifest", required=False, help="Path to evaluation manifest JSON.")
    parser.add_argument("--out-dir", required=False, help="Output directory root for scorecards.")
    parser.add_argument("--run-name", required=False, default="baseline", help="Run folder name under out-dir.")
    parser.add_argument("--cache-dir", required=False, help="Cache directory.")
    parser.set_defaults(command="eval")


def run(args: argparse.Namespace) -> None:
    manifest_raw = getattr(args, "manifest", None)
    if not manifest_raw:
        raise ConfigError("dyana eval requires --manifest <path>.")

    manifest_path = Path(manifest_raw)
    run_name = str(getattr(args, "run_name", "baseline"))
    config = load_config(Path.cwd())
    root_out_dir = resolve_out_dir(config, Path(args.out_dir) if getattr(args, "out_dir", None) else None)
    run_out_dir = root_out_dir / run_name
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None

    results = evaluate_manifest(manifest_path, out_dir=run_out_dir, cache_dir=cache_dir)
    write_scorecard(run_out_dir / "scorecard.json", run_out_dir / "scorecard.csv", results, aggregate(results))
    print(f"scorecard written: {run_out_dir / 'scorecard.json'}")
