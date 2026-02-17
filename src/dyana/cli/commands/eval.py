"""`dyana eval` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dyana.decode.params import DecodeTuningParams
from dyana.errors import ConfigError
from dyana.errors.config import load_config, resolve_out_dir
from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import aggregate, write_scorecard
from dyana.eval.suite import load_suite_items, write_manifest


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Run evaluation harness.")
    parser.add_argument("--manifest", required=False, help="Path to evaluation manifest JSON.")
    parser.add_argument("--suite", required=False, help="Named suite (e.g. week1).")
    parser.add_argument("--segments", nargs="*", default=[], help="Subset segments (synthetic_leakage hard easy).")
    parser.add_argument("--out-dir", required=False, help="Output directory root for scorecards.")
    parser.add_argument("--run-name", required=False, default="baseline", help="Run folder name under out-dir.")
    parser.add_argument("--cache-dir", required=False, help="Cache directory.")
    parser.set_defaults(command="eval")


def run(args: argparse.Namespace) -> None:
    manifest_raw = getattr(args, "manifest", None)
    suite_raw = getattr(args, "suite", None)
    if not manifest_raw and not suite_raw:
        raise ConfigError("dyana eval requires --manifest <path> or --suite week1.")

    run_name = str(getattr(args, "run_name", "baseline"))
    config = load_config(Path.cwd())
    root_out_dir = resolve_out_dir(config, Path(args.out_dir) if getattr(args, "out_dir", None) else None)
    run_out_dir = root_out_dir / run_name
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None
    if manifest_raw:
        manifest_path = Path(manifest_raw)
    else:
        suite_items = load_suite_items(Path.cwd(), str(suite_raw), list(getattr(args, "segments", [])))
        manifest_path = write_manifest(suite_items, run_out_dir / "manifest.resolved.json")

    params = DecodeTuningParams()
    results = evaluate_manifest(manifest_path, out_dir=run_out_dir, cache_dir=cache_dir, tuning_params=params)
    write_scorecard(
        run_out_dir / "scorecard.json",
        run_out_dir / "scorecard.csv",
        results,
        aggregate(results),
        metadata={
            "params": {
                "speaker_switch_penalty": params.speaker_switch_penalty,
                "leak_entry_bias": params.leak_entry_bias,
                "ovl_transition_cost": params.ovl_transition_cost,
                "a_to_ovl_cost": params.a_to_ovl_cost,
                "b_to_ovl_cost": params.b_to_ovl_cost,
                "ovl_to_a_cost": params.ovl_to_a_cost,
                "ovl_to_b_cost": params.ovl_to_b_cost,
            }
        },
    )
    print(f"scorecard written: {run_out_dir / 'scorecard.json'}")
