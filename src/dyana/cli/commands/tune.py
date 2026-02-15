"""`dyana tune` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dyana.decode.params import DecodeTuningParams
from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import aggregate, read_scorecard, write_scorecard
from dyana.eval.tuning import compute_delta_report, write_delta_report


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("tune", help="Run evaluation with decode tuning params and compare to baseline.")
    parser.add_argument("--manifest", required=False, help="Evaluation manifest JSON.")
    parser.add_argument("--baseline", required=False, help="Baseline scorecard JSON path.")
    parser.add_argument("--out-dir", required=False, help="Output directory.")
    parser.add_argument("--cache-dir", required=False, help="Cache directory.")
    parser.add_argument("--speaker-switch-penalty", type=float, required=False)
    parser.add_argument("--leak-entry-bias", type=float, required=False)
    parser.add_argument("--ovl-transition-cost", type=float, required=False)
    parser.add_argument("--grid", action="store_true", help="Run a tiny parameter grid.")
    parser.set_defaults(command="tune")


def _build_params(args: argparse.Namespace) -> DecodeTuningParams:
    defaults = DecodeTuningParams()
    return DecodeTuningParams(
        speaker_switch_penalty=args.speaker_switch_penalty
        if args.speaker_switch_penalty is not None
        else defaults.speaker_switch_penalty,
        leak_entry_bias=args.leak_entry_bias if args.leak_entry_bias is not None else defaults.leak_entry_bias,
        ovl_transition_cost=args.ovl_transition_cost
        if args.ovl_transition_cost is not None
        else defaults.ovl_transition_cost,
    )


def run(args: argparse.Namespace) -> None:
    if not getattr(args, "manifest", None) or not getattr(args, "out_dir", None):
        return
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None
    baseline_path = Path(args.baseline) if getattr(args, "baseline", None) else out_dir / "baseline" / "scorecard.json"

    if args.grid:
        candidates = [
            DecodeTuningParams(-7.0, -2.0, -3.0),
            DecodeTuningParams(-6.0, -2.0, -3.0),
            DecodeTuningParams(-8.0, -1.5, -3.5),
        ]
        params = candidates[0]
    else:
        params = _build_params(args)

    run_rows = evaluate_manifest(manifest_path, out_dir=out_dir / "current", cache_dir=cache_dir, tuning_params=params)
    write_scorecard(out_dir / "current" / "scorecard.json", out_dir / "current" / "scorecard.csv", run_rows, aggregate(run_rows))

    baseline_payload = read_scorecard(baseline_path)
    current_payload = read_scorecard(out_dir / "current" / "scorecard.json")
    report = compute_delta_report(
        baseline_payload,
        current_payload,
        params={
            "speaker_switch_penalty": params.speaker_switch_penalty,
            "leak_entry_bias": params.leak_entry_bias,
            "ovl_transition_cost": params.ovl_transition_cost,
        },
        baseline_path=baseline_path,
    )
    write_delta_report(report, out_dir)

    print("tune summary:")
    print(report["summary"])
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
