"""`dyana tune` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dyana.decode.params import DecodeTuningParams
from dyana.errors import ConfigError, PipelineError
from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import aggregate, read_scorecard, write_scorecard
from dyana.eval.tuning import METRIC_KEYS, compute_delta_report, write_delta_report


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("tune", help="Run evaluation with decode tuning params and compare to baseline.")
    parser.add_argument("--manifest", required=False, help="Evaluation manifest JSON.")
    parser.add_argument("--baseline", required=False, help="Baseline scorecard JSON path.")
    parser.add_argument("--out-dir", required=False, help="Output directory root.")
    parser.add_argument("--run-name", required=False, default="current", help="Run folder name under out-dir.")
    parser.add_argument("--cache-dir", required=False, help="Cache directory.")
    parser.add_argument("--speaker-switch-penalty", type=float, required=False)
    parser.add_argument("--leak-entry-bias", type=float, required=False)
    parser.add_argument("--ovl-transition-cost", type=float, required=False)
    parser.add_argument("--grid", action="store_true", help="Run a tiny parameter grid.")
    parser.set_defaults(command="tune")


def _build_params(args: argparse.Namespace) -> DecodeTuningParams:
    defaults = DecodeTuningParams()
    return DecodeTuningParams(
        speaker_switch_penalty=(
            args.speaker_switch_penalty
            if args.speaker_switch_penalty is not None
            else defaults.speaker_switch_penalty
        ),
        leak_entry_bias=args.leak_entry_bias if args.leak_entry_bias is not None else defaults.leak_entry_bias,
        ovl_transition_cost=args.ovl_transition_cost if args.ovl_transition_cost is not None else defaults.ovl_transition_cost,
    )


def _select_grid_params() -> DecodeTuningParams:
    candidates = [
        DecodeTuningParams(-6.0, -2.0, -3.0),
        DecodeTuningParams(-7.0, -2.0, -3.0),
        DecodeTuningParams(-6.0, -2.5, -3.0),
    ]
    return candidates[0]


def _print_summary(report: dict) -> None:
    print("tier metric deltas:")
    tier_delta = report.get("summary", {}).get("tier_delta", {})
    for tier in sorted(tier_delta):
        print(f"- {tier}")
        for key in METRIC_KEYS:
            value = float(tier_delta[tier].get(key, 0.0))
            print(f"  {key}: {value:+.4f}")
    if report.get("warnings"):
        print("warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    if report.get("failed"):
        print("easy guardrail: FAIL")
    else:
        print("easy guardrail: PASS")


def run(args: argparse.Namespace) -> None:
    manifest_raw = getattr(args, "manifest", None)
    out_dir_raw = getattr(args, "out_dir", None)
    baseline_raw = getattr(args, "baseline", None)

    if not manifest_raw:
        raise ConfigError("dyana tune requires --manifest <path>.")
    if not out_dir_raw:
        raise ConfigError("dyana tune requires --out-dir <path>.")
    if not baseline_raw:
        raise ConfigError("dyana tune requires --baseline <baseline_scorecard.json>.")

    manifest_path = Path(manifest_raw)
    baseline_path = Path(baseline_raw)
    run_name = str(getattr(args, "run_name", "current"))
    run_out_dir = Path(out_dir_raw) / run_name
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None

    params = _select_grid_params() if args.grid else _build_params(args)
    run_rows = evaluate_manifest(manifest_path, out_dir=run_out_dir, cache_dir=cache_dir, tuning_params=params)
    write_scorecard(run_out_dir / "scorecard.json", run_out_dir / "scorecard.csv", run_rows, aggregate(run_rows))

    baseline_payload = read_scorecard(baseline_path)
    current_payload = read_scorecard(run_out_dir / "scorecard.json")
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
    write_delta_report(report, run_out_dir)
    _print_summary(report)

    if report.get("failed"):
        message = "; ".join(report.get("failures", [])) or "easy-tier guardrail failed."
        raise PipelineError(message)
