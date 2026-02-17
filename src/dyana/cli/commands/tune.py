"""`dyana tune` command implementation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dyana.decode.params import DecodeTuningParams
from dyana.errors import ConfigError, PipelineError
from dyana.errors.config import load_config, resolve_out_dir
from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import aggregate, read_scorecard, write_scorecard
from dyana.eval.suite import load_suite_items, write_manifest
from dyana.eval.tuning import METRIC_KEYS, compute_delta_report, write_delta_report


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("tune", help="Run evaluation with decode tuning params and compare to baseline.")
    parser.add_argument("--manifest", required=False, help="Evaluation manifest JSON.")
    parser.add_argument("--suite", required=False, help="Named suite (e.g. week1).")
    parser.add_argument("--segments", nargs="*", default=[], help="Subset segments (synthetic_leakage hard easy).")
    parser.add_argument("--baseline", required=False, help="Baseline scorecard JSON path.")
    parser.add_argument("--out-dir", required=False, help="Output directory root.")
    parser.add_argument("--run-name", required=False, default="current", help="Run folder name under out-dir.")
    parser.add_argument("--cache-dir", required=False, help="Cache directory.")
    parser.add_argument("--speaker-switch-penalty", type=float, required=False)
    parser.add_argument("--leak-entry-bias", type=float, required=False)
    parser.add_argument("--ovl-transition-cost", type=float, required=False)
    parser.add_argument("--a-to-ovl-cost", type=float, required=False)
    parser.add_argument("--b-to-ovl-cost", type=float, required=False)
    parser.add_argument("--ovl-to-a-cost", type=float, required=False)
    parser.add_argument("--ovl-to-b-cost", type=float, required=False)
    parser.add_argument("--grid", action="store_true", help="Run a tiny parameter grid.")
    parser.set_defaults(command="tune")


def _build_params(args: argparse.Namespace) -> DecodeTuningParams:
    defaults = DecodeTuningParams()
    speaker_switch_penalty = getattr(args, "speaker_switch_penalty", None)
    leak_entry_bias = getattr(args, "leak_entry_bias", None)
    ovl_transition_cost = getattr(args, "ovl_transition_cost", None)
    return DecodeTuningParams(
        speaker_switch_penalty=(
            speaker_switch_penalty
            if speaker_switch_penalty is not None
            else defaults.speaker_switch_penalty
        ),
        leak_entry_bias=leak_entry_bias if leak_entry_bias is not None else defaults.leak_entry_bias,
        ovl_transition_cost=ovl_transition_cost if ovl_transition_cost is not None else defaults.ovl_transition_cost,
        a_to_ovl_cost=getattr(args, "a_to_ovl_cost", None),
        b_to_ovl_cost=getattr(args, "b_to_ovl_cost", None),
        ovl_to_a_cost=getattr(args, "ovl_to_a_cost", None),
        ovl_to_b_cost=getattr(args, "ovl_to_b_cost", None),
    )


def _grid_candidates() -> list[DecodeTuningParams]:
    return [
        DecodeTuningParams(-6.0, -2.0, -3.0),
        DecodeTuningParams(-7.0, -2.0, -3.0),
        DecodeTuningParams(-6.0, -2.5, -3.0),
        DecodeTuningParams(-7.0, -2.5, -3.5),
    ]


def _print_summary(report: dict) -> None:
    print("Day4 summary:")
    tier_delta = report.get("summary", {}).get("tier_delta", {})
    synthetic_delta = tier_delta.get("synthetic", {})
    hard_delta = tier_delta.get("hard", {})
    easy_delta = tier_delta.get("easy", {})
    print(f"- Synthetic leakage: {'OK' if 'synthetic' in tier_delta else 'SKIPPED'}")
    if hard_delta:
        print(f"- Hard real: OK (micro-IPUs/min Δ={float(hard_delta.get('micro_ipus_per_min', 0.0)):+.4f})")
    else:
        print("- Hard real: SKIPPED")
    if easy_delta:
        print(
            "- Easy real: "
            f"OK (F1@20ms Δ={float(easy_delta.get('boundary_f1_20ms', 0.0)):+.4f}, "
            f"F1@50ms Δ={float(easy_delta.get('boundary_f1_50ms', 0.0)):+.4f})"
        )
    else:
        print("- Easy real: SKIPPED")
    if synthetic_delta:
        print(
            f"  Synthetic switches/min Δ={float(synthetic_delta.get('switches_per_min', 0.0)):+.4f}"
        )

    print("tier metric deltas:")
    for tier in sorted(tier_delta):
        metric_text = ", ".join(f"{key}={float(tier_delta[tier].get(key, 0.0)):+.4f}" for key in METRIC_KEYS)
        print(f"  {tier}: {metric_text}")
    if report.get("warnings"):
        print("warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    if report.get("failed"):
        print(f"Guardrails: FAIL ({'; '.join(report.get('failures', []))})")
    else:
        print("Guardrails: PASS")


def _params_dict(params: DecodeTuningParams) -> dict[str, float | None]:
    return {
        "speaker_switch_penalty": params.speaker_switch_penalty,
        "leak_entry_bias": params.leak_entry_bias,
        "ovl_transition_cost": params.ovl_transition_cost,
        "a_to_ovl_cost": params.a_to_ovl_cost,
        "b_to_ovl_cost": params.b_to_ovl_cost,
        "ovl_to_a_cost": params.ovl_to_a_cost,
        "ovl_to_b_cost": params.ovl_to_b_cost,
    }


def _write_leaderboard(leaderboard_rows: list[dict], out_dir: Path) -> None:
    if not leaderboard_rows:
        return
    sorted_rows = sorted(
        leaderboard_rows,
        key=lambda row: (
            float(row.get("hard_micro_ipus_per_min_delta", 0.0)),
            abs(float(row.get("easy_boundary_f1_20ms_delta", 0.0))),
            float(row.get("switches_per_min_delta", 0.0)),
        ),
    )
    (out_dir / "leaderboard.json").write_text(json.dumps(sorted_rows, indent=2, sort_keys=True))
    with (out_dir / "leaderboard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sorted_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_rows)


def run(args: argparse.Namespace) -> None:
    manifest_raw = getattr(args, "manifest", None)
    suite_raw = getattr(args, "suite", None)
    baseline_raw = getattr(args, "baseline", None)

    if not manifest_raw and not suite_raw:
        raise ConfigError("dyana tune requires --manifest <path> or --suite week1.")
    if not baseline_raw:
        raise ConfigError("dyana tune requires --baseline <baseline_scorecard.json>.")

    baseline_path = Path(baseline_raw)
    run_name = str(getattr(args, "run_name", "current"))
    config = load_config(Path.cwd())
    root_out_dir = resolve_out_dir(config, Path(args.out_dir) if getattr(args, "out_dir", None) else None)
    run_out_dir = root_out_dir / run_name
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None
    if manifest_raw:
        manifest_path = Path(manifest_raw)
    else:
        suite_items = load_suite_items(Path.cwd(), str(suite_raw), list(getattr(args, "segments", [])))
        manifest_path = write_manifest(suite_items, run_out_dir / "manifest.resolved.json")

    baseline_payload = read_scorecard(baseline_path)
    params_list = _grid_candidates() if args.grid else [_build_params(args)]
    leaderboard_rows: list[dict] = []
    first_failure: str | None = None

    for index, params in enumerate(params_list):
        candidate_dir = run_out_dir if len(params_list) == 1 else run_out_dir / f"candidate_{index:02d}"
        run_rows = evaluate_manifest(manifest_path, out_dir=candidate_dir, cache_dir=cache_dir, tuning_params=params)
        write_scorecard(
            candidate_dir / "scorecard.json",
            candidate_dir / "scorecard.csv",
            run_rows,
            aggregate(run_rows),
            metadata={"params": _params_dict(params)},
        )

        current_payload = read_scorecard(candidate_dir / "scorecard.json")
        report = compute_delta_report(
            baseline_payload,
            current_payload,
            params=_params_dict(params),
            baseline_path=baseline_path,
        )
        write_delta_report(report, candidate_dir)
        _print_summary(report)

        tier_delta = report.get("summary", {}).get("tier_delta", {})
        leaderboard_rows.append(
            {
                "candidate": candidate_dir.name,
                "failed": bool(report.get("failed")),
                "hard_micro_ipus_per_min_delta": float(tier_delta.get("hard", {}).get("micro_ipus_per_min", 0.0)),
                "easy_boundary_f1_20ms_delta": float(tier_delta.get("easy", {}).get("boundary_f1_20ms", 0.0)),
                "switches_per_min_delta": float(report.get("summary", {}).get("overall_delta", {}).get("switches_per_min", 0.0)),
                **_params_dict(params),
            }
        )

        if report.get("failed") and first_failure is None:
            first_failure = "; ".join(report.get("failures", [])) or "easy-tier guardrail failed."

    if len(params_list) > 1:
        _write_leaderboard(leaderboard_rows, run_out_dir)

    if first_failure is not None:
        raise PipelineError(first_failure)
