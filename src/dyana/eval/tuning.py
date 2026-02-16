from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


METRIC_KEYS = [
    "boundary_f1_20ms",
    "boundary_f1_50ms",
    "micro_ipus_per_min",
    "switches_per_min",
]

EASY_BOUNDARY_DROP_THRESHOLD = -0.05
EASY_SWITCH_INCREASE_FACTOR = 1.25
EASY_MICRO_IPU_INCREASE_FACTOR = 1.25
SUSPICIOUS_SWITCH_WORSE_FACTOR = 1.50
SUSPICIOUS_MICRO_WORSE_FACTOR = 1.50


def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _group_by_tier(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("tier", "unknown")), []).append(row)
    return grouped


def compute_delta_report(
    baseline_payload: Dict[str, Any],
    current_payload: Dict[str, Any],
    *,
    params: Dict[str, float],
    baseline_path: Path,
) -> Dict[str, Any]:
    baseline_rows = list(baseline_payload.get("results", []))
    current_rows = list(current_payload.get("results", []))
    baseline_by_id = _index_by_id(baseline_rows)
    current_by_id = _index_by_id(current_rows)

    per_item: List[Dict[str, Any]] = []
    for item_id in sorted(set(baseline_by_id) & set(current_by_id)):
        baseline_row = baseline_by_id[item_id]
        current_row = current_by_id[item_id]
        delta_row: Dict[str, Any] = {
            "id": item_id,
            "tier": current_row.get("tier", baseline_row.get("tier", "unknown")),
        }
        for key in METRIC_KEYS:
            baseline_value = float(baseline_row.get(key, 0.0))
            current_value = float(current_row.get(key, 0.0))
            delta_row[f"{key}_baseline"] = baseline_value
            delta_row[f"{key}_current"] = current_value
            delta_row[f"{key}_delta"] = current_value - baseline_value
        per_item.append(delta_row)

    failures: List[str] = []
    warnings: List[str] = []

    for row in [item for item in per_item if item.get("tier") == "easy"]:
        if row["boundary_f1_20ms_delta"] < EASY_BOUNDARY_DROP_THRESHOLD:
            failures.append(f"easy regression: boundary_f1_20ms drop > 0.05 for {row['id']}")
        if row["boundary_f1_50ms_delta"] < EASY_BOUNDARY_DROP_THRESHOLD:
            failures.append(f"easy regression: boundary_f1_50ms drop > 0.05 for {row['id']}")
        baseline_switch = max(row["switches_per_min_baseline"], 1e-9)
        if row["switches_per_min_current"] > baseline_switch * EASY_SWITCH_INCREASE_FACTOR:
            failures.append(f"easy regression: switches_per_min increase > 25% for {row['id']}")
        baseline_micro = max(row["micro_ipus_per_min_baseline"], 1e-9)
        if row["micro_ipus_per_min_current"] > baseline_micro * EASY_MICRO_IPU_INCREASE_FACTOR:
            failures.append(f"easy regression: micro_ipus_per_min increase > 25% for {row['id']}")

    for row in [item for item in per_item if item.get("tier") == "hard"]:
        if row["boundary_f1_20ms_delta"] > 0:
            baseline_switch = max(row["switches_per_min_baseline"], 1e-9)
            baseline_micro = max(row["micro_ipus_per_min_baseline"], 1e-9)
            switch_worse = row["switches_per_min_current"] > baseline_switch * SUSPICIOUS_SWITCH_WORSE_FACTOR
            micro_worse = row["micro_ipus_per_min_current"] > baseline_micro * SUSPICIOUS_MICRO_WORSE_FACTOR
            if switch_worse or micro_worse:
                warnings.append(
                    f"suspicious improvement: hard boundary improved but instability worsened for {row['id']}"
                )

    baseline_bytes = baseline_path.read_bytes()
    baseline_hash = hashlib.sha1(baseline_bytes).hexdigest()

    summary = {
        "overall_delta": {key: _mean([row[f"{key}_delta"] for row in per_item]) for key in METRIC_KEYS},
        "tier_delta": {
            tier: {key: _mean([row[f"{key}_delta"] for row in rows]) for key in METRIC_KEYS}
            for tier, rows in _group_by_tier(per_item).items()
        },
    }

    return {
        "params": params,
        "baseline": {
            "path": str(baseline_path),
            "sha1": baseline_hash,
            "mtime": baseline_path.stat().st_mtime,
        },
        "rows": per_item,
        "summary": summary,
        "failed": bool(failures),
        "failures": failures,
        "warnings": warnings,
    }


def write_delta_report(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    rows = list(report.get("rows", []))
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_dir / "delta.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
