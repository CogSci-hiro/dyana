from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


METRIC_KEYS = [
    "boundary_f1_20",
    "boundary_f1_50",
    "micro_ipus_per_min",
    "speaker_switches_per_min",
]

EASY_BOUNDARY_DROP_THRESHOLD = -0.05
EASY_SWITCH_INCREASE_FACTOR = 1.20
EASY_MICRO_IPU_INCREASE_FACTOR = 1.20
SUSPICIOUS_SWITCH_WORSE_FACTOR = 1.50
SUSPICIOUS_MICRO_WORSE_FACTOR = 1.50


def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["id"]): r for r in rows}


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _tier_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_tier.setdefault(str(row.get("tier", "unknown")), []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for tier, tier_rows in by_tier.items():
        out[tier] = {k: _mean([float(r[k]) for r in tier_rows if k in r]) for k in METRIC_KEYS}
    return out


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

    ids = sorted(set(baseline_by_id) & set(current_by_id))
    per_item: List[Dict[str, Any]] = []
    for item_id in ids:
        b = baseline_by_id[item_id]
        c = current_by_id[item_id]
        row: Dict[str, Any] = {"id": item_id, "tier": c.get("tier", b.get("tier", "unknown"))}
        for key in METRIC_KEYS:
            b_val = float(b.get(key, 0.0))
            c_val = float(c.get(key, 0.0))
            row[f"{key}_baseline"] = b_val
            row[f"{key}_current"] = c_val
            row[f"{key}_delta"] = c_val - b_val
        per_item.append(row)

    warnings: List[str] = []
    easy_rows = [r for r in per_item if r.get("tier") == "easy"]
    for row in easy_rows:
        if row["boundary_f1_20_delta"] < EASY_BOUNDARY_DROP_THRESHOLD:
            warnings.append(f"easy regression: boundary_f1_20 drop too large for {row['id']}")
        baseline_switch = max(row["speaker_switches_per_min_baseline"], 1e-9)
        if row["speaker_switches_per_min_current"] > baseline_switch * EASY_SWITCH_INCREASE_FACTOR:
            warnings.append(f"easy regression: switches increase too large for {row['id']}")
        baseline_micro = max(row["micro_ipus_per_min_baseline"], 1e-9)
        if row["micro_ipus_per_min_current"] > baseline_micro * EASY_MICRO_IPU_INCREASE_FACTOR:
            warnings.append(f"easy regression: micro-IPU rate increase too large for {row['id']}")

    hard_rows = [r for r in per_item if r.get("tier") == "hard"]
    for row in hard_rows:
        if row["boundary_f1_20_delta"] > 0:
            baseline_switch = max(row["speaker_switches_per_min_baseline"], 1e-9)
            baseline_micro = max(row["micro_ipus_per_min_baseline"], 1e-9)
            switch_worse = row["speaker_switches_per_min_current"] > baseline_switch * SUSPICIOUS_SWITCH_WORSE_FACTOR
            micro_worse = row["micro_ipus_per_min_current"] > baseline_micro * SUSPICIOUS_MICRO_WORSE_FACTOR
            if switch_worse or micro_worse:
                warnings.append(
                    f"suspicious improvement: hard boundary improved but instability worsened for {row['id']}"
                )

    baseline_bytes = baseline_path.read_bytes()
    baseline_hash = hashlib.sha1(baseline_bytes).hexdigest()
    baseline_mtime = baseline_path.stat().st_mtime

    summary = {
        "overall_delta": {k: _mean([r[f"{k}_delta"] for r in per_item]) for k in METRIC_KEYS},
        "tier_delta": {
            tier: {k: _mean([r[f"{k}_delta"] for r in tier_rows]) for k in METRIC_KEYS}
            for tier, tier_rows in _group_by_tier(per_item).items()
        },
    }

    return {
        "params": params,
        "baseline": {"path": str(baseline_path), "sha1": baseline_hash, "mtime": baseline_mtime},
        "rows": per_item,
        "summary": summary,
        "warnings": warnings,
    }


def _group_by_tier(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_tier.setdefault(str(row.get("tier", "unknown")), []).append(row)
    return by_tier


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
