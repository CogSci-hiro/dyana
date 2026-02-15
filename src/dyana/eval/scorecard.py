from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    keys = [k for k in results[0] if k not in ("id", "tier")]
    return {k: float(sum(r[k] for r in results) / len(results)) for k in keys}


def aggregate_by_tier(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[Dict[str, float]]] = {}
    for row in results:
        tier = str(row.get("tier", "unknown"))
        by.setdefault(tier, []).append(row)
    return {tier: aggregate(rows) for tier, rows in by.items()}


def write_scorecard(
    path_json: Path,
    path_csv: Path,
    rows: List[Dict[str, float]],
    summary: Dict[str, Any],
) -> None:
    payload = {
        "results": rows,
        "summary": summary,
        "by_tier": aggregate_by_tier(rows),
    }
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    fieldnames = list(rows[0].keys()) if rows else []
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(path_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_scorecard(path_json: Path) -> Dict[str, Any]:
    return json.loads(path_json.read_text())


def write_scorecard_to_dir(rows: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    write_scorecard(out_dir / "scorecard.json", out_dir / "scorecard.csv", rows, summary)
