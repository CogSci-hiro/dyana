from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    keys = [k for k in results[0] if k not in ("id", "tier")]
    return {k: float(sum(r[k] for r in results) / len(results)) for k in keys}


def aggregate_by_tier(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by = {}
    for r in results:
        tier = r.get("tier", "unknown")
        by.setdefault(tier, []).append(r)
    return {tier: aggregate(items) for tier, items in by.items()}


def write_scorecard(results: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(results)
    by_tier = aggregate_by_tier(results)
    payload = {"results": results, "summary": summary, "by_tier": by_tier}
    (out_dir / "scorecard.json").write_text(json.dumps(payload, indent=2))

    # CSV
    fieldnames = list(results[0].keys()) if results else []
    if fieldnames:
        with open(out_dir / "scorecard.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
