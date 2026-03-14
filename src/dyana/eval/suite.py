"""Helpers for named evaluation suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dyana.eval.synthetic_cases import LEAKAGE_STRESS_ID


def _default_week1_items() -> list[dict[str, Any]]:
    return [
        {
            "id": "synthetic_leakage",
            "tier": "synthetic",
            "scenario": LEAKAGE_STRESS_ID,
            "audio_path": None,
            "ref_path": None,
        }
    ]


def load_suite_items(workspace_root: Path, suite_name: str, segments: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Return manifest items for a named suite.

    ``workspace_root`` is accepted for compatibility with the CLI call sites and
    future suite discovery, even though the built-in suites are currently static.
    """

    del workspace_root

    normalized = suite_name.strip().lower()
    if normalized == "week1":
        items = _default_week1_items()
    else:
        raise ValueError(f"Unknown evaluation suite '{suite_name}'.")

    selected = set(segments or [])
    if not selected:
        return items

    filtered = [
        item for item in items if item["id"] in selected or str(item.get("tier", "")).lower() in selected
    ]
    return filtered


def write_manifest(items: list[dict[str, Any]], path: Path) -> Path:
    """Write a resolved evaluation manifest to disk and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, sort_keys=True))
    return path
