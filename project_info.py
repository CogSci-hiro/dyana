#!/usr/bin/env python3

"""Generate a lightweight project snapshot markdown report.

This script is intended for quick reproducibility/debugging snapshots. It gathers:
- Environment info (Python/platform/cwd)
- Git state (HEAD + dirty file list)
- A shallow scan of Snakemake rule names (if present)
- A size-sorted repository file summary (excluding common caches)
- An optional inventory of output artifacts (derived from config)
- Optional best-effort probes of known artifact formats

The output is written as a single Markdown file (default: `project_snapshot.md`).
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


# ##################################################################################################
# Helpers
# ##################################################################################################

def _run(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _now_iso() -> str:
    """Current local time as an ISO-8601 string (seconds precision)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_text(path: Path, max_chars: int = 200_000) -> str:
    """Read UTF-8 text (lossy) and truncate to `max_chars`."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    return txt[:max_chars]


def _safe_rel(path: Path, root: Path) -> str:
    """Best-effort relative path for display; falls back to absolute on failure."""
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _human_bytes(n: int) -> str:
    """Format bytes as a short human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{n} B"


def _is_probably_binary(path: Path) -> bool:
    """Return True if `path` likely contains binary data (NUL-byte heuristic)."""
    try:
        chunk = path.read_bytes()[:2048]
    except Exception:
        return True
    return b"\x00" in chunk


# ##################################################################################################
# Config
# ##################################################################################################

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from `path`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping at top level.")
    return raw


def _get_out_dir(cfg: dict[str, Any]) -> Optional[Path]:
    """Extract an output directory from a config dict (best-effort)."""
    # Prefer the schema location: io.out_dir
    io = cfg.get("io", {})
    if isinstance(io, dict) and "out_dir" in io:
        return Path(str(io["out_dir"]))
    # Fallback: paths.out_dir if present
    paths = cfg.get("paths", {})
    if isinstance(paths, dict) and "out_dir" in paths:
        return Path(str(paths["out_dir"]))
    return None


# ##################################################################################################
# Snakemake parsing (lightweight)
# ##################################################################################################

_RULE_RE = re.compile(r"^\s*rule\s+([A-Za-z0-9_]+)\s*:", re.MULTILINE)

def _find_snakemake_rules(workflow_dir: Path) -> list[str]:
    """Return a sorted list of rule names found under a Snakemake workflow dir."""
    rules: set[str] = set()

    candidates: list[Path] = []
    snakefile = workflow_dir / "Snakefile"
    if snakefile.exists():
        candidates.append(snakefile)

    rules_dir = workflow_dir / "rules"
    if rules_dir.exists():
        candidates.extend(sorted(rules_dir.rglob("*.smk")))

    for p in candidates:
        if _is_probably_binary(p):
            continue
        text = _read_text(p, max_chars=300_000)
        for m in _RULE_RE.finditer(text):
            rules.add(m.group(1))

    return sorted(rules)


# ##################################################################################################
# Tree summary
# ##################################################################################################

@dataclass(frozen=True)
class TreeItem:
    relpath: str
    size_bytes: int

def _tree_summary(root: Path, max_items: int = 200) -> list[TreeItem]:
    """Return up to `max_items` largest files under `root` (excluding caches)."""
    items: list[TreeItem] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        # skip huge hidden caches
        if any(part in {".git", ".venv", "__pycache__", ".snakemake"} for part in p.parts):
            continue
        try:
            size = p.stat().st_size
        except Exception:
            continue
        items.append(TreeItem(relpath=_safe_rel(p, root), size_bytes=size))
    items.sort(key=lambda x: x.size_bytes, reverse=True)
    return items[:max_items]


# ##################################################################################################
# Artifact inventory
# ##################################################################################################

@dataclass(frozen=True)
class ArtifactItem:
    relpath: str
    size_bytes: int

def _artifact_inventory(out_dir: Path, max_items: int = 500) -> list[ArtifactItem]:
    """Return an alphabetized inventory of output artifacts under `out_dir`."""
    if not out_dir.exists():
        return []
    items: list[ArtifactItem] = []
    for p in out_dir.rglob("*"):
        if p.is_dir():
            continue
        try:
            size = p.stat().st_size
        except Exception:
            continue
        items.append(ArtifactItem(relpath=_safe_rel(p, out_dir), size_bytes=size))
    items.sort(key=lambda x: x.relpath)
    return items[:max_items]


# ##################################################################################################
# Optional probing (MNE)
# ##################################################################################################

def _probe_mne_evokeds(path: Path) -> Optional[dict[str, Any]]:
    """Probe an MNE evoked `.fif` file if `mne` is installed (best-effort)."""
    try:
        import mne  # type: ignore
    except Exception:
        return None

    if not path.exists():
        return {"exists": False}

    try:
        evs = mne.read_evokeds(path, condition=None, verbose="ERROR")
        return {
            "exists": True,
            "n_evokeds": len(evs),
            "comments_preview": [e.comment for e in evs[:5]],
            "ch_count": len(evs[0].ch_names) if len(evs) > 0 else None,
            "tmin": float(evs[0].times[0]) if len(evs) > 0 else None,
            "tmax": float(evs[0].times[-1]) if len(evs) > 0 else None,
        }
    except Exception as e:
        return {"exists": True, "error": repr(e)}


# ##################################################################################################
# Report
# ##################################################################################################

def main() -> None:
    """Parse CLI args and write the snapshot report."""
    parser = argparse.ArgumentParser(description="Collect project snapshot for reproducibility/debugging.")
    parser.add_argument("--root", type=str, default=".", help="Project root (default: .)")
    parser.add_argument("--config", type=str, default="workflow/config.yaml", help="Path to config.yaml")
    parser.add_argument("--out", type=str, default="project_snapshot.md", help="Output markdown file")
    parser.add_argument("--probe", action="store_true", help="Probe known artifacts (e.g., MNE fif) if possible.")
    parser.add_argument("--max_tree_items", type=int, default=150)
    parser.add_argument("--max_artifacts", type=int, default=400)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg_path = (root / args.config).resolve()
    out_path = (root / args.out).resolve()
    workflow_dir = root / "workflow"

    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)

    out_dir = _get_out_dir(cfg)

    # Git info
    git_ok = (root / ".git").exists()
    git_head = ""
    git_status = ""
    if git_ok:
        _, git_head, _ = _run(["git", "rev-parse", "HEAD"], cwd=root)
        _, git_status, _ = _run(["git", "status", "--porcelain"], cwd=root)

    # Python / env
    py_info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }

    # Optional pip freeze (can be large)
    pip_freeze = ""
    if shutil.which("pip"):
        _, pip_freeze, _ = _run([sys.executable, "-m", "pip", "freeze"])

    # Snakemake rules
    rules = _find_snakemake_rules(workflow_dir) if workflow_dir.exists() else []

    # Trees
    tree_items = _tree_summary(root, max_items=args.max_tree_items)
    artifact_items: list[ArtifactItem] = []
    if out_dir is not None:
        artifact_items = _artifact_inventory(out_dir, max_items=args.max_artifacts)

    # Probing (optional)
    probes: dict[str, Any] = {}
    if args.probe and cfg:
        # Example probe: ERP timecourse files if present in config.
        # Keep probes conservative (optional, best-effort) to avoid hard failures.
        viz = cfg.get("viz", {})
        if isinstance(viz, dict):
            erp_tc = viz.get("erp_timecourse", {})
            if isinstance(erp_tc, dict):
                for k in ["duration_long_fif", "duration_short_fif", "latency_fast_fif", "latency_slow_fif"]:
                    v = erp_tc.get(k)
                    if v:
                        probes[k] = _probe_mne_evokeds(Path(str(v)))

    # Write report
    lines: list[str] = []
    lines.append(f"# Project Snapshot\n")
    lines.append(f"- Generated: `{_now_iso()}`\n")
    lines.append(f"- Root: `{root}`\n")
    lines.append(f"- Config: `{cfg_path if cfg_path.exists() else 'MISSING'}`\n")

    lines.append("## Environment\n")
    lines.append("```json")
    lines.append(json.dumps(py_info, indent=2))
    lines.append("```\n")

    if git_ok:
        lines.append("## Git\n")
        lines.append(f"- HEAD: `{git_head}`\n")
        lines.append(f"- Dirty files: `{len(git_status.splitlines()) if git_status else 0}`\n")
        if git_status:
            lines.append("```text")
            lines.append(git_status)
            lines.append("```\n")

    lines.append("## Snakemake Rules\n")
    if rules:
        for r in rules:
            lines.append(f"- `{r}`")
        lines.append("")
    else:
        lines.append("_No Snakefile/rules detected._\n")

    lines.append("## Key Config Excerpts\n")
    if cfg:
        # Keep the snapshot compact and reduce the risk of leaking secrets
        # by avoiding full config dumps.
        lines.append(f"- Top-level keys: `{sorted(cfg.keys())}`\n")
        if out_dir is not None:
            lines.append(f"- Resolved out_dir: `{out_dir}`\n")
        # show viz section if present
        if isinstance(cfg.get("viz"), dict):
            lines.append("### viz\n")
            lines.append("```yaml")
            lines.append(yaml.safe_dump({"viz": cfg["viz"]}, sort_keys=False))
            lines.append("```\n")
    else:
        lines.append("_Config not found or invalid._\n")

    lines.append("## Repository File Summary\n")
    lines.append(f"Top {len(tree_items)} files by size (excluding .git/.venv/__pycache__/.snakemake):\n")
    lines.append("| file | size |")
    lines.append("|---|---:|")
    for it in tree_items:
        lines.append(f"| `{it.relpath}` | {_human_bytes(it.size_bytes)} |")
    lines.append("")

    lines.append("## Output Artifact Inventory\n")
    if out_dir is None:
        lines.append("_Could not determine out_dir from config (expected io.out_dir or paths.out_dir)._")
    elif not out_dir.exists():
        lines.append(f"_out_dir does not exist: `{out_dir}`_")
    else:
        lines.append(f"out_dir: `{out_dir}`\n")
        lines.append(f"Listing first {len(artifact_items)} files:\n")
        lines.append("| file | size |")
        lines.append("|---|---:|")
        for it in artifact_items:
            lines.append(f"| `{it.relpath}` | {_human_bytes(it.size_bytes)} |")
    lines.append("")

    if args.probe:
        lines.append("## Probes\n")
        if probes:
            lines.append("```json")
            lines.append(json.dumps(probes, indent=2))
            lines.append("```\n")
        else:
            lines.append("_No probes run (missing config keys or dependencies)._")

    # pip freeze at end (can be huge)
    lines.append("\n## pip freeze\n")
    if pip_freeze:
        lines.append("```text")
        lines.append(pip_freeze)
        lines.append("```")
    else:
        lines.append("_pip not available._")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
