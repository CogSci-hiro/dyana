from __future__ import annotations

import os
from pathlib import Path

_ARTIFACT_ENV_VAR = "DYANA_ARTIFACT_ROOT"


def get_artifact_root() -> Path:
    root = os.environ.get(_ARTIFACT_ENV_VAR)
    path = Path(root) if root else Path.cwd() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    (path / "cache").mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    cache_dir = get_artifact_root() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_run_dir(run_id: str) -> Path:
    run_dir = get_artifact_root() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
