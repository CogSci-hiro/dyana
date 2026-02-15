"""Tiny file-based cache helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _hash_dict(obj: Mapping[str, Any]) -> str:
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def make_cache_key(audio_path: Path, func: str, params: Mapping[str, Any]) -> str:
    stat = audio_path.stat()
    base = {
        "func": func,
        "path": str(audio_path.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "params": params,
    }
    return _hash_dict(base)


def cache_get(cache_dir: Optional[Path], key: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{key}.npz"
    if npz_path.exists():
        return npz_path
    return None


def cache_put(cache_dir: Optional[Path], key: str, arrays: Mapping[str, Any]) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{key}.npz"
    np.savez_compressed(npz_path, **arrays)
    return npz_path
