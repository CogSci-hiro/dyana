from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_numpy(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = b"|".join(
        [
            str(contiguous.dtype).encode("utf-8"),
            json.dumps(contiguous.shape).encode("utf-8"),
            contiguous.tobytes(),
        ]
    )
    return hash_bytes(payload)


def hash_json(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hash_bytes(payload)
