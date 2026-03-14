from __future__ import annotations

import numpy as np

from dyana.artifacts.hashing import hash_bytes, hash_json, hash_numpy


def test_hash_bytes_is_stable_for_same_input() -> None:
    assert hash_bytes(b"dyana") == hash_bytes(b"dyana")


def test_hash_bytes_changes_for_different_input() -> None:
    assert hash_bytes(b"dyana") != hash_bytes(b"artifact")


def test_hash_numpy_is_stable_for_same_input() -> None:
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert hash_numpy(array) == hash_numpy(array.copy())


def test_hash_json_sorts_keys_deterministically() -> None:
    left = {"a": 1, "b": {"c": 2}}
    right = {"b": {"c": 2}, "a": 1}
    assert hash_json(left) == hash_json(right)


def test_hash_json_changes_when_payload_changes() -> None:
    assert hash_json({"a": 1}) != hash_json({"a": 2})
