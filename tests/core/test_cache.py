from pathlib import Path

import numpy as np

from dyana.core.cache import cache_get, cache_put, make_cache_key


def test_cache_roundtrip(tmp_path: Path) -> None:
    audio = tmp_path / "dummy.wav"
    audio.write_bytes(b"abc")
    key = make_cache_key(audio, "fn", {"p": 1})
    assert cache_get(tmp_path, key) is None
    arr = np.array([1, 2, 3])
    path = cache_put(tmp_path, key, {"values": arr})
    assert path is not None and path.exists()
    hit = cache_get(tmp_path, key)
    assert hit == path
    with np.load(hit) as npz:
        loaded = npz["values"]
    assert np.allclose(loaded, arr)
