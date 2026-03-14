from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dyana.version import __version__

from .hashing import hash_bytes, hash_json, hash_numpy
from .metadata import ArtifactMetadata
from .paths import get_artifact_root
from .types import Artifact


class ArtifactStore:
    def __init__(self, root: Path | None = None):
        self.root = (root or get_artifact_root()).resolve()
        self.cache_dir = self.root / "cache"
        self.runs_dir = self.root / "runs"
        self.logs_dir = self.root / "logs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(
        self,
        name: str,
        obj: Any,
        metadata: ArtifactMetadata,
    ) -> Artifact:
        artifact_hash = self.compute_step_hash(
            step=metadata.step,
            inputs=metadata.inputs,
            parameters=metadata.parameters,
        )
        payload = self._serialize_object(obj)
        payload_hash = hash_bytes(payload)
        artifact_path = self._artifact_path(artifact_hash)
        metadata_path = self._metadata_path(artifact_hash)
        artifact_path.write_bytes(payload)
        metadata_payload = asdict(metadata)
        metadata_payload.update(
            {
                "artifact_name": name,
                "artifact_hash": artifact_hash,
                "payload_hash": payload_hash,
                "serializer": "pickle",
            }
        )
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))
        return Artifact(name=name, path=artifact_path, hash=artifact_hash, metadata=metadata_payload)

    def load_artifact(self, hash: str) -> Any:
        return pickle.loads(self._artifact_path(hash).read_bytes())

    def artifact_exists(self, hash: str) -> bool:
        return self._artifact_path(hash).exists()

    def compute_step_hash(
        self,
        step: str,
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> str:
        payload = {
            "code_version": __version__,
            "inputs": self._normalize_for_json(inputs),
            "parameters": self._normalize_for_json(parameters),
            "step": step,
        }
        return hash_json(payload)

    def hash_object(self, obj: Any) -> str:
        if isinstance(obj, np.ndarray):
            return hash_numpy(obj)
        if isinstance(obj, bytes):
            return hash_bytes(obj)
        if isinstance(obj, Path):
            return hash_json({"path": str(obj.resolve())})
        if is_dataclass(obj) and not isinstance(obj, type):
            return hash_json(self._normalize_for_json(asdict(obj)))
        if isinstance(obj, dict):
            return hash_json(self._normalize_for_json(obj))
        if isinstance(obj, (list, tuple, str, int, float, bool)) or obj is None:
            return hash_json({"value": self._normalize_for_json(obj)})
        return hash_bytes(self._serialize_object(obj))

    def read_metadata(self, hash: str) -> dict[str, Any]:
        return json.loads(self._metadata_path(hash).read_text())

    def _artifact_path(self, hash: str) -> Path:
        return self.cache_dir / f"{hash}.pkl"

    def _metadata_path(self, hash: str) -> Path:
        return self.cache_dir / f"{hash}.json"

    def _serialize_object(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _normalize_for_json(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return {
                "dtype": str(value.dtype),
                "hash": hash_numpy(value),
                "shape": list(value.shape),
            }
        if is_dataclass(value) and not isinstance(value, type):
            return self._normalize_for_json(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._normalize_for_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [self._normalize_for_json(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
