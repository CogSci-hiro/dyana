from __future__ import annotations

from pathlib import Path

from dyana.errors.config import ErrorHandlingConfig


def test_resolved_run_id_returns_explicit_id() -> None:
    cfg = ErrorHandlingConfig(run_id="myrun")
    assert cfg.resolved_run_id() == "myrun"


def test_resolved_run_id_auto_is_non_empty_and_changes() -> None:
    cfg = ErrorHandlingConfig(run_id="auto")
    a = cfg.resolved_run_id()
    b = cfg.resolved_run_id()
    assert isinstance(a, str) and len(a) > 0
    assert isinstance(b, str) and len(b) > 0
    # Very low probability of collision, but still essentially safe for unit testing here
    assert a != b


def test_from_env_reads_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ERROR_MODE", "debug")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "mylogs"))
    monkeypatch.setenv("WRITE_JSONL", "0")
    monkeypatch.setenv("MAX_FAILURES", "7")

    cfg = ErrorHandlingConfig.from_env()

    assert cfg.mode == "debug"
    assert cfg.log_dir == tmp_path / "mylogs"
    assert cfg.write_jsonl is False
    assert cfg.max_failures == 7


def test_from_env_invalid_values_fall_back(monkeypatch) -> None:
    base = ErrorHandlingConfig(mode="run", write_jsonl=True, max_failures=None, log_dir=Path("logs"))

    monkeypatch.setenv("ERROR_MODE", "nonsense")
    monkeypatch.setenv("WRITE_JSONL", "maybe")  # treated as truthy in our implementation
    monkeypatch.setenv("MAX_FAILURES", "abc")   # invalid -> fallback to base.max_failures

    cfg = ErrorHandlingConfig.from_env(default=base)

    assert cfg.mode == "run"              # fallback
    assert cfg.max_failures is None       # fallback
    assert cfg.write_jsonl is True        # "maybe" is not in ("0","false","False","")


def test_from_env_respects_prefix(monkeypatch, tmp_path: Path) -> None:
    base = ErrorHandlingConfig(env_prefix="DYANA_", log_dir=Path("logs"))

    monkeypatch.setenv("DYANA_ERROR_MODE", "debug")
    monkeypatch.setenv("DYANA_LOG_DIR", str(tmp_path / "pref_logs"))

    cfg = ErrorHandlingConfig.from_env(default=base)

    assert cfg.mode == "debug"
    assert cfg.log_dir == tmp_path / "pref_logs"
