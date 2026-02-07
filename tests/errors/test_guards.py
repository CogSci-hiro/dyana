from pathlib import Path
import json

import pytest

from dyana.errors import ErrorHandlingConfig, configure_logging, ErrorReporter
from dyana.errors.guards import guard
from dyana.errors.types import StepStatus


def _make_reporter(*, tmp_path: Path, mode: str, write_jsonl: bool = False) -> ErrorReporter:
    cfg = ErrorHandlingConfig(
        mode=mode,  # type: ignore[arg-type]
        log_dir=tmp_path / "logs",
        run_id="testrun",
        write_jsonl=write_jsonl,
    )
    logger, event_logger = configure_logging(cfg=cfg)
    return ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)


def test_guard_success_marks_ok_and_returns_value(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run")

    out = guard("step_ok", reporter, lambda: 123, default=None)

    assert out == 123
    assert reporter.status("step_ok") == StepStatus.OK
    assert reporter.has_failures() is False


def test_guard_failure_run_mode_records_failed_and_returns_default(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run")

    def boom() -> int:
        raise ValueError("nope")

    out = guard("step_fail", reporter, boom, default=999)

    assert out == 999
    assert reporter.status("step_fail") == StepStatus.FAILED
    assert reporter.has_failures() is True


def test_guard_failure_debug_mode_raises(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="debug")

    def boom() -> int:
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        _ = guard("step_fail", reporter, boom, default=999)

    # It should still have recorded the failure before re-raising
    assert reporter.status("step_fail") == StepStatus.FAILED


def test_guard_writes_jsonl_events_if_enabled(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run", write_jsonl=True)

    _ = guard("step_ok", reporter, lambda: "hi", default=None)

    jsonl_path = reporter.cfg.log_dir / "events_testrun.jsonl"
    assert jsonl_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    payload = json.loads(lines[0])
    assert payload["run_id"] == "testrun"
    assert payload["event"] in {"step_ok"}  # could include other initial events in future
    assert payload["step"] == "step_ok"
