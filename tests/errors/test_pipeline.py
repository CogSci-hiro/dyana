from pathlib import Path
import logging
import pytest

from dyana.errors import ErrorHandlingConfig, configure_logging, ErrorReporter, Pipeline
from dyana.errors.types import StepStatus


def _make_reporter(*, tmp_path: Path, mode: str, max_failures: int | None = None) -> ErrorReporter:
    cfg = ErrorHandlingConfig(
        mode=mode,  # type: ignore[arg-type]
        log_dir=tmp_path / "logs",
        run_id="testrun",
        write_jsonl=False,
        max_failures=max_failures,
        console_level=logging.CRITICAL,  # keep test output quiet
    )
    logger, event_logger = configure_logging(cfg=cfg)
    return ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)


def test_pipeline_happy_path_runs_all_and_returns_results(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run")
    pipe = Pipeline(reporter)

    pipe.add("A", lambda: 1)
    pipe.add("B", lambda: 2, deps=["A"])
    pipe.add("C", lambda: 3, deps=["B"])

    results = pipe.run()

    assert results == {"A": 1, "B": 2, "C": 3}
    assert reporter.status("A") == StepStatus.OK
    assert reporter.status("B") == StepStatus.OK
    assert reporter.status("C") == StepStatus.OK
    assert reporter.has_failures() is False


def test_pipeline_failure_skips_dependents_but_runs_independent(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run")
    pipe = Pipeline(reporter)

    ran: list[str] = []

    def a_fail() -> int:
        ran.append("A")
        raise ValueError("nope")

    def b_should_skip() -> int:
        ran.append("B")
        return 2

    def c_independent_ok() -> int:
        ran.append("C")
        return 3

    pipe.add("A", a_fail)
    pipe.add("B", b_should_skip, deps=["A"])
    pipe.add("C", c_independent_ok)

    results = pipe.run()

    assert "C" in results and results["C"] == 3
    assert "A" not in results
    assert "B" not in results

    assert reporter.status("A") == StepStatus.FAILED
    assert reporter.status("B") == StepStatus.SKIPPED
    assert reporter.status("C") == StepStatus.OK

    assert ran == ["A", "C"]  # deterministic because steps are sorted by name


def test_pipeline_debug_mode_raises_on_failure(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="debug")
    pipe = Pipeline(reporter)

    def boom() -> None:
        raise RuntimeError("stop")

    pipe.add("A", boom)

    with pytest.raises(RuntimeError, match="stop"):
        pipe.run()

    # Failure should be recorded before raising
    assert reporter.status("A") == StepStatus.FAILED


def test_pipeline_cycle_detection_raises(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run")
    pipe = Pipeline(reporter)

    pipe.add("A", lambda: 1, deps=["B"])
    pipe.add("B", lambda: 2, deps=["A"])

    with pytest.raises(RuntimeError, match="could not make progress"):
        pipe.run()


def test_pipeline_undefined_dependency_raises(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run")
    pipe = Pipeline(reporter)

    # "B" is never added, so "A" can never be decided -> no progress -> raises
    pipe.add("A", lambda: 1, deps=["B"])

    with pytest.raises(RuntimeError, match="could not make progress"):
        pipe.run()


def test_pipeline_max_failures_stops_scheduling_and_marks_remaining_skipped(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, mode="run", max_failures=1)
    pipe = Pipeline(reporter)

    ran: list[str] = []

    def a_fail() -> None:
        ran.append("A_fail")
        raise ValueError("first")

    def b_ok() -> str:
        ran.append("B_ok")
        return "ok"

    def c_ok() -> str:
        ran.append("C_ok")
        return "ok"

    # Deterministic order: A_fail, B_ok, C_ok
    # After A_fail, failures_count == 1, then remaining steps get skipped due to max_failures.
    pipe.add("A_fail", a_fail)
    pipe.add("B_ok", b_ok)
    pipe.add("C_ok", c_ok)

    results = pipe.run()

    assert ran == ["A_fail"]
    assert results == {}

    assert reporter.status("A_fail") == StepStatus.FAILED
    assert reporter.status("B_ok") == StepStatus.SKIPPED
    assert reporter.status("C_ok") == StepStatus.SKIPPED
    assert reporter.failures_count() == 1
