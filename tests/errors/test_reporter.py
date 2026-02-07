from pathlib import Path
import logging
import json

from dyana.errors import ErrorHandlingConfig, configure_logging, ErrorReporter
from dyana.errors.types import StepStatus


def _make_reporter(*, tmp_path: Path, write_jsonl: bool = False) -> ErrorReporter:
    cfg = ErrorHandlingConfig(
        mode="run",
        log_dir=tmp_path / "logs",
        run_id="testrun",
        write_jsonl=write_jsonl,
        console_level=logging.CRITICAL,
    )
    logger, event_logger = configure_logging(cfg=cfg)
    return ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)


def test_mark_ok_and_queries(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path)

    reporter.mark_ok("step1")

    assert reporter.status("step1") == StepStatus.OK
    assert reporter.ok("step1") is True
    assert reporter.failed("step1") is False
    assert reporter.skipped("step1") is False
    assert reporter.has_failures() is False


def test_mark_failed_records_failure(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path)

    try:
        raise ValueError("boom")
    except ValueError as exc:
        reporter.mark_failed(step_name="step_fail", exc=exc, context={"x": 1})

    assert reporter.status("step_fail") == StepStatus.FAILED
    assert reporter.failed("step_fail") is True
    assert reporter.has_failures() is True
    assert reporter.failures_count() == 1

    # Inspect stored record via rendered summary (black-box)
    summary = reporter.render_summary()
    assert "FAIL step_fail" in summary
    assert "ValueError" in summary
    assert "boom" in summary


def test_mark_skipped_records_cause(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path)

    reporter.mark_failed(step_name="A", exc=RuntimeError("nope"))
    reporter.mark_skipped(step_name="B", caused_by="A")

    assert reporter.status("B") == StepStatus.SKIPPED
    assert reporter.skipped("B") is True

    summary = reporter.render_summary()
    assert "SKIP B" in summary
    assert "dependency 'A'" in summary


def test_render_summary_includes_artifacts(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path)

    reporter.mark_ok("ok1")
    reporter.mark_failed(step_name="fail1", exc=ValueError("bad"))

    text = reporter.render_summary()

    assert "Run summary" in text
    assert "OK:" in text
    assert "FAIL:" in text
    assert "Artifacts:" in text
    assert "run_testrun.log" in text


def test_exit_code(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path)

    reporter.mark_ok("ok")
    assert reporter.exit_code() == 0

    reporter.mark_failed(step_name="fail", exc=ValueError("x"))
    assert reporter.exit_code() == 1


def test_jsonl_events_written(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path=tmp_path, write_jsonl=True)

    reporter.mark_ok("ok_step")
    reporter.mark_failed(step_name="fail_step", exc=RuntimeError("bad"))

    jsonl_path = reporter.cfg.log_dir / "events_testrun.jsonl"
    assert jsonl_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2

    payloads = [json.loads(l) for l in lines]
    events = {p["event"] for p in payloads}

    assert "step_ok" in events
    assert "step_failed" in events
