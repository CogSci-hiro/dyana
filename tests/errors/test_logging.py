from __future__ import annotations

import json
import logging
from pathlib import Path

from dyana.errors.config import ErrorHandlingConfig
from dyana.errors.logging import JsonlEventLogger, configure_logging


def test_jsonl_event_logger_writes_valid_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "events_abc.jsonl"
    ev = JsonlEventLogger(path=path, run_id="abc")

    ev.write(event="step_ok", step="load", level="INFO")
    ev.write(event="step_failed", step="parse", level="ERROR", context={"paper_id": "p1"}, exc=ValueError("nope"))

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    a = json.loads(lines[0])
    assert a["run_id"] == "abc"
    assert a["event"] == "step_ok"
    assert a["step"] == "load"
    assert a["level"] == "INFO"
    assert "time_utc" in a

    b = json.loads(lines[1])
    assert b["event"] == "step_failed"
    assert b["step"] == "parse"
    assert b["level"] == "ERROR"
    assert b["context"]["paper_id"] == "p1"
    assert b["exc_type"] == "ValueError"
    assert b["exc_msg"] == "nope"


def test_configure_logging_creates_log_file_and_writes(tmp_path: Path) -> None:
    cfg = ErrorHandlingConfig(
        mode="run",
        log_dir=tmp_path / "logs",
        run_id="testrun",
        write_jsonl=False,
        console_level=logging.CRITICAL,  # keep test output quiet
        file_level=logging.DEBUG,
    )

    logger, event_logger = configure_logging(cfg=cfg)
    assert event_logger is None

    log_path = cfg.log_dir / "run_testrun.log"
    assert log_path.exists()

    # Write without specifying extra={"step": ...} to ensure filter injects defaults
    logger.info("hello world")

    text = log_path.read_text(encoding="utf-8")
    assert "hello world" in text
    assert "run=testrun" in text
    assert "step=-" in text  # injected default from filter


def test_configure_logging_step_extra_is_used_in_file(tmp_path: Path) -> None:
    cfg = ErrorHandlingConfig(
        mode="run",
        log_dir=tmp_path / "logs",
        run_id="testrun",
        write_jsonl=False,
        console_level=logging.CRITICAL,
        file_level=logging.DEBUG,
    )
    logger, _ = configure_logging(cfg=cfg)

    logger.error("boom", extra={"step": "parse_pdf"})

    log_path = cfg.log_dir / "run_testrun.log"
    text = log_path.read_text(encoding="utf-8")

    assert "boom" in text
    assert "step=parse_pdf" in text


def test_configure_logging_returns_event_logger_when_enabled(tmp_path: Path) -> None:
    cfg = ErrorHandlingConfig(
        mode="run",
        log_dir=tmp_path / "logs",
        run_id="testrun",
        write_jsonl=True,
        console_level=logging.CRITICAL,
        file_level=logging.DEBUG,
    )
    _, event_logger = configure_logging(cfg=cfg)
    assert event_logger is not None
    assert event_logger.path == cfg.log_dir / "events_testrun.jsonl"
    assert event_logger.run_id == "testrun"
