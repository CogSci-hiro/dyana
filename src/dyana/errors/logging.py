from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import ErrorHandlingConfig


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JsonlEventLogger:
    """
    Writes structured events as JSON lines.

    Each line is a dict that includes at least:
    - time_utc
    - run_id
    - event
    - step
    - level
    - context (optional)
    - exc_type, exc_msg (optional)

    Usage example
    -------------
        ev = JsonlEventLogger(path=Path("logs/events_abc.jsonl"), run_id="abc")
        ev.write(event="step_failed", step="parse", level="ERROR", exc=exc, context={"paper_id": "p1"})
    """
    path: Path
    run_id: str

    def write(
        self,
        *,
        event: str,
        step: Optional[str],
        level: str,
        context: Optional[Mapping[str, Any]] = None,
        exc: Optional[BaseException] = None,
        message: Optional[str] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "time_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "event": event,
            "step": step,
            "level": level,
        }
        if message:
            payload["message"] = message
        if context:
            payload["context"] = dict(context)
        if exc is not None:
            payload["exc_type"] = type(exc).__name__
            payload["exc_msg"] = str(exc)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class _RunContextFilter(logging.Filter):
    def __init__(self, *, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        # Ensure `run_id` exists for formatter
        if not hasattr(record, "run_id"):
            setattr(record, "run_id", self._run_id)
        if not hasattr(record, "step"):
            setattr(record, "step", "-")
        return True


def configure_logging(*, cfg: ErrorHandlingConfig) -> tuple[logging.Logger, Optional[JsonlEventLogger]]:
    """
    Configure console + file logging, plus optional JSONL event logger.

    Returns
    -------
    logger
        A configured logger named "app".
    event_logger
        JsonlEventLogger if cfg.write_jsonl else None.

    Usage example
    -------------
        logger, event_logger = configure_logging(cfg=cfg)
        logger.info("Hello")
    """
    run_id = cfg.resolved_run_id()
    log_dir = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("app")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    logger.addFilter(_RunContextFilter(run_id=run_id))

    # Console handler (try rich if available)
    console_handler: logging.Handler
    try:
        from rich.logging import RichHandler  # type: ignore

        console_handler = RichHandler(rich_tracebacks=(cfg.mode == "debug"), show_path=False)
        console_fmt = "%(message)s"
    except Exception:
        console_handler = logging.StreamHandler()
        console_fmt = "[%(levelname)s] %(message)s"

    console_handler.setLevel(cfg.console_level)
    console_handler.setFormatter(logging.Formatter(console_fmt))
    logger.addHandler(console_handler)

    # File handler (always plain)
    file_path = log_dir / f"run_{run_id}.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(cfg.file_level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)sZ | run=%(run_id)s | step=%(step)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    event_logger = None
    if cfg.write_jsonl:
        event_logger = JsonlEventLogger(path=log_dir / f"events_{run_id}.jsonl", run_id=run_id)

    logger.debug("Logging configured (run_id=%s, mode=%s, log_dir=%s)", run_id, cfg.mode, str(log_dir))
    return logger, event_logger
