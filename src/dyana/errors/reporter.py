from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .config import ErrorHandlingConfig
from .logging import JsonlEventLogger
from .types import FailureRecord, StepStatus


@dataclass
class ErrorReporter:
    """
    Collects step outcomes (OK / FAILED / SKIPPED) and renders end-of-run summaries.

    Design notes
    ------------
    - This is the single source of truth for "what happened"
    - Pipeline and guards should report here; they shouldn't decide formatting.

    Usage example
    -------------
        reporter = ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)
        reporter.mark_ok("load")
        reporter.mark_failed("parse", exc, context={"paper_id": "p1"})
        print(reporter.render_summary())
    """

    cfg: ErrorHandlingConfig
    logger: logging.Logger
    event_logger: Optional[JsonlEventLogger] = None

    def __post_init__(self) -> None:
        """Initialize run-scoped state."""
        self._run_id = self.cfg.resolved_run_id()
        self._records: list[FailureRecord] = []
        self._status: dict[str, StepStatus] = {}

    def status(self, step_name: str) -> Optional[StepStatus]:
        """Return the current status for a step, if present."""
        return self._status.get(step_name)

    def ok(self, step_name: str) -> bool:
        """Return True if the step completed successfully."""
        return self._status.get(step_name) == StepStatus.OK

    def failed(self, step_name: str) -> bool:
        """Return True if the step failed."""
        return self._status.get(step_name) == StepStatus.FAILED

    def skipped(self, step_name: str) -> bool:
        """Return True if the step was skipped."""
        return self._status.get(step_name) == StepStatus.SKIPPED

    def failures_count(self) -> int:
        """Return the total number of failed steps."""
        return sum(1 for s in self._status.values() if s == StepStatus.FAILED)

    def has_failures(self) -> bool:
        """Return True when any step has failed."""
        return self.failures_count() > 0

    def mark_ok(self, step_name: str) -> None:
        """Record a successful step."""
        self._status[step_name] = StepStatus.OK
        if self.event_logger is not None:
            self.event_logger.write(event="step_ok", step=step_name, level="INFO")

    def mark_skipped(self, *, step_name: str, caused_by: str, context: Optional[Mapping[str, Any]] = None) -> None:
        """Record a skipped step and its dependency cause."""
        self._status[step_name] = StepStatus.SKIPPED
        rec = FailureRecord(
            step_name=step_name,
            status=StepStatus.SKIPPED,
            message=f"Skipped because dependency '{caused_by}' failed or was skipped.",
            context=context,
            caused_by=caused_by,
        )
        self._records.append(rec)
        self.logger.warning("Skipping step '%s' (caused_by=%s)", step_name, caused_by, extra={"step": step_name})
        if self.event_logger is not None:
            self.event_logger.write(
                event="step_skipped",
                step=step_name,
                level="WARNING",
                context=context,
                message=rec.message,
            )

    def mark_failed(self, *, step_name: str, exc: BaseException, context: Optional[Mapping[str, Any]] = None) -> None:
        """Record a failed step and associated exception details."""
        self._status[step_name] = StepStatus.FAILED
        rec = FailureRecord.from_exception(step_name=step_name, exc=exc, context=context)
        self._records.append(rec)

        self.logger.error(
            "Step '%s' failed: %s (%s)",
            step_name,
            str(exc),
            type(exc).__name__,
            extra={"step": step_name},
        )
        self.logger.debug("Traceback for step '%s':\n%s", step_name, rec.traceback, extra={"step": step_name})

        if self.event_logger is not None:
            self.event_logger.write(
                event="step_failed",
                step=step_name,
                level="ERROR",
                context=context,
                exc=exc,
            )

    def render_summary(self) -> str:
        """Render a human-readable summary with details and artifact paths."""
        ok_n = sum(1 for s in self._status.values() if s == StepStatus.OK)
        fail_n = sum(1 for s in self._status.values() if s == StepStatus.FAILED)
        skip_n = sum(1 for s in self._status.values() if s == StepStatus.SKIPPED)

        lines: list[str] = []
        lines.append(f"Run summary (run_id={self._run_id}, mode={self.cfg.mode})")
        lines.append(f"  OK:   {ok_n}")
        lines.append(f"  FAIL: {fail_n}")
        lines.append(f"  SKIP: {skip_n}")

        if fail_n + skip_n == 0:
            return "\n".join(lines)

        lines.append("")
        lines.append("Details:")
        for rec in self._records:
            if rec.status == StepStatus.FAILED:
                lines.append(f"  - FAIL {rec.step_name}: {rec.exc_type}: {rec.message}")
            else:
                lines.append(f"  - SKIP {rec.step_name}: {rec.message}")

        lines.append("")
        lines.append(f"Artifacts:")
        lines.append(f"  - {str(self.cfg.log_dir / f'run_{self._run_id}.log')}")
        if self.cfg.write_jsonl:
            lines.append(f"  - {str(self.cfg.log_dir / f'events_{self._run_id}.jsonl')}")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """
        Print summary to console, using Rich if available.

        Usage example
        -------------
            reporter.print_summary()
        """
        text = self.render_summary()
        try:
            from rich.console import Console  # type: ignore

            Console().print(text)
        except Exception:
            print(text)

    def exit_code(self) -> int:
        """Return a conventional process exit code: 0 if success, else 1."""
        return 0 if not self.has_failures() else 1
