from dyana.errors.types import FailureRecord, StepStatus


def test_stepstatus_values_are_stable() -> None:
    assert StepStatus.OK.value == "ok"
    assert StepStatus.FAILED.value == "failed"
    assert StepStatus.SKIPPED.value == "skipped"


def test_failure_record_from_exception_captures_fields() -> None:
    try:
        raise ValueError("nope")
    except ValueError as exc:
        rec = FailureRecord.from_exception(
            step_name="parse",
            exc=exc,
            context={"paper_id": "p1"},
        )

    assert rec.step_name == "parse"
    assert rec.status == StepStatus.FAILED
    assert rec.message == "nope"
    assert rec.exc_type == "ValueError"
    assert rec.context == {"paper_id": "p1"}
    assert rec.traceback is not None
    assert "ValueError" in rec.traceback
    assert "nope" in rec.traceback
