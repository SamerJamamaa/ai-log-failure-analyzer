import io
import json
import os

import pandas as pd

from src.ingestion import load_and_validate

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_logs_small.csv")


def _json_file(records):
    return io.StringIO(json.dumps(records))


def test_valid_rows_and_rejection_counts():
    df, report = load_and_validate(FIXTURE)

    assert report.ok
    assert report.total_rows == 10
    assert report.valid_rows == 6
    assert report.rejected_rows == 4
    assert report.rejection_reasons["duplicate execution_id"] == 1
    assert report.rejection_reasons["unparsable start_time"] == 1
    assert report.rejection_reasons["invalid duration_seconds"] == 1
    assert report.rejection_reasons["invalid status"] == 1

    assert len(df) == 6
    assert set(df["execution_id"]) == {"exec_1", "exec_2", "exec_3", "exec_4", "exec_9", "exec_10"}


def test_csv_without_any_optional_columns_still_gets_clean_defaults():
    """Backward compatibility: an existing plain CSV (only the 7 required
    columns, no environment/service/http_status/...) must keep working
    exactly as before, with every optional column present downstream anyway
    so fingerprinting/grouping code never needs a KeyError guard."""
    df, report = load_and_validate(FIXTURE)

    assert report.ok
    from src.models import OPTIONAL_NUMERIC_COLUMNS, OPTIONAL_TEXT_COLUMNS

    for col in OPTIONAL_TEXT_COLUMNS:
        assert col in df.columns
        assert (df[col] == "").all()
    for col in OPTIONAL_NUMERIC_COLUMNS:
        assert col in df.columns
        assert df[col].isna().all()


def test_csv_with_extra_optional_columns_incorporates_them():
    csv_text = (
        "execution_id,workflow_name,start_time,status,duration_seconds,error_type,error_message,service,http_status\n"
        "exec_1,WF,2026-01-01 10:00:00,failure,5.0,API_TIMEOUT,timed out,whatsapp_gateway,504\n"
    )
    df, report = load_and_validate(io.StringIO(csv_text))

    assert report.ok
    assert report.valid_rows == 1
    assert df.iloc[0]["service"] == "whatsapp_gateway"
    assert df.iloc[0]["http_status"] == 504


def test_missing_required_column_is_fatal():
    bad_csv = (
        "execution_id,workflow_name,start_time,status,duration_seconds\n"
        "exec_1,WF,2026-01-01,success,5\n"
    )
    df, report = load_and_validate(io.StringIO(bad_csv))

    assert not report.ok
    assert "error_type" in report.fatal_error
    assert df.empty


def test_unreadable_file_is_fatal():
    df, report = load_and_validate(io.StringIO(""))

    assert not report.ok
    assert df.empty


def test_failed_execution_without_error_info_is_rejected():
    csv_text = (
        "execution_id,workflow_name,start_time,status,duration_seconds,error_type,error_message\n"
        "exec_1,WF,2026-01-01 10:00:00,failure,5.0,,\n"
        "exec_2,WF,2026-01-01 10:05:00,failure,5.0,TimeoutError,timed out\n"
    )
    df, report = load_and_validate(io.StringIO(csv_text))

    assert report.ok
    assert report.valid_rows == 1
    assert report.rejected_rows == 1
    assert report.rejection_reasons["failed execution missing error information"] == 1
    assert df.iloc[0]["execution_id"] == "exec_2"


def test_rejected_row_details_have_correct_row_numbers_and_messages():
    # FIXTURE data rows, 1-indexed with header as row 1:
    #   row 6  = duplicate exec_2      -> "duplicate execution_id"
    #   row 7  = exec_6, bad timestamp -> "unparsable start_time"
    #   row 8  = exec_7, duration -3.0 -> "invalid duration_seconds"
    #   row 9  = exec_8, bad status    -> "invalid status"
    df, report = load_and_validate(FIXTURE)

    by_row = {r.row_number: r.reasons for r in report.rejected_row_details}
    assert by_row[6] == ["duplicate execution_id"]
    assert by_row[7] == ["unparsable start_time"]
    assert by_row[8] == ["invalid duration_seconds"]
    assert by_row[9] == ["invalid status"]

    messages = [r.message() for r in report.rejected_row_details]
    assert "Row 6: duplicate execution_id" in messages


# --- JSON format -----------------------------------------------------------------


def test_valid_json_loads_and_translates_failed_to_failure():
    records = [
        {
            "execution_id": "EXE-1001",
            "workflow_name": "WhatsApp Lead Intake",
            "start_time": "2026-09-20T10:15:00Z",
            "status": "success",
            "duration_seconds": 2.4,
            "error": None,
        },
        {
            "execution_id": "EXE-1002",
            "workflow_name": "WhatsApp Lead Intake",
            "workflow_step": "Send WhatsApp Message",
            "start_time": "2026-09-20T10:17:00Z",
            "status": "failed",
            "duration_seconds": 15.8,
            "environment": "production",
            "service": "whatsapp_gateway",
            "provider": "external_whatsapp_provider",
            "endpoint": "/messages/send",
            "http_method": "POST",
            "http_status": 504,
            "correlation_id": "CORR-83921",
            "retry_count": 2,
            "application_version": "1.4.2",
            "error": {
                "type": "API_TIMEOUT",
                "code": "GATEWAY_TIMEOUT",
                "message": "Provider did not respond within 15 seconds",
                "stack_summary": "TimeoutError in WhatsAppClient.send_message",
            },
        },
    ]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.ok
    assert report.valid_rows == 2
    assert report.rejected_rows == 0
    assert list(df["status"]) == ["success", "failure"]  # "failed" -> "failure", internal vocabulary
    assert df.iloc[1]["error_type"] == "API_TIMEOUT"
    assert df.iloc[1]["error_message"] == "Provider did not respond within 15 seconds"

    # optional technical fields are now carried into the model, top-level ones
    # verbatim and the nested error.code/error.stack_summary flattened
    row = df.iloc[1]
    assert row["workflow_step"] == "Send WhatsApp Message"
    assert row["environment"] == "production"
    assert row["service"] == "whatsapp_gateway"
    assert row["provider"] == "external_whatsapp_provider"
    assert row["endpoint"] == "/messages/send"
    assert row["http_method"] == "POST"
    assert row["http_status"] == 504
    assert row["correlation_id"] == "CORR-83921"
    assert row["retry_count"] == 2
    assert row["application_version"] == "1.4.2"
    assert row["error_code"] == "GATEWAY_TIMEOUT"
    assert row["error_stack_summary"] == "TimeoutError in WhatsAppClient.send_message"

    # a record that provides none of the optional fields still gets clean defaults
    first = df.iloc[0]
    assert first["environment"] == ""
    assert first["service"] == ""
    assert pd.isna(first["http_status"])
    assert pd.isna(first["retry_count"])


def test_malformed_json_syntax_is_fatal_without_raw_exception_text():
    df, report = load_and_validate(io.StringIO("not valid json"), filename="log.json")

    assert not report.ok
    assert df.empty
    assert "json" in report.fatal_error.lower()
    # never leak the parser's raw exception text (e.g. "Expecting value: line 1 column 1...")
    assert "line" not in report.fatal_error.lower()
    assert "column" not in report.fatal_error.lower()


def test_json_top_level_must_be_an_array():
    df, report = load_and_validate(_json_file({"a": 1}), filename="log.json")  # a single object, not an array

    assert not report.ok
    assert df.empty
    assert "array" in report.fatal_error.lower()


def _base_record(**overrides):
    record = {
        "execution_id": "A",
        "workflow_name": "WF",
        "start_time": "2026-09-20T10:00:00Z",
        "status": "success",
        "duration_seconds": 1.0,
        "error": None,
    }
    record.update(overrides)
    return record


def test_json_rejects_error_present_on_successful_execution():
    records = [_base_record(status="success", error={"type": "X", "message": "Y"})]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.rejected_rows == 1
    assert report.rejection_reasons["error must be null for a successful execution"] == 1


def test_json_requires_error_type_and_message_for_failed_execution():
    records = [_base_record(status="failed", duration_seconds=5.0, error=None)]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.rejected_rows == 1
    assert report.rejection_reasons["failed execution missing error information"] == 1


def test_json_rejects_negative_or_non_integer_retry_count():
    # retry_count is a top-level field (alongside service, http_status, ...),
    # not nested inside `error`
    records = [
        _base_record(execution_id="A", status="failed", duration_seconds=5.0, error={"type": "X", "message": "Y"}, retry_count=-1),
        _base_record(execution_id="B", status="failed", duration_seconds=5.0, error={"type": "X", "message": "Y"}, retry_count=1.5),
        _base_record(execution_id="C", status="failed", duration_seconds=5.0, error={"type": "X", "message": "Y"}, retry_count=True),
    ]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.rejected_rows == 3
    assert report.rejection_reasons["retry_count must be a non-negative integer"] == 3


def test_json_accepts_valid_retry_count_and_other_optional_fields():
    records = [
        _base_record(
            status="failed",
            duration_seconds=5.0,
            error={"type": "X", "message": "Y"},
            retry_count=2,
            http_status=504,
            service="whatsapp_gateway",
        )
    ]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.ok
    assert report.rejected_rows == 0
    assert df.iloc[0]["retry_count"] == 2
    assert df.iloc[0]["http_status"] == 504
    assert df.iloc[0]["service"] == "whatsapp_gateway"


def test_json_requires_strictly_positive_duration():
    records = [_base_record(duration_seconds=0)]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.rejected_rows == 1
    assert report.rejection_reasons["invalid duration_seconds"] == 1


def test_json_requires_iso8601_timestamp():
    records = [_base_record(start_time="09/20/2026")]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.rejected_rows == 1
    assert report.rejection_reasons["start_time is not a valid ISO 8601 timestamp"] == 1


def test_json_rejects_duplicate_execution_ids():
    records = [_base_record(execution_id="A"), _base_record(execution_id="A", start_time="2026-09-20T10:05:00Z")]
    df, report = load_and_validate(_json_file(records), filename="log.json")

    assert report.valid_rows == 1
    assert report.rejection_reasons["duplicate execution_id"] == 1


def test_json_rejects_missing_required_field():
    record = _base_record()
    del record["execution_id"]
    df, report = load_and_validate(_json_file([record]), filename="log.json")

    assert report.rejected_rows == 1
    assert report.rejection_reasons["missing execution_id"] == 1


def test_extension_detection_falls_back_to_csv_when_unknown():
    """No filename and no .json extension -> treated as CSV, same as before
    JSON support existed. Keeps every pre-existing call site unaffected."""
    csv_text = "execution_id,workflow_name,start_time,status,duration_seconds,error_type,error_message\n"
    df, report = load_and_validate(io.StringIO(csv_text))

    assert report.ok  # parsed as an (empty) CSV, not rejected as bad JSON
