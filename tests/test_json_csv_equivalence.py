"""Proves that the same executions, described in JSON vs CSV, produce
identical normalized records, KPIs, and detected anomalies — the core
guarantee behind 'analytics stay independent of upload format'.
"""

import io
import json

import pandas as pd
import pytest

from src.anomalies import detect_all
from src.db import create_connection, replace_executions
from src.ingestion import load_and_validate
from src.metrics import compute_kpis

# Same 12 executions, two representations. Includes both statuses, a
# recurring error, and a duplicate-free set of IDs — enough to exercise
# KPIs and the recurring-error detection rule identically in both formats.
_EXECUTIONS = [
    {"execution_id": "E1", "workflow_name": "WF", "start_time": "2026-09-20T10:00:00Z", "status": "success", "duration_seconds": 4.0, "error": None},
    {"execution_id": "E2", "workflow_name": "WF", "start_time": "2026-09-20T10:05:00Z", "status": "success", "duration_seconds": 4.2, "error": None},
    {"execution_id": "E3", "workflow_name": "WF", "start_time": "2026-09-20T10:10:00Z", "status": "success", "duration_seconds": 3.8, "error": None},
    {"execution_id": "E4", "workflow_name": "WF", "start_time": "2026-09-20T10:15:00Z", "status": "success", "duration_seconds": 4.1, "error": None},
    {"execution_id": "E5", "workflow_name": "WF", "start_time": "2026-09-20T10:20:00Z", "status": "success", "duration_seconds": 3.9, "error": None},
    {"execution_id": "E6", "workflow_name": "WF", "start_time": "2026-09-20T10:25:00Z", "status": "success", "duration_seconds": 4.0, "error": None},
    {"execution_id": "E7", "workflow_name": "WF", "start_time": "2026-09-20T10:30:00Z", "status": "success", "duration_seconds": 4.3, "error": None},
    {
        "execution_id": "E8",
        "workflow_name": "WF",
        "start_time": "2026-09-20T10:35:00Z",
        "status": "failed",
        "duration_seconds": 15.8,
        "error": {"type": "API_TIMEOUT", "message": "Provider did not respond within 15 seconds"},
    },
    {
        "execution_id": "E9",
        "workflow_name": "WF",
        "start_time": "2026-09-20T10:40:00Z",
        "status": "failed",
        "duration_seconds": 16.2,
        "error": {"type": "API_TIMEOUT", "message": "Provider did not respond within 15 seconds", "service": "provider", "retry_count": 1},
    },
    {
        "execution_id": "E10",
        "workflow_name": "WF",
        "start_time": "2026-09-20T10:45:00Z",
        "status": "failed",
        "duration_seconds": 15.5,
        "error": {"type": "API_TIMEOUT", "message": "Provider did not respond within 15 seconds"},
    },
    {"execution_id": "E11", "workflow_name": "WF", "start_time": "2026-09-20T10:50:00Z", "status": "success", "duration_seconds": 4.0, "error": None},
    {"execution_id": "E12", "workflow_name": "WF", "start_time": "2026-09-20T10:55:00Z", "status": "success", "duration_seconds": 3.7, "error": None},
]


def _as_csv() -> io.StringIO:
    rows = []
    for e in _EXECUTIONS:
        status = "failure" if e["status"] == "failed" else "success"
        error = e["error"] or {}
        rows.append(
            {
                "execution_id": e["execution_id"],
                "workflow_name": e["workflow_name"],
                "start_time": e["start_time"].replace("T", " ").replace("Z", ""),
                "status": status,
                "duration_seconds": e["duration_seconds"],
                "error_type": error.get("type", ""),
                "error_message": error.get("message", ""),
            }
        )
    return io.StringIO(pd.DataFrame(rows).to_csv(index=False))


def _as_json() -> io.StringIO:
    return io.StringIO(json.dumps(_EXECUTIONS))


@pytest.fixture
def csv_result():
    return load_and_validate(_as_csv(), filename="executions.csv")


@pytest.fixture
def json_result():
    return load_and_validate(_as_json(), filename="executions.json")


def test_both_formats_validate_with_zero_rejections(csv_result, json_result):
    _, csv_report = csv_result
    _, json_report = json_result

    assert csv_report.ok and csv_report.rejected_rows == 0
    assert json_report.ok and json_report.rejected_rows == 0
    assert csv_report.valid_rows == json_report.valid_rows == len(_EXECUTIONS)


def test_normalized_records_are_identical(csv_result, json_result):
    csv_df, _ = csv_result
    json_df, _ = json_result

    pd.testing.assert_frame_equal(
        csv_df.sort_values("execution_id").reset_index(drop=True),
        json_df.sort_values("execution_id").reset_index(drop=True),
    )


def test_kpis_are_identical(csv_result, json_result):
    csv_df, _ = csv_result
    json_df, _ = json_result

    csv_conn = create_connection()
    replace_executions(csv_conn, csv_df)
    csv_kpis = compute_kpis(csv_conn)

    json_conn = create_connection()
    replace_executions(json_conn, json_df)
    json_kpis = compute_kpis(json_conn)

    assert csv_kpis == json_kpis


def test_detected_anomalies_are_identical(csv_result, json_result):
    csv_df, _ = csv_result
    json_df, _ = json_result

    csv_anomalies = detect_all(csv_df)
    json_anomalies = detect_all(json_df)

    assert csv_anomalies == json_anomalies
    assert any(a.type == "recurring_error" for a in csv_anomalies)  # sanity: the fixture actually exercises a rule
