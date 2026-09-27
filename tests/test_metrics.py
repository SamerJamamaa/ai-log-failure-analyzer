import os

import pandas as pd
import pytest

from src.db import create_connection, create_schema, replace_executions
from src.ingestion import load_and_validate
from src.metrics import compute_kpis
from src.models import REQUIRED_COLUMNS

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_logs_small.csv")


def test_kpis_match_hand_computed_values():
    df, report = load_and_validate(FIXTURE)
    assert report.valid_rows == 6

    conn = create_connection()
    replace_executions(conn, df)
    kpis = compute_kpis(conn)

    # Valid rows (durations): 5.0, 6.0, 4.0, 5.5, 5.0, 4.5 -> 4 success, 2 failure
    assert kpis.total_executions == 6
    assert kpis.failure_count == 2
    assert kpis.success_rate == pytest.approx(66.666, abs=0.01)
    assert kpis.avg_duration_seconds == pytest.approx(5.0, abs=0.001)
    assert kpis.p95_duration_seconds == pytest.approx(5.875, abs=0.001)
    assert kpis.most_common_error_type == "TimeoutError"

    assert len(kpis.by_workflow) == 1
    wf = kpis.by_workflow[0]
    assert wf.workflow_name == "WF_A"
    assert wf.total_executions == 6
    assert wf.failure_count == 2


def test_compute_kpis_handles_empty_dataset_safely():
    conn = create_connection()
    replace_executions(conn, pd.DataFrame(columns=REQUIRED_COLUMNS))
    kpis = compute_kpis(conn)

    assert kpis.total_executions == 0
    assert kpis.success_rate == 0.0
    assert kpis.failure_count == 0
    assert kpis.avg_duration_seconds == 0.0
    assert kpis.p95_duration_seconds == 0.0
    assert kpis.most_common_error_type is None
    assert kpis.by_workflow == []


def test_compute_kpis_handles_table_with_no_rows_inserted_yet():
    # create_schema() only — replace_executions() was never called
    conn = create_connection()
    create_schema(conn)

    kpis = compute_kpis(conn)

    assert kpis.total_executions == 0
    assert kpis.by_workflow == []


def test_compute_kpis_handles_single_record_workflow():
    conn = create_connection()
    df = pd.DataFrame(
        [
            {
                "execution_id": "e1",
                "workflow_name": "WF",
                "start_time": "2026-01-01 00:00:00",
                "status": "success",
                "duration_seconds": 7.5,
                "error_type": "",
                "error_message": "",
            }
        ]
    )
    replace_executions(conn, df)

    kpis = compute_kpis(conn)

    assert kpis.total_executions == 1
    assert kpis.success_rate == 100.0
    assert kpis.avg_duration_seconds == pytest.approx(7.5)
    assert kpis.p95_duration_seconds == pytest.approx(7.5)
