import os
import sqlite3

import pandas as pd
import pytest

from src.db import (
    count_executions,
    create_connection,
    create_schema,
    fetch_all_executions,
    fetch_executions_by_workflow,
    replace_executions,
)
from src.ingestion import load_and_validate

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_logs_small.csv")


def _row(execution_id, workflow_name, status="success", duration=1.0, error_type="", error_message="", start="2026-01-01 00:00:00"):
    return {
        "execution_id": execution_id,
        "workflow_name": workflow_name,
        "start_time": start,
        "status": status,
        "duration_seconds": duration,
        "error_type": error_type,
        "error_message": error_message,
    }


def test_create_schema_creates_executions_table():
    conn = create_connection()
    create_schema(conn)

    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("executions",) in tables


def test_replace_executions_stores_only_the_given_validated_rows():
    df, report = load_and_validate(FIXTURE)
    assert report.valid_rows == 6

    conn = create_connection()
    replace_executions(conn, df)

    assert count_executions(conn) == 6
    stored = fetch_all_executions(conn)
    assert set(stored["execution_id"]) == set(df["execution_id"])


def test_replace_executions_replaces_the_active_dataset():
    conn = create_connection()
    first_df = pd.DataFrame([_row("a1", "WF")])
    second_df = pd.DataFrame(
        [
            _row("b1", "WF", start="2026-01-02 00:00:00"),
            _row("b2", "WF", status="failure", error_type="TimeoutError", error_message="x", start="2026-01-02 01:00:00"),
        ]
    )

    replace_executions(conn, first_df)
    assert count_executions(conn) == 1

    replace_executions(conn, second_df)

    assert count_executions(conn) == 2
    assert set(fetch_all_executions(conn)["execution_id"]) == {"b1", "b2"}  # a1 is gone


def test_fetch_executions_by_workflow_uses_a_parameterized_filter():
    conn = create_connection()
    df = pd.DataFrame([_row("a1", "WF_A"), _row("b1", "WF_B")])
    replace_executions(conn, df)

    result = fetch_executions_by_workflow(conn, "WF_A")

    assert len(result) == 1
    assert result.iloc[0]["execution_id"] == "a1"


def test_execution_id_is_enforced_as_primary_key():
    conn = create_connection()
    create_schema(conn)
    conn.execute(
        "INSERT INTO executions (execution_id, workflow_name, start_time, status, duration_seconds, error_type, error_message) "
        "VALUES ('dup', 'WF', '2026-01-01', 'success', 1.0, '', '')"
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO executions (execution_id, workflow_name, start_time, status, duration_seconds, error_type, error_message) "
            "VALUES ('dup', 'WF', '2026-01-02', 'success', 1.0, '', '')"
        )


def test_workflow_names_containing_sql_syntax_are_stored_and_queried_safely():
    """Proves the parameterized queries treat this as literal data, never as
    SQL to execute — a naive string-formatted query would either break or,
    worse, actually drop the table."""
    conn = create_connection()
    malicious_name = "WF'; DROP TABLE executions; --"
    df = pd.DataFrame([_row("a1", malicious_name)])

    replace_executions(conn, df)
    result = fetch_executions_by_workflow(conn, malicious_name)

    assert len(result) == 1
    assert result.iloc[0]["workflow_name"] == malicious_name
    assert count_executions(conn) == 1  # table still exists and is intact
