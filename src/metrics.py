import sqlite3

import numpy as np
import pandas as pd

from src.models import KPISet, WorkflowKPI


def _p95(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    return float(np.percentile(clean, 95))


def compute_kpis(conn: sqlite3.Connection) -> KPISet:
    """All aggregate counts/rates/averages are computed via SQL against the
    executions table; P95 is computed in Python (numpy) over the SQL-sourced
    duration column, since SQLite has no built-in percentile function."""
    overall = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_executions,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) AS failure_count,
            AVG(duration_seconds) AS avg_duration_seconds
        FROM executions
        """,
        conn,
    ).iloc[0]

    total = int(overall["total_executions"])
    success_count = int(overall["success_count"] or 0)
    failure_count = int(overall["failure_count"] or 0)
    success_rate = (success_count / total * 100) if total else 0.0
    avg_duration = float(overall["avg_duration_seconds"] or 0.0)

    durations = pd.read_sql_query("SELECT duration_seconds FROM executions", conn)["duration_seconds"]
    p95_duration = _p95(durations)

    most_common_error = pd.read_sql_query(
        """
        SELECT error_type, COUNT(*) AS cnt
        FROM executions
        WHERE status = 'failure' AND error_type != ''
        GROUP BY error_type
        ORDER BY cnt DESC, error_type ASC
        LIMIT 1
        """,
        conn,
    )
    most_common_error_type = (
        most_common_error.iloc[0]["error_type"] if not most_common_error.empty else None
    )

    by_workflow_df = pd.read_sql_query(
        """
        SELECT
            workflow_name,
            COUNT(*) AS total_executions,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) AS failure_count,
            AVG(duration_seconds) AS avg_duration_seconds
        FROM executions
        GROUP BY workflow_name
        ORDER BY workflow_name
        """,
        conn,
    )

    by_workflow = []
    for _, row in by_workflow_df.iterrows():
        wf_durations = pd.read_sql_query(
            "SELECT duration_seconds FROM executions WHERE workflow_name = ?",
            conn,
            params=(row["workflow_name"],),
        )["duration_seconds"]
        wf_total = int(row["total_executions"])
        wf_success = int(row["success_count"] or 0)
        by_workflow.append(
            WorkflowKPI(
                workflow_name=row["workflow_name"],
                total_executions=wf_total,
                success_rate=(wf_success / wf_total * 100) if wf_total else 0.0,
                failure_count=int(row["failure_count"] or 0),
                avg_duration_seconds=float(row["avg_duration_seconds"] or 0.0),
                p95_duration_seconds=_p95(wf_durations),
            )
        )

    return KPISet(
        total_executions=total,
        success_rate=success_rate,
        failure_count=failure_count,
        avg_duration_seconds=avg_duration,
        p95_duration_seconds=p95_duration,
        most_common_error_type=most_common_error_type,
        by_workflow=by_workflow,
    )
