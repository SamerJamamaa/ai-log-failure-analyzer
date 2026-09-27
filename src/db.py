import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    start_time TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    error_type TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT ''
)
"""

INSERT_SQL = """
INSERT INTO executions
    (execution_id, workflow_name, start_time, status, duration_seconds, error_type, error_message)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def create_connection() -> sqlite3.Connection:
    """Fresh in-memory SQLite DB. A new connection is created per upload —
    this app intentionally does not persist data across uploads/sessions."""
    return sqlite3.connect(":memory:", check_same_thread=False)


def create_schema(conn: sqlite3.Connection) -> None:
    """Creates the executions table if it doesn't already exist."""
    conn.execute(SCHEMA)
    conn.commit()


def replace_executions(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Replaces the active dataset with `df` (all prior rows are discarded).

    `df` must already be validated (e.g. via ingestion.load_and_validate) —
    this function stores what it's given, it does not re-validate. Insertion
    uses a parameterized executemany, never string-formatted SQL.
    """
    create_schema(conn)
    records = [
        (
            row.execution_id,
            row.workflow_name,
            str(row.start_time),
            row.status,
            float(row.duration_seconds),
            row.error_type,
            row.error_message,
        )
        for row in df.itertuples(index=False)
    ]
    conn.execute("DELETE FROM executions")
    conn.executemany(INSERT_SQL, records)
    conn.commit()


def fetch_all_executions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM executions", conn)


def fetch_executions_by_workflow(conn: sqlite3.Connection, workflow_name: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM executions WHERE workflow_name = ?", conn, params=(workflow_name,)
    )


def count_executions(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("SELECT COUNT(*) FROM executions")
    return cursor.fetchone()[0]
