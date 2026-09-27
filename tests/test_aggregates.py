import pandas as pd

from src.aggregates import (
    compute_category_summary,
    compute_group_pareto,
    compute_performance_anomalies,
    compute_period_comparison,
    compute_retry_outcomes,
    compute_version_comparison,
    compute_workflow_impact_matrix,
)
from src.exception_groups import build_exception_groups
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS


def _row(i, status="failure", error_type="ERR", http_status=None, retry_count=None, version="", workflow="WF"):
    row = {c: (pd.NA if c in OPTIONAL_NUMERIC_COLUMNS else "") for c in OPTIONAL_COLUMNS}
    row.update(
        execution_id=f"e{i}",
        workflow_name=workflow,
        start_time=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
        status=status,
        duration_seconds=2.0,
        error_type=error_type if status == "failure" else "",
        error_message=f"{error_type} occurred" if status == "failure" else "",
        http_status=http_status,
        retry_count=retry_count,
        application_version=version,
    )
    return row


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


# --- period comparison ---------------------------------------------------------


def test_period_comparison_detects_a_real_increase():
    rows = [_row(i, status="success") for i in range(10)]
    rows += [_row(10 + i, status="failure") for i in range(10)]
    result = compute_period_comparison(_df(rows))

    assert result["valid"] is True
    assert result["change_pp"] > 0


def test_period_comparison_invalid_with_too_few_records():
    rows = [_row(i, status="failure") for i in range(4)]
    result = compute_period_comparison(_df(rows))
    assert result["valid"] is False


def test_period_comparison_invalid_when_empty():
    result = compute_period_comparison(_df([]))
    assert result["valid"] is False


def test_period_comparison_rates_are_json_serializable_native_floats():
    """Regression test: pandas .mean() returns np.float64, which breaks
    json.dumps() downstream (e.g. when this feeds an LLM evidence payload)
    unless explicitly cast back to a native float."""
    import json

    rows = [_row(i, status="success") for i in range(10)] + [_row(10 + i, status="failure") for i in range(10)]
    result = compute_period_comparison(_df(rows))
    assert type(result["first_period_failure_rate"]) is float
    assert type(result["change_pp"]) is float
    json.dumps(result)


# --- version comparison ---------------------------------------------------------


def test_version_comparison_empty_with_single_version():
    rows = [_row(i, version="1.0") for i in range(5)]
    assert compute_version_comparison(_df(rows)) == []


def test_version_comparison_with_two_versions():
    rows = [_row(i, status="success", version="1.0") for i in range(10)]
    rows += [_row(10 + i, status="failure", error_type="X", version="2.0") for i in range(5)]
    result = compute_version_comparison(_df(rows))

    assert {r["version"] for r in result} == {"1.0", "2.0"}
    v2 = next(r for r in result if r["version"] == "2.0")
    assert v2["failures"] == 5
    assert v2["failure_rate"] == 100.0


# --- workflow impact matrix ------------------------------------------------------


def test_workflow_impact_matrix_one_row_per_group():
    rows = [_row(i, error_type="A", workflow="WF1") for i in range(3)]
    rows += [_row(3 + i, error_type="B", workflow="WF2") for i in range(3)]
    groups = build_exception_groups(_df(rows))
    matrix = compute_workflow_impact_matrix(groups)

    assert len(matrix) == 2
    assert {r["workflow"] for r in matrix} == {"WF1", "WF2"}


# --- category summary / pareto ---------------------------------------------------


def test_category_summary_sums_and_cumulative_percentage_reaches_100():
    rows = [_row(i, error_type="AUTHENTICATION_ERROR", http_status=401) for i in range(6)]
    rows += [_row(6 + i, error_type="API_TIMEOUT") for i in range(4)]
    groups = build_exception_groups(_df(rows))
    summary = compute_category_summary(groups)

    assert summary[0]["occurrence_count"] >= summary[-1]["occurrence_count"]
    assert summary[-1]["cumulative_percentage"] == 100.0


def test_category_summary_empty_for_no_groups():
    assert compute_category_summary([]) == []


def test_group_pareto_uses_human_titles_and_reaches_100():
    rows = [_row(i, error_type="AUTHENTICATION_ERROR", http_status=401) for i in range(6)]
    rows += [_row(6 + i, error_type="API_TIMEOUT") for i in range(4)]
    groups = build_exception_groups(_df(rows))
    pareto = compute_group_pareto(groups)

    assert len(pareto) == 2
    assert pareto[0]["occurrence_count"] >= pareto[1]["occurrence_count"]
    assert pareto[-1]["cumulative_percentage"] == 100.0
    assert all("_" not in row["title"] for row in pareto)  # human title, not raw code


# --- retry outcomes ---------------------------------------------------------------


def test_retry_outcomes_classifies_each_bucket_correctly():
    rows = [
        _row(0, status="success", retry_count=0),  # successful without retry
        _row(1, status="success", retry_count=2),  # recovered after retry
        _row(2, status="failure", error_type="X", retry_count=0),  # failed without retry
        _row(3, status="failure", error_type="X", retry_count=1),  # failed after retry
        _row(4, status="failure", error_type="DUP", http_status=409, retry_count=1),  # duplicate after retry
        _row(5, status="failure", error_type="X", retry_count=None),  # unavailable
    ]
    counts = compute_retry_outcomes(_df(rows))

    assert counts["Successful without retry"] == 1
    assert counts["Recovered after retry"] == 1
    assert counts["Failed without retry"] == 1
    assert counts["Failed after retry"] == 1
    assert counts["Duplicate created after retry"] == 1
    assert counts["Retry information unavailable"] == 1


def test_retry_outcomes_empty_dataframe_returns_zeroed_buckets():
    counts = compute_retry_outcomes(_df([]))
    assert all(v == 0 for v in counts.values())


# --- performance anomalies ---------------------------------------------------------


def test_slow_successful_executions_are_counted_but_slow_failures_are_not():
    """A slow FAILURE (e.g. a timeout that also errored) must never be
    double-counted as a performance anomaly — only successes count here."""
    rows = [_row(i, status="success") for i in range(10)]  # normal-duration successes
    rows[0]["duration_seconds"] = 2.0
    for r in rows:
        r["duration_seconds"] = 2.0
    # inject one dramatically slow success and one dramatically slow failure
    slow_success = _row(100, status="success")
    slow_success["duration_seconds"] = 100.0
    slow_failure = _row(101, status="failure", error_type="API_TIMEOUT")
    slow_failure["duration_seconds"] = 100.0
    rows += [slow_success, slow_failure]

    result = compute_performance_anomalies(_df(rows))
    assert result["slow_successful_count"] == 1


def test_performance_anomalies_empty_dataframe_does_not_crash():
    result = compute_performance_anomalies(_df([]))
    assert result["slow_successful_count"] == 0
    assert result["latency_degradation"]["valid"] is False


def test_latency_degradation_detected_independently_of_failures():
    rows = [_row(i, status="success") for i in range(10)]
    for r in rows[:5]:
        r["duration_seconds"] = 2.0
    for r in rows[5:]:
        r["duration_seconds"] = 20.0
    result = compute_performance_anomalies(_df(rows))
    assert result["latency_degradation"]["valid"] is True
    assert result["latency_degradation"]["change_pct"] > 0
