import pandas as pd

from src.anomalies import (
    FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF,
    FAILURE_RATE_INCREASE_THRESHOLD_PP,
    RECURRING_FAILURE_MIN_COUNT,
    SLOW_EXECUTION_MIN_SAMPLES,
    SLOW_EXECUTION_STD_MULTIPLIER,
    assess_system_status,
    detect_all,
    detect_failure_rate_increase,
    detect_recurring_errors,
    detect_slow_executions,
    select_primary_issue,
)
from src.models import REQUIRED_COLUMNS, Anomaly


def _rows(n, workflow, status="success", duration=5.0, error_type="", start="2026-01-01 00:00:00", minutes_step=1):
    start_ts = pd.Timestamp(start)
    return [
        {
            "execution_id": f"{workflow}_{start}_{i}",
            "workflow_name": workflow,
            "start_time": start_ts + pd.Timedelta(minutes=i * minutes_step),
            "status": status,
            "duration_seconds": duration,
            "error_type": error_type,
            "error_message": "" if not error_type else f"{error_type} occurred",
        }
        for i in range(n)
    ]


def test_detect_slow_executions_flags_mean_plus_std_outliers():
    # A large baseline cluster at 5s (so the two outliers below don't
    # dominate the mean/std), plus two clear outliers.
    baseline = [5.0] * 20
    outliers = [50.0, 55.0]
    rows = [
        {
            "execution_id": f"e{i}",
            "workflow_name": "WF",
            "start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
            "status": "success",
            "duration_seconds": d,
            "error_type": "",
            "error_message": "",
        }
        for i, d in enumerate(baseline + outliers)
    ]
    df = pd.DataFrame(rows)

    result = detect_slow_executions(df)

    assert len(result) == 1
    assert result[0].workflow_name == "WF"
    assert result[0].evidence["outlier_count"] == 2
    mean = df["duration_seconds"].mean()
    std = df["duration_seconds"].std(ddof=0)
    assert result[0].evidence["threshold_seconds"] == round(float(mean + SLOW_EXECUTION_STD_MULTIPLIER * std), 1)


def test_detect_recurring_errors_flags_repeated_error_type_only():
    rows = _rows(RECURRING_FAILURE_MIN_COUNT, "WF", status="failure", error_type="APITimeoutError")
    rows += _rows(RECURRING_FAILURE_MIN_COUNT - 1, "WF", status="failure", error_type="AuthenticationError", start="2026-01-02 00:00:00")
    df = pd.DataFrame(rows)

    result = detect_recurring_errors(df)

    assert len(result) == 1
    assert result[0].evidence["error_type"] == "APITimeoutError"
    assert result[0].evidence["occurrence_count"] == RECURRING_FAILURE_MIN_COUNT


def test_detect_failure_rate_increase_flags_second_half_only():
    n_per_half = FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF
    stable_first = _rows(n_per_half, "Stable", status="success", start="2026-01-01 00:00:00")
    stable_second = _rows(n_per_half, "Stable", status="success", start="2026-01-02 00:00:00")

    degrading_first = _rows(n_per_half, "Degrading", status="success", start="2026-01-01 00:00:00")
    degrading_second = _rows(n_per_half, "Degrading", status="failure", start="2026-01-02 00:00:00")

    df = pd.DataFrame(stable_first + stable_second + degrading_first + degrading_second)

    result = detect_failure_rate_increase(df)

    assert len(result) == 1
    assert result[0].workflow_name == "Degrading"
    assert result[0].evidence["increase_percentage_points"] >= FAILURE_RATE_INCREASE_THRESHOLD_PP
    assert result[0].evidence["first_half_n"] == n_per_half
    assert result[0].evidence["second_half_n"] == n_per_half


def test_detect_failure_rate_increase_requires_minimum_records_per_half():
    too_few = FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF - 1
    first = _rows(too_few, "Small", status="success", start="2026-01-01 00:00:00")
    second = _rows(too_few, "Small", status="failure", start="2026-01-02 00:00:00")
    df = pd.DataFrame(first + second)

    result = detect_failure_rate_increase(df)

    assert result == []


def test_detect_all_handles_empty_dataframe_safely():
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    assert detect_all(empty_df) == []


def test_detect_slow_executions_ignores_zero_variance_workflow():
    # All executions take exactly the same time -> std == 0. mean + 2*std
    # would equal the mean itself, which would (wrongly) flag everything;
    # the zero-variance guard must skip this workflow instead.
    rows = _rows(10, "WF", status="success", duration=5.0)
    df = pd.DataFrame(rows)

    assert detect_slow_executions(df) == []


def test_detect_slow_executions_requires_minimum_samples():
    # Only 4 samples (< SLOW_EXECUTION_MIN_SAMPLES) even though one value
    # looks like an obvious outlier — too small a sample to trust mean/std.
    assert SLOW_EXECUTION_MIN_SAMPLES > 4
    rows = [
        {
            "execution_id": f"e{i}",
            "workflow_name": "WF",
            "start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
            "status": "success",
            "duration_seconds": d,
            "error_type": "",
            "error_message": "",
        }
        for i, d in enumerate([5.0, 5.0, 5.0, 100.0])
    ]
    df = pd.DataFrame(rows)

    assert detect_slow_executions(df) == []


def test_detect_recurring_errors_handles_dataframe_with_no_failures():
    rows = _rows(5, "WF", status="success")
    df = pd.DataFrame(rows)

    assert detect_recurring_errors(df) == []


def test_detect_failure_rate_increase_includes_dominant_error_type_and_affected_count():
    n_per_half = FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF
    first = _rows(n_per_half, "WF", status="success", start="2026-01-01 00:00:00")

    second_success = _rows(2, "WF", status="success", start="2026-01-02 00:00:00")
    second_timeouts = _rows(5, "WF", status="failure", error_type="API_TIMEOUT", start="2026-01-02 01:00:00")
    second_auth = _rows(3, "WF", status="failure", error_type="AUTH_ERROR", start="2026-01-02 02:00:00")
    second = second_success + second_timeouts + second_auth  # n=10

    df = pd.DataFrame(first + second)

    result = detect_failure_rate_increase(df)

    assert len(result) == 1
    assert result[0].evidence["dominant_error_type"] == "API_TIMEOUT"
    assert result[0].evidence["affected_execution_count"] == 8  # 5 timeouts + 3 auth failures


def test_select_primary_issue_prioritizes_failure_rate_increase_over_others():
    increase = Anomaly(type="failure_rate_increase", workflow_name="A", summary="", evidence={"increase_percentage_points": 15.0})
    recurring = Anomaly(type="recurring_error", workflow_name="B", summary="", evidence={"occurrence_count": 50})
    slow = Anomaly(type="slow_execution", workflow_name="C", summary="", evidence={"outlier_count": 99})

    assert select_primary_issue([recurring, slow, increase]) is increase


def test_select_primary_issue_picks_the_most_severe_within_one_type():
    small = Anomaly(type="recurring_error", workflow_name="A", summary="", evidence={"occurrence_count": 3})
    large = Anomaly(type="recurring_error", workflow_name="B", summary="", evidence={"occurrence_count": 20})

    assert select_primary_issue([small, large]) is large


def test_select_primary_issue_returns_none_when_no_anomalies():
    assert select_primary_issue([]) is None


def test_assess_system_status_critical_when_failure_rate_increase_present():
    anomalies = [Anomaly(type="failure_rate_increase", workflow_name="A", summary="", evidence={})]
    assert assess_system_status(anomalies) == "Critical"


def test_assess_system_status_attention_required_for_other_anomaly_types():
    anomalies = [Anomaly(type="recurring_error", workflow_name="A", summary="", evidence={})]
    assert assess_system_status(anomalies) == "Attention Required"


def test_assess_system_status_healthy_when_no_anomalies():
    assert assess_system_status([]) == "Healthy"
