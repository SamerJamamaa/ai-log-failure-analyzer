import pandas as pd
import pytest

from src.exception_groups import (
    TREND_INCREASE_MIN_RATIO,
    TREND_ISOLATED_MAX,
    TREND_MIN_RECORDS_PER_HALF,
    build_exception_groups,
)
from src.models import OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS, OPTIONAL_COLUMNS


def _rows(n, workflow="WF", status="failure", error_type="API_TIMEOUT", duration=5.0, start="2026-01-01 00:00:00", **extra):
    start_ts = pd.Timestamp(start)
    rows = []
    for i in range(n):
        # mirror src.ingestion's post-validation defaults: "" for text
        # columns, NaN for numeric ones — never a bare "" in a numeric column
        row = {col: (pd.NA if col in OPTIONAL_NUMERIC_COLUMNS else "") for col in OPTIONAL_COLUMNS}
        row.update(
            execution_id=f"{workflow}_{start}_{i}",
            workflow_name=workflow,
            start_time=start_ts + pd.Timedelta(minutes=i),
            status=status,
            duration_seconds=duration,
            error_type=error_type if status == "failure" else "",
            error_message=f"{error_type} occurred" if status == "failure" else "",
        )
        row.update(extra)
        rows.append(row)
    return rows


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def test_returns_empty_list_when_no_failures():
    df = _df(_rows(10, status="success"))
    assert build_exception_groups(df) == []


def test_groups_by_fingerprint_not_raw_error_type():
    """Two different services failing with the same error_type must stay
    separate groups — this is the exact behavior the taxonomy spec requires
    ("API timeout from WhatsApp provider" vs "... CRM provider" must not merge)."""
    rows = _rows(4, workflow="WF", error_type="API_TIMEOUT", service="whatsapp_gateway")
    rows += _rows(4, workflow="WF", error_type="API_TIMEOUT", service="crm_gateway", start="2026-01-02 00:00:00")
    df = _df(rows)

    groups = build_exception_groups(df)

    assert len(groups) == 2
    services = {g.service for g in groups}
    assert services == {"whatsapp_gateway", "crm_gateway"}
    for g in groups:
        assert g.occurrence_count == 4


def test_varying_representative_message_does_not_fragment_a_group_when_error_type_present():
    """Regression test for a real bug caught during implementation: 4
    rotating error messages for the SAME error_type/service/endpoint must
    still consolidate into ONE group, not 4."""
    messages = [
        "Required field 'invoice_number' is missing.",
        "Field 'amount' was returned as text instead of a number.",
        "Field 'due_date' does not match the expected date format.",
        "Response JSON has an unexpected nested structure.",
    ]
    rows = []
    for i, msg in enumerate(messages * 2):  # 8 total, 4 distinct messages
        rows.append(
            {
                **{col: "" for col in OPTIONAL_COLUMNS},
                "execution_id": f"e{i}",
                "workflow_name": "Invoice Parser",
                "start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
                "status": "failure",
                "duration_seconds": 8.0,
                "error_type": "INVALID_JSON_SCHEMA",
                "error_message": msg,
            }
        )
    df = _df(rows)

    groups = build_exception_groups(df)

    assert len(groups) == 1
    assert groups[0].occurrence_count == 8


def test_percentage_of_all_failures_and_of_workflow_executions():
    rows = _rows(2, workflow="WF", error_type="A") + _rows(6, workflow="WF", error_type="B", start="2026-01-02 00:00:00")
    rows += _rows(2, workflow="WF", status="success", start="2026-01-03 00:00:00")
    df = _df(rows)

    groups = {g.error_type: g for g in build_exception_groups(df)}

    # 8 total failures (2+6); workflow has 10 total executions (8 failures + 2 success)
    assert groups["A"].percentage_of_all_failures == pytest.approx(25.0)
    assert groups["B"].percentage_of_all_failures == pytest.approx(75.0)
    assert groups["A"].percentage_of_workflow_executions == pytest.approx(20.0)
    assert groups["B"].percentage_of_workflow_executions == pytest.approx(60.0)


def test_trend_isolated_for_small_count():
    rows = _rows(TREND_ISOLATED_MAX)
    df = _df(rows)
    group = build_exception_groups(df)[0]
    assert group.trend_label == "isolated"
    assert group.is_increasing is False


def test_trend_recurring_when_stable_across_halves():
    n = TREND_MIN_RECORDS_PER_HALF * 2  # equal halves, no growth
    rows = _rows(n)
    df = _df(rows)
    group = build_exception_groups(df)[0]
    assert group.trend_label == "recurring"
    assert group.is_increasing is False


def test_trend_increasing_when_second_half_clears_ratio_threshold():
    first_half = _rows(TREND_MIN_RECORDS_PER_HALF, start="2026-01-01 00:00:00")
    second_half = _rows(
        int(TREND_MIN_RECORDS_PER_HALF * TREND_INCREASE_MIN_RATIO), start="2026-01-02 00:00:00"
    )
    df = _df(first_half + second_half)
    group = build_exception_groups(df)[0]
    assert group.trend_label == "increasing"
    assert group.is_increasing is True


def test_trend_not_increasing_without_minimum_sample_per_half():
    """Even a dramatic-looking jump must not be called 'increasing' if
    either half has too few records — misleading-data-prevention rule."""
    rows = _rows(1, start="2026-01-01 00:00:00") + _rows(10, start="2026-01-02 00:00:00")
    df = _df(rows)
    group = build_exception_groups(df)[0]
    assert group.trend_label != "increasing"


def test_http_status_distribution_and_retry_info_populate_when_present():
    rows = _rows(3, http_status=504, retry_count=2) + _rows(2, http_status=500, retry_count=0, start="2026-01-02 00:00:00")
    df = _df(rows)
    group = build_exception_groups(df)[0]

    assert group.http_status_distribution == {"504": 3, "500": 2}
    assert group.retry_info["executions_with_retries"] == 3
    assert group.retry_info["all_retries_still_failed"] is True


def test_optional_fields_absent_produce_empty_not_crash():
    df = _df(_rows(3))
    group = build_exception_groups(df)[0]

    assert group.http_status_distribution == {}
    assert group.retry_info == {}
    assert group.application_versions == []
    assert group.hosts_affected == []  # never populated — no host field in the schema


def test_application_versions_collected_and_deduplicated():
    rows = _rows(2, application_version="1.4.2") + _rows(2, application_version="1.4.2", start="2026-01-02 00:00:00")
    rows += _rows(2, application_version="1.5.0", start="2026-01-03 00:00:00")
    df = _df(rows)
    group = build_exception_groups(df)[0]

    assert group.application_versions == ["1.4.2", "1.5.0"]


def test_workflows_sharing_this_issue_counts_across_workflows():
    rows = _rows(3, workflow="A", error_type="API_TIMEOUT", service="shared_gw")
    rows += _rows(3, workflow="B", error_type="API_TIMEOUT", service="shared_gw", start="2026-01-02 00:00:00")
    df = _df(rows)

    groups = build_exception_groups(df)

    assert len(groups) == 2
    assert all(g.workflows_sharing_this_issue == 2 for g in groups)


def test_classification_is_applied_to_each_group():
    df = _df(_rows(3, error_type="AUTHENTICATION_ERROR"))
    group = build_exception_groups(df)[0]
    assert group.primary_category == "Authentication and authorization"
