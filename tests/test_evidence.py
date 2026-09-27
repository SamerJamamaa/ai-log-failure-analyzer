import pandas as pd

from src.evidence import build_evidence_table, missing_evidence_for, redact_evidence, redact_text
from src.exception_groups import build_exception_groups
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS


def _row(i, status="failure", error_type="AUTHENTICATION_ERROR", version="1.0", http_status=None, retry_count=None, workflow="WF"):
    row = {c: (pd.NA if c in OPTIONAL_NUMERIC_COLUMNS else "") for c in OPTIONAL_COLUMNS}
    row.update(
        execution_id=f"e{i}",
        workflow_name=workflow,
        start_time=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
        status=status,
        duration_seconds=2.0,
        error_type=error_type if status == "failure" else "",
        error_message=f"{error_type} occurred" if status == "failure" else "",
        application_version=version,
        http_status=http_status,
        retry_count=retry_count,
    )
    return row


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def _first_group(df):
    return build_exception_groups(df)[0]


# --- evidence table --------------------------------------------------------------


def test_baseline_occurrence_evidence_always_present():
    df = _df([_row(i) for i in range(3)])
    items = build_evidence_table(_first_group(df), df)

    assert items[0].source_field == "occurrence_count"
    assert items[0].occurrence_count == 3


def test_http_status_evidence_included_when_distribution_present():
    df = _df([_row(i, http_status=401) for i in range(3)])
    items = build_evidence_table(_first_group(df), df)

    http_items = [i for i in items if i.source_field == "http_status"]
    assert len(http_items) == 1
    assert "401" in http_items[0].observed_value


def test_version_regression_evidence_matches_the_spec_example():
    """The exact CRM-auth-regression story from your spec: all failures on
    the new version, none of that error type on the previous version."""
    rows = [_row(i, status="success", version="3.4.0") for i in range(10)]
    rows += [_row(10 + i, version="3.5.0", http_status=401) for i in range(8)]
    df = _df(rows)

    items = build_evidence_table(_first_group(df), df)
    version_items = [i for i in items if i.source_field == "application_version"]

    assert any("3.5.0" in i.observed_value and i.occurrence_count == 8 for i in version_items)
    assert any("3.4.0" in i.observed_value and i.occurrence_count == 0 for i in version_items)


def test_no_contrastive_version_evidence_when_only_one_version_exists():
    """Backward compatibility: no version data at all (the common case
    today) must not produce a fabricated version comparison."""
    df = _df([_row(i, version="") for i in range(3)])
    items = build_evidence_table(_first_group(df), df)

    assert not any(i.source_field == "application_version" for i in items)


def test_retry_evidence_included_when_retries_present():
    df = _df([_row(i, retry_count=2) for i in range(3)])
    items = build_evidence_table(_first_group(df), df)

    retry_items = [i for i in items if i.source_field == "retry_count"]
    assert len(retry_items) == 1
    assert retry_items[0].occurrence_count == 3


def test_trend_evidence_absent_for_isolated_group():
    df = _df([_row(0)])
    items = build_evidence_table(_first_group(df), df)
    assert not any(i.source_field == "trend" for i in items)


def test_trend_evidence_present_for_recurring_group():
    df = _df([_row(i) for i in range(3)])
    items = build_evidence_table(_first_group(df), df)
    assert any(i.source_field == "trend" for i in items)


def test_cross_workflow_spread_evidence_when_shared():
    rows = [_row(i, workflow="A") for i in range(3)]
    rows += [_row(3 + i, workflow="B") for i in range(3)]
    df = _df(rows)
    group_a = [g for g in build_exception_groups(df) if g.workflow_name == "A"][0]

    items = build_evidence_table(group_a, df)
    assert any(i.source_field == "workflows_sharing_this_issue" for i in items)


def test_representative_ids_are_capped():
    df = _df([_row(i) for i in range(20)])
    items = build_evidence_table(_first_group(df), df)
    assert all(len(i.representative_execution_ids) <= 5 for i in items)


# --- missing evidence --------------------------------------------------------------


def test_missing_evidence_is_category_specific():
    df = _df([_row(i, error_type="AUTHENTICATION_ERROR") for i in range(3)])
    group = _first_group(df)
    missing = missing_evidence_for(group)

    assert any("token expiration" in item.lower() for item in missing)
    assert "Request headers" in missing  # universal item always included


def test_missing_evidence_falls_back_for_unclassified():
    df = _df([_row(i, error_type="SOME_NOVEL_ERROR") for i in range(3)])
    group = _first_group(df)
    missing = missing_evidence_for(group)
    assert len(missing) > 0


# --- redaction ------------------------------------------------------------------


def test_redact_text_strips_bearer_token():
    text = "Authorization: Bearer sk-ant-secret123"
    redacted = redact_text(text)
    assert "sk-ant-secret123" not in redacted


def test_redact_text_strips_api_key():
    redacted = redact_text("api_key=sk-live-abc123xyz")
    assert "sk-live-abc123xyz" not in redacted


def test_redact_text_strips_password():
    redacted = redact_text("password=hunter2")
    assert "hunter2" not in redacted


def test_redact_text_strips_email():
    redacted = redact_text("failed for user alice@example.com")
    assert "alice@example.com" not in redacted


def test_redact_text_strips_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    redacted = redact_text(jwt)
    assert jwt not in redacted


def test_redact_text_handles_empty():
    assert redact_text("") == ""
    assert redact_text(None) is None


def test_redact_evidence_strips_always_strip_keys():
    payload = {"error_message": "timed out", "request_body": {"user": "secret data"}, "nested": {"password": "x"}}
    redacted = redact_evidence(payload)

    assert "request_body" not in redacted
    assert "password" not in redacted["nested"]
    assert redacted["error_message"] == "timed out"


def test_redact_evidence_recurses_into_lists():
    payload = {"messages": ["Bearer sk-secret-token-value", "normal message"]}
    redacted = redact_evidence(payload)

    assert "sk-secret-token-value" not in redacted["messages"][0]
    assert redacted["messages"][1] == "normal message"


def test_redact_evidence_leaves_non_string_values_alone():
    payload = {"count": 5, "rate": 12.5, "flag": True, "nothing": None}
    assert redact_evidence(payload) == payload
