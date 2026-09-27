import pandas as pd

from src.exception_groups import build_exception_groups
from src.incident_brief import generate_incident_brief
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS
from src.prioritization import prioritize_groups


def _row(i, status="failure", error_type="AUTHENTICATION_ERROR", version="", http_status=None, workflow="WF", service=""):
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
        service=service,
    )
    return row


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def test_healthy_file_produces_a_healthy_brief():
    df = _df([_row(i, status="success") for i in range(20)])
    brief = generate_incident_brief([], df, total_executions=20, total_failures=0)

    assert "no" in brief.primary_incident.lower() or "none" in brief.primary_incident.lower()
    assert brief.secondary_incidents == []
    assert not brief.likely_explanations


def test_incident_brief_names_the_primary_exception_and_its_reason():
    rows = [_row(i, status="success", version="3.4.0", service="crm") for i in range(10)]
    rows += [_row(10 + i, version="3.5.0", http_status=401, service="crm") for i in range(8)]
    df = _df(rows)
    groups = prioritize_groups(build_exception_groups(df))

    brief = generate_incident_brief(groups, df, total_executions=len(df), total_failures=8)

    assert "crm" in brief.primary_incident.lower()
    assert "priority because" in brief.primary_incident
    assert "3.5.0" in brief.time_version_relationship
    assert "correlation" in brief.time_version_relationship.lower()
    assert "confirmed causation" not in brief.time_version_relationship or "not confirmed causation" in brief.time_version_relationship.lower()


def test_incident_brief_lists_secondary_incidents_when_multiple_groups_exist():
    rows = [_row(i, error_type="AUTHENTICATION_ERROR", http_status=401, workflow="WF1") for i in range(10)]
    rows += [_row(10 + i, error_type="API_TIMEOUT", workflow="WF2") for i in range(3)]
    df = _df(rows)
    groups = prioritize_groups(build_exception_groups(df))

    brief = generate_incident_brief(groups, df, total_executions=len(df), total_failures=13)

    assert len(brief.secondary_incidents) == 1
    assert "WF2" in brief.secondary_incidents[0] or "timing" in brief.secondary_incidents[0].lower()


def test_incident_brief_truncates_and_notes_remaining_patterns_beyond_the_cap():
    rows = []
    for i, (etype, status) in enumerate(
        [("A", 401), ("B", 409), ("C", 429), ("D", 500), ("E", 504)]
    ):
        rows += [_row(100 * i + j, error_type=etype, http_status=status, workflow=f"WF{i}") for j in range(3)]
    df = _df(rows)
    groups = prioritize_groups(build_exception_groups(df))
    assert len(groups) == 5  # sanity: 5 distinct exception groups

    brief = generate_incident_brief(groups, df, total_executions=len(df), total_failures=15)

    assert len(brief.secondary_incidents) == 4  # 3 shown + 1 "...and N more" note
    assert "more exception pattern" in brief.secondary_incidents[-1]
    assert any("top 4 of 5" in item for item in brief.limitations)


def test_technical_impact_includes_exact_business_impact_disclaimer():
    rows = [_row(i) for i in range(5)]
    df = _df(rows)
    groups = prioritize_groups(build_exception_groups(df))
    brief = generate_incident_brief(groups, df, total_executions=5, total_failures=5)

    assert "Technical impact is visible, but business impact cannot be determined from the supplied logs." in brief.technical_impact


def test_time_version_relationship_does_not_claim_correlation_with_only_one_version_ever_seen():
    """Regression test caught via smoke-testing: a workflow that only ever
    has ONE version in the whole upload has nothing to correlate the
    exception against, even though the group's own application_versions
    list trivially has length 1 — must not be phrased as a 'correlation'."""
    rows = [_row(i, version="4.5.0", http_status=504, workflow="WhatsApp", service="wa") for i in range(10)]
    df = _df(rows)
    groups = prioritize_groups(build_exception_groups(df))
    brief = generate_incident_brief(groups, df, total_executions=len(df), total_failures=10)

    assert "correlation observed" not in brief.time_version_relationship
    assert "no other version" in brief.time_version_relationship.lower()


def test_incident_brief_never_claims_confirmed_causation():
    rows = [_row(i, status="success", version="1.0") for i in range(10)]
    rows += [_row(10 + i, version="2.0") for i in range(8)]
    df = _df(rows)
    groups = prioritize_groups(build_exception_groups(df))
    brief = generate_incident_brief(groups, df, total_executions=len(df), total_failures=8)

    full_text = " ".join([brief.what_happened, brief.primary_incident, brief.time_version_relationship, *brief.likely_explanations])
    assert "caused this" not in full_text.lower()
    assert "root cause" not in full_text.lower()
