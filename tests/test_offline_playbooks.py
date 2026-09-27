"""Category-specific offline playbook coverage — built against real
ExceptionGroup/evidence data (not hand-typed evidence dicts) for the 9
categories the spec explicitly requires at minimum: Authentication,
Timeouts, HTTP 429, HTTP 5xx dependency, Schema-validation, Duplicate
records, Retry exhaustion, Configuration errors, Slow successful executions.
"""

import pandas as pd

from src.evidence import build_evidence_table, missing_evidence_for
from src.exception_groups import build_exception_groups
from src.llm.base import build_exception_evidence
from src.llm.offline import generate_offline_investigation
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS
from src.prioritization import prioritize_groups


def _row(i, workflow="WF", status="failure", error_type="", http_status=None, retry_count=None, version="", provider="", duration=2.0):
    row = {c: (pd.NA if c in OPTIONAL_NUMERIC_COLUMNS else "") for c in OPTIONAL_COLUMNS}
    row.update(
        execution_id=f"{workflow}-{i}",
        workflow_name=workflow,
        start_time=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
        status=status,
        duration_seconds=duration,
        error_type=error_type if status == "failure" else "",
        error_message=f"{error_type} occurred" if status == "failure" else "",
        http_status=http_status,
        retry_count=retry_count,
        application_version=version,
        provider=provider,
    )
    return row


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def _investigate_first_group(df):
    groups = prioritize_groups(build_exception_groups(df))
    group = groups[0]
    items = build_evidence_table(group, df)
    missing = missing_evidence_for(group)
    evidence = build_exception_evidence(group, items, missing)
    return group, generate_offline_investigation(evidence)


def test_authentication_playbook_produces_credential_hypothesis():
    rows = [_row(i, status="success", version="3.4.0") for i in range(10)]
    rows += [_row(10 + i, error_type="AUTHENTICATION_ERROR", http_status=401, version="3.5.0") for i in range(8)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "Authentication and authorization"
    assert "credential" in inv.hypotheses[0].hypothesis.lower() or "token" in inv.hypotheses[0].hypothesis.lower()
    assert "root cause" not in inv.exception_explanation.lower()
    assert "caused this" not in inv.exception_explanation.lower()


def test_timeout_playbook_offers_four_remediation_options():
    rows = [_row(i, error_type="API_TIMEOUT", provider="whatsapp", duration=30.0) for i in range(6)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "Timeout and latency"
    assert len(inv.remediation_options) >= 4


def test_rate_limit_429_playbook():
    rows = [_row(i, error_type="RATE_LIMITED", http_status=429, provider="crm") for i in range(6)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "Rate limiting and capacity"
    assert any("limit" in opt.option.lower() or "quota" in opt.option.lower() for opt in inv.remediation_options)


def test_http_5xx_dependency_playbook():
    rows = [_row(i, error_type="DEPENDENCY_UNAVAILABLE", http_status=503, provider="billing") for i in range(6)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.http_status_distribution
    assert any("circuit breaker" in opt.option.lower() or "fallback" in opt.option.lower() for opt in inv.remediation_options)


def test_schema_validation_playbook():
    rows = [_row(i, error_type="INVALID_JSON_SCHEMA") for i in range(8)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "AI and structured-output"
    assert any("schema" in opt.option.lower() or "contract" in opt.option.lower() or "structured" in opt.option.lower() for opt in inv.remediation_options)


def test_duplicate_records_playbook_flags_idempotency():
    rows = [_row(i, error_type="DUPLICATE_RECORD", http_status=409, retry_count=1) for i in range(6)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "Data integrity and persistence"
    assert any("idempotency" in opt.option.lower() for opt in inv.remediation_options)


def test_retry_exhaustion_playbook():
    rows = [_row(i, error_type="RETRY_EXHAUSTED", retry_count=3) for i in range(6)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "Retry and idempotency"
    assert any("retry" in opt.option.lower() or "dead-letter" in opt.option.lower() for opt in inv.remediation_options)


def test_configuration_playbook_mentions_version_when_present():
    rows = [_row(i, error_type="MISSING_ENV_VARIABLE", version="2.1.0") for i in range(6)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.primary_category == "Configuration and deployment"
    assert "[" not in inv.exception_explanation  # no raw list repr leaking into the text
    assert "2.1.0" in inv.exception_explanation


def test_slow_successful_executions_are_not_classified_as_exception_groups():
    """Spec requirement: slow-but-successful executions must never appear as
    an exception group — only status='failure' rows are grouped."""
    rows = [_row(i, status="success", duration=45.0) for i in range(3)]
    rows += [_row(3 + i, status="success", duration=3.0) for i in range(10)]
    groups = build_exception_groups(_df(rows))
    assert groups == []


def test_containment_only_present_for_high_or_critical_priority():
    rows = [_row(i, error_type="AUTHENTICATION_ERROR", http_status=401) for i in range(1)]
    group, inv = _investigate_first_group(_df(rows))

    assert group.priority_label not in ("Critical", "High")
    assert inv.containment_actions == []


def test_every_hypothesis_evidence_id_reference_is_real():
    rows = [_row(i, error_type="AUTHENTICATION_ERROR", http_status=401, version="3.5.0") for i in range(8)]
    df = _df(rows)
    group = build_exception_groups(df)[0]
    items = build_evidence_table(group, df)
    evidence = build_exception_evidence(group, items, missing_evidence_for(group))
    inv = generate_offline_investigation(evidence)

    valid_ids = {item["evidence_id"] for item in evidence["evidence"]}
    for hyp in inv.hypotheses:
        assert all(eid in valid_ids for eid in hyp.supporting_evidence_ids)
    for action in inv.investigation_actions:
        assert all(eid in valid_ids for eid in action.evidence_reference)
