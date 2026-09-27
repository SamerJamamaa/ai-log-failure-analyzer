"""Coverage for the expanded 15-category offline playbook set: the new
DNS/TLS/database-transaction/internal-error/provider-degradation playbooks,
correct dispatch between them, and diagnostic_commands being present and
placeholder-safe (never a real value from the evidence)."""

import pandas as pd

from src.evidence import build_evidence_table, missing_evidence_for
from src.exception_groups import build_exception_groups
from src.llm.base import build_exception_evidence
from src.llm.offline import generate_offline_investigation
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS
from src.prioritization import prioritize_groups


def _row(i, error_type, http_status=None, service="", provider="", workflow="WF", correlation_id=""):
    row = {c: (pd.NA if c in OPTIONAL_NUMERIC_COLUMNS else "") for c in OPTIONAL_COLUMNS}
    row.update(
        execution_id=f"{workflow}-{i}",
        workflow_name=workflow,
        start_time=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
        status="failure",
        duration_seconds=2.0,
        error_type=error_type,
        error_message=f"{error_type} occurred",
        http_status=http_status,
        service=service,
        provider=provider,
        correlation_id=correlation_id or f"CORR-{i}",
    )
    return row


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def _investigate(rows):
    df = _df(rows)
    group = prioritize_groups(build_exception_groups(df))[0]
    items = build_evidence_table(group, df)
    missing = missing_evidence_for(group)
    evidence = build_exception_evidence(group, items, missing)
    return group, generate_offline_investigation(evidence)


def test_internal_500_gets_internal_error_playbook_not_provider_degradation():
    """Regression test for a real dispatch bug: a plain internal HTTP 500
    used to get the 'downstream dependency is experiencing an outage'
    narrative, which is wrong for a bug in this service's own code."""
    group, inv = _investigate([_row(i, "UNHANDLED_EXCEPTION", http_status=500, service="checkout-service") for i in range(3)])

    assert group.primary_category == "Internal application error"
    assert "own code" in inv.exception_explanation or "own request-handling" in inv.hypotheses[0].hypothesis
    assert "experiencing degraded availability" not in inv.exception_explanation  # the provider-degradation narrative


def test_provider_502_gets_provider_degradation_playbook():
    group, inv = _investigate([_row(i, "SERVICE_UNAVAILABLE", http_status=502, provider="billing-api") for i in range(3)])

    assert group.primary_category == "Connectivity and dependency"
    assert "billing-api" in inv.exception_explanation
    assert any("circuit breaker" in r.option.lower() for r in inv.remediation_options)


def test_undeliverable_400_does_not_get_mislabeled_as_a_server_side_outage():
    """Regression test caught while preparing portfolio screenshots: a
    WhatsApp-style 'recipient unreachable' failure (HTTP 400, not a server
    error) was going through the same playbook as a genuine 502/503
    provider outage, producing the factually wrong claim 'server-side
    failures (HTTP 400)... experiencing degraded availability or an
    outage.' A <500 status must get the reachability-specific narrative."""
    group, inv = _investigate([_row(i, "MESSAGE_UNDELIVERABLE", http_status=400, provider="whatsapp-business-api") for i in range(3)])

    assert group.primary_category == "Connectivity and dependency"
    assert "server-side failures" not in inv.exception_explanation
    assert "outage" not in inv.hypotheses[0].hypothesis.lower()
    assert "unreachable" in inv.hypotheses[0].hypothesis.lower() or "unreachable" in inv.exception_explanation.lower()


def test_dns_failure_gets_dns_playbook():
    group, inv = _investigate([_row(i, "DNS_RESOLUTION_FAILED", provider="partner-api") for i in range(3)])

    assert group.secondary_category == "DNS failure"
    assert "DNS" in inv.exception_explanation
    assert any("dig" in c.command for c in inv.diagnostic_commands)


def test_tls_failure_gets_tls_playbook():
    group, inv = _investigate([_row(i, "TLS_HANDSHAKE_FAILED", provider="partner-api") for i in range(3)])

    assert group.secondary_category == "TLS failure"
    assert "TLS" in inv.exception_explanation or "certificate" in inv.exception_explanation.lower()
    assert any("openssl" in c.command for c in inv.diagnostic_commands)


def test_database_transaction_failure_gets_db_playbook_not_duplicate_playbook():
    group, inv = _investigate([_row(i, "TRANSACTION_FAILURE", service="orders-service") for i in range(3)])

    assert group.primary_category == "Data integrity and persistence"
    assert group.secondary_category == "Transaction failure"
    assert "idempotency" not in inv.remediation_options[0].option.lower()  # the duplicate-record remediation
    assert "transaction" in inv.exception_explanation.lower()


def test_duplicate_record_still_gets_duplicate_playbook():
    group, inv = _investigate([_row(i, "DUPLICATE_RECORD", http_status=409, service="orders-service") for i in range(3)])

    assert group.secondary_category == "Duplicate record"
    assert any("idempotency" in r.option.lower() for r in inv.remediation_options)


def test_missing_http_status_never_renders_as_the_literal_string_http_none():
    """Regression test caught via live browser testing: a duplicate-record
    exception classified without an http_status printed 'HTTP None' in its
    explanation — a confusing, unprofessional label."""
    group, inv = _investigate([_row(i, "DUPLICATE_RECORD", http_status=None, service="orders-service") for i in range(3)])
    assert "HTTP None" not in inv.exception_explanation


def test_every_playbook_produces_at_least_one_diagnostic_command():
    scenarios = [
        ("AUTHENTICATION_ERROR", 401),
        ("API_TIMEOUT", None),
        ("RATE_LIMITED", 429),
        ("SERVICE_UNAVAILABLE", 502),
        ("UNHANDLED_EXCEPTION", 500),
        ("INVALID_JSON_SCHEMA", None),
        ("DUPLICATE_RECORD", 409),
        ("RETRY_EXHAUSTED", None),
        ("DNS_RESOLUTION_FAILED", None),
        ("TLS_HANDSHAKE_FAILED", None),
        ("MISSING_ENV_VARIABLE", None),
        ("TRANSACTION_FAILURE", None),
        ("SOME_NOVEL_ERROR", None),
    ]
    for error_type, http_status in scenarios:
        _, inv = _investigate([_row(i, error_type, http_status=http_status) for i in range(3)])
        assert len(inv.diagnostic_commands) >= 1, f"no diagnostic command for {error_type}"


def test_business_rule_failure_gets_dedicated_playbook_not_generic_fallback():
    """Regression test: after the taxonomy expansion correctly classified
    real-platform business-rule rejections (card declines, re-engagement
    windows, opted-out recipients) as 'Business-rule failure', they still
    fell through to the generic 'insufficient evidence' playbook because no
    dedicated playbook existed for that category — thin content right where
    the classification had just gotten more specific."""
    group, inv = _investigate([_row(i, "RE_ENGAGEMENT_WINDOW_EXPIRED", http_status=400, workflow="WhatsApp Sender") for i in range(3)])

    assert group.primary_category == "Business-rule failure"
    assert "Insufficient evidence is available to classify this" not in inv.exception_explanation
    assert "business-rule rejection" in inv.exception_explanation.lower()
    assert inv.hypotheses[0].confidence != "low"


def test_diagnostic_commands_never_contain_real_correlation_ids():
    """Every command must be placeholder-based — never leak an actual
    correlation ID that happened to be in this group's evidence."""
    group, inv = _investigate(
        [_row(i, "AUTHENTICATION_ERROR", http_status=401, correlation_id=f"REAL-SECRET-CORR-{i}") for i in range(3)]
    )
    for command in inv.diagnostic_commands:
        assert "REAL-SECRET-CORR" not in command.command
        assert "<" in command.command  # uses a placeholder
