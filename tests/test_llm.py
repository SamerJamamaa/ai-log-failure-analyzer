import json

from src.evidence import build_evidence_table, missing_evidence_for
from src.exception_groups import build_exception_groups
from src.llm.base import build_exception_evidence, parse_exception_investigation
from src.llm.offline import generate_offline_investigation
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS, ExceptionInvestigation

import pandas as pd

VALID_PAYLOAD = {
    "exception_explanation": "Authentication failures concentrated on the new version.",
    "known_facts": ["8 occurrences on version 3.5.0"],
    "unknowns": ["Token expiration timestamp"],
    "hypotheses": [
        {
            "rank": 1,
            "hypothesis": "A version-specific OAuth configuration regression is plausible",
            "confidence": "medium",
            "supporting_evidence_ids": ["E1"],
            "contradicting_evidence_ids": [],
            "missing_evidence": ["Configuration diff"],
            "verification_steps": ["Compare OAuth config across versions"],
        }
    ],
    "containment_actions": [],
    "investigation_actions": [
        {
            "priority": 1,
            "action": "Trace correlation IDs for the failed executions",
            "component": "CRM service",
            "evidence_reference": ["E1"],
            "expected_findings": ["A consistent failure point"],
            "decision_from_result": "If consistent, treat as systemic",
        }
    ],
    "remediation_options": [
        {
            "option": "Correct the OAuth audience configuration",
            "appropriate_when": "A config diff confirms the mismatch",
            "expected_benefit": "Restores authorization",
            "risks": ["Misconfiguration could grant excess access"],
            "tradeoffs": ["Needs a confirmed diff first"],
            "complexity": "medium",
            "change_type": "configuration",
            "temporary_or_permanent": "permanent",
        }
    ],
    "verification_plan": [
        {
            "test": "Re-run affected executions after the fix",
            "expected_result": "No further authentication failures",
            "success_metric": "Occurrence count",
            "success_threshold": "Zero new occurrences",
            "observation_period": "7 days after the fix is deployed",
            "rollback_condition": "Failures continue",
        }
    ],
    "prevention_actions": ["Add token-expiration monitoring"],
    "limitations": ["Configuration evidence is required for confirmation"],
}


def _row(i, status="failure", error_type="AUTHENTICATION_ERROR", version="3.5.0", http_status=401):
    row = {c: (pd.NA if c in OPTIONAL_NUMERIC_COLUMNS else "") for c in OPTIONAL_COLUMNS}
    row.update(
        execution_id=f"e{i}",
        workflow_name="CRM Sync",
        start_time=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
        status=status,
        duration_seconds=2.0,
        error_type=error_type if status == "failure" else "",
        error_message=f"{error_type} occurred" if status == "failure" else "",
        application_version=version,
        http_status=http_status,
    )
    return row


def _df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def _sample_evidence() -> dict:
    df = _df([_row(i) for i in range(8)])
    group = build_exception_groups(df)[0]
    group.priority_score, group.priority_label, group.priority_factors = 60.0, "High", {}
    items = build_evidence_table(group, df)
    missing = missing_evidence_for(group)
    return build_exception_evidence(group, items, missing)


# --- parsing -----------------------------------------------------------------


def test_parse_exception_investigation_accepts_valid_schema():
    report = parse_exception_investigation(json.dumps(VALID_PAYLOAD), source="llm")

    assert report is not None
    assert isinstance(report, ExceptionInvestigation)
    assert report.source == "llm"
    assert report.hypotheses[0].confidence == "medium"


def test_parse_exception_investigation_rejects_invalid_confidence():
    bad_payload = json.loads(json.dumps(VALID_PAYLOAD))
    bad_payload["hypotheses"][0]["confidence"] = "extreme"

    assert parse_exception_investigation(json.dumps(bad_payload), source="llm") is None


def test_parse_exception_investigation_rejects_missing_field():
    bad_payload = json.loads(json.dumps(VALID_PAYLOAD))
    del bad_payload["remediation_options"]

    assert parse_exception_investigation(json.dumps(bad_payload), source="llm") is None


def test_parse_exception_investigation_rejects_malformed_json():
    assert parse_exception_investigation("not json at all", source="llm") is None


def test_parse_exception_investigation_strips_markdown_fences():
    fenced = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    report = parse_exception_investigation(fenced, source="llm")
    assert report is not None


# --- evidence payload construction --------------------------------------------


def test_build_exception_evidence_contains_expected_top_level_keys():
    evidence = _sample_evidence()
    assert set(evidence.keys()) == {
        "exception_identity",
        "occurrence",
        "impact",
        "priority",
        "evidence",
        "missing_evidence",
    }


def test_build_exception_evidence_redacts_secrets_in_representative_message():
    df = _df([_row(i) for i in range(3)])
    group = build_exception_groups(df)[0]
    group.representative_message = "Authorization: Bearer sk-secret-value failed"
    items = build_evidence_table(group, df)
    evidence = build_exception_evidence(group, items, [])

    assert "sk-secret-value" not in json.dumps(evidence)


# --- offline generator ---------------------------------------------------------


def test_offline_investigation_is_valid_and_labeled_offline():
    evidence = _sample_evidence()
    report = generate_offline_investigation(evidence)

    assert isinstance(report, ExceptionInvestigation)
    assert report.source == "offline"
    assert any("Rule-based investigation guidance" in item for item in report.limitations)


def test_offline_investigation_hypotheses_reference_real_evidence_ids():
    evidence = _sample_evidence()
    report = generate_offline_investigation(evidence)
    valid_ids = {item["evidence_id"] for item in evidence["evidence"]}

    for hypothesis in report.hypotheses:
        assert all(eid in valid_ids for eid in hypothesis.supporting_evidence_ids)


def test_offline_investigation_containment_empty_for_low_priority():
    evidence = _sample_evidence()
    evidence["priority"]["label"] = "Low"
    report = generate_offline_investigation(evidence)
    assert report.containment_actions == []
