import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm.anthropic_client import AnthropicLLMClient
from src.llm.base import LLMProviderError, ResilientLLMClient, get_client
from src.llm.offline import OfflineLLMClient
from src.models import ExceptionInvestigation

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

SAMPLE_EVIDENCE = {
    "exception_identity": {
        "primary_category": "Authentication and authorization",
        "secondary_category": "Invalid/expired token",
        "error_type": "AUTHENTICATION_ERROR",
        "error_code": "",
        "representative_message": "Authentication failed",
        "workflow_name": "CRM Sync",
        "workflow_step": "",
        "service": "crm-service",
        "endpoint": "",
        "http_method": "",
        "http_status": 401,
        "provider": "",
        "environment": "",
    },
    "occurrence": {
        "occurrence_count": 8,
        "affected_execution_count": 8,
        "first_occurrence": "2026-01-01",
        "last_occurrence": "2026-01-02",
        "percentage_of_all_failures": 100.0,
        "percentage_of_workflow_executions": 100.0,
        "trend": "recurring",
        "application_versions": ["3.5.0"],
        "workflows_sharing_this_issue": 1,
    },
    "impact": {
        "avg_duration_seconds": 2.0,
        "p95_duration_seconds": 2.0,
        "http_status_distribution": {"401": 8},
        "retry_info": {},
    },
    "priority": {"score": 60.0, "label": "High", "factors": {}},
    "evidence": [
        {
            "evidence_id": "E1",
            "source_field": "occurrence_count",
            "observed_value": "8 occurrences",
            "occurrence_count": 8,
            "why_it_matters": "Represents all failures in this upload.",
            "representative_execution_ids": ["e1", "e2"],
            "representative_correlation_ids": [],
        }
    ],
    "missing_evidence": ["Token expiration timestamp"],
}


def _fake_anthropic_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


# --- Valid AI response ------------------------------------------------------


def test_valid_ai_response_returns_llm_sourced_report():
    client = AnthropicLLMClient(api_key="test-key")
    client.client.messages.create = MagicMock(return_value=_fake_anthropic_message(json.dumps(VALID_PAYLOAD)))

    report = client.analyze(SAMPLE_EVIDENCE)

    assert isinstance(report, ExceptionInvestigation)
    assert report.source == "llm"
    assert report.exception_explanation == VALID_PAYLOAD["exception_explanation"]
    client.client.messages.create.assert_called_once()  # no retry needed


def test_openai_client_also_returns_llm_sourced_report():
    """Same behavior through the other concrete provider — proves the
    interface is genuinely provider-agnostic, not just Anthropic-shaped."""
    from src.llm.openai_client import OpenAILLMClient

    client = OpenAILLMClient(api_key="test-key")
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(VALID_PAYLOAD)))]
    )
    client.client.chat.completions.create = MagicMock(return_value=fake_response)

    report = client.analyze(SAMPLE_EVIDENCE)

    assert report.source == "llm"
    assert report.exception_explanation == VALID_PAYLOAD["exception_explanation"]


# --- Invalid structured response --------------------------------------------


def test_invalid_structured_response_raises_after_one_retry():
    client = AnthropicLLMClient(api_key="test-key")
    client.client.messages.create = MagicMock(return_value=_fake_anthropic_message("not valid json"))

    with pytest.raises(LLMProviderError):
        client.analyze(SAMPLE_EVIDENCE)

    assert client.client.messages.create.call_count == 2  # first attempt + one retry


def test_invalid_structured_response_falls_back_to_offline_via_resilient_wrapper():
    client = AnthropicLLMClient(api_key="test-key")
    client.client.messages.create = MagicMock(return_value=_fake_anthropic_message("not valid json"))
    resilient = ResilientLLMClient(client, provider_name="Anthropic Claude")

    report = resilient.analyze(SAMPLE_EVIDENCE)

    assert report.source == "offline"
    assert any("Anthropic Claude call failed" in item for item in report.limitations)


# --- Missing API key ---------------------------------------------------------


def test_missing_api_key_returns_offline_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = get_client()

    assert isinstance(client, OfflineLLMClient)


# --- Provider timeout ---------------------------------------------------------


def test_provider_timeout_falls_back_to_offline():
    client = AnthropicLLMClient(api_key="test-key")
    client.client.messages.create = MagicMock(side_effect=TimeoutError("request timed out after 30s"))
    resilient = ResilientLLMClient(client, provider_name="Anthropic Claude")

    report = resilient.analyze(SAMPLE_EVIDENCE)

    assert report.source == "offline"
    assert any("TimeoutError" in item for item in report.limitations)
    assert not any("timed out after 30s" in item for item in report.limitations)  # never the raw exception text


# --- Provider failure ---------------------------------------------------------


def test_provider_failure_falls_back_to_offline_without_leaking_details():
    client = AnthropicLLMClient(api_key="test-key")
    client.client.messages.create = MagicMock(
        side_effect=ConnectionError("connection reset while using key=sk-should-never-appear")
    )
    resilient = ResilientLLMClient(client, provider_name="Anthropic Claude")

    report = resilient.analyze(SAMPLE_EVIDENCE)

    assert report.source == "offline"
    assert any("ConnectionError" in item for item in report.limitations)
    assert not any("sk-should-never-appear" in item for item in report.limitations)  # credential never leaked


# --- Offline fallback ---------------------------------------------------------


def test_offline_fallback_produces_a_valid_investigation():
    report = OfflineLLMClient().analyze(SAMPLE_EVIDENCE)

    assert isinstance(report, ExceptionInvestigation)
    assert report.source == "offline"
    assert any("Rule-based investigation guidance" in item for item in report.limitations)
