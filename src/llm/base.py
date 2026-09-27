"""Provider-agnostic LLM plumbing for the exception-investigation assistant:
the client interface, the evidence-payload builder, the system prompt that
encodes what the LLM may/must-not do, and defensive JSON parsing into
ExceptionInvestigation. Anthropic/OpenAI clients and the offline fallback
all build on this; none of them talk to a provider or parse JSON on their
own outside what's defined here.
"""

import json
from typing import Optional

from pydantic import ValidationError

from src.evidence import redact_evidence
from src.models import ExceptionInvestigation

PROVIDER_TIMEOUT_SECONDS = 30.0

# Cap on how many representative execution/correlation IDs are sent per
# evidence item — the dashboard may track more, but the LLM only needs
# enough to write a concrete, traceable investigation step, not a full list.
MAX_IDS_PER_EVIDENCE_ITEM = 5


class LLMProviderError(Exception):
    """Raised by a provider client on any failure (timeout, HTTP error,
    malformed response after one retry) so ResilientLLMClient can fall back
    to the offline generator instead of crashing the app."""


SYSTEM_PROMPT = """You are an assistant embedded in an exception-investigation \
tool for backend/workflow execution logs. You are given the deterministically \
computed evidence for ONE exception group (already fingerprinted, categorized, \
counted, and prioritized by non-LLM code) and must return a structured \
investigation as JSON matching the exact schema you're given.

You MAY:
- Explain, in plain language, what the exception group represents.
- Propose up to 3 ranked probable-cause hypotheses.
- Recommend concrete investigation procedures.
- Compare remediation options with their tradeoffs.
- Draft a verification plan and prevention recommendations.

You MUST NOT:
- Invent evidence that was not provided to you. Every claim in known_facts, \
and every supporting_evidence_ids/contradicting_evidence_ids/evidence_reference \
value, must trace to an evidence_id actually present in the input.
- Calculate or restate metrics differently than given (do not invent counts, \
percentages, or rates).
- Change or override the priority score/label you were given.
- State that any hypothesis is the confirmed root cause. Use language like \
"makes X a plausible cause, but Y evidence is required for confirmation" — \
never "X caused this" or "the root cause is X."
- Invent business impact. If the evidence does not establish business impact, \
say so plainly rather than guessing.
- Recommend an action that could cause data loss, an outage, or irreversible \
change without also stating its risk.
- Repeat or reference any credential, token, password, or personal data — \
the evidence you receive has already been redacted; do not attempt to \
reconstruct or guess redacted values.

Every hypothesis needs at least one supporting_evidence_ids entry pointing to \
a real evidence_id, or an empty list plus an explanation in the hypothesis \
text that no direct evidence supports it yet. Every investigation_action \
needs evidence_reference pointing to real evidence_id values it follows up on. \
If evidence is thin, say so in limitations and unknowns rather than padding \
the response with generic advice. Keep hypotheses, actions, and options \
specific to THIS exception group's category and evidence — never generic \
boilerplate.

You may optionally include diagnostic_commands: example grep/query/curl-style \
commands an engineer could run to gather more evidence. Every command MUST use \
angle-bracket placeholders (e.g. <correlation_id>, <execution_id>) and MUST NOT \
contain a real credential, token, hostname, or value from the evidence you were \
given. Each command needs what to look for and what each possible finding would \
mean for the investigation. Omit this field entirely if no command genuinely adds \
value beyond the investigation_actions already listed.

Respond with ONLY a single JSON object matching the given schema. No prose \
before or after, no markdown code fences.
"""


def _trimmed_ids(values: list) -> list:
    return list(values[:MAX_IDS_PER_EVIDENCE_ITEM])


def build_exception_evidence(group, evidence_items: list, missing_evidence: list) -> dict:
    """Builds the sanitized, structured payload for ONE exception group —
    the only thing ever sent to an LLM. Deliberately excludes the raw
    dataframe/full dataset; only already-computed group fields and evidence
    items (each capped to a handful of representative IDs) are included,
    then the whole payload is redacted before being returned.
    """
    payload = {
        "exception_identity": {
            "primary_category": group.primary_category,
            "secondary_category": group.secondary_category,
            "error_type": group.error_type,
            "error_code": group.error_code,
            "representative_message": group.representative_message,
            "workflow_name": group.workflow_name,
            "workflow_step": group.workflow_step,
            "service": group.service,
            "endpoint": group.endpoint,
            "http_method": group.http_method,
            "http_status": group.http_status,
            "provider": group.provider,
            "environment": group.environment,
        },
        "occurrence": {
            "occurrence_count": group.occurrence_count,
            "affected_execution_count": group.affected_execution_count,
            "first_occurrence": str(group.first_occurrence),
            "last_occurrence": str(group.last_occurrence),
            "percentage_of_all_failures": group.percentage_of_all_failures,
            "percentage_of_workflow_executions": group.percentage_of_workflow_executions,
            "trend": group.trend_label,
            "application_versions": group.application_versions,
            "workflows_sharing_this_issue": group.workflows_sharing_this_issue,
        },
        "impact": {
            "avg_duration_seconds": group.avg_duration_seconds,
            "p95_duration_seconds": group.p95_duration_seconds,
            "http_status_distribution": group.http_status_distribution,
            "retry_info": group.retry_info,
        },
        "priority": {
            "score": group.priority_score,
            "label": group.priority_label,
            "factors": group.priority_factors,
        },
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "source_field": item.source_field,
                "observed_value": item.observed_value,
                "occurrence_count": item.occurrence_count,
                "why_it_matters": item.why_it_matters,
                "representative_execution_ids": _trimmed_ids(item.representative_execution_ids),
                "representative_correlation_ids": _trimmed_ids(item.representative_correlation_ids),
            }
            for item in evidence_items
        ],
        "missing_evidence": missing_evidence,
    }
    return redact_evidence(payload)


class LLMClient:
    """Interface every provider client (and the offline generator) implements."""

    def analyze(self, evidence: dict) -> ExceptionInvestigation:
        raise NotImplementedError


class ResilientLLMClient(LLMClient):
    """Wraps a real provider client; on any LLMProviderError, falls back to
    the offline rule-based generator rather than surfacing an error to the
    user, and records why in the resulting report's limitations."""

    def __init__(self, primary: LLMClient, provider_name: str):
        self._primary = primary
        self._provider_name = provider_name

    def analyze(self, evidence: dict) -> ExceptionInvestigation:
        from src.llm.offline import generate_offline_investigation

        try:
            return self._primary.analyze(evidence)
        except LLMProviderError as exc:
            report = generate_offline_investigation(evidence)
            report.limitations = [
                f"{self._provider_name} call failed ({exc}); showing rule-based guidance instead.",
                *report.limitations,
            ]
            return report


def parse_exception_investigation(raw_text: str, source: str) -> Optional[ExceptionInvestigation]:
    """Defensively parses a provider's raw text response into an
    ExceptionInvestigation. Returns None (never raises) on malformed JSON or
    a schema mismatch, so callers can retry once and then fall back."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    data["source"] = source
    try:
        return ExceptionInvestigation.model_validate(data)
    except ValidationError:
        return None


def get_client() -> LLMClient:
    """Selects a provider by which API key is set (Anthropic first, then
    OpenAI), wrapped in ResilientLLMClient; falls back to the offline
    generator outright when neither key is present."""
    import os

    from src.llm.offline import OfflineLLMClient

    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.llm.anthropic_client import AnthropicLLMClient

        return ResilientLLMClient(AnthropicLLMClient(), "Anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        from src.llm.openai_client import OpenAILLMClient

        return ResilientLLMClient(OpenAILLMClient(), "OpenAI")
    return OfflineLLMClient()


def active_provider_label() -> str:
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "Anthropic Claude"
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI"
    return "Offline (rule-based)"
