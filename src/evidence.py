"""Builds the deterministic evidence table (section D) and missing-evidence
list (section E) for one ExceptionGroup — every item here is computed from
already-validated data, never invented, and never produced by an LLM.

Also owns redact_evidence(), the sanitization step every evidence payload
must pass through before it's ever sent to an LLM (src/llm/base.py calls
this — see DATA SAFETY requirements).
"""

import re

import pandas as pd

from src.models import EvidenceItem

REPRESENTATIVE_SAMPLE_SIZE = 5

# Deterministic, per-category missing-evidence checklist — what would be
# needed to actually CONFIRM a cause, tailored to what kind of exception
# this is rather than one generic list dumped on everything. Never filled
# with assumptions; these are always explicitly "not available," not guessed.
_MISSING_EVIDENCE_BY_CATEGORY = {
    "Authentication and authorization": [
        "Token expiration timestamp",
        "Configuration diff (OAuth audience/scope) across the affected versions",
        "Identity-provider logs for the affected time window",
    ],
    "Connectivity and dependency": [
        "Provider response body",
        "Infrastructure metrics (network/DNS/TLS handshake logs)",
        "Distributed trace across the request path",
    ],
    "Timeout and latency": [
        "Dependency latency metrics for the affected time window",
        "Provider response body",
        "Distributed trace across the request path",
    ],
    "Rate limiting and capacity": [
        "Provider-side rate-limit/quota configuration",
        "Infrastructure metrics (request volume, worker saturation)",
    ],
    "Input and validation": [
        "The exact request payload that failed validation",
        "Schema/contract version in effect at the time of failure",
    ],
    "AI and structured-output": [
        "The exact raw model response that failed validation",
        "Model/prompt version in effect at the time of failure",
    ],
    "Data integrity and persistence": [
        "Database transaction status for the affected records",
        "Whether the original (non-duplicate) request ultimately succeeded",
    ],
    "Business-rule failure": [
        "The business rule/policy configuration in effect at the time",
    ],
    "Configuration and deployment": [
        "Deployment timestamp",
        "Configuration diff between the last-known-good and current state",
    ],
    "Internal application error": [
        "Full stack trace (beyond the truncated summary)",
        "Deployment timestamp",
    ],
    "Retry and idempotency": [
        "Retry logs (attempt-by-attempt timing and responses)",
        "Whether the underlying operation is idempotency-protected",
    ],
    "Performance anomaly without failure": [
        "Dependency latency metrics for the affected time window",
        "Infrastructure metrics (resource utilization)",
    ],
    "Unclassified": [
        "Additional log fields or context to classify this exception with confidence",
    ],
}

_UNIVERSAL_MISSING_EVIDENCE = ["Request headers"]

# --- redaction ------------------------------------------------------------------

_REDACTION_PATTERNS = [
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_.]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9\-_]+"), r"\1=<redacted>"),
    (re.compile(r"(?i)\b(password|passwd|secret|token)\s*[:=]\s*['\"]?[^\s'\",;]+"), r"\1=<redacted>"),
    (re.compile(r"(?i)\bauthorization\s*:\s*\S+"), "Authorization: <redacted>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "<redacted-jwt>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<redacted-email>"),
]

# Keys that are always stripped outright if present, rather than redacted
# in place — these are the kind of field that should never even be
# structurally sent, not just have their value masked.
_ALWAYS_STRIP_KEYS = {
    "request_body",
    "response_body",
    "headers",
    "request_headers",
    "response_headers",
    "authorization",
    "password",
    "api_key",
    "secret",
    "token",
}


def redact_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_evidence(payload):
    """Recursively redacts credential/secret-shaped substrings from every
    string in `payload` and strips known-sensitive keys outright. Safe to
    call on any JSON-shaped structure (dict/list/str/number)."""
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, dict):
        return {k: redact_evidence(v) for k, v in payload.items() if k.lower() not in _ALWAYS_STRIP_KEYS}
    if isinstance(payload, list):
        return [redact_evidence(v) for v in payload]
    return payload


# --- evidence table --------------------------------------------------------------


def _sample(values, n: int = REPRESENTATIVE_SAMPLE_SIZE) -> list:
    return list(values[:n])


def build_evidence_table(group, full_df: pd.DataFrame) -> list:
    """Deterministic evidence items for one ExceptionGroup. `full_df` is the
    complete validated dataset (all workflows/statuses), needed for the
    contrastive items (e.g. "no failures on version X") that compare this
    group against the rest of the data, not just its own rows.
    """
    items: list = []
    next_id = 1

    def add(source_field, observed_value, occurrence_count, why, exec_ids=None, corr_ids=None):
        nonlocal next_id
        items.append(
            EvidenceItem(
                evidence_id=f"E{next_id}",
                source_field=source_field,
                observed_value=str(observed_value),
                occurrence_count=occurrence_count,
                why_it_matters=why,
                representative_execution_ids=_sample(exec_ids or group.representative_execution_ids),
                representative_correlation_ids=_sample(corr_ids or group.representative_correlation_ids),
            )
        )
        next_id += 1

    # 1. baseline occurrence evidence — always present
    add(
        "occurrence_count",
        f"{group.occurrence_count} occurrences",
        group.occurrence_count,
        f"Represents {group.percentage_of_all_failures}% of all failures and "
        f"{group.percentage_of_workflow_executions}% of '{group.workflow_name}' executions in this upload.",
    )

    # 2. HTTP status, if a dominant value covers all/most occurrences
    if group.http_status_distribution:
        dominant_status, dominant_count = max(group.http_status_distribution.items(), key=lambda kv: kv[1])
        add(
            "http_status",
            f"HTTP {dominant_status}",
            dominant_count,
            "A consistent HTTP status narrows down where in the request lifecycle the failure occurred.",
        )

    # 3. application_version — uniform-version + contrastive "not on other versions"
    if len(group.application_versions) == 1:
        version = group.application_versions[0]
        workflow_rows = full_df[full_df["workflow_name"] == group.workflow_name]
        other_versions = sorted(
            {v for v in workflow_rows["application_version"] if v and v != version}
        )
        add(
            "application_version",
            version,
            group.occurrence_count,
            f"All {group.occurrence_count} failures in this group occurred on version {version}.",
        )
        for other_version in other_versions:
            other_version_rows = workflow_rows[workflow_rows["application_version"] == other_version]
            same_error_on_other = other_version_rows[
                (other_version_rows["status"] == "failure") & (other_version_rows["error_type"] == group.error_type)
            ]
            if same_error_on_other.empty and not other_version_rows.empty:
                add(
                    "application_version",
                    f"no '{group.error_type}' on {other_version}",
                    0,
                    f"No '{group.error_type}' failures appeared on version {other_version} "
                    f"({len(other_version_rows)} execution(s) observed on that version).",
                )

    # 4. retry behavior
    if group.retry_info.get("executions_with_retries"):
        add(
            "retry_count",
            f"{group.retry_info['executions_with_retries']} execution(s) retried (up to {group.retry_info['max_retry_count']}x)",
            group.retry_info["executions_with_retries"],
            "These executions retried and still failed — the cause is unlikely to be a single transient blip.",
        )

    # 5. trend
    if group.trend_label != "isolated":
        add(
            "trend",
            group.trend_label,
            group.occurrence_count,
            "Repeated or worsening occurrence over time, not a one-off." if group.trend_label == "increasing"
            else "This pattern has occurred more than once — not an isolated incident.",
        )

    # 6. cross-workflow spread
    if group.workflows_sharing_this_issue > 1:
        add(
            "workflows_sharing_this_issue",
            group.workflows_sharing_this_issue,
            group.workflows_sharing_this_issue,
            "The same service/error combination is affecting more than one workflow.",
        )

    return items


def missing_evidence_for(group) -> list:
    """Category-specific list of what's needed to CONFIRM a cause — never
    filled with assumptions, always explicit about what's unavailable."""
    category_items = _MISSING_EVIDENCE_BY_CATEGORY.get(group.primary_category, _MISSING_EVIDENCE_BY_CATEGORY["Unclassified"])
    return list(category_items) + list(_UNIVERSAL_MISSING_EVIDENCE)
