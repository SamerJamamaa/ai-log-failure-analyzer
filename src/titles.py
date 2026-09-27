"""Human-readable exception titles — deterministic, rule-based, never an LLM
call. Turns a raw error_type/category/context combination (e.g.
'AUTHENTICATION_ERROR' on version 3.5.0) into a sentence a human would
actually read in an incident queue ('CRM access token rejected after
version 3.5.0'), the way a production-support engineer would describe it.

Every branch here reads only already-computed ExceptionGroup fields —
nothing is invented, and a category with no dedicated template still gets a
readable (if plainer) sentence via the fallback.
"""


def _subject(group) -> str:
    """Prefers `provider` over `service` — for an orchestrated workflow
    (e.g. n8n), `service` is often just the generic orchestrator itself
    ('n8n-worker') for every node, while `provider` names the actual
    external system a given failure came from (google-sheets, gmail,
    stripe, ...), which is what a reader investigating actually needs."""
    return group.provider or group.service or group.workflow_name


def _version_suffix(group) -> str:
    if len(group.application_versions) == 1:
        return f" after version {group.application_versions[0]}"
    return ""


def _authentication_title(group) -> str:
    subject = _subject(group)
    if group.http_status == 403:
        return f"{subject} denying access (HTTP 403){_version_suffix(group)}"
    return f"{subject} access token rejected{_version_suffix(group)}"


def _timeout_title(group) -> str:
    subject = _subject(group)
    if group.avg_duration_seconds:
        return f"{subject} requests timing out after ~{group.avg_duration_seconds:.0f}s"
    return f"{subject} requests timing out"


def _connectivity_title(group) -> str:
    subject = _subject(group)
    if group.http_status in (502, 503):
        return f"{subject} unavailable (HTTP {group.http_status})"
    return f"{subject} connection failures"


def _rate_limit_title(group) -> str:
    subject = _subject(group)
    return f"{subject} rate limit exceeded" + (" (HTTP 429)" if group.http_status == 429 else "")


def _validation_title(group) -> str:
    subject = group.workflow_name
    detail = (group.secondary_category or "invalid input").lower()
    return f"{subject} rejecting {detail}"


def _ai_output_title(group) -> str:
    subject = group.workflow_name
    detail = (group.secondary_category or "malformed output").lower()
    return f"{subject} producing {detail}"


def _data_integrity_title(group) -> str:
    subject = group.workflow_name
    if group.http_status == 409 or (group.secondary_category or "").lower() == "duplicate record":
        return f"Duplicate record conflicts in {subject}"
    return f"Data integrity failures in {subject}"


def _retry_title(group) -> str:
    return f"{group.workflow_name} retries exhausted without success"


def _configuration_title(group) -> str:
    subject = group.workflow_name
    return f"{subject} misconfigured{_version_suffix(group)}"


def _internal_error_title(group) -> str:
    subject = _subject(group)
    status = f" (HTTP {group.http_status})" if group.http_status else ""
    return f"{subject} internal error{status}"


def _business_rule_title(group) -> str:
    detail = group.secondary_category or "a business rule"
    return f"{group.workflow_name}: {detail}"


def _performance_title(group) -> str:
    return f"{group.workflow_name} executions running slower than normal"


_TITLE_BUILDERS = {
    "Authentication and authorization": _authentication_title,
    "Timeout and latency": _timeout_title,
    "Connectivity and dependency": _connectivity_title,
    "Rate limiting and capacity": _rate_limit_title,
    "Input and validation": _validation_title,
    "AI and structured-output": _ai_output_title,
    "Data integrity and persistence": _data_integrity_title,
    "Retry and idempotency": _retry_title,
    "Configuration and deployment": _configuration_title,
    "Internal application error": _internal_error_title,
    "Business-rule failure": _business_rule_title,
    "Performance anomaly without failure": _performance_title,
}


def human_title(group) -> str:
    """Returns a readable, one-line description of the exception group.
    Falls back to a plain 'workflow: error_type' sentence for Unclassified
    or any category without a dedicated template, rather than guessing."""
    builder = _TITLE_BUILDERS.get(group.primary_category)
    if builder is not None:
        return builder(group)
    label = group.error_type or group.representative_message[:40] or "an unclassified error"
    return f"{group.workflow_name}: {label}"
