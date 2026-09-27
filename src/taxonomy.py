"""Deterministic exception-category classification — no LLM involved.

Matches error_type / error_code / http_status / error_message against a
fixed rule table, checked in a specific priority order (more specific
categories first, so e.g. a retry-exhaustion error isn't swallowed by the
generic timeout rule it also happens to mention). Falls back to
"Unclassified" rather than forcing a confident category onto ambiguous data.
"""

import re
from dataclasses import dataclass
from typing import Optional

UNCLASSIFIED = "Unclassified"
UNCLASSIFIED_SECONDARY = "Additional evidence required"

# Assigned separately (see src/exception_groups.py) for slow-but-successful
# executions — never reached via classify_exception, since those rows have
# no error_type at all.
PERFORMANCE_ANOMALY_CATEGORY = "Performance anomaly without failure"


@dataclass(frozen=True)
class _Rule:
    primary: str
    secondary: str
    keywords: tuple = ()  # any substring match in error_type/code/message triggers this rule
    http_statuses: tuple = ()  # or an http_status match triggers it


# Order matters: earlier rules win. Retry/idempotency and AI/structured-
# output are checked before the generic categories they could otherwise be
# absorbed into (a "retry exhaustion" error also mentions retries, but isn't
# a rate-limiting issue; an "invalid JSON schema" error is a structured-
# output problem, not generic input validation).
_RULES: tuple = (
    _Rule("Retry and idempotency", "Retry exhaustion", ("RETRY_EXHAUST", "MAX_RETRIES", "RETRY_STORM")),
    _Rule("Retry and idempotency", "Non-idempotent operation", ("NON_IDEMPOTENT", "IDEMPOTENCY")),
    _Rule("Retry and idempotency", "Incorrect backoff", ("BACKOFF",)),
    _Rule("Retry and idempotency", "Duplicate caused by retry", ("RETRY_DUPLICATE",)),
    _Rule("AI and structured-output", "Invalid JSON", ("INVALID_JSON", "MALFORMED_JSON", "JSON_PARSE", "UNEXPECTED_TOKEN")),
    _Rule("AI and structured-output", "Output-schema violation", ("JSON_SCHEMA", "OUTPUT_SCHEMA", "SCHEMA_VIOLATION")),
    _Rule("AI and structured-output", "Type mismatch", ("TYPE_MISMATCH",)),
    _Rule("AI and structured-output", "Hallucinated value", ("HALLUCINAT",)),
    _Rule("AI and structured-output", "Response truncation", ("TRUNCAT", "CONTEXT_LENGTH")),
    _Rule("AI and structured-output", "Model refusal", ("MODEL_REFUS", "REFUSAL", "CONTENT_POLICY", "CONTENT_FILTER", "SAFETY_SYSTEM")),
    _Rule("Authentication and authorization", "Invalid or expired token", ("INVALID_TOKEN", "EXPIRED_TOKEN", "TOKEN_EXPIR")),
    _Rule("Authentication and authorization", "Incorrect audience or scope", ("AUDIENCE", "INVALID_SCOPE")),
    _Rule("Authentication and authorization", "Permission denied", ("PERMISSION_DENIED", "FORBIDDEN")),
    _Rule("Authentication and authorization", "Missing credentials", ("MISSING_CREDENTIAL", "NO_CREDENTIAL")),
    _Rule("Authentication and authorization", "Secret or certificate expiration", ("CERT_EXPIR", "SECRET_EXPIR", "CERTIFICATE_EXPIR")),
    _Rule("Authentication and authorization", "Invalid or expired token", ("AUTH",), (401,)),
    _Rule("Authentication and authorization", "Permission denied", (), (403,)),
    _Rule("Rate limiting and capacity", "HTTP 429", (), (429,)),
    _Rule("Rate limiting and capacity", "Quota exceeded", ("QUOTA",)),
    _Rule("Rate limiting and capacity", "Worker saturation", ("SATURAT",)),
    _Rule("Rate limiting and capacity", "Queue backlog", ("QUEUE_BACKLOG", "BACKLOG")),
    _Rule("Rate limiting and capacity", "Resource exhaustion", ("RESOURCE_EXHAUST",)),
    _Rule("Rate limiting and capacity", "Rate limited", ("RATE_LIMIT", "THROTTL")),
    _Rule("Timeout and latency", "Gateway timeout", ("GATEWAY_TIMEOUT",), (504,)),
    _Rule("Timeout and latency", "Connection timeout", ("CONNECT_TIMEOUT", "CONNECTION_TIMEOUT")),
    _Rule("Timeout and latency", "Read timeout", ("READ_TIMEOUT",)),
    _Rule("Timeout and latency", "Processing timeout", ("PROCESSING_TIMEOUT",)),
    _Rule("Timeout and latency", "Gateway timeout", (), (408, 504)),
    _Rule("Timeout and latency", "Slow dependency", ("TIMEOUT", "TIMED_OUT")),
    _Rule("Connectivity and dependency", "Connection refused", ("CONNECTION_REFUSED", "ECONNREFUSED")),
    _Rule("Connectivity and dependency", "DNS failure", ("DNS",)),
    _Rule("Connectivity and dependency", "TLS failure", ("TLS", "SSL")),
    _Rule("Connectivity and dependency", "External dependency unavailable", ("UNAVAILABLE", "SERVICE_UNAVAILABLE", "UNDELIVERABLE", "UNREACHABLE")),
    _Rule("Connectivity and dependency", "Network interruption", ("NETWORK_ERROR", "ECONNRESET", "MEDIA_DOWNLOAD", "DOWNLOAD_ERROR")),
    _Rule("Connectivity and dependency", "External dependency unavailable", (), (502, 503)),
    _Rule("Data integrity and persistence", "Duplicate record", ("DUPLICATE_RECORD", "DUPLICATE")),
    _Rule("Data integrity and persistence", "Unique-key conflict", ("UNIQUE_KEY", "UNIQUE_CONSTRAINT")),
    _Rule("Data integrity and persistence", "Foreign-key failure", ("FOREIGN_KEY",)),
    _Rule("Data integrity and persistence", "Transaction failure", ("TRANSACTION",)),
    _Rule("Data integrity and persistence", "Inconsistent state", ("INCONSISTENT_STATE",)),
    _Rule("Data integrity and persistence", "Partial write", ("PARTIAL_WRITE",)),
    _Rule("Data integrity and persistence", "Unique-key conflict", (), (409,)),
    _Rule("Business-rule failure", "Ineligible operation", ("INELIGIBLE", "CARD_DECLINED", "INSUFFICIENT_FUNDS", "EXPIRED_CARD")),
    _Rule("Business-rule failure", "Invalid workflow state", ("INVALID_WORKFLOW_STATE", "INVALID_STATE", "RE_ENGAGEMENT", "REENGAGEMENT")),
    _Rule("Business-rule failure", "Duplicate business event", ("DUPLICATE_EVENT",)),
    _Rule("Business-rule failure", "Business constraint violation", ("BUSINESS_RULE", "BUSINESS_CONSTRAINT", "UNSUBSCRIBED", "OPTED_OUT")),
    _Rule("Configuration and deployment", "Missing environment variable", ("MISSING_ENV", "ENV_VAR")),
    _Rule("Configuration and deployment", "Invalid configuration", ("INVALID_CONFIG", "CHANNEL_NOT_FOUND", "RANGE_ERROR", "NO_LONGER_EXISTS")),
    _Rule("Configuration and deployment", "Version regression", ("VERSION_REGRESSION",)),
    _Rule("Configuration and deployment", "Incompatible contract", ("INCOMPATIBLE_CONTRACT", "CONTRACT_MISMATCH")),
    _Rule("Configuration and deployment", "Feature-flag issue", ("FEATURE_FLAG",)),
    _Rule("Input and validation", "Missing required field", ("REQUIRED_FIELD", "MISSING_FIELD", "PARAMETER_MISSING", "PARAM_MISSING", "MISSING_REQUIRED")),
    _Rule("Input and validation", "Invalid value", ("INVALID_VALUE", "INVALID_PHONE")),
    _Rule("Input and validation", "Unsupported format", ("UNSUPPORTED_FORMAT",)),
    _Rule("Input and validation", "Invalid date or number format", ("INVALID_DATE", "INVALID_NUMBER", "INVALID_FORMAT")),
    _Rule("Input and validation", "Schema mismatch", ("SCHEMA_MISMATCH", "VALIDATION", "PARAM_MISMATCH", "TEMPLATE_MISMATCH")),
    _Rule("Internal application error", "Null reference", ("NULL_REFERENCE", "NULLPOINTER", "NONE_TYPE")),
    _Rule("Internal application error", "Serialization failure", ("SERIALIZATION",)),
    _Rule("Internal application error", "Unhandled exception", ("UNHANDLED", "UNEXPECTED_ERROR")),
    _Rule("Internal application error", "Internal HTTP 500", (), (500,)),
)


_SEPARATOR_PATTERN = re.compile(r"[_\-\s]+")


def _normalize(text: str) -> str:
    """Strips underscores/hyphens/whitespace so keyword matching isn't
    fooled by naming convention alone — a real-world error_type is just as
    likely to arrive as 'InvalidJSONSchemaError' (PascalCase) as
    'INVALID_JSON_SCHEMA' (SNAKE_CASE), and both must hit the same rule."""
    return _SEPARATOR_PATTERN.sub("", text.upper())


def _clean_http_status(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if value != value:  # NaN != NaN; avoids importing math/pandas just for this
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_exception(
    error_type: str, error_code: str = "", http_status=None, error_message: str = ""
) -> tuple:
    """Returns (primary_category, secondary_category). Pure keyword/status
    matching, never an LLM call. Returns the Unclassified pair when nothing
    matches confidently rather than forcing a guess."""
    haystack = _normalize(" ".join(str(v) for v in (error_type, error_code, error_message) if v))
    status = _clean_http_status(http_status)

    for rule in _RULES:
        keyword_hit = any(_normalize(kw) in haystack for kw in rule.keywords)
        status_hit = status is not None and status in rule.http_statuses
        if keyword_hit or status_hit:
            return rule.primary, rule.secondary

    return UNCLASSIFIED, UNCLASSIFIED_SECONDARY
