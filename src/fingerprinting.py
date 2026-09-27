"""Deterministic exception fingerprinting — groups related failures without
an LLM. Two failures get the same fingerprint iff they agree on workflow,
step, service, endpoint, and error type/code, and their error messages
match after dynamic values (IDs, timestamps, ...) are normalized out — so
"timeout for order 84213" and "timeout for order 91007" still fingerprint
identically, while genuinely different exceptions never collide.

http_status is deliberately NOT a fingerprint field: the same underlying
issue can surface slightly different status codes across occurrences (a 504
from one proxy hop vs. a 502 from another), and folding status into the
identity key would make the separately-tracked "HTTP status distribution"
per group structurally pointless (it could only ever hold one value). It's
still exposed to the classifier and shown as evidence — see exception_groups.py.
"""

import hashlib
import re
from typing import Any

import pandas as pd

# Applied in this order. Each pattern replaces one class of dynamic,
# request-specific value with a stable placeholder before the message
# contributes to a fingerprint.
_NORMALIZATION_PATTERNS = [
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"), "<timestamp>"),
    (re.compile(r"\b[A-Za-z]+-\d+\b"), "<ref_id>"),  # EXE-1002, CORR-83921, ORDER-4471, ...
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "<hex>"),  # long hex tokens (request IDs, hashes) — before <number>
    (re.compile(r"\b\d{4,}\b"), "<number>"),  # long numeric record IDs / large millisecond values
]


def normalize_dynamic_value(text: str) -> str:
    """Strips request-specific dynamic values (UUIDs, ref-style IDs like
    EXE-1002/CORR-83921, ISO timestamps, long numeric/hex IDs) from free
    text, replacing each with a stable placeholder — so two occurrences of
    "the same" underlying failure, differing only in which specific record
    triggered it, normalize to identical text. Short numbers (e.g. "15
    seconds", a 3-digit http code mentioned in prose) are deliberately left
    alone — only 4+ digit runs are treated as record-identifying.
    """
    if not text:
        return ""
    normalized = text
    for pattern, placeholder in _NORMALIZATION_PATTERNS:
        normalized = pattern.sub(placeholder, normalized)
    return normalized.strip().lower()


# The fields that define an exception's identity. Order is fixed so the
# fingerprint is reproducible; each contributes even when empty (an absent
# field is a fixed placeholder, never a wildcard that could merge two
# genuinely different exceptions that both happen to be missing the same field).
_FINGERPRINT_FIELDS = [
    "workflow_name",
    "workflow_step",
    "service",
    "endpoint",
    "error_type",
    "error_code",
]


def _field_str(row: Any, field: str) -> str:
    value = row.get(field) if hasattr(row, "get") else row[field]
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()


def compute_fingerprint(row: Any) -> str:
    """Deterministic fingerprint for one failed execution row (a pandas
    Series or dict carrying at least src.models.ALL_COLUMNS). Never uses
    the LLM — this is pure, reproducible string/hash logic.

    Normalized error_message is used as a fingerprint component only when
    error_type is absent — when a structured error_type IS present, it
    already identifies the exception on its own, and the specific message
    text is supplementary detail (e.g. which field was missing), not
    identity. Including full message text unconditionally would otherwise
    fragment one real exception type into many near-duplicate groups
    whenever a source system varies wording per occurrence (a rotating
    "field X is required" / "field Y is required" message from the same
    schema-validation error, for example) — verified empirically against
    real generated data, not just assumed.
    """
    parts = [_field_str(row, field) for field in _FINGERPRINT_FIELDS]
    if not _field_str(row, "error_type"):
        parts.append(normalize_dynamic_value(_field_str(row, "error_message")))
    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
