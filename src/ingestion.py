import json
import os
from typing import Any, Optional

import pandas as pd

from src.models import (
    ALL_COLUMNS,
    OPTIONAL_NUMERIC_COLUMNS,
    OPTIONAL_TEXT_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_STATUSES,
    RowRejection,
    ValidationReport,
)

# JSON's wire-format status value differs from the engine's internal
# vocabulary (VALID_STATUSES in src/models.py) — "failed" there means the
# same thing "failure" means everywhere else in this app. Translated here,
# at the ingestion boundary, so nothing downstream (src/anomalies.py,
# src/metrics.py, ...) ever needs to know two spellings exist.
_JSON_STATUS_ALIASES = {"failed": "failure", "success": "success"}


def load_and_validate(file: Any, filename: Optional[str] = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Loads a CSV or JSON execution log and validates it into the common
    execution model (REQUIRED_COLUMNS + OPTIONAL_COLUMNS, see src/models.py)
    — the same shape regardless of source format, so everything downstream
    (SQLite storage, KPI/anomaly calculation, exception grouping, the LLM
    evidence builder) is completely format-independent. Optional technical
    fields default to "" / NaN when a record or format doesn't provide them.

    Format is detected from `filename` if given, else from `file.name`
    (a Streamlit UploadedFile), else from `file` itself if it's a path
    string. Falls back to CSV if nothing indicates otherwise — this keeps
    every existing CSV call site and test working unchanged.

    Returns (valid_df, ValidationReport). On a fatal error (missing/invalid
    structure, unreadable file) valid_df is empty and report.fatal_error is
    set to a generic message — never the raw parser exception text.
    """
    fmt = _detect_format(file, filename)
    if fmt == "json":
        return _load_json(file)
    return _load_csv(file)


def _detect_format(file: Any, filename: Optional[str]) -> str:
    name = filename or getattr(file, "name", None) or (file if isinstance(file, str) else "")
    ext = os.path.splitext(str(name))[1].lower()
    return "json" if ext == ".json" else "csv"


# --- CSV -----------------------------------------------------------------------


def _load_csv(file: Any) -> tuple[pd.DataFrame, ValidationReport]:
    try:
        if hasattr(file, "seek"):
            file.seek(0)  # a Streamlit UploadedFile can persist across reruns; always read from the start
        df = pd.read_csv(file)
    except Exception:
        return pd.DataFrame(), ValidationReport(
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            fatal_error="Could not parse this file as CSV. Check that it's valid CSV with a header row.",
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return pd.DataFrame(), ValidationReport(
            total_rows=len(df),
            valid_rows=0,
            rejected_rows=len(df),
            fatal_error=f"Missing required column(s): {', '.join(missing)}",
        )

    total_rows = len(df)
    df, row_reasons = _validate_rows(
        df.copy(),
        min_duration_exclusive=False,
        iso8601_strict=False,
        reject_error_present_on_success=False,
    )
    return _build_report(df, row_reasons, total_rows, row_offset=2)  # +2: 1-indexed, plus header row


# --- JSON ------------------------------------------------------------------------


def _read_text(file: Any) -> str:
    if hasattr(file, "seek"):
        file.seek(0)
    content = file.read() if hasattr(file, "read") else open(file, "r", encoding="utf-8").read()
    return content.decode("utf-8") if isinstance(content, bytes) else content


def _extract_json_record(obj: Any) -> tuple[Optional[dict], list[str]]:
    """Flattens one JSON execution object into the common column shape.
    Missing simple fields (execution_id, workflow_name, ...) are left as
    None here and caught generically by _validate_rows (same as an empty
    CSV cell would be) — only the nested `error` object's shape is JSON-
    specific enough to need its own structural check. The optional
    technical fields (environment, service, http_status, ...) are all
    top-level; only type/code/message/stack_summary live inside `error`.
    """
    if not isinstance(obj, dict):
        return None, ["record is not a JSON object"]

    reasons: list[str] = []
    status_raw = obj.get("status")
    status_key = str(status_raw).strip().lower() if status_raw is not None else ""
    status = _JSON_STATUS_ALIASES.get(status_key, status_key)

    error = obj.get("error", "__missing__")
    error_type, error_message, error_code, error_stack_summary = "", "", "", ""
    if error == "__missing__":
        reasons.append("missing 'error' field")
    elif error is None:
        pass  # valid for a successful execution
    elif isinstance(error, dict):
        error_type = str(error.get("type") or "").strip()
        error_message = str(error.get("message") or "").strip()
        error_code = str(error.get("code") or "").strip()
        error_stack_summary = str(error.get("stack_summary") or "").strip()
    else:
        reasons.append("'error' must be null or an object")

    candidate = {
        "execution_id": obj.get("execution_id"),
        "workflow_name": obj.get("workflow_name"),
        "start_time": obj.get("start_time"),
        "status": status,
        "duration_seconds": obj.get("duration_seconds"),
        "error_type": error_type,
        "error_message": error_message,
        "environment": obj.get("environment"),
        "workflow_step": obj.get("workflow_step"),
        "service": obj.get("service"),
        "endpoint": obj.get("endpoint"),
        "http_method": obj.get("http_method"),
        "http_status": obj.get("http_status"),
        "correlation_id": obj.get("correlation_id"),
        "retry_count": obj.get("retry_count"),
        "application_version": obj.get("application_version"),
        "provider": obj.get("provider"),
        "error_code": error_code,
        "error_stack_summary": error_stack_summary,
    }
    return candidate, reasons


def _load_json(file: Any) -> tuple[pd.DataFrame, ValidationReport]:
    try:
        data = json.loads(_read_text(file))
    except Exception:
        return pd.DataFrame(), ValidationReport(
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            fatal_error="Could not parse this file as JSON. Check that it contains valid JSON syntax.",
        )

    if not isinstance(data, list):
        return pd.DataFrame(), ValidationReport(
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            fatal_error="Expected a JSON array of execution objects at the top level.",
        )

    total_rows = len(data)
    candidates = []
    row_reasons: dict[int, list[str]] = {}
    for i, obj in enumerate(data):
        candidate, structural_reasons = _extract_json_record(obj)
        candidates.append(candidate or {col: None for col in ALL_COLUMNS})
        if structural_reasons:
            row_reasons[i] = list(structural_reasons)

    df = pd.DataFrame(candidates, columns=ALL_COLUMNS)
    df, extra_reasons = _validate_rows(
        df,
        min_duration_exclusive=True,
        iso8601_strict=True,
        reject_error_present_on_success=True,
    )
    for idx, reasons in extra_reasons.items():
        row_reasons.setdefault(idx, []).extend(reasons)

    return _build_report(df, row_reasons, total_rows, row_offset=1)  # 1-indexed, no header row


# --- shared row-level validation ------------------------------------------------


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantees every optional column exists, whichever format produced
    `df` and however many of them it actually provided — so nothing
    downstream (fingerprinting, exception grouping, ...) has to special-case
    a missing optional field with a KeyError guard."""
    for col in OPTIONAL_TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _is_bad_retry_count(value: Any) -> bool:
    """True if `value` is present but not a non-negative integer. bool is
    deliberately rejected even though Python treats it as an int subtype —
    a retry count is never semantically true/false."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or (value is pd.NA):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return value < 0 or float(value) != int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return numeric < 0 or numeric != int(numeric)


def _validate_rows(
    df: pd.DataFrame,
    *,
    min_duration_exclusive: bool,
    iso8601_strict: bool,
    reject_error_present_on_success: bool,
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """The validation rules genuinely shared by both formats (uniqueness,
    status enum, missing-error-info), plus the handful of points where CSV
    and JSON are deliberately held to different strictness — parametrized
    here rather than duplicated in two call sites.
    """
    df = _ensure_optional_columns(df)
    row_reasons: dict[int, list[str]] = {}

    def flag(mask: pd.Series, reason: str) -> None:
        for idx in df.index[mask]:
            row_reasons.setdefault(idx, []).append(reason)

    empty_id = df["execution_id"].isna() | (df["execution_id"].astype(str).str.strip() == "")
    flag(empty_id, "missing execution_id")

    dup_id = df["execution_id"].duplicated(keep="first") & ~empty_id
    flag(dup_id, "duplicate execution_id")

    if iso8601_strict:
        parsed_time = pd.to_datetime(df["start_time"], format="ISO8601", errors="coerce", utc=True)
        if parsed_time.notna().any():
            parsed_time = parsed_time.dt.tz_localize(None)
        flag(parsed_time.isna(), "start_time is not a valid ISO 8601 timestamp")
    else:
        parsed_time = pd.to_datetime(df["start_time"], errors="coerce", utc=False)
        flag(parsed_time.isna(), "unparsable start_time")
    df["start_time"] = parsed_time

    parsed_duration = pd.to_numeric(df["duration_seconds"], errors="coerce")
    bad_duration = parsed_duration.isna() | (parsed_duration <= 0 if min_duration_exclusive else parsed_duration < 0)
    flag(bad_duration, "invalid duration_seconds")
    df["duration_seconds"] = parsed_duration

    normalized_status = df["status"].astype(str).str.strip().str.lower()
    flag(~normalized_status.isin(VALID_STATUSES), "invalid status")
    df["status"] = normalized_status

    df["workflow_name"] = df["workflow_name"].astype(str).str.strip()
    df["error_type"] = df["error_type"].fillna("").astype(str).str.strip()
    df["error_message"] = df["error_message"].fillna("").astype(str).str.strip()

    missing_error_info = (df["status"] == "failure") & (df["error_type"] == "") & (df["error_message"] == "")
    flag(missing_error_info, "failed execution missing error information")

    if reject_error_present_on_success:
        success_with_error = (df["status"] == "success") & ((df["error_type"] != "") | (df["error_message"] != ""))
        flag(success_with_error, "error must be null for a successful execution")

    for col in OPTIONAL_TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["http_status"] = pd.to_numeric(df["http_status"], errors="coerce")

    bad_retry = df["retry_count"].apply(_is_bad_retry_count)
    flag(bad_retry, "retry_count must be a non-negative integer")
    df["retry_count"] = pd.to_numeric(df["retry_count"], errors="coerce")

    return df, row_reasons


def _build_report(
    df: pd.DataFrame, row_reasons: dict[int, list[str]], total_rows: int, row_offset: int
) -> tuple[pd.DataFrame, ValidationReport]:
    valid_mask = ~df.index.isin(row_reasons.keys())
    valid_df = df.loc[valid_mask, ALL_COLUMNS].reset_index(drop=True)

    rejection_reasons: dict[str, int] = {}
    for reasons in row_reasons.values():
        for reason in reasons:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    rejected_row_details = [
        RowRejection(row_number=idx + row_offset, reasons=sorted(set(reasons)))
        for idx, reasons in sorted(row_reasons.items())
    ]

    report = ValidationReport(
        total_rows=total_rows,
        valid_rows=len(valid_df),
        rejected_rows=total_rows - len(valid_df),
        rejection_reasons=rejection_reasons,
        rejected_row_details=rejected_row_details,
    )
    return valid_df, report
