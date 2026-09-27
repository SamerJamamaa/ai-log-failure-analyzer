"""Groups failed executions into ExceptionGroups by deterministic
fingerprint (src.fingerprinting), computes every occurrence/impact/evidence
statistic the investigation dashboard needs, and classifies each group
(src.taxonomy). No LLM involved anywhere in this module.
"""

import numpy as np
import pandas as pd

from src.fingerprinting import compute_fingerprint
from src.models import ExceptionGroup
from src.taxonomy import classify_exception

# --- Configurable thresholds -------------------------------------------------
# A group with this many occurrences or fewer is "isolated" — a one-off,
# not (yet) a pattern.
TREND_ISOLATED_MAX = 2

# Need at least this many occurrences in EACH chronological half of a
# group's own timeline before judging a trend at all — guards against
# calling a 4-occurrence group "increasing" off a 1-vs-3 split.
TREND_MIN_RECORDS_PER_HALF = 3

# The second half must have at least this many times the first half's
# occurrence count to be called "increasing" (not just "recurring").
TREND_INCREASE_MIN_RATIO = 2.0
# -----------------------------------------------------------------------------

REPRESENTATIVE_SAMPLE_SIZE = 5


def _p95(series: pd.Series) -> float:
    clean = series.dropna()
    return float(np.percentile(clean, 95)) if not clean.empty else 0.0


def _mode_or_empty(series: pd.Series) -> str:
    clean = series.dropna()
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    return str(clean.mode().iloc[0])


def _compute_trend(rows_sorted: pd.DataFrame) -> tuple:
    """Isolated / recurring / increasing, based only on this group's own
    occurrence timeline — never on the LLM. See threshold constants above.

    Splits at the CHRONOLOGICAL midpoint (halfway between first and last
    occurrence), not at the row-count midpoint — a count-based split always
    yields two nearly-equal-sized halves by construction (floor/ceil of n/2
    can differ by at most 1), which can never clear a "2x more occurrences"
    ratio threshold. A time-based split lets genuine clustering (many
    occurrences packed into a recent window) actually show up as a count
    imbalance between the two halves.
    """
    n = len(rows_sorted)
    if n <= TREND_ISOLATED_MAX:
        return "isolated", False

    start, end = rows_sorted["start_time"].min(), rows_sorted["start_time"].max()
    if end <= start:
        return "recurring", False  # all occurrences at effectively the same instant

    midpoint_time = start + (end - start) / 2
    first_half = rows_sorted[rows_sorted["start_time"] < midpoint_time]
    second_half = rows_sorted[rows_sorted["start_time"] >= midpoint_time]
    if len(first_half) < TREND_MIN_RECORDS_PER_HALF or len(second_half) < TREND_MIN_RECORDS_PER_HALF:
        return "recurring", False

    if len(second_half) >= len(first_half) * TREND_INCREASE_MIN_RATIO:
        return "increasing", True
    return "recurring", False


def _retry_info(retry_values: pd.Series) -> dict:
    clean = retry_values.dropna()
    if clean.empty:
        return {}
    return {
        "executions_with_retries": int((clean > 0).sum()),
        "avg_retry_count": round(float(clean.mean()), 1),
        "max_retry_count": int(clean.max()),
        "all_retries_still_failed": bool((clean > 0).any()),  # every row here is a failure by construction
    }


def _http_status_distribution(status_values: pd.Series) -> dict:
    clean = status_values.dropna()
    if clean.empty:
        return {}
    return {str(int(status)): int(count) for status, count in clean.astype(int).value_counts().items()}


def build_exception_groups(df: pd.DataFrame) -> list:
    """One ExceptionGroup per distinct fingerprint among df's failed rows.
    `df` is the full validated dataset (all statuses) — needed so each
    group can compute its share of its workflow's TOTAL executions, not
    just of its failures.
    """
    failures = df[df["status"] == "failure"].copy()
    if failures.empty:
        return []

    # defensive: coerce to numeric even if a caller bypassed ingestion.py's
    # own coercion (e.g. handed us a DataFrame built some other way) — "" or
    # other junk becomes NaN rather than crashing the mode()/int() below
    failures["http_status"] = pd.to_numeric(failures["http_status"], errors="coerce")
    failures["retry_count"] = pd.to_numeric(failures["retry_count"], errors="coerce")

    failures["_fingerprint"] = failures.apply(compute_fingerprint, axis=1)
    total_failures = len(failures)

    # cross-group signal: how many distinct workflows see this same
    # (service, error_type) pair, regardless of fingerprint — a real
    # "problem is spreading" signal, since fingerprint itself always
    # includes workflow_name (so a single fingerprint is always one workflow)
    workflow_spread = failures.groupby(["service", "error_type"])["workflow_name"].nunique()

    groups = []
    for fingerprint, rows in failures.groupby("_fingerprint"):
        rows = rows.sort_values("start_time")
        workflow_name = rows["workflow_name"].iloc[0]
        workflow_total = int((df["workflow_name"] == workflow_name).sum())

        error_type = _mode_or_empty(rows["error_type"])
        error_code = _mode_or_empty(rows["error_code"])
        representative_message = rows["error_message"].iloc[0]
        http_status_values = rows["http_status"].dropna()
        representative_http_status = int(http_status_values.mode().iloc[0]) if not http_status_values.empty else None

        primary_category, secondary_category = classify_exception(
            error_type, error_code, representative_http_status, representative_message
        )
        trend_label, is_increasing = _compute_trend(rows)

        service_key = rows["service"].iloc[0]
        spread = int(workflow_spread.get((service_key, error_type), 1))

        groups.append(
            ExceptionGroup(
                fingerprint=fingerprint,
                primary_category=primary_category,
                secondary_category=secondary_category,
                error_type=error_type,
                error_code=error_code,
                workflow_name=workflow_name,
                workflow_step=_mode_or_empty(rows["workflow_step"]),
                service=service_key,
                endpoint=_mode_or_empty(rows["endpoint"]),
                http_method=_mode_or_empty(rows["http_method"]),
                http_status=representative_http_status,
                provider=_mode_or_empty(rows["provider"]),
                environment=_mode_or_empty(rows["environment"]),
                representative_message=representative_message,
                occurrence_count=len(rows),
                affected_execution_count=int(rows["execution_id"].nunique()),
                first_occurrence=rows["start_time"].min(),
                last_occurrence=rows["start_time"].max(),
                percentage_of_all_failures=round(len(rows) / total_failures * 100, 1) if total_failures else 0.0,
                percentage_of_workflow_executions=round(len(rows) / workflow_total * 100, 1) if workflow_total else 0.0,
                trend_label=trend_label,
                is_increasing=is_increasing,
                avg_duration_seconds=round(float(rows["duration_seconds"].mean()), 1),
                p95_duration_seconds=round(_p95(rows["duration_seconds"]), 1),
                http_status_distribution=_http_status_distribution(rows["http_status"]),
                retry_info=_retry_info(rows["retry_count"]),
                application_versions=sorted({v for v in rows["application_version"] if v}),
                hosts_affected=[],  # no host field exists in the input schema today
                workflows_sharing_this_issue=spread,
                representative_execution_ids=rows["execution_id"].head(REPRESENTATIVE_SAMPLE_SIZE).tolist(),
                representative_correlation_ids=sorted({c for c in rows["correlation_id"] if c})[:REPRESENTATIVE_SAMPLE_SIZE],
                rows=rows.drop(columns=["_fingerprint"]),
            )
        )

    return groups
