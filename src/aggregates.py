"""Dataset-wide and queue-wide deterministic aggregations that power the
KPI cards and the operational-visuals charts — never per-exception-group
evidence (see src/evidence.py) and never LLM-touched. Every function here
either returns a well-defined result or explicitly signals "not enough
data" / "not available" rather than guessing.
"""

import numpy as np
import pandas as pd

from src.titles import human_title

# mirrors the minimum-per-half guard already used for exception-group trend
# classification (src/exception_groups.py) — a period comparison built on
# too few records in either half is not a comparison worth showing
MIN_RECORDS_PER_PERIOD = 5

# same thresholds src/anomalies.py uses for its slow_execution rule, kept in
# sync deliberately — this module computes a SUCCESS-ONLY variant of the
# same idea (see compute_performance_anomalies), so the two numbers should
# never silently drift apart
SLOW_EXECUTION_STD_MULTIPLIER = 2.0
SLOW_EXECUTION_MIN_SAMPLES = 5


def _p95(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    return round(float(np.percentile(clean, 95)), 1)


def compute_period_comparison(df: pd.DataFrame) -> dict:
    """Splits the WHOLE dataset at its chronological midpoint and compares
    failure rate between the two halves. Returns valid=False (and no rate
    numbers) when either half has too few records to be a meaningful
    comparison — never silently substitutes a misleading small-sample delta.
    """
    if df.empty:
        return {"valid": False, "reason": "no data"}

    start, end = df["start_time"].min(), df["start_time"].max()
    if end <= start:
        return {"valid": False, "reason": "all executions occurred at the same time"}

    midpoint = start + (end - start) / 2
    first_half = df[df["start_time"] < midpoint]
    second_half = df[df["start_time"] >= midpoint]

    if len(first_half) < MIN_RECORDS_PER_PERIOD or len(second_half) < MIN_RECORDS_PER_PERIOD:
        return {"valid": False, "reason": "insufficient records in one or both periods"}

    def _rate(subset: pd.DataFrame) -> float:
        return round(float((subset["status"] == "failure").mean()) * 100, 1)

    first_rate, second_rate = _rate(first_half), _rate(second_half)
    return {
        "valid": True,
        "first_period_count": int(len(first_half)),
        "second_period_count": int(len(second_half)),
        "first_period_failure_rate": first_rate,
        "second_period_failure_rate": second_rate,
        "change_pp": round(second_rate - first_rate, 1),
    }


def compute_version_comparison(df: pd.DataFrame) -> list:
    """Per-version executions/failures/failure-rate/category breakdown.
    Returns [] when fewer than two distinct, non-blank versions are
    present — a single-version dataset has nothing to compare."""
    versions = sorted(v for v in df["application_version"].unique() if v)
    if len(versions) < 2:
        return []

    rows = []
    for version in versions:
        subset = df[df["application_version"] == version]
        failures = subset[subset["status"] == "failure"]
        rows.append(
            {
                "version": version,
                "executions": int(len(subset)),
                "failures": int(len(failures)),
                "failure_rate": round((len(failures) / len(subset) * 100) if len(subset) else 0.0, 1),
                "error_types": sorted(t for t in failures["error_type"].unique() if t),
            }
        )
    return rows


def compute_workflow_impact_matrix(groups: list) -> list:
    """One row per (category, workflow) pair actually present among the
    exception groups, with occurrence/affected counts — the raw material
    for the workflow/service impact-matrix chart. A category's spread
    (isolated to one workflow vs. shared across several) is directly
    readable from how many rows it produces here."""
    rows = []
    for g in groups:
        rows.append(
            {
                "category": g.primary_category,
                "workflow": g.workflow_name,
                "service": g.service or "",
                "provider": g.provider or "",
                "occurrence_count": g.occurrence_count,
                "affected_execution_count": g.affected_execution_count,
            }
        )
    return rows


def compute_category_summary(groups: list) -> list:
    """Per primary_category totals, sorted by occurrence count descending
    with a running cumulative percentage — the raw material for both the
    category-breakdown chart and the error-Pareto chart."""
    if not groups:
        return []

    totals: dict = {}
    for g in groups:
        entry = totals.setdefault(
            g.primary_category, {"category": g.primary_category, "group_count": 0, "occurrence_count": 0, "affected_execution_count": 0}
        )
        entry["group_count"] += 1
        entry["occurrence_count"] += g.occurrence_count
        entry["affected_execution_count"] += g.affected_execution_count

    total_occurrences = sum(e["occurrence_count"] for e in totals.values())
    rows = sorted(totals.values(), key=lambda e: e["occurrence_count"], reverse=True)

    cumulative = 0
    for row in rows:
        row["percentage_of_failures"] = round(row["occurrence_count"] / total_occurrences * 100, 1) if total_occurrences else 0.0
        cumulative += row["occurrence_count"]
        row["cumulative_percentage"] = round(cumulative / total_occurrences * 100, 1) if total_occurrences else 0.0
    return rows


# --- performance anomalies ----------------------------------------------------


def compute_performance_anomalies(df: pd.DataFrame) -> dict:
    """Slow-but-SUCCESSFUL executions only — deliberately excludes failures,
    so a slow timeout that also errored is never double-counted as both an
    exception AND a performance anomaly (src.anomalies.detect_slow_executions
    does not make this distinction; this is the precise, success-only
    variant the console's Performance Anomalies KPI needs).
    """
    successes = df[df["status"] == "success"]

    slow_count = 0
    for _workflow, group in successes.groupby("workflow_name"):
        durations = group["duration_seconds"].dropna()
        if len(durations) < SLOW_EXECUTION_MIN_SAMPLES:
            continue
        mean, std = durations.mean(), durations.std(ddof=0)
        if std <= 0:
            continue
        threshold = mean + SLOW_EXECUTION_STD_MULTIPLIER * std
        slow_count += int((durations > threshold).sum())

    return {
        "slow_successful_count": slow_count,
        "p95_duration_seconds": _p95(successes["duration_seconds"]),
        "latency_degradation": _compute_latency_degradation(successes),
    }


def _compute_latency_degradation(successes: pd.DataFrame) -> dict:
    """Whether P95 duration among SUCCESSFUL executions rose meaningfully
    between the first and second half of the upload — i.e. latency
    degrading independently of any failure. Same insufficient-sample guard
    as compute_period_comparison; never claims degradation on a small
    sample."""
    if successes.empty:
        return {"valid": False, "reason": "no successful executions"}

    start, end = successes["start_time"].min(), successes["start_time"].max()
    if end <= start:
        return {"valid": False, "reason": "all executions occurred at the same time"}

    midpoint = start + (end - start) / 2
    first_half = successes[successes["start_time"] < midpoint]
    second_half = successes[successes["start_time"] >= midpoint]
    if len(first_half) < MIN_RECORDS_PER_PERIOD or len(second_half) < MIN_RECORDS_PER_PERIOD:
        return {"valid": False, "reason": "insufficient successful executions in one or both periods"}

    first_p95, second_p95 = _p95(first_half["duration_seconds"]), _p95(second_half["duration_seconds"])
    change_pct = round((second_p95 - first_p95) / first_p95 * 100, 1) if first_p95 else 0.0
    return {
        "valid": True,
        "first_period_p95": first_p95,
        "second_period_p95": second_p95,
        "change_pct": change_pct,
    }


def compute_group_pareto(groups: list) -> list:
    """Per-exception-GROUP (not per-category) occurrence counts with a
    cumulative percentage — deliberately finer-grained than
    compute_category_summary so the Pareto chart tells a different story
    than the category-breakdown chart (which exception PATTERNS, not
    categories, account for most failures)."""
    if not groups:
        return []

    total = sum(g.occurrence_count for g in groups)
    ranked = sorted(groups, key=lambda g: g.occurrence_count, reverse=True)

    cumulative = 0
    rows = []
    for g in ranked:
        cumulative += g.occurrence_count
        rows.append(
            {
                "title": human_title(g),
                "occurrence_count": g.occurrence_count,
                "percentage_of_failures": round(g.occurrence_count / total * 100, 1) if total else 0.0,
                "cumulative_percentage": round(cumulative / total * 100, 1) if total else 0.0,
            }
        )
    return rows


# --- retry outcomes ----------------------------------------------------------

RETRY_OUTCOME_LABELS = [
    "Successful without retry",
    "Recovered after retry",
    "Failed without retry",
    "Failed after retry",
    "Duplicate created after retry",
    "Retry information unavailable",
]


def compute_retry_outcomes(df: pd.DataFrame) -> dict:
    """Classifies every execution into a retry-outcome bucket using only
    that row's own status + retry_count (+ http_status for the duplicate
    case) — never a fabricated cross-row correlation. retry_count is the
    number of retries that execution's own flow needed before reaching its
    final status, so a successful row with retry_count > 0 genuinely means
    'recovered after retry'; a NaN retry_count means the source log simply
    didn't report retry behavior for that row.
    """
    counts = {label: 0 for label in RETRY_OUTCOME_LABELS}
    if df.empty:
        return counts

    retry_count = pd.to_numeric(df["retry_count"], errors="coerce")
    is_success = df["status"] == "success"
    is_failure = df["status"] == "failure"
    is_unavailable = retry_count.isna()
    has_retried = ~is_unavailable & (retry_count > 0)
    no_retry = ~is_unavailable & (retry_count == 0)
    http_status = pd.to_numeric(df["http_status"], errors="coerce")

    counts["Retry information unavailable"] = int(is_unavailable.sum())
    counts["Successful without retry"] = int((is_success & no_retry).sum())
    counts["Recovered after retry"] = int((is_success & has_retried).sum())
    counts["Failed without retry"] = int((is_failure & no_retry).sum())
    counts["Duplicate created after retry"] = int((is_failure & has_retried & (http_status == 409)).sum())
    counts["Failed after retry"] = int((is_failure & has_retried & (http_status != 409)).sum())
    return counts
