import pandas as pd

from src.models import Anomaly

# --- Configurable detection thresholds -------------------------------------
# Centralized here so the detection rules below never hardcode a magic
# number inline; tune sensitivity by changing these constants only.

# Slow execution: duration_seconds > per-workflow mean + N * std.
SLOW_EXECUTION_STD_MULTIPLIER = 2.0
SLOW_EXECUTION_MIN_SAMPLES = 5  # minimum samples per workflow to trust mean/std

# Recurring failure: same (workflow, error_type) appearing >= N times.
RECURRING_FAILURE_MIN_COUNT = 3

# Failure-rate increase: second chronological half's failure rate is >= N
# percentage points higher than the first half's, with >= M records in each half.
FAILURE_RATE_INCREASE_THRESHOLD_PP = 10.0
FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF = 10
# -----------------------------------------------------------------------------


def detect_slow_executions(df: pd.DataFrame) -> list[Anomaly]:
    """Per-workflow outliers: duration_seconds > mean + SLOW_EXECUTION_STD_MULTIPLIER * std.

    Computed per workflow (not globally) since normal duration varies by workflow.
    """
    anomalies = []
    for workflow, group in df.groupby("workflow_name"):
        durations = group["duration_seconds"].dropna()
        if len(durations) < SLOW_EXECUTION_MIN_SAMPLES:
            continue
        mean, std = durations.mean(), durations.std(ddof=0)
        if std <= 0:
            continue
        threshold = mean + SLOW_EXECUTION_STD_MULTIPLIER * std
        outliers = group[group["duration_seconds"] > threshold]
        if outliers.empty:
            continue
        anomalies.append(
            Anomaly(
                type="slow_execution",
                workflow_name=workflow,
                summary=(
                    f"{len(outliers)} execution(s) took longer than {threshold:.1f}s "
                    f"(mean {mean:.1f}s + {SLOW_EXECUTION_STD_MULTIPLIER:.0f} std for this "
                    f"workflow), up to {outliers['duration_seconds'].max():.1f}s."
                ),
                evidence={
                    "mean_duration_seconds": round(float(mean), 1),
                    "std_duration_seconds": round(float(std), 1),
                    "threshold_seconds": round(float(threshold), 1),
                    "outlier_count": int(len(outliers)),
                    "max_duration_seconds": round(float(outliers["duration_seconds"].max()), 1),
                    "example_execution_ids": outliers["execution_id"].head(5).tolist(),
                },
            )
        )
    return anomalies


def detect_recurring_errors(df: pd.DataFrame) -> list[Anomaly]:
    """Same (workflow, error_type) appearing >= RECURRING_FAILURE_MIN_COUNT times."""
    anomalies = []
    failures = df[(df["status"] == "failure") & (df["error_type"] != "")]
    grouped = failures.groupby(["workflow_name", "error_type"]).size().reset_index(name="count")
    for _, row in grouped[grouped["count"] >= RECURRING_FAILURE_MIN_COUNT].iterrows():
        anomalies.append(
            Anomaly(
                type="recurring_error",
                workflow_name=row["workflow_name"],
                summary=(
                    f"Error type '{row['error_type']}' occurred {int(row['count'])} times "
                    f"in workflow '{row['workflow_name']}'."
                ),
                evidence={
                    "error_type": row["error_type"],
                    "occurrence_count": int(row["count"]),
                },
            )
        )
    return anomalies


def detect_failure_rate_increase(df: pd.DataFrame) -> list[Anomaly]:
    """Splits each workflow's records (sorted by start_time) into a first and
    second chronological half, and flags a workflow where the second half's
    failure rate is >= FAILURE_RATE_INCREASE_THRESHOLD_PP percentage points
    higher than the first half's, provided both halves have at least
    FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF records.
    """
    anomalies = []
    for workflow, group in df.groupby("workflow_name"):
        group = group.sort_values("start_time")
        midpoint = len(group) // 2
        first_half, second_half = group.iloc[:midpoint], group.iloc[midpoint:]
        if (
            len(first_half) < FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF
            or len(second_half) < FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF
        ):
            continue
        first_failure_rate = (first_half["status"] == "failure").mean() * 100
        second_failure_rate = (second_half["status"] == "failure").mean() * 100
        increase = second_failure_rate - first_failure_rate
        if increase >= FAILURE_RATE_INCREASE_THRESHOLD_PP:
            second_half_failures = second_half[second_half["status"] == "failure"]
            error_counts = second_half_failures["error_type"].value_counts()
            dominant_error_type = error_counts.index[0] if not error_counts.empty else None
            anomalies.append(
                Anomaly(
                    type="failure_rate_increase",
                    workflow_name=workflow,
                    summary=(
                        f"Failure rate for '{workflow}' rose from {first_failure_rate:.1f}% "
                        f"(first half, n={len(first_half)}) to {second_failure_rate:.1f}% "
                        f"(second half, n={len(second_half)})."
                    ),
                    evidence={
                        "first_half_failure_rate": round(float(first_failure_rate), 1),
                        "second_half_failure_rate": round(float(second_failure_rate), 1),
                        "first_half_n": int(len(first_half)),
                        "second_half_n": int(len(second_half)),
                        "increase_percentage_points": round(float(increase), 1),
                        "dominant_error_type": dominant_error_type,
                        "affected_execution_count": int(len(second_half_failures)),
                    },
                )
            )
    return anomalies


def detect_all(df: pd.DataFrame) -> list[Anomaly]:
    return (
        detect_failure_rate_increase(df)
        + detect_recurring_errors(df)
        + detect_slow_executions(df)
    )


def select_primary_issue(anomalies: list[Anomaly]) -> Anomaly | None:
    """Picks the single most important anomaly to lead the dashboard with,
    using a fixed priority order: an ongoing reliability trend
    (failure_rate_increase) outranks a recurring error, which outranks a
    duration outlier — the same relative severity src/llm/offline.py already
    uses. Within one type, picks the most severe by that type's own
    evidence. Returns None when there's nothing to report (a healthy system).
    """
    increases = [a for a in anomalies if a.type == "failure_rate_increase"]
    if increases:
        return max(increases, key=lambda a: a.evidence["increase_percentage_points"])

    recurring = [a for a in anomalies if a.type == "recurring_error"]
    if recurring:
        return max(recurring, key=lambda a: a.evidence["occurrence_count"])

    slow = [a for a in anomalies if a.type == "slow_execution"]
    if slow:
        return max(slow, key=lambda a: a.evidence["outlier_count"])

    return None


def assess_system_status(anomalies: list[Anomaly]) -> str:
    """Deterministic, evidence-only status — never LLM-derived. "Critical"
    means at least one workflow is actively trending worse right now;
    "Attention Required" means real issues exist but none are an active
    trend; "Healthy" means no anomalies were detected at all."""
    if any(a.type == "failure_rate_increase" for a in anomalies):
        return "Critical"
    if anomalies:
        return "Attention Required"
    return "Healthy"
