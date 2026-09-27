"""Deterministic exception-group priority scoring — never LLM-assigned.

A simple, explainable points system: each factor contributes a bounded
number of points (max 100 total); the total maps to Critical/High/Medium/Low
via fixed thresholds. Every contributing factor is returned alongside the
score so the dashboard can show exactly why a group was ranked where it is.
"""

PRIORITY_THRESHOLDS = [
    (70.0, "Critical"),
    (45.0, "High"),
    (20.0, "Medium"),
    (0.0, "Low"),
]

# Points awarded per trend label — an increasing pattern is weighted as
# heavily as the affected-executions factor, since an actively-worsening
# issue is the clearest signal something needs attention now.
_TREND_POINTS = {"isolated": 0.0, "recurring": 10.0, "increasing": 25.0}

# avg_duration_seconds below this is never treated as a duration-impact
# signal on its own (a fast workflow's "slow" execution may still be quick
# in absolute terms).
_DURATION_IMPACT_FLOOR_SECONDS = 10.0
_DURATION_IMPACT_SCALE_SECONDS = 30.0

# floor for the affected-executions scale denominator — see compute_priority
_MIN_SCALE = 10

# a one-off, by definition, is not yet a pattern — capped just under the
# Medium threshold regardless of other factors, so an isolated blip never
# reads as more urgent than a real recurring or increasing issue
_ISOLATED_SCORE_CAP = 15.0


def _label_for_score(score: float) -> str:
    for threshold, label in PRIORITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low"


def compute_priority(group, max_affected_executions: int) -> tuple:
    """Returns (score, label, factors) for one ExceptionGroup.
    `max_affected_executions` is the largest affected_execution_count among
    ALL groups in this upload, scaling this factor relative to the dataset
    — but never below _MIN_SCALE, so a lone 1-occurrence blip in an
    otherwise-healthy file doesn't score as "proportionally the worst thing
    here" just because nothing else in the file is worse either.
    """
    factors = {}

    scale = max(max_affected_executions, _MIN_SCALE)
    factors["affected_executions"] = round(25 * min(group.affected_execution_count / scale, 1.0), 1)

    factors["trend"] = _TREND_POINTS.get(group.trend_label, 0.0)

    factors["failure_percentage"] = round(min(group.percentage_of_workflow_executions / 50 * 15, 15), 1)

    if group.avg_duration_seconds > _DURATION_IMPACT_FLOOR_SECONDS:
        factors["duration_impact"] = round(min(group.avg_duration_seconds / _DURATION_IMPACT_SCALE_SECONDS * 10, 10), 1)
    else:
        factors["duration_impact"] = 0.0

    retry_points = 0.0
    if group.retry_info.get("executions_with_retries"):
        retry_points = 10.0 if group.retry_info.get("all_retries_still_failed") else 5.0
    factors["retry_failure"] = retry_points

    http_points = 0.0
    if group.http_status is not None:
        http_points = 10.0 if group.http_status >= 500 else (5.0 if group.http_status >= 400 else 0.0)
    factors["http_severity"] = http_points

    factors["cross_workflow_spread"] = round(min((group.workflows_sharing_this_issue - 1) * 5, 5), 1)

    score = round(sum(factors.values()), 1)
    if group.trend_label == "isolated":
        score = min(score, _ISOLATED_SCORE_CAP)
    return score, _label_for_score(score), factors


def build_priority_reason(group, label: str) -> str:
    """A plain-language sentence justifying the priority label, built only
    from already-computed group fields — never a raw factor-score dump.
    Clauses are included only when they actually apply to this group."""
    clauses = [f"affects {group.affected_execution_count} execution(s)"]
    clauses.append(f"represents {group.percentage_of_all_failures}% of all failures")
    if group.trend_label != "isolated":
        clauses.append(f"is {group.trend_label}")
    if len(group.application_versions) == 1:
        clauses.append(f"appears on version {group.application_versions[0]}")
    if group.workflows_sharing_this_issue > 1:
        clauses.append(f"spans {group.workflows_sharing_this_issue} workflows")
    if group.http_status is not None and group.http_status >= 500:
        clauses.append(f"returns a server error (HTTP {group.http_status})")
    if group.retry_info.get("all_retries_still_failed"):
        clauses.append("retries did not recover it")

    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return f"{label} priority because this exception {body}."


def prioritize_groups(groups: list) -> list:
    """Scores every group (filling in its priority_score/priority_label/
    priority_factors/priority_reason fields) and returns them sorted
    highest-priority first.
    """
    if not groups:
        return []
    max_affected = max(g.affected_execution_count for g in groups)
    for group in groups:
        score, label, factors = compute_priority(group, max_affected)
        group.priority_score = score
        group.priority_label = label
        group.priority_factors = factors
        group.priority_reason = build_priority_reason(group, label)
    return sorted(groups, key=lambda g: g.priority_score, reverse=True)
