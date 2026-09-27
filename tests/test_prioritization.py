from src.models import ExceptionGroup
from src.prioritization import build_priority_reason, compute_priority, prioritize_groups


def _group(**overrides) -> ExceptionGroup:
    base = dict(
        fingerprint="fp1",
        primary_category="Timeout and latency",
        secondary_category="Gateway timeout",
        error_type="API_TIMEOUT",
        error_code="",
        workflow_name="WF",
        workflow_step="",
        service="",
        endpoint="",
        http_method="",
        http_status=None,
        provider="",
        environment="",
        representative_message="timed out",
        occurrence_count=5,
        affected_execution_count=5,
        first_occurrence="2026-01-01",
        last_occurrence="2026-01-02",
        percentage_of_all_failures=10.0,
        percentage_of_workflow_executions=10.0,
        trend_label="recurring",
        is_increasing=False,
        avg_duration_seconds=5.0,
        p95_duration_seconds=6.0,
    )
    base.update(overrides)
    return ExceptionGroup(**base)


def test_isolated_low_impact_group_scores_low():
    group = _group(trend_label="isolated", affected_execution_count=1, percentage_of_workflow_executions=1.0)
    score, label, factors = compute_priority(group, max_affected_executions=20)
    assert label == "Low"


def test_increasing_high_volume_group_scores_critical():
    group = _group(
        trend_label="increasing",
        is_increasing=True,
        affected_execution_count=20,
        percentage_of_workflow_executions=50.0,
        http_status=500,
        retry_info={"executions_with_retries": 5, "all_retries_still_failed": True},
    )
    score, label, factors = compute_priority(group, max_affected_executions=20)
    assert label == "Critical"
    assert score >= 70.0


def test_factors_are_all_returned_and_non_negative():
    group = _group()
    score, label, factors = compute_priority(group, max_affected_executions=10)
    assert set(factors.keys()) == {
        "affected_executions",
        "trend",
        "failure_percentage",
        "duration_impact",
        "retry_failure",
        "http_severity",
        "cross_workflow_spread",
    }
    assert all(v >= 0 for v in factors.values())
    assert score == round(sum(factors.values()), 1)


def test_5xx_scores_higher_http_severity_than_4xx():
    group_5xx = _group(http_status=503)
    group_4xx = _group(http_status=404)
    _, _, factors_5xx = compute_priority(group_5xx, max_affected_executions=10)
    _, _, factors_4xx = compute_priority(group_4xx, max_affected_executions=10)
    assert factors_5xx["http_severity"] > factors_4xx["http_severity"]


def test_retry_failure_scores_higher_when_retries_still_failed():
    retried_and_failed = _group(retry_info={"executions_with_retries": 3, "all_retries_still_failed": True})
    retried_some_succeeded = _group(retry_info={"executions_with_retries": 3, "all_retries_still_failed": False})
    no_retries = _group(retry_info={})

    _, _, f1 = compute_priority(retried_and_failed, max_affected_executions=10)
    _, _, f2 = compute_priority(retried_some_succeeded, max_affected_executions=10)
    _, _, f3 = compute_priority(no_retries, max_affected_executions=10)

    assert f1["retry_failure"] > f2["retry_failure"] > f3["retry_failure"]


def test_short_duration_contributes_no_duration_impact_points():
    group = _group(avg_duration_seconds=2.0)
    _, _, factors = compute_priority(group, max_affected_executions=10)
    assert factors["duration_impact"] == 0.0


def test_prioritize_groups_sorts_highest_first_and_fills_in_fields():
    low = _group(fingerprint="low", trend_label="isolated", affected_execution_count=1, percentage_of_workflow_executions=1.0)
    high = _group(
        fingerprint="high",
        trend_label="increasing",
        is_increasing=True,
        affected_execution_count=20,
        percentage_of_workflow_executions=60.0,
    )

    ranked = prioritize_groups([low, high])

    assert ranked[0].fingerprint == "high"
    assert ranked[0].priority_score >= ranked[1].priority_score
    assert ranked[0].priority_label in {"Critical", "High", "Medium", "Low"}
    assert ranked[0].priority_factors  # populated, not left empty


def test_prioritize_groups_handles_empty_list():
    assert prioritize_groups([]) == []


def test_isolated_group_never_exceeds_low_regardless_of_other_factors():
    """Regression test for a real bug caught via manual verification: a
    single-occurrence blip in an otherwise-quiet dataset was scoring
    'Medium' purely because it was proportionally the largest thing present
    — even with severe-looking other factors, an isolated one-off must stay
    capped as Low."""
    group = _group(
        trend_label="isolated",
        affected_execution_count=1,
        percentage_of_workflow_executions=100.0,
        http_status=500,
        avg_duration_seconds=60.0,
        retry_info={"executions_with_retries": 1, "all_retries_still_failed": True},
    )
    score, label, factors = compute_priority(group, max_affected_executions=1)
    assert label == "Low"


def test_affected_executions_factor_has_an_absolute_floor_not_just_relative_scaling():
    """Regression test: when every group in the upload is tiny (max=1), a
    1-occurrence group must not score as if it were proportionally 'the
    worst thing here' (25/25 points) — the scale has a minimum floor."""
    group = _group(trend_label="recurring", affected_execution_count=1)
    _, _, factors = compute_priority(group, max_affected_executions=1)
    assert factors["affected_executions"] < 25.0


def test_cross_workflow_spread_increases_score():
    isolated_to_one_workflow = _group(workflows_sharing_this_issue=1)
    spread_across_three = _group(workflows_sharing_this_issue=3)

    _, _, f1 = compute_priority(isolated_to_one_workflow, max_affected_executions=10)
    _, _, f2 = compute_priority(spread_across_three, max_affected_executions=10)

    assert f2["cross_workflow_spread"] > f1["cross_workflow_spread"]


def test_priority_reason_is_a_readable_sentence_not_a_factor_dump():
    group = _group(
        affected_execution_count=8,
        percentage_of_all_failures=62.0,
        trend_label="recurring",
        application_versions=["3.5.0"],
    )
    reason = build_priority_reason(group, "High")

    assert reason.startswith("High priority because")
    assert "8 execution" in reason
    assert "62.0%" in reason
    assert "recurring" in reason
    assert "3.5.0" in reason
    assert ":" not in reason  # never a raw "key: value" factor dump


def test_priority_reason_omits_clauses_that_do_not_apply():
    group = _group(trend_label="isolated", application_versions=[], workflows_sharing_this_issue=1, http_status=None, retry_info={})
    reason = build_priority_reason(group, "Low")

    assert "isolated" not in reason  # isolated trend clause is skipped, not stated as a factor
    assert "version" not in reason
    assert "workflows" not in reason


def test_prioritize_groups_fills_in_priority_reason():
    group = _group(fingerprint="only")
    ranked = prioritize_groups([group])
    assert ranked[0].priority_reason
    assert ranked[0].priority_reason.startswith(ranked[0].priority_label)
