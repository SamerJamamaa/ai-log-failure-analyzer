import pandas as pd

from scripts.generate_sample_data import WORKFLOW_SPECS, generate
from src.ingestion import load_and_validate
from src.models import REQUIRED_COLUMNS

EXPECTED_WORKFLOWS = {
    "WhatsApp Lead Intake",
    "Invoice Parser",
    "CRM Synchronization",
    "Customer Support Classifier",
}

REQUIRED_ERROR_TYPES = {
    "APITimeoutError",
    "AuthenticationError",
    "InvalidJSONSchemaError",
    "DuplicateRecordError",
}

FAILURE_RATE_INCREASE_MIN_PP = 10.0


def test_generate_produces_exactly_200_records_across_required_workflows():
    rows = generate()

    assert len(rows) == 200
    assert {r["workflow_name"] for r in rows} == EXPECTED_WORKFLOWS


def test_all_required_columns_exist():
    rows = generate()

    for row in rows:
        assert set(row.keys()) == set(REQUIRED_COLUMNS)


def test_execution_ids_are_unique():
    rows = generate()
    ids = [r["execution_id"] for r in rows]

    assert len(ids) == len(set(ids))


def test_successful_executions_have_empty_error_fields():
    rows = generate()
    successes = [r for r in rows if r["status"] == "success"]

    assert successes  # sanity: there are successful executions at all
    for row in successes:
        assert row["error_type"] == ""
        assert row["error_message"] == ""


def test_failed_executions_contain_error_information():
    rows = generate()
    failures = [r for r in rows if r["status"] == "failure"]

    assert failures  # sanity: there are failed executions at all
    for row in failures:
        assert row["error_type"] != ""
        assert row["error_message"] != ""


def test_generate_includes_every_required_error_category_recurring():
    """Each required error type must appear >= 3 times somewhere, so it's
    detectable as a recurring failure once the analytics stage is built."""
    rows = generate()
    df = pd.DataFrame(rows)
    failures = df[df["status"] == "failure"]

    assert set(failures["error_type"]) == REQUIRED_ERROR_TYPES
    counts = failures["error_type"].value_counts()
    for error_type in REQUIRED_ERROR_TYPES:
        assert counts[error_type] >= 3


def test_one_workflow_shows_a_deliberate_second_half_failure_rate_increase():
    """Verifies the actual data pattern (not just the generator's config
    flag): sorted by start_time, the flagged workflow's second half must
    have a failure rate at least FAILURE_RATE_INCREASE_MIN_PP points higher
    than its first half, with the other workflows staying well under that."""
    degrading_workflows = [spec["name"] for spec in WORKFLOW_SPECS if spec["degrade_second_half"]]
    assert len(degrading_workflows) == 1
    degrading_workflow = degrading_workflows[0]

    df = pd.DataFrame(generate())
    increases = {}
    for workflow, group in df.groupby("workflow_name"):
        group = group.sort_values("start_time")
        midpoint = len(group) // 2
        first_half, second_half = group.iloc[:midpoint], group.iloc[midpoint:]
        first_rate = (first_half["status"] == "failure").mean() * 100
        second_rate = (second_half["status"] == "failure").mean() * 100
        increases[workflow] = second_rate - first_rate

    assert increases[degrading_workflow] >= FAILURE_RATE_INCREASE_MIN_PP
    for workflow, increase in increases.items():
        if workflow != degrading_workflow:
            assert increase < FAILURE_RATE_INCREASE_MIN_PP


def test_generate_output_passes_ingestion_validation_with_zero_rejections():
    import io

    df = pd.DataFrame(generate())
    csv_bytes = df.to_csv(index=False)

    valid_df, report = load_and_validate(io.StringIO(csv_bytes))

    assert report.ok
    assert report.rejected_rows == 0
    assert len(valid_df) == len(df)


def test_generate_is_deterministic_across_runs():
    """Composition (which record fails, with what error, what duration, in
    what order) is pinned by RNG_SEED and must be reproducible. start_time
    is deliberately anchored to datetime.now() so the sample data always
    looks recent, so it's excluded from this comparison by design."""
    first = [{k: v for k, v in row.items() if k != "start_time"} for row in generate()]
    second = [{k: v for k, v in row.items() if k != "start_time"} for row in generate()]

    assert first == second
