import io
import json
import os

import pytest

from scripts.generate_manual_test_logs import FILES, _to_csv_dataframe, _to_json_record, compute_findings, generate_all
from src.exception_groups import build_exception_groups
from src.ingestion import load_and_validate
from src.prioritization import prioritize_groups

MANUAL_TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "manual_tests")
EXPECTED_FINDINGS_PATH = os.path.join(MANUAL_TESTS_DIR, "expected_findings.json")

SCENARIO_NAMES = [name for name, _ in FILES]

EXPECTED_RECORD_COUNTS = {
    "01_healthy_baseline": 78,
    "02_whatsapp_provider_timeouts": 80,
    "03_crm_authentication_regression": 40,
    "04_invoice_schema_failures": 70,
    "05_crm_duplicate_records": 36,
    "06_slow_successful_executions": 60,
    "07_mixed_production_incident": 132,
}


@pytest.fixture(scope="module")
def generated() -> dict:
    return generate_all()


@pytest.fixture(scope="module")
def json_records(generated) -> dict:
    return {name: [_to_json_record(r) for r in rows] for name, rows in generated.items()}


def test_generator_produces_all_seven_files(generated):
    assert set(generated.keys()) == set(SCENARIO_NAMES)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_record_count_matches_spec(generated, name):
    assert len(generated[name]) == EXPECTED_RECORD_COUNTS[name]


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_execution_ids_are_unique_within_file(generated, name):
    ids = [r["execution_id"] for r in generated[name]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_status_values_are_only_success_or_failure(generated, name):
    assert {r["status"] for r in generated[name]} <= {"success", "failure"}


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_success_and_failure_rows_have_consistent_error_fields(generated, name):
    rows = generated[name]
    for r in rows:
        if r["status"] == "success":
            assert r["error_type"] == "" and r["error_message"] == ""
        else:
            assert r["error_type"] != "" and r["error_message"] != ""


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_durations_are_always_positive(generated, name):
    assert all(r["duration_seconds"] > 0 for r in generated[name])


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_records_are_sorted_chronologically(generated, name):
    timestamps = [r["start_time"] for r in generated[name]]
    assert timestamps == sorted(timestamps)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_span_covers_up_to_seven_days(generated, name):
    timestamps = [r["start_time"] for r in generated[name]]
    span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
    assert 0 <= span_days <= 7


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_json_form_passes_validation_with_zero_rejections(json_records, name):
    text = json.dumps(json_records[name])
    valid_df, report = load_and_validate(io.StringIO(text), filename=f"{name}.json")

    assert report.ok
    assert report.rejected_rows == 0
    assert report.valid_rows == len(json_records[name])


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_csv_form_passes_validation_with_zero_rejections(generated, name):
    csv_text = _to_csv_dataframe(generated[name]).to_csv(index=False)
    valid_df, report = load_and_validate(io.StringIO(csv_text), filename=f"{name}.csv")

    assert report.ok
    assert report.rejected_rows == 0
    assert report.valid_rows == len(generated[name])


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_json_and_csv_forms_produce_identical_exception_groups(generated, json_records, name):
    """The same scenario, uploaded as JSON vs CSV, must lead to the exact
    same exception groups — proves format independence on real generated
    scenario data, not just a hand-built fixture."""
    csv_text = _to_csv_dataframe(generated[name]).to_csv(index=False)
    csv_df, _ = load_and_validate(io.StringIO(csv_text), filename=f"{name}.csv")
    json_df, _ = load_and_validate(io.StringIO(json.dumps(json_records[name])), filename=f"{name}.json")

    csv_groups = prioritize_groups(build_exception_groups(csv_df))
    json_groups = prioritize_groups(build_exception_groups(json_df))

    assert csv_groups == json_groups


def test_expected_findings_file_matches_freshly_computed_results(json_records):
    """expected_findings.json must exist and reflect the analyzer's ACTUAL
    output for the current generator, not a stale or hand-typed snapshot.
    Regenerate it (scripts/generate_manual_test_logs.py) if this fails
    after a legitimate change to the generator or the detection engine."""
    assert os.path.exists(EXPECTED_FINDINGS_PATH), "run scripts/generate_manual_test_logs.py to create it"
    with open(EXPECTED_FINDINGS_PATH) as f:
        committed = json.load(f)

    for name, records in json_records.items():
        fresh = compute_findings(name, records)
        assert fresh == committed[f"{name}.json"], f"expected_findings.json is stale for {name}"


def test_generator_is_deterministic_across_runs():
    """Composition (which record fails, with what error, what duration, in
    what order) is pinned by RNG_SEED and must be reproducible. start_time
    is anchored to datetime.now(), so it's excluded from this comparison."""
    first = generate_all()
    second = generate_all()

    for name in first:
        first_no_ts = [{k: v for k, v in r.items() if k != "start_time"} for r in first[name]]
        second_no_ts = [{k: v for k, v in r.items() if k != "start_time"} for r in second[name]]
        assert first_no_ts == second_no_ts


# --- per-scenario intent checks: each scenario's exception groups must show ---
# --- exactly the expected story when run through the real, unmodified engine --


def _groups_for(json_records, name):
    df, _ = load_and_validate(io.StringIO(json.dumps(json_records[name])), filename=f"{name}.json")
    return prioritize_groups(build_exception_groups(df))


def test_healthy_baseline_has_no_high_priority_groups(json_records):
    """01: isolated one-off failures only — nothing should ever read as an
    actionable investigation on a healthy file."""
    groups = _groups_for(json_records, "01_healthy_baseline")

    assert all(g.trend_label == "isolated" for g in groups)
    assert all(g.priority_label == "Low" for g in groups)


def test_whatsapp_provider_timeouts_groups_as_one_timeout_pattern(json_records):
    """02: all gateway-timeout failures collapse into a single fingerprint,
    classified under Timeout and latency, with correlation IDs available as
    evidence for tracing."""
    from src.evidence import build_evidence_table

    df, _ = load_and_validate(
        io.StringIO(json.dumps(json_records["02_whatsapp_provider_timeouts"])),
        filename="02_whatsapp_provider_timeouts.json",
    )
    groups = prioritize_groups(build_exception_groups(df))

    assert len(groups) == 1
    group = groups[0]
    assert group.primary_category == "Timeout and latency"
    assert group.occurrence_count == 14
    assert group.priority_label in ("High", "Critical")

    items = build_evidence_table(group, df)
    assert any(item.representative_correlation_ids for item in items)


def test_crm_authentication_regression_is_correlated_with_new_version_only(json_records):
    """03: the authentication-failure group must be entirely on version
    3.5.0, with zero of that error type on the prior version — the
    deterministic evidence a 'correlation, not confirmed causation' framing
    is built on."""
    groups = _groups_for(json_records, "03_crm_authentication_regression")

    assert len(groups) == 1
    group = groups[0]
    assert group.primary_category == "Authentication and authorization"
    assert group.application_versions == ["3.5.0"]


def test_invoice_schema_failures_are_split_into_separate_subtypes_not_performance(json_records):
    """04: three distinct schema-validation subtypes must stay separate
    (different fingerprints) and none may be classified as a timeout or
    performance issue."""
    groups = _groups_for(json_records, "04_invoice_schema_failures")

    assert len(groups) == 3
    error_types = {g.error_type for g in groups}
    assert error_types == {"MISSING_REQUIRED_FIELD", "INVALID_JSON_SCHEMA", "TYPE_MISMATCH"}
    assert all(g.primary_category not in ("Timeout and latency", "Performance anomaly without failure") for g in groups)


def test_crm_duplicate_records_shows_retry_correlation(json_records):
    """05: every 409 conflict was already retried once and still failed —
    the evidence must reflect 'retry didn't help,' pointing toward
    idempotency rather than blind retrying."""
    groups = _groups_for(json_records, "05_crm_duplicate_records")

    assert len(groups) == 1
    group = groups[0]
    assert group.primary_category == "Data integrity and persistence"
    assert group.retry_info.get("executions_with_retries") == group.occurrence_count
    assert group.retry_info.get("all_retries_still_failed") is True


def test_slow_successful_executions_produce_zero_exception_groups(json_records):
    """06: 100% success rate — must never appear as an exception group,
    regardless of how slow individual executions were."""
    groups = _groups_for(json_records, "06_slow_successful_executions")
    assert groups == []


def test_mixed_production_incident_keeps_issues_separate_and_prioritizes_by_impact(json_records):
    """07: the WhatsApp timeout, CRM auth regression, and three invoice
    schema subtypes must remain five distinct groups with no shared category
    or fingerprint, and the highest-volume issue (WhatsApp) must rank first."""
    groups = _groups_for(json_records, "07_mixed_production_incident")

    assert len(groups) == 5
    assert len({g.fingerprint for g in groups}) == 5
    assert groups[0].workflow_name == "WhatsApp Lead Intake"
    assert groups[0].priority_label == "Critical"

    categories = {g.primary_category for g in groups}
    assert categories == {
        "Timeout and latency",
        "Authentication and authorization",
        "Input and validation",
        "AI and structured-output",
    }
