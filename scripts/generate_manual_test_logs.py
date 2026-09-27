"""Generates deterministic JSON (+ CSV) manual test logs for the exception-
investigation dashboard — one scenario per file, each engineered to exercise
a specific part of the deterministic engine (fingerprinting, taxonomy,
trend/priority scoring, evidence, offline playbooks) with REALISTIC optional
fields (service, endpoint, http_status, correlation_id, retry_count,
application_version, provider), not just the 7 required columns.

This replaces the earlier six-file, anomalies-only generator (which left
every optional field blank) now that the app is an exception-investigation
assistant, not a KPI/anomaly dashboard. JSON is the primary format (the
realistic ingestion format — nested `error`, top-level technical fields);
CSV is written alongside for quick manual edits, using the same flat
optional columns src/ingestion.py already accepts for CSV.

Regenerate everything with:
    python scripts/generate_manual_test_logs.py
Deterministic (fixed seed) except timestamps, which are anchored to "now" so
manually-uploaded files always look recent.
"""

import json
import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RNG_SEED = 20260926
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "manual_tests")
SPAN_DAYS = 7


# --- low-level row builders --------------------------------------------------


def _clipped_normal(rng: random.Random, mean: float, std: float, max_std: float = 1.0) -> float:
    value = rng.gauss(mean, std)
    value = max(mean - max_std * std, min(mean + max_std * std, value))
    return round(max(0.5, value), 2)


def _row(
    execution_id: str,
    workflow_name: str,
    start_time: datetime,
    status: str,
    duration_seconds: float,
    *,
    error_type: str = "",
    error_message: str = "",
    error_code: str = "",
    error_stack_summary: str = "",
    environment: str = "production",
    workflow_step: str = "",
    service: str = "",
    endpoint: str = "",
    http_method: str = "",
    http_status=None,
    correlation_id: str = "",
    retry_count=0,
    application_version: str = "1.0.0",
    provider: str = "",
) -> dict:
    return dict(
        execution_id=execution_id,
        workflow_name=workflow_name,
        start_time=start_time,
        status=status,
        duration_seconds=duration_seconds,
        error_type=error_type,
        error_message=error_message,
        error_code=error_code,
        error_stack_summary=error_stack_summary,
        environment=environment,
        workflow_step=workflow_step,
        service=service,
        endpoint=endpoint,
        http_method=http_method,
        http_status=http_status,
        correlation_id=correlation_id,
        retry_count=retry_count,
        application_version=application_version,
        provider=provider,
    )


def _interleave_evenly(base: list, inserts: list) -> list:
    if not inserts:
        return list(base)
    result = list(base)
    step = len(result) / (len(inserts) + 1)
    for offset, item in enumerate(inserts, start=1):
        position = min(len(result), round(step * offset))
        result.insert(position, item)
    return result


def _timestamps_in_day_range(n: int, days_ago_start: float, days_ago_end: float, seed: int) -> list:
    end = datetime.now()
    range_start = end - timedelta(days=days_ago_start)
    range_end = end - timedelta(days=days_ago_end)
    span_minutes = (range_end - range_start).total_seconds() / 60
    step = span_minutes / n
    rng = random.Random(RNG_SEED + seed)
    return [range_start + timedelta(minutes=i * step + rng.uniform(0, step * 0.8)) for i in range(n)]


def _finalize(rows: list) -> list:
    return sorted(rows, key=lambda r: r["start_time"])


# --- scenario 1: healthy baseline ---------------------------------------------


def build_healthy_baseline() -> list:
    """No investigation-worthy signal: near-perfect reliability across three
    workflows, a couple of isolated one-off failures (never recurring),
    stable single application versions."""
    rng = random.Random(RNG_SEED + 1)
    rows = []

    profiles = [
        ("WhatsApp Lead Intake", "send_message", "messaging-service", "whatsapp-business-api", "/v1/messages", "POST", 4.0, 1.0, "4.2.0"),
        ("CRM Synchronization", "sync_contact", "crm-sync-service", "salesforce-crm", "/contacts", "POST", 3.0, 0.8, "3.4.0"),
        ("Invoice Parser", "parse_ai_response", "invoice-extraction-service", "internal-llm", "/extract", "POST", 8.0, 2.0, "2.1.0"),
    ]
    for idx, (wf, step, service, provider, endpoint, method, mean, std, version) in enumerate(profiles):
        n = 25
        ts = _timestamps_in_day_range(n, SPAN_DAYS, 0, seed=10 + idx)
        for i in range(n):
            rows.append(
                _row(
                    f"healthy_{wf[:3].lower()}_{i:04d}", wf, ts[i], "success", _clipped_normal(rng, mean, std),
                    workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                    application_version=version, correlation_id=f"CORR-healthy-{idx}{i:04d}",
                )
            )
        # one isolated failure per workflow — never a second occurrence of the same fingerprint
        fail_ts = ts[n // 2]
        rows.append(
            _row(
                f"healthy_{wf[:3].lower()}_fail", wf, fail_ts, "failure", _clipped_normal(rng, mean, std),
                error_type="TRANSIENT_ERROR", error_message="A single transient error occurred.",
                workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                http_status=500, application_version=version, correlation_id=f"CORR-healthy-{idx}-fail", retry_count=0,
            )
        )
    return _finalize(rows)


# --- scenario 2: WhatsApp provider timeouts -----------------------------------


def build_whatsapp_provider_timeouts() -> list:
    """Clean first 5 days, then a cluster of gateway-timeout failures from the
    WhatsApp provider concentrated in the final 2 days — an increasing trend,
    single fingerprint, evidence should show correlation IDs and retries."""
    rng = random.Random(RNG_SEED + 2)
    wf, step, service, provider = "WhatsApp Lead Intake", "send_message", "messaging-service", "whatsapp-business-api"
    endpoint, method, version = "/v1/messages", "POST", "4.5.0"
    mean, std = 4.0, 1.0

    rows = []
    early_ts = _timestamps_in_day_range(40, SPAN_DAYS, 2, seed=20)
    for i in range(40):
        rows.append(
            _row(
                f"wa_timeout_early_{i:04d}", wf, early_ts[i], "success", _clipped_normal(rng, mean, std),
                workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                application_version=version, correlation_id=f"CORR-wa-early-{i:04d}",
            )
        )

    late_ts = _timestamps_in_day_range(40, 2, 0, seed=21)
    late_success = 26
    late_failures = 14
    late_records = _interleave_evenly(
        [{"kind": "success"} for _ in range(late_success)],
        [{"kind": "failure"} for _ in range(late_failures)],
    )
    fail_idx = 0
    for i, rec in enumerate(late_records):
        if rec["kind"] == "success":
            rows.append(
                _row(
                    f"wa_timeout_late_s{i:04d}", wf, late_ts[i], "success", _clipped_normal(rng, mean, std),
                    workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                    application_version=version, correlation_id=f"CORR-wa-late-{i:04d}",
                )
            )
        else:
            rows.append(
                _row(
                    f"wa_timeout_late_f{i:04d}", wf, late_ts[i], "failure", round(mean + 25 + rng.uniform(0, 3), 2),
                    error_type="GATEWAY_TIMEOUT", error_message="Gateway timeout returned after 30000ms.",
                    workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                    http_status=504, application_version=version, correlation_id=f"CORR-wa-late-f{fail_idx:04d}",
                    retry_count=1,
                )
            )
            fail_idx += 1
    return _finalize(rows)


# --- scenario 3: CRM authentication regression --------------------------------


def build_crm_auth_regression() -> list:
    """Version 3.4.0 is clean; after the version bumps to 3.5.0, authentication
    failures appear consistently — a version-correlated (not confirmed-caused)
    regression story."""
    rng = random.Random(RNG_SEED + 3)
    wf, step, service, provider = "CRM Synchronization", "authenticate", "crm-sync-service", "salesforce-crm"
    endpoint, method = "/oauth/token", "POST"
    mean, std = 3.0, 0.8

    rows = []
    v1_ts = _timestamps_in_day_range(20, SPAN_DAYS, 3, seed=30)
    for i in range(20):
        rows.append(
            _row(
                f"crm_auth_v1_{i:04d}", wf, v1_ts[i], "success", _clipped_normal(rng, mean, std),
                workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                application_version="3.4.0", correlation_id=f"CORR-crm-v1-{i:04d}",
            )
        )

    v2_success_ts = _timestamps_in_day_range(12, 3, 0, seed=31)
    for i in range(12):
        rows.append(
            _row(
                f"crm_auth_v2_s{i:04d}", wf, v2_success_ts[i], "success", _clipped_normal(rng, mean, std),
                workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                application_version="3.5.0", correlation_id=f"CORR-crm-v2-{i:04d}",
            )
        )

    v2_fail_ts = _timestamps_in_day_range(8, 3, 0, seed=32)
    for i in range(8):
        rows.append(
            _row(
                f"crm_auth_v2_f{i:04d}", wf, v2_fail_ts[i], "failure", _clipped_normal(rng, mean, std),
                error_type="AUTHENTICATION_ERROR", error_message="Authentication with the upstream service failed (401 Unauthorized).",
                workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                http_status=401, application_version="3.5.0", correlation_id=f"CORR-crm-v2-f{i:04d}", retry_count=0,
            )
        )
    return _finalize(rows)


# --- scenario 4: Invoice schema failures --------------------------------------


def build_invoice_schema_failures() -> list:
    """Three distinct schema-validation subtypes, each its own fingerprint —
    durations kept tight throughout so this cannot be misread as a
    performance incident."""
    rng = random.Random(RNG_SEED + 4)
    wf, step, service, provider = "Invoice Parser", "parse_ai_response", "invoice-extraction-service", "internal-llm"
    endpoint, method, version = "/extract", "POST", "2.3.0"
    mean, std = 8.0, 2.0

    subtypes = [
        ("MISSING_REQUIRED_FIELD", "Required field 'invoice_number' is missing from the response.", 5),
        ("INVALID_JSON_SCHEMA", "Response JSON has an unexpected nested structure for 'line_items'.", 4),
        ("TYPE_MISMATCH", "Field 'amount' was returned as text instead of a number.", 3),
    ]
    total_failures = sum(n for *_, n in subtypes)
    successes = 58

    rows = []
    success_ts = _timestamps_in_day_range(successes, SPAN_DAYS, 0, seed=40)
    for i in range(successes):
        rows.append(
            _row(
                f"invoice_schema_s{i:04d}", wf, success_ts[i], "success", _clipped_normal(rng, mean, std),
                workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                application_version=version, correlation_id=f"CORR-invoice-s{i:04d}",
            )
        )

    fail_ts = _timestamps_in_day_range(total_failures, SPAN_DAYS, 0, seed=41)
    idx = 0
    for error_type, message, count in subtypes:
        for _ in range(count):
            rows.append(
                _row(
                    f"invoice_schema_f{idx:04d}", wf, fail_ts[idx], "failure", _clipped_normal(rng, mean, std),
                    error_type=error_type, error_message=message,
                    workflow_step=step, service=service, provider=provider, endpoint=endpoint, http_method=method,
                    application_version=version, correlation_id=f"CORR-invoice-f{idx:04d}", retry_count=0,
                )
            )
            idx += 1
    return _finalize(rows)


# --- scenario 5: CRM duplicate records -----------------------------------------


def build_crm_duplicate_records() -> list:
    """Every failure is a 409 conflict that was already retried once and
    still failed — the evidence should read as 'retry made it worse,' not
    'needs more retries.'"""
    rng = random.Random(RNG_SEED + 5)
    wf, step, service = "CRM Synchronization", "create_contact", "crm-sync-service"
    endpoint, method, version = "/contacts", "POST", "3.5.0"
    mean, std = 3.0, 0.8

    rows = []
    success_ts = _timestamps_in_day_range(30, SPAN_DAYS, 0, seed=50)
    for i in range(30):
        rows.append(
            _row(
                f"crm_dup_s{i:04d}", wf, success_ts[i], "success", _clipped_normal(rng, mean, std),
                workflow_step=step, service=service, endpoint=endpoint, http_method=method,
                application_version=version, correlation_id=f"CORR-crmdup-s{i:04d}",
            )
        )

    fail_ts = _timestamps_in_day_range(6, SPAN_DAYS, 0, seed=51)
    for i in range(6):
        rows.append(
            _row(
                f"crm_dup_f{i:04d}", wf, fail_ts[i], "failure", _clipped_normal(rng, mean, std),
                error_type="DUPLICATE_RECORD", error_message="Record already exists; skipped to avoid duplicate processing.",
                workflow_step=step, service=service, endpoint=endpoint, http_method=method,
                http_status=409, application_version=version, correlation_id=f"CORR-crmdup-f{i:04d}", retry_count=1,
            )
        )
    return _finalize(rows)


# --- scenario 6: slow successful executions -------------------------------------


def build_slow_successful_executions() -> list:
    """100% success, but several executions are dramatically slower than
    their workflow's norm — must never appear as an exception group."""
    rng = random.Random(RNG_SEED + 6)
    profiles = [
        ("WhatsApp Lead Intake", "send_message", "messaging-service", 4.0, 1.0, "4.5.0"),
        ("CRM Synchronization", "sync_contact", "crm-sync-service", 3.0, 0.8, "3.5.0"),
        ("Invoice Parser", "parse_ai_response", "invoice-extraction-service", 8.0, 2.0, "2.3.0"),
    ]
    rows = []
    for idx, (wf, step, service, mean, std, version) in enumerate(profiles):
        n = 18
        ts = _timestamps_in_day_range(n + 2, SPAN_DAYS, 0, seed=60 + idx)
        for i in range(n):
            rows.append(
                _row(
                    f"slow_exec_{idx}_{i:04d}", wf, ts[i], "success", _clipped_normal(rng, mean, std),
                    workflow_step=step, service=service, application_version=version,
                    correlation_id=f"CORR-slow-{idx}-{i:04d}",
                )
            )
        for j in range(2):
            slow_duration = round(mean + 4.5 * std + rng.uniform(0, std), 2)
            rows.append(
                _row(
                    f"slow_exec_{idx}_slow{j}", wf, ts[n + j], "success", slow_duration,
                    workflow_step=step, service=service, application_version=version,
                    correlation_id=f"CORR-slow-{idx}-slow{j}",
                )
            )
    return _finalize(rows)


# --- scenario 7: mixed production incident -------------------------------------


def build_mixed_production_incident() -> list:
    """Combines a subset of the WhatsApp timeout, CRM auth-regression, and
    invoice schema scenarios in one file, plus a healthy workflow — each
    must remain a distinct, separately-prioritized exception group."""
    rows = []
    rows += [r for r in build_whatsapp_provider_timeouts() if r["execution_id"].startswith("wa_timeout_late")]
    rows += [r for r in build_crm_auth_regression()]
    rows += [r for r in build_invoice_schema_failures() if r["execution_id"].startswith("invoice_schema_f")]

    rng = random.Random(RNG_SEED + 7)
    wf, step, service, version = "Customer Support Classifier", "classify_ticket", "support-classifier-service", "1.8.0"
    mean, std = 5.0, 1.5
    n = 40
    ts = _timestamps_in_day_range(n, SPAN_DAYS, 0, seed=70)
    for i in range(n):
        rows.append(
            _row(
                f"mixed_healthy_{i:04d}", wf, ts[i], "success", _clipped_normal(rng, mean, std),
                workflow_step=step, service=service, application_version=version,
                correlation_id=f"CORR-mixed-healthy-{i:04d}",
            )
        )
    return _finalize(rows)


FILES: list = [
    ("01_healthy_baseline", build_healthy_baseline),
    ("02_whatsapp_provider_timeouts", build_whatsapp_provider_timeouts),
    ("03_crm_authentication_regression", build_crm_auth_regression),
    ("04_invoice_schema_failures", build_invoice_schema_failures),
    ("05_crm_duplicate_records", build_crm_duplicate_records),
    ("06_slow_successful_executions", build_slow_successful_executions),
    ("07_mixed_production_incident", build_mixed_production_incident),
]


def generate_all() -> dict:
    return {name: builder() for name, builder in FILES}


# --- serialization -------------------------------------------------------------


def _to_json_record(row: dict) -> dict:
    is_failure = row["status"] == "failure"
    error = None
    if is_failure:
        error = {
            "type": row["error_type"],
            "message": row["error_message"],
            "code": row["error_code"] or None,
            "stack_summary": row["error_stack_summary"] or None,
        }
    return {
        "execution_id": row["execution_id"],
        "workflow_name": row["workflow_name"],
        "start_time": row["start_time"].strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "status": "failed" if is_failure else "success",
        "duration_seconds": row["duration_seconds"],
        "error": error,
        "environment": row["environment"],
        "workflow_step": row["workflow_step"],
        "service": row["service"],
        "endpoint": row["endpoint"],
        "http_method": row["http_method"],
        "http_status": row["http_status"],
        "correlation_id": row["correlation_id"],
        "retry_count": row["retry_count"],
        "application_version": row["application_version"],
        "provider": row["provider"],
    }


def _to_csv_dataframe(rows: list) -> pd.DataFrame:
    flat = []
    for row in rows:
        flat.append({**row, "start_time": row["start_time"].strftime("%Y-%m-%d %H:%M:%S")})
    return pd.DataFrame(flat)


def compute_findings(filename: str, json_records: list) -> dict:
    """Runs the real exception-investigation engine (ingestion, fingerprinting,
    taxonomy, trend/priority scoring — all deterministic, no LLM) against one
    generated file's JSON records and summarizes the resulting exception
    groups. This is the single source of truth for expected_findings.json."""
    import io

    from src.exception_groups import build_exception_groups
    from src.ingestion import load_and_validate
    from src.prioritization import prioritize_groups

    buf = io.StringIO(json.dumps(json_records))
    valid_df, report = load_and_validate(buf, filename=f"{filename}.json")

    groups = prioritize_groups(build_exception_groups(valid_df))

    return {
        "file": f"{filename}.json",
        "validation": {"ok": report.ok, "total_rows": report.total_rows, "valid_rows": report.valid_rows, "rejected_rows": report.rejected_rows},
        "total_executions": int(len(valid_df)),
        "total_failures": int((valid_df["status"] == "failure").sum()),
        "exception_group_count": len(groups),
        "exception_groups": [
            {
                "fingerprint": g.fingerprint,
                "primary_category": g.primary_category,
                "secondary_category": g.secondary_category,
                "error_type": g.error_type,
                "workflow_name": g.workflow_name,
                "service": g.service,
                "http_status": g.http_status,
                "occurrence_count": g.occurrence_count,
                "affected_execution_count": g.affected_execution_count,
                "percentage_of_all_failures": g.percentage_of_all_failures,
                "percentage_of_workflow_executions": g.percentage_of_workflow_executions,
                "trend_label": g.trend_label,
                "application_versions": g.application_versions,
                "priority_score": g.priority_score,
                "priority_label": g.priority_label,
            }
            for g in groups
        ],
    }


def main() -> None:
    out_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    generated = generate_all()
    findings = {}
    for name, rows in generated.items():
        json_records = [_to_json_record(r) for r in rows]
        json_path = os.path.join(out_dir, f"{name}.json")
        with open(json_path, "w") as f:
            json.dump(json_records, f, indent=2)
        print(f"Wrote {len(json_records)} records to {json_path}")

        csv_path = os.path.join(out_dir, f"{name}.csv")
        _to_csv_dataframe(rows).to_csv(csv_path, index=False)
        print(f"Wrote {len(rows)} rows to {csv_path}")

        findings[f"{name}.json"] = compute_findings(name, json_records)

    findings_path = os.path.join(out_dir, "expected_findings.json")
    with open(findings_path, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"Wrote expected findings to {findings_path}")


if __name__ == "__main__":
    main()
