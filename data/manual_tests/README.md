# Manual Test Logs

Seven deterministic scenario files, each provided as both `.json` and `.csv`, for manually uploading into the Exception Investigation Assistant one at a time. Each carries realistic optional technical fields (`service`, `endpoint`, `http_method`, `http_status`, `correlation_id`, `retry_count`, `application_version`, `provider`) so the fingerprinting, taxonomy, trend/priority scoring, and evidence machinery all have something real to work with — not just the 7 required columns.

Upload the `.json` file for the more realistic experience (nested `error` object, the app's actual wire format) — the `.csv` is there for quick manual edits.

Regenerate everything (JSON + CSV + `expected_findings.json`) with:

```bash
python scripts/generate_manual_test_logs.py
```

Deterministic (fixed seed) — regenerating produces byte-identical data (aside from timestamps, which are anchored to "now" so the files always look recent). Every number in `expected_findings.json` is computed by actually running the real deterministic engine (`src/ingestion.py` → `src/exception_groups.py` → `src/prioritization.py`) against the generated data — never hand-typed — and `tests/test_manual_test_logs.py` re-derives it on every test run to catch drift.

---

### 01_healthy_baseline

**What it tests:** the baseline/no-incident case — does the assistant stay quiet when nothing is actually wrong?
**Records:** 78, across three workflows.
**Injected issue:** one isolated, never-recurring failure per workflow (3 total) — never enough to form a recurring pattern.
**Expected result:** 3 exception groups, all `isolated` trend, all `Low` priority. No investigation-worthy signal anywhere.

### 02_whatsapp_provider_timeouts

**What it tests:** a single recurring exception pattern, concentrated late in the window.
**Records:** 80, WhatsApp Lead Intake only.
**Injected issue:** clean first 5 days; 14 `GATEWAY_TIMEOUT` failures (HTTP 504) in the final 2 days, each carrying a correlation ID and one retry.
**Expected result:** 1 exception group, `Timeout and latency` / `Gateway timeout`, `Critical` priority. Evidence includes representative correlation IDs for tracing.

### 03_crm_authentication_regression

**What it tests:** a version-correlated authentication regression.
**Records:** 40, CRM Synchronization only.
**Injected issue:** version 3.4.0 is clean; after the bump to 3.5.0, `AUTHENTICATION_ERROR` (HTTP 401) failures appear consistently, alongside continuing 3.5.0 successes.
**Expected result:** 1 exception group, `Authentication and authorization`, entirely on version 3.5.0, contrastive evidence shows zero of that error type on 3.4.0. The investigation must frame this as a plausible, version-correlated cause — never a confirmed one.

### 04_invoice_schema_failures

**What it tests:** structural/format failures with multiple subtypes, cleanly separated from any performance signal.
**Records:** 70, Invoice Parser only.
**Injected issue:** three distinct schema-validation subtypes (`MISSING_REQUIRED_FIELD` ×5, `INVALID_JSON_SCHEMA` ×4, `TYPE_MISMATCH` ×3), durations kept tight throughout.
**Expected result:** 3 separate exception groups (one per subtype) — never merged, never misclassified as a timeout or performance issue.

### 05_crm_duplicate_records

**What it tests:** duplicate-record conflicts correlated with retry behavior.
**Records:** 36, CRM Synchronization only.
**Injected issue:** 6 `DUPLICATE_RECORD` failures (HTTP 409), each already retried once and still failing.
**Expected result:** 1 exception group, `Data integrity and persistence` / `Duplicate record`, `High` priority. Evidence shows every occurrence retried and still failed — pointing the investigation toward idempotency, not blind retrying.

### 06_slow_successful_executions

**What it tests:** that a slow-but-successful execution is never mistaken for a failure.
**Records:** 60, across three workflows, 100% success rate — several executions are dramatically slower than their workflow's norm.
**Expected result:** zero exception groups. Slow-but-successful executions are a performance signal, not an exception.

### 07_mixed_production_incident

**What it tests:** the full portfolio-demo case — several simultaneous, distinguishable issues in one file, and whether the assistant correctly keeps them apart.
**Records:** 132: a subset of the WhatsApp timeout scenario, the full CRM auth-regression scenario, a subset of the invoice schema failures, plus a fourth, healthy workflow (Customer Support Classifier).
**Expected result:** 5 distinct exception groups (no shared fingerprint or implied shared root cause), correctly prioritized by deterministic impact — WhatsApp's timeout pattern ranks highest (`Critical`) since it's the largest, most severe issue in the file. Customer Support Classifier produces no exception group at all.

---

## Cross-checking a file's expected results

`expected_findings.json` in this folder holds the exact exception-group breakdown for each file — computed by actually running the real engine against the generated data. `tests/test_manual_test_logs.py` re-runs that computation on every test run and asserts it still matches, plus a per-scenario test asserting the specific story above (grouping, category, trend, evidence) holds when run through the unmodified pipeline.
