# AI Log Failure Analyzer

A focused portfolio POC for automation developers, technical PMs, and ops teams: upload workflow execution logs (n8n/Make/Zapier-style CSVs), get deterministic reliability KPIs and anomaly detection, plus an AI-assisted incident analysis layered strictly on top of that evidence.

**Design principle:** all KPIs and anomalies are computed deterministically with Python and SQL (SQLite). The LLM never calculates numbers — it only interprets a compact evidence JSON built from those calculations, and its output is always presented as hypotheses, never as facts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py   # writes data/sample_workflow_logs.csv
```

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) to enable LLM-generated analysis:

```bash
cp .env.example .env
# then edit .env and set exactly one of the two keys
```

Without either key set, the app runs in a clearly labeled offline (rule-based) analysis mode — it's fully functional either way. Never commit `.env` or share your key; it's already listed in `.gitignore`.

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. In the sidebar: upload a JSON or CSV log (or click **Load Sample Data**), then click **Analyze**.

## Test

```bash
pytest
```

137 tests, no network calls (LLM provider calls are mocked).

## Architecture

```
JSON upload  ──┐
CSV upload   ──┼──► src/ingestion.py   format detected from the file extension;
sample data  ──┘        each format is validated with its own rules, then
                         normalized into the SAME 7-column execution model
        │                (valid rows only continue — see "Upload formats" below)
        ▼
 src/db.py           SQLite, in-memory, parameterized queries only
        │
        ├──────────────► src/metrics.py     KPIs via SQL (+ numpy for P95)
        │
        └──────────────► src/anomalies.py   3 deterministic detection rules
                                │
                                ▼
                     src/llm/base.py  build_evidence()
                     (KPIs + anomalies + 5 example rows — never the full dataset)
                                │
                                ▼
                     src/llm/{anthropic,openai,offline}_client.py
                     → validated IncidentReport (pydantic schema)
                                │
                                ▼
                          ui/components.py  (Streamlit)
```

- **`app.py`** — page orchestration and session state only; no business logic.
- **`src/`** — the deterministic engine + LLM service layer. No Streamlit imports here, so it's independently testable and reusable.
- **`ui/`** — all Streamlit rendering. No calculations here — it only formats and displays what `src/` already computed.
- **`scripts/generate_sample_data.py`** — deterministic (seeded) synthetic dataset generator.

## Upload formats

The app accepts **JSON** (recommended — matches a realistic ingestion pipeline and captures richer error detail) or **CSV** (supported for quick manual testing). Format is auto-detected from the file extension. Both are normalized into the exact same internal 7-column execution model before anything else runs — `src/metrics.py`, `src/anomalies.py`, and the LLM evidence builder never know or care which format a given upload came from. Proven, not just asserted: `tests/test_json_csv_equivalence.py` builds the same executions in both formats and asserts identical normalized records, KPIs, and detected anomalies.

### JSON schema (recommended)

A JSON array of execution objects. Required fields: `execution_id, workflow_name, start_time, status, duration_seconds, error`.

- `status` is `"success"` or `"failed"` — note this differs from the CSV vocabulary below (`"failed"` vs `"failure"`); `src/ingestion.py` translates it internally, nothing downstream ever sees the JSON spelling.
- `start_time` must be an ISO 8601 timestamp (e.g. `2026-09-20T10:15:00Z`) — validated strictly, unlike CSV's more lenient timestamp parsing.
- A successful execution's `error` must be `null`.
- A failed execution's `error` must be an object with `type` and `message`; it may optionally include `service` and `retry_count` (a non-negative integer).
- `duration_seconds` must be strictly positive (CSV allows exactly `0`; JSON does not).

```json
[
  {
    "execution_id": "EXE-1001",
    "workflow_name": "WhatsApp Lead Intake",
    "start_time": "2026-09-20T10:15:00Z",
    "status": "success",
    "duration_seconds": 2.4,
    "error": null
  },
  {
    "execution_id": "EXE-1002",
    "workflow_name": "WhatsApp Lead Intake",
    "start_time": "2026-09-20T10:17:00Z",
    "status": "failed",
    "duration_seconds": 15.8,
    "error": {
      "type": "API_TIMEOUT",
      "message": "Provider did not respond within 15 seconds",
      "service": "whatsapp_provider",
      "retry_count": 2
    }
  }
]
```

### CSV schema

Required columns: `execution_id, workflow_name, start_time, status, duration_seconds, error_type, error_message`.

- `status` must be `success` or `failure` (case-insensitive) — note the CSV spelling is `failure`, not JSON's `failed`.
- `start_time` must parse as a timestamp (lenient — not restricted to ISO 8601).
- `duration_seconds` must be numeric and ≥ 0.
- A `failure` row must carry `error_type` or `error_message`.

### Shared validation (both formats)

- `execution_id` must be present and unique within the file (first occurrence of a duplicate is kept, later ones rejected).
- Every rejected record is reported individually — which record, and exactly why — not just an aggregate count.
- A structurally broken file (missing required CSV column, JSON that isn't valid syntax, or a JSON top level that isn't an array) is a fatal, whole-file rejection with a plain-language message; the underlying parser's raw exception text is never shown to the user. Everything else is per-record: valid records still load even if others are rejected.

### Normalization flow

```
raw upload (bytes)
      │
      ▼
format detected from filename extension (.json / .csv; unknown → CSV)
      │
      ├── .json → parse array → flatten each record's nested `error` object
      │            into error_type/error_message → translate "failed" → "failure"
      │
      └── .csv  → pandas reads the 7 named columns directly
      │
      ▼
shared row-level validation (uniqueness, status enum, missing-error-info, ...)
      │
      ▼
common execution model: execution_id, workflow_name, start_time, status,
duration_seconds, error_type, error_message — identical shape either way
      │
      ▼
SQLite → KPIs → anomaly detection → LLM evidence (unchanged by upload format)
```

## Sample data

`data/sample_workflow_logs.csv` is a synthetic dataset — exactly 200 executions across 4 workflows (WhatsApp Lead Intake, Invoice Parser, CRM Synchronization, Customer Support Classifier), generated by `scripts/generate_sample_data.py` with a fixed random seed (reproducible on every run). No real customer data anywhere — all names, IDs, and messages are synthetic. It's deliberately constructed so every detection rule has something to catch: recurring API-timeout / authentication / invalid-JSON-schema / duplicate-record errors, a handful of abnormally long executions, and a deliberate second-half failure-rate increase on WhatsApp Lead Intake.

For hand-uploading one scenario at a time instead, see `data/manual_tests/` — 6 focused files (a healthy baseline, and 5 single/combined-issue scenarios), each provided as both `.json` and `.csv`, generated by `scripts/generate_manual_test_logs.py`. Documented in `data/manual_tests/README.md`.

## Detection rules

Thresholds are named constants at the top of `src/anomalies.py` — edit them there to retune sensitivity; nothing is hardcoded inline.

- **Slow execution** — `duration_seconds` > per-workflow mean + `SLOW_EXECUTION_STD_MULTIPLIER` (2) standard deviations. Computed per workflow (durations vary a lot by workflow type). Skipped for workflows with too few samples or zero duration variance, so it never false-positives on a tiny or perfectly uniform dataset.
- **Recurring failure** — the same `(workflow, error_type)` pair appearing ≥ `RECURRING_FAILURE_MIN_COUNT` (3) times.
- **Failure-rate increase** — a workflow's records, sorted by time and split into two equal halves, where the second half's failure rate is ≥ `FAILURE_RATE_INCREASE_THRESHOLD_PP` (10) percentage points higher than the first half's, and both halves have ≥ `FAILURE_RATE_INCREASE_MIN_RECORDS_PER_HALF` (10) records (guards against noisy conclusions from a handful of records).

## AI safeguards (responsible AI design)

- **Never sees the raw dataset.** `build_evidence()` sends only computed KPIs, detected anomalies, and up to 5 example rows for the selected scope — never the full upload.
- **Never computes anything.** The system prompt explicitly forbids inventing or recomputing numbers; all facts come from the deterministic engine.
- **Facts vs. hypotheses, structurally separated.** `IncidentReport` (pydantic) has distinct `observed_evidence` (must trace back to real evidence) and `probable_causes` (explicitly speculative) fields — this isn't just a prompt instruction, it's enforced by the schema shape the UI renders.
- **Validated output.** The LLM's JSON response is parsed against a strict schema (`incident_title`, `severity` ∈ {low,medium,high}, `observed_evidence`, `probable_causes`, `recommended_actions`, `confidence` ∈ {low,medium,high}, `limitations`). One retry on malformed output, then a graceful offline fallback — never a raw traceback.
- **Resilient to provider failure.** A 30-second timeout plus try/except around every provider call. Any failure mode (timeout, auth error, rate limit, network error, or output that still won't validate) falls back to the offline rule-based analysis, clearly labeled as such.
- **Human-review notice is always visible** above the AI panel, not just after a result appears.
- **Credentials are never logged or displayed.** Read once from `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`; provider-error messages carry only the exception's type name, never its text (which could theoretically echo request details) — covered directly by a test that plants a fake credential in a mocked exception and asserts it never appears in what's shown to the user.

## Assumptions

- Each upload is a self-contained window of history for the workflows it covers — there's no cross-upload/session persistence by design (a fresh in-memory SQLite database per analysis run).
- "Reliability" is judged from the fields the schema provides (status, duration, error type) — there's no notion of execution priority/severity coming from the source system.
- The failure-rate-increase rule assumes a workflow's own execution order is a meaningful proxy for time-based drift; it doesn't account for multiple concurrent instances of the same workflow interleaving unpredictably.
- The sample data's timestamps are anchored to "now" at generation time (not a fixed historical date), so the dashboard's date-based chart always looks current on a fresh clone.

## Limitations

- Single-provider LLM integration per run (Anthropic *or* OpenAI, chosen by whichever env var is set — not both simultaneously). The interface (`LLMClient`) is provider-agnostic, so adding a third provider means implementing one class.
- Detection thresholds are fixed constants, not adjustable from the UI (a deliberate scope decision for this POC, documented in `src/anomalies.py`).
- No authentication, no multi-user support, no persistent history across sessions — by design, not an oversight (see Scope in the project's original plan).
- P95 duration is computed with `numpy.percentile` (linear interpolation) since SQLite has no native percentile function — this is standard practice, not an approximation hack, but worth knowing if you're cross-checking against another tool's percentile method.
- The AI's `probable_causes` are genuinely the model's inference — like any LLM output, treat them as a starting point for investigation, not a diagnosis. The offline mode's causes are even more generic (template-matched to the anomaly *type*, not a read of the specific data).

## Screenshots

Not committed to the repo (keeps it lightweight and avoids stale images). To capture your own for a portfolio writeup or PR description:

1. `streamlit run app.py`, click **Load Sample Data** → **Analyze**.
2. Capture: (a) the KPI cards + charts section, (b) the "Detected Issues" section with the three anomaly tables/chart, (c) the AI Incident Analysis panel after clicking **Generate AI Analysis** (once with a key configured, once without, to show both modes).
3. Save under a local `screenshots/` folder (already covered by `.gitignore`'s general patterns — add an exception if you want to commit them) and reference them from this README or your portfolio writeup with standard markdown image syntax: `![KPI dashboard](screenshots/kpis.png)`.

## Deploying (making it accessible to others)

Locally it only runs on your own machine (`streamlit run app.py`, reachable at `localhost:8501`). To give someone else a real link they can open themselves:

1. **Push this repo to GitHub** (a public repo is free on Streamlit Community Cloud; a private repo works too, on any paid plan).
2. **Deploy on [Streamlit Community Cloud](https://share.streamlit.io)** — sign in with GitHub, click **New app**, point it at this repo and `app.py`. It installs `requirements.txt` and gives you a public URL like `your-app.streamlit.app`.
3. **API key (optional):** in the app's **Settings → Secrets** on Streamlit Cloud, paste the contents of [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with a real key filled in. Streamlit exposes top-level secrets as environment variables automatically, so `get_client()` picks it up with no code changes.
   - **Leaving it unset is the safer default for a public link** — the app runs in offline (rule-based) mode, which costs nothing and needs no key from you or any visitor. If you do set a key, remember every visitor's "Generate Investigation" click spends *your* API credits — there's no per-visitor key entry.
4. Any push to the connected branch redeploys automatically.

Other options if you outgrow the free tier: [Hugging Face Spaces](https://huggingface.co/spaces) (also free, supports Streamlit natively) or a container platform (Render, Fly.io, Railway) using a basic `Dockerfile` that runs `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## More

- [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) — a 3-minute walkthrough script for demoing this live.
- [`PORTFOLIO.md`](PORTFOLIO.md) — a short problem/solution/tech/contribution writeup for a portfolio page or README badge.
