# 3-Minute Demo Script

A timed walkthrough for showing the AI Log Failure Analyzer live — in an interview, a portfolio video, or a screen share. Total: ~3 minutes. Have the app already running (`streamlit run app.py`) before you start talking.

---

### 0:00–0:20 — The problem (20s)

> "If you're running automations in n8n or Make, you end up with execution logs but no easy way to see reliability trends — which workflow is failing more than usual, whether a specific error keeps recurring, or whether things are just getting slower. Reviewing that manually doesn't scale. This tool takes those logs and answers that automatically."

Show the empty landing page briefly.

### 0:20–0:40 — Load data (20s)

Click **Load Sample Data** in the sidebar, then **Analyze**.

> "This is a synthetic dataset — 200 executions across four workflows, built to include the kinds of problems you'd actually see: timeouts, auth failures, a workflow that's degrading over time. No real customer data anywhere."

### 0:40–1:20 — Deterministic KPIs (40s)

Point at the KPI cards and workflow table.

> "Everything up here — total executions, success rate, P95 duration, failure rate per workflow — is calculated with plain Python and SQL against a SQLite table. No AI involved yet. This is just arithmetic on the data you gave it."

Scroll to the charts.

> "Same for the charts — success rate by workflow, duration distribution, the trend over time."

### 1:20–2:00 — Anomaly detection (40s)

Scroll to "Detected Issues."

> "This is still 100% deterministic — three specific rules: is a duration more than 2 standard deviations above that workflow's normal, has the same error type shown up 3+ times, and has a workflow's failure rate jumped by more than 10 points between the first and second half of the data. Here you can see WhatsApp Lead Intake went from a 4% failure rate to 46% — that's the story I built into the sample data."

Point at the failure-rate comparison chart specifically.

### 2:00–2:40 — AI interpretation, with guardrails (40s)

Scroll to "AI Incident Analysis."

> "Now — and only now — does an LLM get involved. And critically, it never sees the raw log data. It only gets this evidence summary: the KPIs, the anomalies, a handful of example rows. It can't invent numbers because it's never given the full dataset to make numbers up from."

Click **Generate AI Analysis** (in offline mode if no key is configured, or with a live key if you have one set up).

> "It comes back as a structured object — title, severity, the evidence restated, probable causes clearly labeled as hypotheses, recommended actions, a confidence level, and limitations. And this banner up here — 'always have a human review it' — is permanent, not just a caveat buried in the text."

If running in offline mode:
> "And if there's no API key configured — like right now — it still works. You get a rule-based analysis instead, clearly labeled, so the tool is never dead in the water without a paid API key."

### 2:40–3:00 — Close (20s)

> "So: deterministic facts first, AI interpretation second, and the two are never allowed to blur together. That separation — and being explicit about it in the UI — is the part I think matters most here, more than the specific detection thresholds."

---

## If asked follow-up questions

- **"Why SQLite instead of just pandas?"** — Wanted to demonstrate SQL directly, and it's a natural fit for "store then query" even at this scale; parameterized queries throughout, verified with a dedicated SQL-injection test.
- **"How do you stop the AI from hallucinating?"** — Structural, not just prompted: it only ever receives pre-computed evidence, its output is validated against a strict schema before being trusted, and a failed/invalid response falls back to the offline analysis rather than showing something unverified.
- **"What would you build next?"** — Configurable thresholds from the UI, a second LLM provider running side-by-side for comparison, and exporting the incident report as a shareable doc.
