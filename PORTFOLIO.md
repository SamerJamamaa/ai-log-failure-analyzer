# AI Log Failure Analyzer — Portfolio Summary

## The problem

I build automation and AI workflow solutions with tools like n8n, WhatsApp integrations, Google Sheets, and various LLM APIs. Execution failures happen across every layer of that stack — flaky external APIs, expired credentials, malformed AI response formats, webhook misconfigurations — and reviewing execution logs by hand to catch recurring problems, unusual slowdowns, or a workflow's reliability quietly degrading over time doesn't scale past a handful of workflows.

## The solution

A focused tool that takes a CSV of workflow execution logs and turns it into an actual reliability read: KPIs, three specific anomaly-detection rules (recurring errors, abnormal duration, and a rising failure rate over time), and an AI-generated incident summary — with a hard architectural rule that the AI only interprets facts the system has already calculated, and never computes or invents them itself. Every AI-generated claim is visibly separated from deterministic fact and labeled as a hypothesis requiring human review. It also runs fully offline (a rule-based fallback) when no LLM API key is configured, so it's never a dead end.

## Technologies

Python 3.12, Streamlit, pandas, SQLite (via `sqlite3`, parameterized queries throughout), Plotly, pydantic (for validating LLM output against a strict schema), pytest (46 tests, fully mocked LLM calls — no network dependency in CI), and a provider-agnostic LLM layer supporting Anthropic Claude or OpenAI behind one interface.

## My contribution

I defined the problem from firsthand experience running automation workflows, specified the product requirements and scope boundaries (what this POC would and wouldn't do), and made the key technical decisions: the exact statistical definitions behind each anomaly rule (why mean + 2σ for duration outliers, why a first-half/second-half split for failure-rate drift, and the specific thresholds), the evidence-only boundary between the deterministic engine and the LLM, and the fact/hypothesis separation in both the data schema and the UI. I worked stage-by-stage through implementation with Claude Code as an AI pair-programmer — reviewing and approving each stage's design before moving to the next, catching and directing fixes for real bugs along the way (a detection-rule dilution bug against the actual sample data, a raw-HTML injection risk in the severity badge, a crash path in the upload flow, an inconsistent number-formatting bug between the AI evidence and the dashboard), and running a final structured quality pass across correctness, security, and test coverage before calling it done. The result reflects my product and architecture decisions, verified and refined through that review process rather than accepted as-delivered.
