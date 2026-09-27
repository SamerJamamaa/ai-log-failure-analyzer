"""Streamlit rendering for the AI Exception Investigation Console — a
single continuous page (Sections 1-5), never tabs. Every fact rendered here
comes from an already-computed ExceptionGroup / evidence item / KPISet /
aggregate — never calculated in this module. AI/offline investigation
content is always visually distinguished and carries its source label and
a human-review notice, per the fact/hypothesis separation this app is
built around. `st.expander` is used ONLY for raw execution-level detail and
secondary technical evidence, never for primary investigation content —
the whole story must read top-to-bottom without a single tab click.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.aggregates import (
    compute_category_summary,
    compute_group_pareto,
    compute_performance_anomalies,
    compute_period_comparison,
    compute_retry_outcomes,
    compute_version_comparison,
    compute_workflow_impact_matrix,
)
from src.evidence import build_evidence_table, missing_evidence_for
from src.models import ExceptionGroup, IncidentBrief, ValidationReport
from src.titles import human_title
from ui.format import format_duration

PRIORITY_BADGE_COLOR = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "gray"}
OVERALL_STATUS_COLOR = {"Critical": "red", "Attention Required": "orange", "Healthy": "green"}
OVERALL_STATUS_ICON = {"Critical": "🔴", "Attention Required": "🟠", "Healthy": "🟢"}
CONFIDENCE_BADGE_COLOR = {"high": "red", "medium": "orange", "low": "gray"}
NOT_AVAILABLE = "Not available in supplied logs"
BUSINESS_IMPACT_DISCLAIMER = "Technical impact is visible, but business impact cannot be determined from the supplied logs."


def render_validation_report(report: ValidationReport) -> None:
    if not report.ok:
        st.error(f"Upload rejected: {report.fatal_error}")
        st.caption(
            "Required columns: execution_id, workflow_name, start_time, status, duration_seconds, "
            "error_type, error_message. See 'How to read this console' below for the full field list."
        )
        return
    if report.rejected_rows:
        reasons = ", ".join(f"{k}: {v}" for k, v in report.rejection_reasons.items())
        st.warning(
            f"{report.rejected_rows} of {report.total_rows} row(s) were skipped as invalid "
            f"({reasons}). {report.valid_rows} valid row(s) loaded — analysis below covers only the valid rows."
        )
        if report.rejected_row_details:
            with st.expander(f"Show the {len(report.rejected_row_details)} skipped row(s)"):
                for row in report.rejected_row_details:
                    st.text(row.message())
    else:
        st.success(f"Loaded {report.valid_rows} valid execution record(s).")


def render_ai_mode_indicator(provider_label: str) -> None:
    if provider_label.startswith("Offline"):
        st.badge(provider_label, color="gray", icon="⚪")
    else:
        st.badge(f"Connected — {provider_label}", color="green", icon="🟢")


def render_human_review_notice() -> None:
    st.warning(
        "Hypotheses and recommendations below require human review before acting on them — "
        "nothing here is a confirmed root cause or a guaranteed fix.",
        icon="⚠️",
    )


def _redacted_fields_note() -> None:
    st.caption(
        "🔒 Before any evidence is sent to an AI provider, credentials, tokens, authorization headers, "
        "cookies, personal data, message contents, and request/response bodies are stripped or redacted. "
        "Only aggregated counts, categories, timestamps, and a capped sample of execution/correlation IDs "
        "are ever sent."
    )


# ===================================================================================
# SECTION 1 — Incident Overview
# ===================================================================================


def _overall_status(groups: list) -> str:
    if not groups:
        return "Healthy"
    top = groups[0].priority_label
    if top == "Critical":
        return "Critical"
    if top in ("High", "Medium"):
        return "Attention Required"
    return "Healthy"


def _kpi_card(column, title: str, value: str, lines: list, tooltip: str) -> None:
    with column:
        with st.container(border=True):
            st.metric(title, value, help=tooltip)
            for line in lines:
                st.caption(line)


def render_incident_overview(groups: list, df: pd.DataFrame, total_executions: int, total_failures: int, kpis) -> None:
    status = _overall_status(groups)
    st.badge(status, color=OVERALL_STATUS_COLOR[status], icon=OVERALL_STATUS_ICON[status])

    period = compute_period_comparison(df)
    perf = compute_performance_anomalies(df)

    if not groups:
        st.markdown(
            f"**{total_executions} execution(s) analyzed, {total_failures} failure(s)** — "
            "no exception pattern in this data warranted investigation."
        )
    else:
        top = groups[0]
        secondary_count = len(groups) - 1
        st.markdown(
            f"**{status}.** The uploaded file contains {total_failures} failed execution(s) grouped into "
            f"{len(groups)} distinct exception pattern(s). **{human_title(top)}** is the primary incident, "
            f"representing {top.percentage_of_all_failures}% of all failures"
            + (f", with {secondary_count} separate secondary issue(s)." if secondary_count else ".")
        )

    row1 = st.columns(3)
    row2 = st.columns(3)

    period_lines = [f"{report_period(df)}", f"{len(df['workflow_name'].unique())} workflow(s) represented"]
    _kpi_card(row1[0], "Total executions", f"{total_executions:,}", period_lines, "Every row in the uploaded file that passed validation.")

    if period["valid"]:
        change_line = (
            f"{'+' if period['change_pp'] >= 0 else ''}{period['change_pp']} pp vs. earlier period "
            f"({period['first_period_count']} vs. {period['second_period_count']} executions)"
        )
    else:
        change_line = f"Change vs. earlier period: {NOT_AVAILABLE.lower()} ({period.get('reason', 'insufficient data')})"
    failure_rate = round(total_failures / total_executions * 100, 1) if total_executions else 0.0
    _kpi_card(
        row1[1],
        "Failed executions",
        f"{total_failures:,}",
        [f"{failure_rate}% of {total_executions:,} executions", change_line],
        "Count of rows with status='failure'. Rate = failures / total executions in this upload.",
    )

    new_patterns = sum(1 for g in groups if g.trend_label == "isolated")
    recurring_patterns = sum(1 for g in groups if g.trend_label != "isolated")
    _kpi_card(
        row1[2],
        "Exception patterns",
        f"{len(groups)}",
        [f"{new_patterns} new (isolated)", f"{recurring_patterns} recurring/increasing"],
        "Distinct deterministically-fingerprinted exception groups (workflow + step + service + error type + endpoint).",
    )

    high_priority = [g for g in groups if g.priority_label in ("Critical", "High")]
    if high_priority:
        top_issue = high_priority[0]
        reason_line = top_issue.priority_reason
    else:
        reason_line = "No Critical or High priority exceptions in this upload."
    _kpi_card(
        row2[0],
        "Critical/High issues",
        f"{len(high_priority)}",
        [reason_line[:110] + ("…" if len(reason_line) > 110 else "")],
        "Count of exception groups whose deterministic priority score lands in the Critical or High band.",
    )

    affected_workflows = sorted({g.workflow_name for g in groups})
    if affected_workflows:
        by_workflow_failures = {}
        for g in groups:
            by_workflow_failures[g.workflow_name] = by_workflow_failures.get(g.workflow_name, 0) + g.affected_execution_count
        most_affected = max(by_workflow_failures, key=by_workflow_failures.get)
        wf_total = len(df[df["workflow_name"] == most_affected])
        wf_rate = round(by_workflow_failures[most_affected] / wf_total * 100, 1) if wf_total else 0.0
        wf_lines = [f"Most affected: {most_affected}", f"{wf_rate}% failure rate ({by_workflow_failures[most_affected]}/{wf_total})"]
    else:
        wf_lines = ["No workflow shows an exception pattern"]
    _kpi_card(
        row2[1],
        "Affected workflows",
        f"{len(affected_workflows)}",
        wf_lines,
        "Distinct workflows that appear in at least one exception group.",
    )

    degradation = perf["latency_degradation"]
    if degradation["valid"]:
        degradation_line = f"P95 rose {degradation['change_pct']}% ({degradation['first_period_p95']}s → {degradation['second_period_p95']}s)" if degradation["change_pct"] > 0 else "No independent latency degradation detected"
    else:
        degradation_line = f"Latency trend: {NOT_AVAILABLE.lower()}"
    _kpi_card(
        row2[2],
        "Performance anomalies",
        f"{perf['slow_successful_count']}",
        [f"P95 duration: {format_duration(perf['p95_duration_seconds'])}", degradation_line],
        "Successful (non-failed) executions whose duration exceeded their own workflow's mean + 2 standard deviations. Never counted as exceptions.",
    )


def report_period(df: pd.DataFrame) -> str:
    if df.empty:
        return "no data"
    start, end = df["start_time"].min(), df["start_time"].max()
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"


# ===================================================================================
# SECTION 2 — AI Investigation Brief
# ===================================================================================


def render_investigation_brief(brief: IncidentBrief) -> None:
    st.subheader("Investigation Brief")
    st.caption("Deterministic, evidence-based — computed from the exception groups below, not a free-form AI narrative.")

    st.markdown(f"**What happened.** {brief.what_happened}")
    st.markdown(f"**Primary incident.** {brief.primary_incident}")

    if brief.secondary_incidents:
        st.markdown("**Secondary incidents:**")
        for item in brief.secondary_incidents:
            st.markdown(f"- {item}")

    if brief.systems_affected:
        st.markdown(f"**Systems affected:** {', '.join(brief.systems_affected)}")

    st.markdown(f"**Time and version relationship.** {brief.time_version_relationship}")
    st.markdown(f"**Technical impact.** {brief.technical_impact}")

    if brief.likely_explanations:
        st.markdown("**Most likely explanations:**")
        for item in brief.likely_explanations:
            st.markdown(f"- {item}")

    if brief.immediate_actions:
        st.markdown("**Immediate actions:**")
        for item in brief.immediate_actions:
            st.markdown(f"- {item}")

    if brief.limitations:
        with st.expander("Key limitations of this brief"):
            for item in brief.limitations:
                st.caption(item)


# ===================================================================================
# SECTION 3 — Exception Landscape
# ===================================================================================


def _group_retry_summary(group: ExceptionGroup) -> str:
    retried = group.retry_info.get("executions_with_retries")
    if not retried:
        return "No retries observed"
    if group.retry_info.get("all_retries_still_failed"):
        return f"All {retried} retried, still failed"
    return f"{retried} retried, some recovered"


def _recommended_next_action(group: ExceptionGroup) -> str:
    from src.evidence import build_evidence_table as _bet
    from src.llm.base import build_exception_evidence as _bee
    from src.llm.offline import generate_offline_investigation as _goi

    # Uses the SAME rule-based playbook the full investigation panel uses,
    # so the one-line recommendation here never contradicts the detailed
    # action plan shown once this exception is selected.
    items = _bet(group, group.rows if group.rows is not None else pd.DataFrame())
    evidence = _bee(group, items, [])
    investigation = _goi(evidence)
    if investigation.investigation_actions:
        action = investigation.investigation_actions[0].action
        return action[:90] + ("…" if len(action) > 90 else "")
    return "Review evidence before acting."


def render_exception_landscape(groups: list) -> ExceptionGroup:
    st.subheader("Exception Landscape")
    st.caption("One row per normalized exception pattern, sorted Critical → High → Medium → Low. Select a row to open its investigation below.")

    rows = []
    for g in groups:
        rows.append(
            {
                "Priority": g.priority_label,
                "Exception": human_title(g),
                "Category": g.primary_category,
                "Error type/code": g.error_type + (f" / {g.error_code}" if g.error_code else ""),
                "Workflow": g.workflow_name,
                "Failed step": g.workflow_step or "—",
                "Service/provider": g.service or g.provider or "—",
                "HTTP status": g.http_status if g.http_status is not None else "—",
                "Occurrences": g.occurrence_count,
                "Affected executions": g.affected_execution_count,
                "% of failures": g.percentage_of_all_failures,
                "First seen": str(g.first_occurrence),
                "Last seen": str(g.last_occurrence),
                "Trend": g.trend_label,
                "Versions": ", ".join(g.application_versions) or "—",
                "Retry outcome": _group_retry_summary(g),
                "Recommended next action": _recommended_next_action(g),
            }
        )

    event = st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="exception_landscape_table",
    )
    selected_rows = event.selection["rows"] if event and event.selection else []
    return groups[selected_rows[0]] if selected_rows else groups[0]


# ===================================================================================
# SECTION 4 — Selected Exception Investigation (single continuous page)
# ===================================================================================


def _priority_factors_text(group: ExceptionGroup) -> str:
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in group.priority_factors.items() if v]
    return "; ".join(parts) if parts else "no contributing factors above zero"


def render_exception_summary(group: ExceptionGroup, investigation) -> None:
    """A. Exception summary."""
    st.markdown(f"### {human_title(group)}")
    badge_cols = st.columns([1, 1, 3])
    with badge_cols[0]:
        st.badge(group.priority_label, color=PRIORITY_BADGE_COLOR.get(group.priority_label, "gray"))
    with badge_cols[1]:
        status = "Investigated" if investigation is not None else "Not started"
        st.badge(status, color="green" if investigation else "gray")
    with badge_cols[2]:
        st.caption(f"{group.primary_category}" + (f" / {group.secondary_category}" if group.secondary_category else ""))

    if investigation is not None:
        st.markdown(investigation.exception_explanation)
    else:
        st.caption("Click **Generate Investigation** below for a full explanation, hypotheses, and action plan.")

    info_cols = st.columns(4)
    info_cols[0].markdown(f"**Workflow**\n\n{group.workflow_name}")
    info_cols[1].markdown(f"**Step**\n\n{group.workflow_step or NOT_AVAILABLE}")
    info_cols[2].markdown(f"**Service**\n\n{group.service or NOT_AVAILABLE}")
    info_cols[3].markdown(f"**Provider**\n\n{group.provider or NOT_AVAILABLE}")

    info_cols2 = st.columns(4)
    info_cols2[0].markdown(f"**Endpoint**\n\n{group.endpoint or NOT_AVAILABLE}")
    info_cols2[1].markdown(f"**HTTP status**\n\n{group.http_status if group.http_status is not None else NOT_AVAILABLE}")
    info_cols2[2].markdown(f"**Version(s)**\n\n{', '.join(group.application_versions) or NOT_AVAILABLE}")
    info_cols2[3].markdown(f"**Total occurrences**\n\n{group.occurrence_count} ({group.first_occurrence} → {group.last_occurrence})")

    st.caption(f"Priority {group.priority_label} ({group.priority_score}/100): {_priority_factors_text(group)}")


def render_why_it_matters(group: ExceptionGroup) -> None:
    """B. Why it matters."""
    st.markdown("#### Why it matters")
    lines = [
        f"**Technical impact:** {group.affected_execution_count} execution(s) directly failed with this exception "
        f"({group.percentage_of_all_failures}% of all failures, {group.percentage_of_workflow_executions}% of "
        f"'{group.workflow_name}' executions).",
        f"**Reliability impact:** this pattern is {group.trend_label}"
        + (f" and affects {group.workflows_sharing_this_issue} workflows" if group.workflows_sharing_this_issue > 1 else " within a single workflow")
        + ".",
    ]
    if group.avg_duration_seconds > 10:
        lines.append(f"**Performance impact:** affected executions average {format_duration(group.avg_duration_seconds)} (P95 {format_duration(group.p95_duration_seconds)}).")
    else:
        lines.append("**Performance impact:** no unusual duration impact observed for this exception.")

    if group.primary_category == "Data integrity and persistence":
        lines.append("**Data-integrity risk:** this category indicates a possible duplicate or inconsistent record — verify before assuming data is clean.")
    else:
        lines.append("**Data-integrity risk:** not indicated by this exception's category.")

    retried = group.retry_info.get("executions_with_retries")
    if retried:
        lines.append(
            "**Retry risk:** "
            + ("all retried executions still failed — retrying further is unlikely to help without a fix." if group.retry_info.get("all_retries_still_failed") else "some retried executions recovered; retry behavior is inconsistent.")
        )
    else:
        lines.append(f"**Retry risk:** {NOT_AVAILABLE.lower()} — no retries observed for this exception.")

    lines.append(
        "**Downstream effect:** "
        + (f"shared with {group.workflows_sharing_this_issue - 1} other workflow(s) — a broader spread than an isolated issue." if group.workflows_sharing_this_issue > 1 else f"{NOT_AVAILABLE} — no downstream system is captured in this data.")
    )

    for line in lines:
        st.markdown(f"- {line}")
    st.info(BUSINESS_IMPACT_DISCLAIMER)


def render_evidence_found(group: ExceptionGroup, full_df: pd.DataFrame) -> None:
    """C. Evidence found."""
    st.markdown("#### Evidence found")
    items = build_evidence_table(group, full_df)
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.evidence_id}.** {item.observed_value} — {item.why_it_matters}")
            cols = st.columns(4)
            cols[0].caption(f"Source field: {item.source_field}")
            cols[1].caption(f"Occurrences: {item.occurrence_count}")
            cols[2].caption(f"Execution IDs: {', '.join(item.representative_execution_ids) or NOT_AVAILABLE}")
            cols[3].caption(f"Correlation IDs: {', '.join(item.representative_correlation_ids) or NOT_AVAILABLE}")

    missing = missing_evidence_for(group)
    if missing:
        with st.expander(f"Missing evidence ({len(missing)})"):
            for item in missing:
                st.caption(f"{NOT_AVAILABLE}: {item}")
    _redacted_fields_note()


def render_probable_causes(investigation) -> None:
    """D. Probable causes."""
    st.markdown("#### Probable causes")
    if investigation is None:
        st.caption("Click **Generate Investigation** above to see ranked probable causes.")
        return
    if not investigation.hypotheses:
        st.caption("No hypotheses were proposed — evidence was too thin to suggest a plausible cause.")
        return

    for h in investigation.hypotheses:
        label = "Leading hypothesis" if h.rank == 1 else f"Hypothesis #{h.rank}"
        with st.container(border=True):
            badge_col, text_col = st.columns([1, 6])
            with badge_col:
                st.badge(h.confidence.upper(), color=CONFIDENCE_BADGE_COLOR.get(h.confidence, "gray"))
            with text_col:
                st.markdown(f"**{label}: {h.hypothesis}**")
            st.markdown(f"Supporting evidence: {', '.join(h.supporting_evidence_ids) or 'none yet'}")
            if h.contradicting_evidence_ids:
                st.markdown(f"Contradicting evidence: {', '.join(h.contradicting_evidence_ids)}")
            st.markdown(f"Missing evidence: {'; '.join(h.missing_evidence) or NOT_AVAILABLE}")
            st.markdown(f"How to confirm or reject: {'; '.join(h.verification_steps)}")
    st.caption("These are ranked hypotheses, not confirmed root causes — never displayed as a 'root cause' unless the logs conclusively prove it.")


def render_action_plan(investigation) -> None:
    """E. Recommended action plan — NOW / NEXT / FIX OPTIONS / VERIFY / PREVENT."""
    st.markdown("#### Recommended action plan")
    if investigation is None:
        st.caption("Click **Generate Investigation** above to see the action plan.")
        return

    if investigation.containment_actions:
        st.markdown("**NOW — Containment**")
        for i, a in enumerate(investigation.containment_actions):
            st.checkbox(a.action, key=f"contain_{i}_{a.action[:20]}")
            st.caption(f"Reason: {a.reason}. Risk: {a.risk}")

    st.markdown("**NEXT — Investigation**")
    for i, a in enumerate(investigation.investigation_actions):
        st.checkbox(f"{a.action} ({a.component})", key=f"invest_{i}_{a.action[:20]}")
        st.caption(f"Evidence: {', '.join(a.evidence_reference)}. Expected findings: {'; '.join(a.expected_findings)}.")
        st.caption(f"Then: {a.decision_from_result}")

    if investigation.diagnostic_commands:
        with st.expander("Example diagnostic commands"):
            for cmd in investigation.diagnostic_commands:
                st.markdown(f"**{cmd.label}**")
                st.code(cmd.command, language="bash")
                st.caption("Look for: " + "; ".join(cmd.look_for))
                st.caption("Interpretation: " + " ".join(cmd.interpretation))

    st.markdown("**FIX OPTIONS — Remediation**")
    for r in investigation.remediation_options:
        with st.container(border=True):
            st.markdown(f"**{r.option}**")
            st.caption(f"Complexity: {r.complexity} · Change owner: {r.change_type} · {r.temporary_or_permanent}")
            st.markdown(f"Appropriate when: {r.appropriate_when}")
            st.markdown(f"Expected benefit: {r.expected_benefit}")
            st.markdown(f"Risks: {'; '.join(r.risks)}")
            st.markdown(f"Tradeoffs: {'; '.join(r.tradeoffs)}")

    st.markdown("**VERIFY — Resolution validation**")
    for v in investigation.verification_plan:
        st.markdown(f"- {v.test} → expect: {v.expected_result}")
        st.caption(f"Metric: {v.success_metric} · Threshold: {v.success_threshold} · Observation period: {v.observation_period}")
        st.caption(f"Rollback if: {v.rollback_condition}")

    st.markdown("**PREVENT — Recurrence protection**")
    for i, p in enumerate(investigation.prevention_actions):
        st.checkbox(p, key=f"prevent_{i}_{p[:20]}")

    if investigation.limitations:
        st.info("**Limitations:** " + " ".join(investigation.limitations))


def render_affected_executions(group: ExceptionGroup) -> None:
    df = group.rows
    if df is None or df.empty:
        st.caption("No affected executions available.")
        return
    with st.expander(f"Raw affected executions ({len(df)})"):
        display = df.copy()
        display["error_message"] = display["error_message"].fillna("")
        columns = {
            "execution_id": "Execution ID",
            "start_time": "Timestamp",
            "workflow_name": "Workflow",
            "workflow_step": "Step",
            "service": "Service",
            "endpoint": "Endpoint",
            "http_status": "HTTP status",
            "duration_seconds": "Duration (s)",
            "retry_count": "Retry count",
            "application_version": "Version",
            "correlation_id": "Correlation ID",
            "error_message": "Error message",
        }
        display = display[list(columns.keys())].rename(columns=columns).sort_values("Timestamp")
        st.dataframe(display, width="stretch", hide_index=True)


def render_investigation_trigger(provider_label: str, existing_investigation) -> bool:
    """The AI-status badge, human-review warning, and Generate/Regenerate
    button as ONE grouped block, placed directly above the AI-populated
    sections (Probable Causes, Action Plan) it controls — not floating
    above the exception summary with no context, and not separated from
    the privacy note a user needs before deciding to click. Returns True
    exactly when the button was clicked this run."""
    with st.container(border=True):
        render_ai_mode_indicator(provider_label)
        render_human_review_notice()

        label = "Regenerate Investigation" if existing_investigation is not None else "Generate Investigation"
        clicked = st.button(label, type="primary")

        if existing_investigation is not None:
            source_label = "AI-assisted investigation" if existing_investigation.source == "llm" else "Rule-based investigation guidance"
            st.caption(f"Below reflects a **{source_label}** already generated for this exception. Click to regenerate.")
        st.caption(
            "Generates ranked probable causes and a tailored action plan below, from only this "
            "exception's redacted evidence — credentials, tokens, message contents, and request/response "
            "bodies are stripped before anything is sent. Usually takes a few seconds."
        )
        return clicked


# ===================================================================================
# SECTION 5 — Operational Visuals
# ===================================================================================


def _chart_block(title: str, subtitle: str) -> None:
    st.markdown(f"##### {title}")
    st.caption(subtitle)


def render_category_chart(groups: list) -> None:
    summary = compute_category_summary(groups)
    if not summary:
        return
    _chart_block("Exceptions by category", "Which kinds of technical problems dominate this dataset.")
    df = pd.DataFrame(summary)
    fig = px.bar(df, x="category", y="occurrence_count", hover_data=["affected_execution_count", "percentage_of_failures"])
    fig.update_layout(yaxis_title="Occurrences", xaxis_title="")
    fig.update_yaxes(tickformat=",d")
    st.plotly_chart(fig, width="stretch")
    top = summary[0]
    st.caption(f"Finding: **{top['category']}** represents {top['percentage_of_failures']}% of failed executions and should be prioritized first.")


def render_failures_over_time_chart(groups: list, df: pd.DataFrame) -> None:
    if not groups:
        return
    _chart_block("Failures over time by exception pattern", "When each problem began and whether it is increasing.")

    frames = []
    for g in groups[:5]:
        if g.rows is None or g.rows.empty:
            continue
        timeline = g.rows.copy()
        timeline["date"] = pd.to_datetime(timeline["start_time"]).dt.date
        counts = timeline.groupby("date").size().reset_index(name="count")
        counts["pattern"] = human_title(g)
        frames.append(counts)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    fig = px.line(combined, x="date", y="count", color="pattern", markers=True)
    fig.update_layout(yaxis_title="Occurrences", xaxis_title="", legend_title="")
    fig.update_yaxes(tickformat=",d")

    # Only the workflows actually plotted (top 5 patterns) — annotating every
    # version across the WHOLE upload floods the chart with version numbers
    # from unrelated workflows/products that have nothing to do with the
    # patterns shown, and with enough of them the on-chart text labels
    # overlap into an unreadable smear. The version->date mapping is listed
    # as plain text below the chart instead, which stays legible regardless
    # of how many versions there are.
    plotted_workflows = {g.workflow_name for g in groups[:5]}
    relevant = df[df["workflow_name"].isin(plotted_workflows)]
    versions = sorted(v for v in relevant["application_version"].unique() if v)
    version_notes = []
    if len(versions) > 1:
        for version in versions:
            first_seen = relevant[relevant["application_version"] == version]["start_time"].min()
            fig.add_vline(x=first_seen, line_dash="dash", line_color="gray")
            version_notes.append(f"**{version}** (first seen {first_seen:%Y-%m-%d})")

    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Finding: **{human_title(groups[0])}** begins at {groups[0].first_occurrence}. "
        "Dashed lines mark when each application version first appears among the workflows plotted above "
        "— this is correlation, not proof of causation."
    )
    if version_notes:
        st.caption("Versions: " + " · ".join(version_notes))


def render_impact_matrix_chart(groups: list) -> None:
    matrix = compute_workflow_impact_matrix(groups)
    if not matrix:
        return
    _chart_block("Workflow and service impact matrix", "Which workflows and services are affected by each exception category.")
    df = pd.DataFrame(matrix)
    pivot = df.pivot_table(index="workflow", columns="category", values="occurrence_count", aggfunc="sum", fill_value=0)
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Reds", showscale=True))
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    workflows_touched = df["workflow"].nunique()
    categories_touched = df["category"].nunique()
    if workflows_touched == 1:
        finding = f"All exceptions are isolated to **{df['workflow'].iloc[0]}**."
    else:
        finding = f"Exceptions span **{workflows_touched} workflows** across **{categories_touched} categories** — this is a broad, not isolated, issue."
    st.caption(f"Finding: {finding}")


def render_pareto_chart(groups: list) -> None:
    pareto = compute_group_pareto(groups)
    if not pareto:
        return
    _chart_block("Error Pareto — which patterns account for most failures", "Bars = occurrences per pattern; line = cumulative % of all failures.")
    df = pd.DataFrame(pareto)
    fig = go.Figure()
    fig.add_bar(x=df["title"], y=df["occurrence_count"], name="Occurrences")
    fig.add_trace(go.Scatter(x=df["title"], y=df["cumulative_percentage"], name="Cumulative %", yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        yaxis=dict(title="Occurrences"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        xaxis=dict(title="", tickangle=-30),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")

    cutoff_idx = next((i for i, row in enumerate(pareto) if row["cumulative_percentage"] >= 80), len(pareto) - 1)
    st.caption(f"Finding: the top {cutoff_idx + 1} of {len(pareto)} pattern(s) account for at least 80% of all failures.")


def render_retry_outcomes_chart(df: pd.DataFrame) -> None:
    counts = compute_retry_outcomes(df)
    if not any(counts.values()):
        return
    _chart_block("Retry outcomes", "Whether retries recover executions or repeat the same error.")
    chart_df = pd.DataFrame([{"outcome": k, "count": v} for k, v in counts.items() if v > 0])
    fig = px.bar(chart_df, x="outcome", y="count")
    fig.update_layout(yaxis_title="Executions", xaxis_title="")
    fig.update_yaxes(tickformat=",d")
    st.plotly_chart(fig, width="stretch")

    if counts["Retry information unavailable"] == sum(counts.values()):
        finding = f"{NOT_AVAILABLE} — no execution in this upload reports a retry_count."
    elif counts["Recovered after retry"] > 0:
        finding = f"{counts['Recovered after retry']} execution(s) recovered after a retry; {counts['Failed after retry'] + counts['Duplicate created after retry']} did not."
    else:
        finding = "No execution recovered after a retry in this upload — retrying alone is not resolving these failures."
    st.caption(f"Finding: {finding}")


def render_version_comparison_chart(df: pd.DataFrame) -> None:
    comparison = compute_version_comparison(df)
    if not comparison:
        return
    _chart_block("Version comparison", "Executions, failures, and failure rate per application version. Shown only when at least two versions exist.")
    chart_df = pd.DataFrame(comparison)
    fig = go.Figure()
    fig.add_bar(x=chart_df["version"], y=chart_df["executions"], name="Executions")
    fig.add_bar(x=chart_df["version"], y=chart_df["failures"], name="Failures")
    fig.update_layout(barmode="group", yaxis_title="Count", xaxis_title="Version")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(chart_df, width="stretch", hide_index=True)
    worst = max(comparison, key=lambda r: r["failure_rate"])
    st.caption(
        f"Finding: version **{worst['version']}** has the highest failure rate ({worst['failure_rate']}%). "
        "A version boundary does not by itself prove the version caused the failures."
    )


def render_operational_visuals(groups: list, df: pd.DataFrame) -> None:
    st.subheader("Operational Visuals")
    render_category_chart(groups)
    render_failures_over_time_chart(groups, df)
    render_impact_matrix_chart(groups)
    render_pareto_chart(groups)
    render_retry_outcomes_chart(df)
    render_version_comparison_chart(df)


# --- Educational section --------------------------------------------------------


def render_how_to_read_expander() -> None:
    with st.expander("How to read this console"):
        st.markdown(
            "- **Exception pattern** — one deterministically-fingerprinted cluster of related failures "
            "(same workflow, step, service, error type, and endpoint) — never grouped by message text alone.\n"
            "- **Priority** — a bounded, explainable points score from occurrence volume, trend, failure "
            "percentage, HTTP severity, retry behavior, and cross-workflow spread. Never AI-assigned.\n"
            "- **Trend** — `isolated` (a one-off), `recurring` (happened more than once, stable), or "
            "`increasing` (occurrence rate is rising within this upload's own timeline).\n"
            "- **Evidence** — every factual claim traces to a numbered evidence item (E1, E2, ...) computed "
            "directly from your data. Anything not available is labeled explicitly, never guessed.\n"
            "- **Leading hypothesis** — the top-ranked probable cause. Never a confirmed root cause unless "
            "the logs conclusively prove it.\n"
            "- **Action plan** — NOW (containment, only when priority justifies it), NEXT (investigation "
            "steps tied to specific evidence), FIX OPTIONS (several remediation options with tradeoffs, "
            "never a single 'the fix'), VERIFY (measurable success criteria), PREVENT (recurrence controls).\n"
            "- **Optional fields** — execution_id, correlation_id, timestamp, environment, workflow, "
            "workflow_step, service, provider, endpoint, http_method, http_status, duration, retry_count, "
            "application_version, host, error type/code/message/stack_summary. Not every field is "
            "mandatory; anything missing is labeled '" + NOT_AVAILABLE + "', never invented.\n"
            "- **Limitations of the investigation** — the AI/rule-based generator only ever sees the "
            "deterministic, redacted evidence for the exception you selected, never your raw upload, and it "
            "never calculates or overrides a metric itself."
        )
