"""Deterministic, incident-level investigation brief — Section 2 of the
console, shown directly below the KPI cards. Built entirely from
already-computed exception groups and dataset-wide aggregates; never an
LLM call, so the console stays fully useful with no API key configured.

Reuses the same category playbooks the per-exception "Generate
Investigation" panel uses (src/llm/offline.py) for the top few groups'
explanations and immediate actions, so the brief and the detailed
per-exception investigation are always consistent with each other.
"""

from src.aggregates import compute_period_comparison
from src.evidence import build_evidence_table, missing_evidence_for
from src.llm.base import build_exception_evidence
from src.llm.offline import generate_offline_investigation
from src.models import IncidentBrief
from src.titles import human_title

MAX_SECONDARY_INCIDENTS = 3
BUSINESS_IMPACT_DISCLAIMER = "Technical impact is visible, but business impact cannot be determined from the supplied logs."


def _explain(group, df):
    items = build_evidence_table(group, df)
    missing = missing_evidence_for(group)
    evidence = build_exception_evidence(group, items, missing)
    return generate_offline_investigation(evidence)


def _time_version_relationship(top, df) -> str:
    """Only claims a version correlation when the workflow actually has
    OTHER versions in the data to contrast against — a single-version
    workflow has nothing to correlate with, even if that one version
    'contains' every failure (trivially true, not evidence of anything)."""
    if len(top.application_versions) != 1:
        return "No single-version correlation is evident for the highest-priority exception from the supplied logs."

    version = top.application_versions[0]
    workflow_versions = {v for v in df[df["workflow_name"] == top.workflow_name]["application_version"] if v}
    if len(workflow_versions) < 2:
        return (
            f"All occurrences of the highest-priority exception ({human_title(top)}) are on version "
            f"{version}, but no other version of '{top.workflow_name}' appears in this upload to compare "
            "against — no version relationship can be established from the supplied logs."
        )
    return (
        f"All occurrences of the highest-priority exception ({human_title(top)}) are on version "
        f"{version}, while other versions of '{top.workflow_name}' are also present in this upload. This "
        "is a correlation observed in the data, not confirmed causation — a configuration or code diff "
        "would be needed to confirm the version caused the regression."
    )


def _healthy_brief(total_executions: int, total_failures: int) -> IncidentBrief:
    return IncidentBrief(
        what_happened=(
            f"{total_executions} execution(s) were analyzed and {total_failures} failed. "
            "No exception pattern in this upload warranted investigation."
        ),
        primary_incident="None — no exception groups were found.",
        secondary_incidents=[],
        systems_affected=[],
        time_version_relationship="Not applicable — no exceptions were found.",
        technical_impact="No investigation-worthy technical impact was found in the supplied logs.",
        likely_explanations=[],
        immediate_actions=["Continue routine monitoring."],
        limitations=[],
    )


def generate_incident_brief(groups: list, df, total_executions: int, total_failures: int) -> IncidentBrief:
    if not groups:
        return _healthy_brief(total_executions, total_failures)

    top = groups[0]
    secondary = groups[1 : 1 + MAX_SECONDARY_INCIDENTS]

    period = compute_period_comparison(df)
    period_sentence = ""
    if period.get("valid") and period["change_pp"] > 0:
        period_sentence = (
            f" The failure rate rose from {period['first_period_failure_rate']}% to "
            f"{period['second_period_failure_rate']}% between the first and second half of this upload "
            f"({period['first_period_count']} vs. {period['second_period_count']} executions compared)."
        )

    what_happened = (
        f"{total_failures} failed execution(s) were grouped into {len(groups)} distinct exception "
        f"pattern(s)." + period_sentence
    )

    top_investigation = _explain(top, df)
    primary_incident = f"{human_title(top)} — {top.priority_reason}"

    secondary_incidents = [
        f"{human_title(g)} ({g.occurrence_count} occurrence(s), {g.priority_label} priority)" for g in secondary
    ]
    remaining = len(groups) - 1 - len(secondary)
    if remaining > 0:
        secondary_incidents.append(f"...and {remaining} more exception pattern(s) in the queue below.")

    systems_affected = sorted({g.workflow_name for g in groups} | {g.service for g in groups if g.service})

    time_version_relationship = _time_version_relationship(top, df)

    technical_impact = (
        f"{sum(g.affected_execution_count for g in groups)} execution(s) across {len(systems_affected)} "
        f"system(s)/workflow(s) were directly affected. {BUSINESS_IMPACT_DISCLAIMER}"
    )

    likely_explanations = [top_investigation.exception_explanation]
    for g in secondary:
        likely_explanations.append(_explain(g, df).exception_explanation)

    immediate_actions = [a.action for a in top_investigation.containment_actions]
    if top_investigation.investigation_actions:
        immediate_actions.append(top_investigation.investigation_actions[0].action)
    if not immediate_actions:
        immediate_actions = ["No immediate containment is required based on current evidence; proceed to investigation."]

    limitations = list(top_investigation.limitations)
    if remaining > 0:
        limitations.append(
            f"This brief covers only the top {1 + len(secondary)} of {len(groups)} exception patterns — "
            "see the full exception landscape below for the rest."
        )

    return IncidentBrief(
        what_happened=what_happened,
        primary_incident=primary_incident,
        secondary_incidents=secondary_incidents,
        systems_affected=systems_affected,
        time_version_relationship=time_version_relationship,
        technical_impact=technical_impact,
        likely_explanations=likely_explanations,
        immediate_actions=immediate_actions,
        limitations=limitations,
    )
