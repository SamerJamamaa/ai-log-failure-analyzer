from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

REQUIRED_COLUMNS = [
    "execution_id",
    "workflow_name",
    "start_time",
    "status",
    "duration_seconds",
    "error_type",
    "error_message",
]

# Richer technical context. Present when the source system provides it,
# never required — ingestion fills sensible defaults ("" / NaN) when absent,
# for both JSON and CSV, so nothing downstream has to special-case a
# missing optional field. error_code/error_stack_summary come from the
# nested `error` object in JSON; the rest are top-level fields.
OPTIONAL_TEXT_COLUMNS = [
    "environment",
    "workflow_step",
    "service",
    "endpoint",
    "http_method",
    "correlation_id",
    "application_version",
    "provider",
    "error_code",
    "error_stack_summary",
]
OPTIONAL_NUMERIC_COLUMNS = ["http_status", "retry_count"]
OPTIONAL_COLUMNS = OPTIONAL_TEXT_COLUMNS + OPTIONAL_NUMERIC_COLUMNS
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

VALID_STATUSES = {"success", "failure"}


@dataclass
class RowRejection:
    """One invalid data row. row_number matches how the row appears in a
    spreadsheet application (row 1 is the header, so the first data row is
    row 2) — that's what a user checking their file against the error
    message will actually see."""

    row_number: int
    reasons: list[str]

    def message(self) -> str:
        return f"Row {self.row_number}: {', '.join(self.reasons)}"


@dataclass
class ValidationReport:
    """Result of ingesting a CSV: how many rows were usable and why others weren't."""

    total_rows: int
    valid_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejected_row_details: list[RowRejection] = field(default_factory=list)
    fatal_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.fatal_error is None


@dataclass
class WorkflowKPI:
    workflow_name: str
    total_executions: int
    success_rate: float
    failure_count: int
    avg_duration_seconds: float
    p95_duration_seconds: float


@dataclass
class KPISet:
    total_executions: int
    success_rate: float
    failure_count: int
    avg_duration_seconds: float
    p95_duration_seconds: float
    most_common_error_type: Optional[str]
    by_workflow: list[WorkflowKPI]


AnomalyType = Literal["slow_execution", "recurring_error", "failure_rate_increase"]


@dataclass
class Anomaly:
    type: AnomalyType
    workflow_name: str
    summary: str
    evidence: dict


TrendLabel = Literal["isolated", "recurring", "increasing"]


@dataclass
class ExceptionGroup:
    """One deterministically-fingerprinted cluster of related failures —
    the unit the whole investigation dashboard is built around. Every field
    here is computed by src/exception_groups.py from the validated data;
    nothing on this object is ever produced or altered by an LLM.
    """

    fingerprint: str
    primary_category: str
    secondary_category: Optional[str]

    # identity
    error_type: str
    error_code: str
    workflow_name: str
    workflow_step: str
    service: str
    endpoint: str
    http_method: str
    http_status: Optional[int]
    provider: str
    environment: str
    representative_message: str

    # occurrence
    occurrence_count: int
    affected_execution_count: int
    first_occurrence: Any  # pandas.Timestamp
    last_occurrence: Any  # pandas.Timestamp
    percentage_of_all_failures: float
    percentage_of_workflow_executions: float
    trend_label: TrendLabel
    is_increasing: bool

    # impact
    avg_duration_seconds: float
    p95_duration_seconds: float
    http_status_distribution: dict = field(default_factory=dict)
    retry_info: dict = field(default_factory=dict)
    application_versions: list = field(default_factory=list)
    hosts_affected: list = field(default_factory=list)  # always [] today — no host field in the input schema
    workflows_sharing_this_issue: int = 1  # distinct workflows seeing the same (service, error_type)

    # evidence
    representative_execution_ids: list = field(default_factory=list)
    representative_correlation_ids: list = field(default_factory=list)

    # priority (filled in by src/prioritization.py, not at construction time)
    priority_score: float = 0.0
    priority_label: str = "Low"
    priority_factors: dict = field(default_factory=dict)
    priority_reason: str = ""

    # the actual matching rows, for the Affected Executions view. Excluded
    # from equality/repr — comparing DataFrames with == doesn't return a
    # plain bool, which would break dataclass-generated __eq__.
    rows: Any = field(default=None, repr=False, compare=False)


class IncidentReport(BaseModel):
    """The LLM's (or offline fallback's) structured incident analysis.

    Built with pydantic so the shape coming back from an LLM call is
    actually validated (required fields, list types, and a closed severity/
    confidence vocabulary) rather than just cast with str()/list().
    """

    incident_title: str
    severity: Literal["low", "medium", "high"]
    observed_evidence: list[str]
    probable_causes: list[str]
    recommended_actions: list[str]
    confidence: Literal["low", "medium", "high"]
    limitations: str
    source: Literal["llm", "offline"] = "llm"


@dataclass
class EvidenceItem:
    """One row of the exception's evidence table (section D) — a single
    factual, traceable observation. Built entirely by src/evidence.py from
    already-computed ExceptionGroup data; never produced by an LLM. Every
    factual claim shown to the user should trace back to one of these.
    """

    evidence_id: str
    source_field: str
    observed_value: str
    occurrence_count: int
    why_it_matters: str
    representative_execution_ids: list = field(default_factory=list)
    representative_correlation_ids: list = field(default_factory=list)


class Hypothesis(BaseModel):
    """One ranked probable-cause hypothesis. Never a confirmed root cause —
    evidence IDs tie it back to what was actually observed, and
    missing_evidence names what would be needed to actually confirm it."""

    rank: int
    hypothesis: str
    confidence: Literal["low", "medium", "high"]
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    missing_evidence: list[str]
    verification_steps: list[str]


class ContainmentAction(BaseModel):
    priority: int
    action: str
    reason: str
    risk: str


class InvestigationAction(BaseModel):
    """One concrete next step, with the evidence it's based on and what
    each possible finding would mean — never a vague 'look into it'."""

    priority: int
    action: str
    component: str
    evidence_reference: list[str]
    expected_findings: list[str]
    decision_from_result: str


class RemediationOption(BaseModel):
    """One of several possible fixes — never presented as the single
    correct answer; complexity/risk/tradeoffs are mandatory, not optional."""

    option: str
    appropriate_when: str
    expected_benefit: str
    risks: list[str]
    tradeoffs: list[str]
    complexity: Literal["low", "medium", "high"]
    change_type: Literal["code", "configuration", "infrastructure", "provider"]
    temporary_or_permanent: Literal["temporary", "permanent"]


class VerificationStep(BaseModel):
    test: str
    expected_result: str
    success_metric: str
    success_threshold: str
    observation_period: str
    rollback_condition: str


class DiagnosticCommand(BaseModel):
    """One example diagnostic command/query — always a placeholder-based
    template (never a real credential/value), clearly labeled, with what to
    look for and how each possible finding should steer the investigation."""

    label: str
    command: str
    look_for: list[str]
    interpretation: list[str]


class ExceptionInvestigation(BaseModel):
    """The LLM's (or offline fallback's) structured investigation of ONE
    exception group. Built with pydantic so the shape coming back from an
    LLM call is actually validated rather than just cast with str()/list().

    This is the sole AI-facing output schema for the exception-investigation
    dashboard: exception_explanation/known_facts/unknowns interpret already-
    computed evidence; hypotheses are ranked and evidence-linked, never
    presented as confirmed; containment/investigation/remediation/
    verification/prevention mirror the 5-stage action plan the dashboard
    renders. See src/llm/base.py's SYSTEM_PROMPT for the behavioral rules
    (may/must-not) this schema is meant to enforce.
    """

    exception_explanation: str
    known_facts: list[str]
    unknowns: list[str]
    hypotheses: list[Hypothesis]
    containment_actions: list[ContainmentAction]
    investigation_actions: list[InvestigationAction]
    remediation_options: list[RemediationOption]
    verification_plan: list[VerificationStep]
    prevention_actions: list[str]
    limitations: list[str]
    diagnostic_commands: list[DiagnosticCommand] = Field(default_factory=list)
    source: Literal["llm", "offline"] = "llm"


class IncidentBrief(BaseModel):
    """The one-page, incident-level investigation brief shown directly
    below the KPI cards (Section 2 of the console). Always built
    deterministically from the already-computed exception groups (see
    src/incident_brief.py) — never an LLM call, so the console stays fully
    useful without an API key. Reuses the same category playbooks the
    per-exception investigation panel uses, so the two never contradict
    each other.
    """

    what_happened: str
    primary_incident: str
    secondary_incidents: list[str]
    systems_affected: list[str]
    time_version_relationship: str
    technical_impact: str
    likely_explanations: list[str]
    immediate_actions: list[str]
    limitations: list[str]
    source: Literal["deterministic"] = "deterministic"
