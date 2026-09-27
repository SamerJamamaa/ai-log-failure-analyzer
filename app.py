import os

import streamlit as st
from dotenv import load_dotenv

from src import db, ingestion, metrics
from src.evidence import build_evidence_table, missing_evidence_for
from src.exception_groups import build_exception_groups
from src.incident_brief import generate_incident_brief
from src.llm.base import active_provider_label, build_exception_evidence, get_client
from src.prioritization import prioritize_groups
from ui import components

load_dotenv()

st.set_page_config(page_title="AI Exception Investigation Console", layout="wide", page_icon="🔍")

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_workflow_logs.csv")


def _run_analysis(source) -> None:
    valid_df, report = ingestion.load_and_validate(source)
    st.session_state["validation_report"] = report
    st.session_state["investigations"] = {}
    if not report.ok or report.valid_rows == 0:
        st.session_state.pop("data", None)
        return
    conn = db.create_connection()
    db.replace_executions(conn, valid_df)
    groups = prioritize_groups(build_exception_groups(valid_df))
    st.session_state["data"] = {
        "df": valid_df,
        "kpis": metrics.compute_kpis(conn),
        "groups": groups,
    }


st.title("AI Exception Investigation Console")
st.caption(
    "Upload your workflow execution logs for a single-page, evidence-based investigation of what "
    "failed, where, when it started, and what to do next — every fact calculated directly from your "
    "data, with hypotheses always clearly separated from confirmed conclusions."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload execution log",
        type=["json", "csv"],
        help="JSON is the recommended format — it matches a realistic ingestion pipeline "
        "and captures richer technical detail (service, endpoint, HTTP status, retries, "
        "application version). CSV is also supported, mainly for quick manual testing.",
    )
    if uploaded is not None:
        st.session_state["pending_source"] = ("upload", uploaded, uploaded.name)
    if st.button("Load Sample Data", width="stretch"):
        st.session_state["pending_source"] = ("sample", SAMPLE_DATA_PATH, "sample dataset")

    if "pending_source" in st.session_state:
        # label is captured at selection time, not re-derived from `uploaded` here —
        # `uploaded` becomes None if the user clears the widget after selecting a file,
        # which would otherwise crash this block on the next rerun
        _kind, source, label = st.session_state["pending_source"]
        st.caption("Ready to analyze:")
        st.text(label)  # plain text, never markdown — a filename is user-controlled input
        if st.button("Analyze", type="primary", width="stretch"):
            with st.spinner("Validating and analyzing..."):
                _run_analysis(source)

    st.divider()
    st.caption("This app never displays or logs your API key.")

if "validation_report" in st.session_state:
    components.render_validation_report(st.session_state["validation_report"])

if "data" not in st.session_state:
    st.info("Upload a JSON or CSV log, or click **Load Sample Data** then **Analyze** in the sidebar to begin.")
    st.stop()

data = st.session_state["data"]
df, kpis, groups = data["df"], data["kpis"], data["groups"]
total_failures = int((df["status"] == "failure").sum())

# --- Section 1: Incident Overview -------------------------------------------------
components.render_incident_overview(groups, df, len(df), total_failures, kpis)

st.divider()

# --- Section 2: AI Investigation Brief ---------------------------------------------
brief = generate_incident_brief(groups, df, len(df), total_failures)
components.render_investigation_brief(brief)

if not groups:
    st.divider()
    components.render_how_to_read_expander()
    st.stop()

st.divider()

# --- Section 3: Exception Landscape -------------------------------------------------
selected_group = components.render_exception_landscape(groups)

st.divider()

# --- Section 4: Selected Exception Investigation (single continuous page) -----------
investigations = st.session_state.setdefault("investigations", {})
existing_investigation = investigations.get(selected_group.fingerprint)

components.render_exception_summary(selected_group, existing_investigation)
components.render_why_it_matters(selected_group)
components.render_evidence_found(selected_group, df)

if components.render_investigation_trigger(active_provider_label(), existing_investigation):
    items = build_evidence_table(selected_group, df)
    missing = missing_evidence_for(selected_group)
    evidence = build_exception_evidence(selected_group, items, missing)
    client = get_client()
    with st.spinner("Analyzing evidence..."):
        report = client.analyze(evidence)
    investigations[selected_group.fingerprint] = report
    existing_investigation = report

components.render_probable_causes(existing_investigation)
components.render_action_plan(existing_investigation)
components.render_affected_executions(selected_group)

st.divider()

# --- Section 5: Operational Visuals -------------------------------------------------
components.render_operational_visuals(groups, df)

st.divider()
components.render_how_to_read_expander()
