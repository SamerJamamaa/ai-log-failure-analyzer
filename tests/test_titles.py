import pandas as pd

from src.exception_groups import build_exception_groups
from src.models import OPTIONAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS
from src.titles import human_title


def _row(i, error_type, http_status=None, version="", service="", provider="", workflow="WF", retry_count=None, duration=2.0):
    row = {c: (pd.NA if c in OPTIONAL_NUMERIC_COLUMNS else "") for c in OPTIONAL_COLUMNS}
    row.update(
        execution_id=f"e{i}",
        workflow_name=workflow,
        start_time=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
        status="failure",
        duration_seconds=duration,
        error_type=error_type,
        error_message=f"{error_type} occurred",
        http_status=http_status,
        application_version=version,
        service=service,
        provider=provider,
        retry_count=retry_count,
    )
    return row


def _first_group(rows):
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    return build_exception_groups(df)[0]


def test_authentication_title_mentions_version_when_present():
    group = _first_group([_row(i, "AUTHENTICATION_ERROR", http_status=401, version="3.5.0", service="crm-service") for i in range(3)])
    title = human_title(group)
    assert "crm-service" in title
    assert "3.5.0" in title
    assert "AUTHENTICATION_ERROR" not in title


def test_authentication_title_403_reads_as_denied_not_rejected():
    group = _first_group([_row(i, "PERMISSION_DENIED", http_status=403, service="crm-service") for i in range(3)])
    title = human_title(group)
    assert "403" in title


def test_timeout_title_mentions_duration():
    group = _first_group([_row(i, "API_TIMEOUT", provider="whatsapp", duration=30.0) for i in range(3)])
    title = human_title(group)
    assert "whatsapp" in title
    assert "timing out" in title


def test_duplicate_title_mentions_workflow():
    group = _first_group([_row(i, "DUPLICATE_RECORD", http_status=409, workflow="CRM Sync") for i in range(3)])
    title = human_title(group)
    assert "Duplicate" in title
    assert "CRM Sync" in title


def test_unclassified_falls_back_to_plain_sentence_not_blank():
    group = _first_group([_row(i, "SOME_NOVEL_ERROR_CODE", workflow="WF") for i in range(3)])
    title = human_title(group)
    assert "WF" in title
    assert title  # never empty


def test_business_rule_title_names_the_specific_secondary_category():
    """Regression test: the business-rule title used to say the identical,
    uninformative 'violating a business rule' for every distinct scenario
    (card decline, re-engagement window, unsubscribed recipient) — must
    name the actual secondary category instead."""
    group = _first_group([_row(i, "RE_ENGAGEMENT_WINDOW_EXPIRED", http_status=400, workflow="WhatsApp Sender") for i in range(3)])
    title = human_title(group)
    assert "violating a business rule" not in title
    assert group.secondary_category in title


def test_title_never_contains_raw_underscore_error_code():
    """The whole point of this module: no queue row should show a raw
    SNAKE_CASE code as its title."""
    group = _first_group([_row(i, "API_TIMEOUT", provider="whatsapp") for i in range(3)])
    title = human_title(group)
    assert "_" not in title


def test_title_prefers_provider_over_a_generic_orchestrator_service():
    """Regression test: for an orchestrated workflow (e.g. n8n), 'service'
    is often the generic orchestrator itself for every node ('n8n-worker'),
    while 'provider' names the actual external system that failed — the
    title must surface the more specific, useful name."""
    group = _first_group(
        [_row(i, "AUTHENTICATION_ERROR", http_status=401, service="n8n-worker", provider="google-sheets") for i in range(3)]
    )
    title = human_title(group)
    assert "google-sheets" in title
    assert "n8n-worker" not in title
