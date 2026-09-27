from src.fingerprinting import compute_fingerprint, normalize_dynamic_value


def _row(**overrides):
    base = {
        "workflow_name": "WhatsApp Lead Intake",
        "workflow_step": "Send WhatsApp Message",
        "service": "whatsapp_gateway",
        "endpoint": "/messages/send",
        "error_type": "API_TIMEOUT",
        "error_code": "GATEWAY_TIMEOUT",
        "http_status": 504,
        "error_message": "Provider did not respond within 15 seconds",
    }
    base.update(overrides)
    return base


# --- normalize_dynamic_value ---------------------------------------------------


def test_normalize_strips_uuid():
    text = "Failed to process request 8f14e45f-ceea-4a5c-8f2b-5b8f3a7a9c11"
    assert "<uuid>" in normalize_dynamic_value(text)
    assert "8f14e45f" not in normalize_dynamic_value(text)


def test_normalize_strips_iso_timestamp():
    text = "Request started at 2026-09-20T10:17:00Z and timed out"
    assert "<timestamp>" in normalize_dynamic_value(text)
    assert "2026-09-20" not in normalize_dynamic_value(text)


def test_normalize_strips_ref_style_ids():
    text = "Correlation CORR-83921 for execution EXE-1002 failed"
    normalized = normalize_dynamic_value(text)
    assert "CORR-83921".lower() not in normalized
    assert "EXE-1002".lower() not in normalized
    assert normalized.count("<ref_id>") == 2


def test_normalize_strips_long_numeric_record_ids():
    text = "Order 8842137 could not be found"
    normalized = normalize_dynamic_value(text)
    assert "8842137" not in normalized
    assert "<number>" in normalized


def test_normalize_leaves_short_numbers_alone():
    # "15 seconds" and an http-style 3-digit code should NOT be treated as
    # record-identifying — only 4+ digit runs are.
    text = "Provider did not respond within 15 seconds (504)"
    normalized = normalize_dynamic_value(text)
    assert "15 seconds" in normalized
    assert "504" in normalized


def test_normalize_strips_long_hex_tokens():
    text = "Trace id a1b2c3d4e5f6a7b8c9d0e1f2 attached"
    normalized = normalize_dynamic_value(text)
    assert "a1b2c3d4e5f6a7b8c9d0e1f2" not in normalized
    assert "<hex>" in normalized


def test_normalize_handles_empty_text():
    assert normalize_dynamic_value("") == ""
    assert normalize_dynamic_value(None) == ""


def test_normalize_is_case_insensitive_for_grouping():
    assert normalize_dynamic_value("Timeout Error") == normalize_dynamic_value("timeout error")


# --- compute_fingerprint --------------------------------------------------------


def test_same_exception_different_ids_produces_same_fingerprint():
    a = _row(error_message="Correlation CORR-83921: provider timed out after 15000ms")
    b = _row(error_message="Correlation CORR-91004: provider timed out after 15000ms")

    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_different_error_type_produces_different_fingerprint():
    a = _row(error_type="API_TIMEOUT")
    b = _row(error_type="AUTHENTICATION_ERROR")

    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_different_service_produces_different_fingerprint():
    a = _row(service="whatsapp_gateway")
    b = _row(service="invoice_service")

    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_different_workflow_step_produces_different_fingerprint():
    a = _row(workflow_step="Send WhatsApp Message")
    b = _row(workflow_step="Validate Recipient")

    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_http_status_alone_does_not_split_the_fingerprint():
    """http_status is deliberately NOT a fingerprint field — the same
    underlying issue can surface slightly different status codes across
    occurrences (504 from one proxy hop vs. 502 from another), and it's
    tracked as a per-group distribution instead (see exception_groups.py)."""
    a = _row(http_status=504)
    b = _row(http_status=500)

    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_missing_optional_fields_still_produce_a_stable_fingerprint():
    row = {
        "workflow_name": "Invoice Parser",
        "workflow_step": None,
        "service": None,
        "endpoint": None,
        "error_type": "INVALID_JSON_SCHEMA",
        "error_code": "",
        "http_status": float("nan"),
        "error_message": "Required field missing",
    }
    fingerprint = compute_fingerprint(row)

    assert isinstance(fingerprint, str) and len(fingerprint) > 0
    assert fingerprint == compute_fingerprint(dict(row))  # deterministic, repeatable


def test_fingerprint_is_deterministic_across_calls():
    row = _row()
    assert compute_fingerprint(row) == compute_fingerprint(row) == compute_fingerprint(dict(row))
