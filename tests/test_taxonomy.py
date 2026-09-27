from src.taxonomy import UNCLASSIFIED, UNCLASSIFIED_SECONDARY, classify_exception


def test_api_timeout_classifies_as_timeout_and_latency():
    primary, secondary = classify_exception("API_TIMEOUT", "GATEWAY_TIMEOUT", 504, "Provider did not respond within 15 seconds")
    assert primary == "Timeout and latency"
    assert secondary == "Gateway timeout"


def test_authentication_error_classifies_as_auth():
    primary, secondary = classify_exception("AUTHENTICATION_ERROR", "", 401, "Invalid or expired token")
    assert primary == "Authentication and authorization"
    assert secondary == "Invalid or expired token"


def test_invalid_json_schema_classifies_as_ai_structured_output_not_generic_validation():
    primary, secondary = classify_exception("INVALID_JSON_SCHEMA", "", None, "Response payload did not match the expected JSON schema")
    assert primary == "AI and structured-output"


def test_duplicate_record_classifies_as_data_integrity():
    primary, secondary = classify_exception("DUPLICATE_RECORD", "", 409, "Record already exists")
    assert primary == "Data integrity and persistence"
    assert secondary == "Duplicate record"


def test_http_429_classifies_as_rate_limiting_even_without_matching_error_type_text():
    primary, secondary = classify_exception("UNKNOWN_ERROR", "", 429, "Too many requests")
    assert primary == "Rate limiting and capacity"
    assert secondary == "HTTP 429"


def test_retry_exhaustion_is_not_absorbed_by_timeout_rule():
    """The message mentions 'timeout' but the error itself is about running
    out of retries — must classify as Retry/idempotency, not Timeout."""
    primary, secondary = classify_exception("RETRY_EXHAUSTED", "MAX_RETRIES", None, "Gave up after repeated timeout retries")
    assert primary == "Retry and idempotency"
    assert secondary == "Retry exhaustion"


def test_missing_required_field_classifies_as_input_validation():
    primary, secondary = classify_exception("VALIDATION_ERROR", "REQUIRED_FIELD_MISSING", None, "Field 'invoice_number' is required")
    assert primary == "Input and validation"


def test_unclassified_when_nothing_matches():
    primary, secondary = classify_exception("SOME_NOVEL_ERROR", "", None, "Something unexpected happened")
    assert primary == UNCLASSIFIED
    assert secondary == UNCLASSIFIED_SECONDARY


def test_classification_is_case_insensitive():
    a = classify_exception("api_timeout", "gateway_timeout", 504, "provider timed out")
    b = classify_exception("API_TIMEOUT", "GATEWAY_TIMEOUT", 504, "PROVIDER TIMED OUT")
    assert a == b


def test_nan_http_status_does_not_crash():
    primary, secondary = classify_exception("API_TIMEOUT", "", float("nan"), "timed out")
    assert primary == "Timeout and latency"


def test_internal_http_500_falls_back_to_internal_application_error():
    primary, secondary = classify_exception("UNKNOWN", "", 500, "")
    assert primary == "Internal application error"
    assert secondary == "Internal HTTP 500"


def test_connection_refused_classifies_as_connectivity():
    primary, secondary = classify_exception("CONNECTION_REFUSED", "", None, "ECONNREFUSED connecting to gateway")
    assert primary == "Connectivity and dependency"
    assert secondary == "Connection refused"


def test_pascal_case_error_type_classifies_same_as_snake_case():
    """Regression test for a bug caught via live-app testing: a keyword
    like 'JSON_SCHEMA' never matched a PascalCase error_type like
    'InvalidJSONSchemaError' because the keyword has an underscore the
    error_type doesn't — both must hit the same rule regardless of naming
    convention."""
    snake = classify_exception("INVALID_JSON_SCHEMA", "", None, "Response payload did not match the expected JSON schema.")
    pascal = classify_exception("InvalidJSONSchemaError", "", None, "Response payload did not match the expected JSON schema.")

    assert snake == pascal
    assert pascal[0] != "Unclassified"


def test_pascal_case_authentication_and_duplicate_errors_classify_correctly():
    auth_primary, _ = classify_exception("AuthenticationError", "", None, "invalid or expired credentials")
    dup_primary, _ = classify_exception("DuplicateRecordError", "", None, "record already exists")

    assert auth_primary == "Authentication and authorization"
    assert dup_primary == "Data integrity and persistence"


# --- real-platform vocabulary (Stripe, WhatsApp, Twilio, OpenAI, n8n, Slack) ------
# Regression tests added after generating realistic multi-tool demo data and
# finding ~40% of exception groups fell into Unclassified — the earlier
# keyword rules were built against this project's own SNAKE_CASE codes, not
# real platforms' actual error vocabulary.


def test_stripe_card_decline_classifies_as_business_rule_failure():
    primary, secondary = classify_exception("card_error", "card_declined", 402, "Your card was declined.")
    assert primary == "Business-rule failure"
    assert secondary == "Ineligible operation"


def test_stripe_missing_param_classifies_as_missing_required_field():
    primary, secondary = classify_exception("invalid_request_error", "parameter_missing", 400, "Missing required param: payment_method.")
    assert primary == "Input and validation"
    assert secondary == "Missing required field"


def test_whatsapp_undeliverable_classifies_as_connectivity():
    primary, _ = classify_exception("MESSAGE_UNDELIVERABLE", "131026", 400, "Message Undeliverable - the recipient phone number is not a WhatsApp user.")
    assert primary == "Connectivity and dependency"


def test_whatsapp_reengagement_window_classifies_as_business_rule():
    primary, secondary = classify_exception(
        "RE_ENGAGEMENT_WINDOW_EXPIRED", "131047", 400, "more than 24 hours have passed since the customer last replied"
    )
    assert primary == "Business-rule failure"
    assert secondary == "Invalid workflow state"


def test_whatsapp_template_param_mismatch_classifies_as_schema_mismatch():
    primary, secondary = classify_exception("TEMPLATE_PARAM_MISMATCH", "132000", 400, "does not match the number of variables in the approved template")
    assert primary == "Input and validation"
    assert secondary == "Schema mismatch"


def test_twilio_unsubscribed_recipient_classifies_as_business_constraint():
    primary, secondary = classify_exception("UNSUBSCRIBED_RECIPIENT", "21610", 400, "the recipient has replied STOP and opted out")
    assert primary == "Business-rule failure"
    assert secondary == "Business constraint violation"


def test_twilio_unreachable_handset_classifies_as_connectivity():
    primary, _ = classify_exception("UNREACHABLE_HANDSET", "30003", 400, "the destination handset is switched off or out of coverage")
    assert primary == "Connectivity and dependency"


def test_openai_context_length_classifies_as_response_truncation():
    primary, secondary = classify_exception(
        "context_length_exceeded", "context_length_exceeded", 400, "This model's maximum context length is 128000 tokens."
    )
    assert primary == "AI and structured-output"
    assert secondary == "Response truncation"


def test_openai_content_policy_classifies_as_model_refusal():
    primary, secondary = classify_exception(
        "content_policy_violation", "content_filter", 400, "rejected as a result of our safety system"
    )
    assert primary == "AI and structured-output"
    assert secondary == "Model refusal"


def test_n8n_json_parse_error_classifies_as_invalid_json():
    primary, secondary = classify_exception("NodeOperationError", "JSON_PARSE_ERROR", None, "Unexpected token '<' is not valid JSON")
    assert primary == "AI and structured-output"
    assert secondary == "Invalid JSON"


def test_n8n_channel_not_found_classifies_as_invalid_configuration():
    primary, secondary = classify_exception("NodeApiError", "SLACK_CHANNEL_NOT_FOUND", 404, "channel_not_found")
    assert primary == "Configuration and deployment"
    assert secondary == "Invalid configuration"


def test_google_sheets_stale_range_classifies_as_invalid_configuration():
    primary, secondary = classify_exception(
        "GOOGLE_SHEETS_RANGE_ERROR", "INVALID_ARGUMENT", 400, "Unable to parse range: Sheet1!A:L — the sheet or range no longer exists."
    )
    assert primary == "Configuration and deployment"
    assert secondary == "Invalid configuration"
