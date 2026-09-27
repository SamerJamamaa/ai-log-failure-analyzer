"""Generates a realistic execution log for the user's own real n8n workflow
("Car insurance and fraud detection.json"): a webhook receives a new
insurance claim, fans out to two GPT-5 agents in parallel — one that saves
the claim to Google Sheets and follows up with the customer over WhatsApp
(via Green API), and one that runs a fraud-risk assessment and emails it
via Gmail. Modeled on each real integration's own documented error
conventions (OpenAI, Google Sheets API, Green API, Gmail API) — not this
project's own earlier synthetic codes.

Reuses the generation engine from generate_realistic_tool_logs.py.

Run it yourself:
    python scripts/generate_claim_fraud_workflow_log.py
"""

import json
import os
import sys

# Lets `python scripts/generate_claim_fraud_workflow_log.py` resolve the
# `scripts.generate_realistic_tool_logs` import regardless of invocation
# style (direct script run vs. `python -m scripts.generate_...`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_realistic_tool_logs import (  # noqa: E402
    _build_tool_log,
    _to_csv_dataframe,
    _to_json_record,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "realistic_tool_logs")
WORKFLOW_NAME = "Car Insurance and Fraud Detection"

STEPS = [
    {
        "workflow_name": WORKFLOW_NAME,
        "workflow_step": "Webhook - A new Claim Received",
        "service": "n8n-worker",
        "provider": "internal",
        "endpoint": "/webhook/82f6341a-c67b-41ba-9f4d-d1e0fe26448c",
        "http_method": "POST",
        "application_version": "1.44.0",
        "success_status": 200,
    },
    {
        "workflow_name": WORKFLOW_NAME,
        "workflow_step": "Fraud Analyst (GPT-5 agent)",
        "service": "ai-agent-service",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
        "http_method": "POST",
        "application_version": "gpt-5",
        "success_status": 200,
    },
    {
        "workflow_name": WORKFLOW_NAME,
        "workflow_step": "Follow up with the customer (GPT-5 agent)",
        "service": "ai-agent-service",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
        "http_method": "POST",
        "application_version": "gpt-5",
        "success_status": 200,
    },
    {
        "workflow_name": WORKFLOW_NAME,
        "workflow_step": "Save the Claim Details",
        "service": "n8n-worker",
        "provider": "google-sheets",
        "endpoint": "/v4/spreadsheets/1LIhWIeUKOYxyqAPG2Ct8FeQOi9oFoZwjlzJO9x3avWY/values/Sheet1:append",
        "http_method": "POST",
        "application_version": "1.44.0",
        "success_status": 200,
    },
    {
        "workflow_name": WORKFLOW_NAME,
        "workflow_step": "HTTP Request - send a message on Whatsapp",
        "service": "n8n-worker",
        "provider": "green-api",
        "endpoint": "/waInstance7103540721/sendMessage",
        "http_method": "POST",
        "application_version": "1.44.0",
        "success_status": 200,
    },
    {
        "workflow_name": WORKFLOW_NAME,
        "workflow_step": "Send an Email - Fraud analysis",
        "service": "n8n-worker",
        "provider": "gmail",
        "endpoint": "/gmail/v1/users/me/messages/send",
        "http_method": "POST",
        "application_version": "1.44.0",
        "success_status": 200,
    },
]

_BOTH_AGENTS = ["Fraud Analyst (GPT-5 agent)", "Follow up with the customer (GPT-5 agent)"]

CATALOG = [
    # OpenAI (GPT-5) — shared by both agents, since both call the same model
    {
        "weight": 14,
        "error_type": "rate_limit_exceeded",
        "error_code": "rate_limit_exceeded",
        "message": "Rate limit reached for gpt-5 in organization org-fnx01 on requests per min (RPM): Limit 500, Used 500, Requested 1.",
        "http_status": 429,
        "retryable": True,
        "step": _BOTH_AGENTS,
    },
    {
        "weight": 6,
        "error_type": "insufficient_quota",
        "error_code": "insufficient_quota",
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "http_status": 429,
        "retryable": False,
        "step": _BOTH_AGENTS,
    },
    {
        "weight": 8,
        "error_type": "TOOL_CALL_VALIDATION_ERROR",
        "error_code": "tool_call_invalid_arguments",
        "message": "Agent tool call to 'Save the Claim Details' failed validation: $fromAI('Total_Claim_Amount') returned an empty string for a required column.",
        "http_status": 200,
        "retryable": False,
        "step": "Follow up with the customer (GPT-5 agent)",
    },
    {
        "weight": 4,
        "error_type": "content_policy_violation",
        "error_code": "content_filter",
        "message": "Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our usage policies.",
        "http_status": 400,
        "retryable": False,
        "step": _BOTH_AGENTS,
    },
    {
        "weight": 4,
        "error_type": "server_error",
        "error_code": "server_error",
        "message": "The server had an error processing your request. Sorry about that! You can retry your request.",
        "http_status": 503,
        "retryable": True,
        "step": _BOTH_AGENTS,
    },
    # Google Sheets — Save the Claim Details
    {
        "weight": 9,
        "error_type": "GOOGLE_SHEETS_AUTH_ERROR",
        "error_code": "invalid_grant",
        "message": "OAuth2 token for 'Google Sheets account - fnx' has expired or been revoked.",
        "http_status": 401,
        "retryable": False,
        "step": "Save the Claim Details",
    },
    {
        "weight": 6,
        "error_type": "GOOGLE_SHEETS_QUOTA_EXCEEDED",
        "error_code": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded for quota metric 'Write requests' for Google Sheets API.",
        "http_status": 429,
        "retryable": True,
        "step": "Save the Claim Details",
    },
    {
        "weight": 3,
        "error_type": "GOOGLE_SHEETS_RANGE_ERROR",
        "error_code": "INVALID_ARGUMENT",
        "message": "Unable to parse range: Sheet1!A:L — the sheet 'Insurance_Payment' or range no longer exists.",
        "http_status": 400,
        "retryable": False,
        "step": "Save the Claim Details",
    },
    # Green API — WhatsApp follow-up message
    {
        "weight": 8,
        "error_type": "INSTANCE_NOT_AUTHORIZED",
        "error_code": "466",
        "message": "Instance7103540721 is not authorized. Please scan the QR code again in the Green API console.",
        "http_status": 401,
        "retryable": False,
        "step": "HTTP Request - send a message on Whatsapp",
    },
    {
        "weight": 5,
        "error_type": "INVALID_CHAT_ID",
        "error_code": "400",
        "message": "Invalid chatId format: expected '{number}@c.us', received a malformed phone number from the claim payload.",
        "http_status": 400,
        "retryable": False,
        "step": "HTTP Request - send a message on Whatsapp",
    },
    {
        "weight": 4,
        "error_type": "OUTGOING_MESSAGE_BLOCKED",
        "error_code": "403",
        "message": "Outgoing message blocked — the recipient's WhatsApp account is unreachable or has blocked messages from this number.",
        "http_status": 403,
        "retryable": False,
        "step": "HTTP Request - send a message on Whatsapp",
    },
    {
        "weight": 5,
        "error_type": "GREEN_API_RATE_LIMIT",
        "error_code": "429",
        "message": "Rate limit exceeded: too many messages sent from instance7103540721 in the last minute.",
        "http_status": 429,
        "retryable": True,
        "step": "HTTP Request - send a message on Whatsapp",
    },
    # Gmail — fraud analysis email
    {
        "weight": 6,
        "error_type": "GMAIL_AUTH_ERROR",
        "error_code": "invalid_grant",
        "message": "OAuth2 token for 'Gmail account - fnx' has expired or been revoked.",
        "http_status": 401,
        "retryable": False,
        "step": "Send an Email - Fraud analysis",
    },
    {
        "weight": 4,
        "error_type": "GMAIL_SEND_QUOTA_EXCEEDED",
        "error_code": "rateLimitExceeded",
        "message": "User-rate limit exceeded for Gmail API send requests.",
        "http_status": 429,
        "retryable": True,
        "step": "Send an Email - Fraud analysis",
    },
    {
        "weight": 2,
        "error_type": "GMAIL_INVALID_RECIPIENT",
        "error_code": "invalidArgument",
        "message": "Invalid recipient address in 'To' field — the address could not be validated.",
        "http_status": 400,
        "retryable": False,
        "step": "Send an Email - Fraud analysis",
    },
    # Webhook entry
    {
        "weight": 3,
        "error_type": "MALFORMED_WEBHOOK_PAYLOAD",
        "error_code": "invalid_body",
        "message": "Request body is not valid JSON or is missing the required field 'phone_number'.",
        "http_status": 400,
        "retryable": False,
        "step": "Webhook - A new Claim Received",
    },
]


def build() -> list:
    return _build_tool_log(
        tool_key="claim_fraud",
        workflow_steps=STEPS,
        success_rate=0.86,
        n_records=260,
        seed=10,
        duration_fn=lambda rng, ok: rng.uniform(1.0, 5.0) if ok else rng.uniform(0.3, 6.0),
        catalog=CATALOG,
        id_prefix="claim-exec",
        correlation_fn=lambda rng, i: f"n8n-exec-{rng.randint(100000, 999999)}",
    )


def main() -> None:
    out_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    rows = build()

    json_records = [_to_json_record(r) for r in rows]
    json_path = os.path.join(out_dir, "car_insurance_fraud_detection.json")
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)
    failures = sum(1 for r in rows if r["status"] == "failure")
    print(f"Wrote {len(json_records)} records ({failures} failures) to {json_path}")

    csv_path = os.path.join(out_dir, "car_insurance_fraud_detection.csv")
    _to_csv_dataframe(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
