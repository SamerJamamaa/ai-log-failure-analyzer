"""Generates realistic, tool-specific execution logs for demoing the AI
Exception Investigation Console against authentic-looking data — not a
single synthetic "WhatsApp Lead Intake" toy scenario, but six separate
files, each modeled on a real integration platform's ACTUAL public error
codes, message wording, endpoint shapes, and ID formats:

  - n8n              (workflow automation — node execution failures)
  - WhatsApp Business Cloud API (Meta Graph API messaging errors)
  - Stripe           (payments API)
  - OpenAI           (chat completions API)
  - Slack            (Web API)
  - Twilio           (Programmable Messaging API)

Every error catalog entry below is a paraphrase of that platform's own
publicly documented error codes/messages (the kind of thing that appears
verbatim in that platform's own API reference docs) — synthetic executions,
authentic error shapes. No real customer data, no real credentials, no
real request/response bodies.

Each file mixes MANY distinct error types at different frequencies (some
recurring 10-20x, some rare 1-3x isolated blips) alongside a realistic
majority of successful executions, so the deterministic engine's
fingerprinting/taxonomy/trend/priority machinery has real diversity to
work with — not one clean signal per file.

Run it yourself:
    python scripts/generate_realistic_tool_logs.py
Deterministic (fixed seed) except timestamps, which are anchored to "now".
Edit RNG_SEED below and re-run for a different random mix.
"""

import json
import os
import random
from datetime import datetime, timedelta

import pandas as pd

RNG_SEED = 20260930
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "realistic_tool_logs")
SPAN_DAYS = 10


# --- shared row builder --------------------------------------------------------


def _row(
    execution_id,
    workflow_name,
    start_time,
    status,
    duration_seconds,
    *,
    error_type="",
    error_code="",
    error_message="",
    error_stack_summary="",
    environment="production",
    workflow_step="",
    service="",
    endpoint="",
    http_method="",
    http_status=None,
    correlation_id="",
    retry_count=0,
    application_version="",
    provider="",
):
    return dict(
        execution_id=execution_id,
        workflow_name=workflow_name,
        start_time=start_time,
        status=status,
        duration_seconds=round(duration_seconds, 3),
        error_type=error_type,
        error_message=error_message,
        error_code=error_code,
        error_stack_summary=error_stack_summary,
        environment=environment,
        workflow_step=workflow_step,
        service=service,
        endpoint=endpoint,
        http_method=http_method,
        http_status=http_status,
        correlation_id=correlation_id,
        retry_count=retry_count,
        application_version=application_version,
        provider=provider,
    )


def _timestamps(n, seed, days=SPAN_DAYS):
    end = datetime.now()
    start = end - timedelta(days=days)
    step = (days * 24 * 60) / n
    rng = random.Random(RNG_SEED + seed)
    return [start + timedelta(minutes=i * step + rng.uniform(0, step * 0.85)) for i in range(n)]


def _weighted_choice(rng, catalog):
    weights = [entry["weight"] for entry in catalog]
    return rng.choices(catalog, weights=weights, k=1)[0]


def _step_for_entry(rng, err, steps_by_name, workflow_steps):
    """Resolves which step a failure catalog entry belongs to: a single
    workflow_step name, a list of candidate names (for an error genuinely
    shared by more than one step, e.g. two agents calling the same LLM
    provider), or — for entries with no "step" key — any step at random,
    which keeps single-provider catalogs (every step interchangeable)
    working unchanged."""
    step_ref = err.get("step")
    if step_ref is None:
        return rng.choice(workflow_steps)
    if isinstance(step_ref, list):
        return steps_by_name[rng.choice(step_ref)]
    return steps_by_name[step_ref]


def _build_tool_log(
    *,
    tool_key,
    workflow_steps,
    success_rate,
    n_records,
    seed,
    duration_fn,
    catalog,
    id_prefix,
    correlation_fn,
):
    """Shared engine: decides success/failure per record by `success_rate`.
    For a success, picks any step at random. For a failure, draws from
    `catalog` (a weighted list of realistic error entries) FIRST, then
    picks the step that entry actually belongs to — a catalog entry with a
    "step" key (matching a workflow_step value in `workflow_steps`) is
    pinned to that step; an entry with no "step" key falls back to a random
    step, for backward compatibility with single-provider catalogs where
    every step is interchangeable. Picking the step independently of the
    catalog entry (the original behavior) let a Gmail auth error land on a
    Google Sheets step in a multi-provider workflow — semantically broken,
    caught by inspecting a real multi-step generated file."""
    rng = random.Random(RNG_SEED + seed)
    timestamps = sorted(_timestamps(n_records, seed))
    rows = []
    steps_by_name = {s["workflow_step"]: s for s in workflow_steps}

    for i, ts in enumerate(timestamps):
        is_success = rng.random() < success_rate
        duration = duration_fn(rng, is_success)
        exec_id = f"{id_prefix}-{i + 1:06d}"

        if is_success:
            step = rng.choice(workflow_steps)
            rows.append(
                _row(
                    exec_id,
                    step["workflow_name"],
                    ts,
                    "success",
                    duration,
                    environment="production",
                    workflow_step=step["workflow_step"],
                    service=step["service"],
                    endpoint=step["endpoint"],
                    http_method=step["http_method"],
                    http_status=step.get("success_status", 200),
                    correlation_id=correlation_fn(rng, i),
                    retry_count=0,
                    application_version=step["application_version"],
                    provider=step["provider"],
                )
            )
            continue

        err = _weighted_choice(rng, catalog)
        step = _step_for_entry(rng, err, steps_by_name, workflow_steps)
        retryable = err.get("retryable", False)
        retry_count = rng.choice([1, 1, 2]) if retryable else 0
        rows.append(
            _row(
                exec_id,
                step["workflow_name"],
                ts,
                "failure",
                duration,
                error_type=err["error_type"],
                error_code=err.get("error_code", ""),
                error_message=err["message"],
                error_stack_summary=err.get("stack", ""),
                environment="production",
                workflow_step=step["workflow_step"],
                service=step["service"],
                endpoint=step["endpoint"],
                http_method=step["http_method"],
                http_status=err.get("http_status"),
                correlation_id=correlation_fn(rng, i),
                retry_count=retry_count,
                application_version=step["application_version"],
                provider=step["provider"],
            )
        )
    return rows


# --- tool 1: n8n (workflow automation) ------------------------------------------

N8N_STEPS = [
    {
        "workflow_name": "Lead Enrichment Pipeline",
        "workflow_step": "HubSpot - Upsert Contact",
        "service": "n8n-worker",
        "provider": "hubspot",
        "endpoint": "/crm/v3/objects/contacts",
        "http_method": "POST",
        "application_version": "1.44.0",
        "success_status": 200,
    },
    {
        "workflow_name": "Invoice Auto-Processing",
        "workflow_step": "Postgres - Insert Invoice",
        "service": "n8n-worker",
        "provider": "postgres",
        "endpoint": "internal:query",
        "http_method": "",
        "application_version": "1.44.0",
        "success_status": None,
    },
    {
        "workflow_name": "Customer Onboarding Sync",
        "workflow_step": "Google Sheets - Append Row",
        "service": "n8n-worker",
        "provider": "google-sheets",
        "endpoint": "/v4/spreadsheets/{spreadsheetId}/values/Sheet1:append",
        "http_method": "POST",
        "application_version": "1.42.1",
        "success_status": 200,
    },
    {
        "workflow_name": "Slack Alert Router",
        "workflow_step": "Slack - Post Message",
        "service": "n8n-worker",
        "provider": "slack",
        "endpoint": "/api/chat.postMessage",
        "http_method": "POST",
        "application_version": "1.44.0",
        "success_status": 200,
    },
    {
        "workflow_name": "Invoice Auto-Processing",
        "workflow_step": "Code - Parse Extracted Fields",
        "service": "n8n-worker",
        "provider": "internal",
        "endpoint": "internal:code-node",
        "http_method": "",
        "application_version": "1.44.0",
        "success_status": None,
    },
]

N8N_CATALOG = [
    {
        "weight": 18,
        "error_type": "NodeApiError",
        "error_code": "ETIMEDOUT",
        "message": "HubSpot node: request timed out after 30000ms while calling https://api.hubapi.com/crm/v3/objects/contacts",
        "stack": "NodeApiError: ETIMEDOUT\n    at HubspotV2.execute (/usr/lib/n8n/nodes/HubspotV2/HubspotV2.node.js:214:11)",
        "http_status": 504,
        "retryable": True,
    },
    {
        "weight": 14,
        "error_type": "NodeApiError",
        "error_code": "RATE_LIMIT_EXCEEDED",
        "message": "HubSpot node: 429 Too Many Requests — rate limit exceeded for this app (100 requests per 10 seconds)",
        "stack": "NodeApiError: Request failed with status code 429\n    at HubspotV2.execute (/usr/lib/n8n/nodes/HubspotV2/HubspotV2.node.js:230:9)",
        "http_status": 429,
        "retryable": True,
    },
    {
        "weight": 6,
        "error_type": "NodeApiError",
        "error_code": "ECONNREFUSED",
        "message": "Postgres node: connect ECONNREFUSED 10.0.4.22:5432",
        "stack": "NodeApiError: connect ECONNREFUSED 10.0.4.22:5432\n    at Postgres.execute (/usr/lib/n8n/nodes/Postgres/Postgres.node.js:88:13)",
        "http_status": None,
        "retryable": True,
    },
    {
        "weight": 8,
        "error_type": "NodeOperationError",
        "error_code": "INVALID_CREDENTIALS",
        "message": "Google Sheets node: OAuth2 credentials for 'Google Sheets account (Marketing)' are invalid or have expired",
        "stack": "NodeOperationError: The Google Sheets credentials are not valid\n    at GoogleSheetsV2.execute (/usr/lib/n8n/nodes/GoogleSheets/GoogleSheetsV2.node.js:151:15)",
        "http_status": 401,
        "retryable": False,
    },
    {
        "weight": 5,
        "error_type": "NodeOperationError",
        "error_code": "JSON_PARSE_ERROR",
        "message": "Code node: Unexpected token '<', \"<html>...\" is not valid JSON at position 0",
        "stack": "SyntaxError: Unexpected token < in JSON at position 0\n    at Code.execute (/usr/lib/n8n/nodes/Code/Code.node.js:97:22)",
        "http_status": None,
        "retryable": False,
    },
    {
        "weight": 3,
        "error_type": "WorkflowActivationError",
        "error_code": "WEBHOOK_REGISTRATION_FAILED",
        "message": "Could not register webhook: the configured URL returned HTTP 500 during handshake",
        "stack": "WorkflowActivationError: Webhook registration failed\n    at ActiveWorkflowRunner.add (/usr/lib/n8n/dist/ActiveWorkflowRunner.js:342:11)",
        "http_status": 500,
        "retryable": False,
    },
    {
        "weight": 2,
        "error_type": "NodeApiError",
        "error_code": "SLACK_CHANNEL_NOT_FOUND",
        "message": "Slack node: channel_not_found — the channel #ops-alerts-legacy no longer exists",
        "stack": "NodeApiError: channel_not_found\n    at SlackV2.execute (/usr/lib/n8n/nodes/Slack/SlackV2.node.js:176:9)",
        "http_status": 404,
        "retryable": False,
    },
]


def build_n8n() -> list:
    return _build_tool_log(
        tool_key="n8n",
        workflow_steps=N8N_STEPS,
        success_rate=0.88,
        n_records=220,
        seed=1,
        duration_fn=lambda rng, ok: rng.uniform(0.2, 2.5) if ok else rng.uniform(0.5, 31.0),
        catalog=N8N_CATALOG,
        id_prefix="n8n-exec",
        correlation_fn=lambda rng, i: f"n8n-wf-{rng.randint(100000, 999999)}",
    )


# --- tool 2: WhatsApp Business Cloud API (Meta) ---------------------------------

WHATSAPP_STEPS = [
    {
        "workflow_name": "WhatsApp Order Confirmation Sender",
        "workflow_step": "send_template_message",
        "service": "messaging-service",
        "provider": "whatsapp-business-api",
        "endpoint": "/v19.0/{phone_number_id}/messages",
        "http_method": "POST",
        "application_version": "3.2.0",
        "success_status": 200,
    },
]

WHATSAPP_CATALOG = [
    {
        "weight": 16,
        "error_type": "MESSAGE_UNDELIVERABLE",
        "error_code": "131026",
        "message": "(#131026) Message Undeliverable - the recipient phone number is not a WhatsApp user or is unreachable.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 12,
        "error_type": "RE_ENGAGEMENT_WINDOW_EXPIRED",
        "error_code": "131047",
        "message": "(#131047) Message failed to send because more than 24 hours have passed since the customer last replied to this number.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 10,
        "error_type": "TEMPLATE_PARAM_MISMATCH",
        "error_code": "132000",
        "message": "(#132000) Number of parameters provided in the template request (2) does not match the number of variables in the approved template (3).",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 9,
        "error_type": "OAUTH_TOKEN_EXPIRED",
        "error_code": "190",
        "message": "(#190) Error validating access token: Session has expired on Tuesday, 23-Sep-26 04:00:00 PDT.",
        "http_status": 401,
        "retryable": False,
    },
    {
        "weight": 7,
        "error_type": "RATE_LIMIT_HIT",
        "error_code": "80007",
        "message": "(#80007) Too many messages sent from this phone number in a short period of time. Retry after a short delay.",
        "http_status": 429,
        "retryable": True,
    },
    {
        "weight": 3,
        "error_type": "MEDIA_DOWNLOAD_ERROR",
        "error_code": "131053",
        "message": "(#131053) Media download error - the referenced media file could not be retrieved from the provided URL.",
        "http_status": 400,
        "retryable": False,
    },
]


def build_whatsapp() -> list:
    return _build_tool_log(
        tool_key="whatsapp",
        workflow_steps=WHATSAPP_STEPS,
        success_rate=0.85,
        n_records=260,
        seed=2,
        duration_fn=lambda rng, ok: rng.uniform(0.3, 1.8) if ok else rng.uniform(0.2, 2.5),
        catalog=WHATSAPP_CATALOG,
        id_prefix="wa-exec",
        correlation_fn=lambda rng, i: f"wamid.HBgL{rng.randint(10**14, 10**15 - 1)}",
    )


# --- tool 3: Stripe (payments) --------------------------------------------------

STRIPE_STEPS = [
    {
        "workflow_name": "Stripe Subscription Billing",
        "workflow_step": "create_payment_intent",
        "service": "billing-service",
        "provider": "stripe",
        "endpoint": "/v1/payment_intents",
        "http_method": "POST",
        "application_version": "2.9.4",
        "success_status": 200,
    },
]

STRIPE_CATALOG = [
    {
        "weight": 20,
        "error_type": "card_error",
        "error_code": "card_declined",
        "message": "Your card was declined. Your request was in test mode, but used a non test (live) card.",
        "http_status": 402,
        "retryable": False,
    },
    {
        "weight": 10,
        "error_type": "card_error",
        "error_code": "insufficient_funds",
        "message": "Your card has insufficient funds to complete this purchase.",
        "http_status": 402,
        "retryable": False,
    },
    {
        "weight": 6,
        "error_type": "card_error",
        "error_code": "expired_card",
        "message": "Your card has expired.",
        "http_status": 402,
        "retryable": False,
    },
    {
        "weight": 8,
        "error_type": "invalid_request_error",
        "error_code": "parameter_missing",
        "message": "Missing required param: payment_method.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 4,
        "error_type": "idempotency_error",
        "error_code": "idempotency_key_in_use",
        "message": "Keys for idempotent requests can only be used with the same parameters they were first used with. Try using a key that hasn't been used before.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 5,
        "error_type": "rate_limit_error",
        "error_code": "rate_limit",
        "message": "Too many requests hit the API too quickly. We recommend an exponential backoff of your requests.",
        "http_status": 429,
        "retryable": True,
    },
    {
        "weight": 3,
        "error_type": "api_error",
        "error_code": "api_connection_error",
        "message": "An error occurred with our API. This is on Stripe's end. Please retry your request.",
        "http_status": 500,
        "retryable": True,
    },
]


def build_stripe() -> list:
    return _build_tool_log(
        tool_key="stripe",
        workflow_steps=STRIPE_STEPS,
        success_rate=0.90,
        n_records=240,
        seed=3,
        duration_fn=lambda rng, ok: rng.uniform(0.4, 1.5) if ok else rng.uniform(0.3, 2.0),
        catalog=STRIPE_CATALOG,
        id_prefix="pi-exec",
        correlation_fn=lambda rng, i: f"req_{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=14))}",
    )


# --- tool 4: OpenAI (chat completions) ------------------------------------------

OPENAI_STEPS = [
    {
        "workflow_name": "AI Ticket Classification Pipeline",
        "workflow_step": "classify_support_ticket",
        "service": "ai-classification-service",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
        "http_method": "POST",
        "application_version": "gpt-4.1-mini",
        "success_status": 200,
    },
]

OPENAI_CATALOG = [
    {
        "weight": 14,
        "error_type": "rate_limit_exceeded",
        "error_code": "rate_limit_exceeded",
        "message": "Rate limit reached for gpt-4.1-mini in organization org-8f2K on requests per min (RPM): Limit 500, Used 500, Requested 1.",
        "http_status": 429,
        "retryable": True,
    },
    {
        "weight": 10,
        "error_type": "insufficient_quota",
        "error_code": "insufficient_quota",
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "http_status": 429,
        "retryable": False,
    },
    {
        "weight": 8,
        "error_type": "context_length_exceeded",
        "error_code": "context_length_exceeded",
        "message": "This model's maximum context length is 128000 tokens. However, your messages resulted in 131502 tokens.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 6,
        "error_type": "INVALID_JSON_SCHEMA",
        "error_code": "response_format_invalid",
        "message": "Model returned malformed JSON in function-call arguments; failed to parse response against the 'classify_ticket' schema.",
        "http_status": 200,
        "retryable": False,
    },
    {
        "weight": 5,
        "error_type": "content_policy_violation",
        "error_code": "content_filter",
        "message": "Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our usage policies.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 5,
        "error_type": "server_error",
        "error_code": "server_error",
        "message": "The server had an error processing your request. Sorry about that! You can retry your request.",
        "http_status": 503,
        "retryable": True,
    },
]


def build_openai() -> list:
    return _build_tool_log(
        tool_key="openai",
        workflow_steps=OPENAI_STEPS,
        success_rate=0.87,
        n_records=300,
        seed=4,
        duration_fn=lambda rng, ok: rng.uniform(1.2, 6.0) if ok else rng.uniform(0.5, 8.0),
        catalog=OPENAI_CATALOG,
        id_prefix="chatcmpl-exec",
        correlation_fn=lambda rng, i: "chatcmpl-" + "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=20)),
    )


# --- tool 5: Slack (Web API) -----------------------------------------------------

SLACK_STEPS = [
    {
        "workflow_name": "Slack Incident Alert Bot",
        "workflow_step": "post_incident_message",
        "service": "alerting-service",
        "provider": "slack",
        "endpoint": "/api/chat.postMessage",
        "http_method": "POST",
        "application_version": "1.8.2",
        "success_status": 200,
    },
]

SLACK_CATALOG = [
    {
        "weight": 10,
        "error_type": "invalid_auth",
        "error_code": "invalid_auth",
        "message": "invalid_auth: the bot token provided is invalid or has been revoked.",
        "http_status": 401,
        "retryable": False,
    },
    {
        "weight": 8,
        "error_type": "channel_not_found",
        "error_code": "channel_not_found",
        "message": "channel_not_found: value passed for channel was invalid (#ops-alerts-legacy).",
        "http_status": 404,
        "retryable": False,
    },
    {
        "weight": 6,
        "error_type": "not_in_channel",
        "error_code": "not_in_channel",
        "message": "not_in_channel: cannot post message; the app's bot user is not a member of this channel.",
        "http_status": 403,
        "retryable": False,
    },
    {
        "weight": 9,
        "error_type": "ratelimited",
        "error_code": "ratelimited",
        "message": "ratelimited: Too many requests, Retry-After: 42 seconds.",
        "http_status": 429,
        "retryable": True,
    },
    {
        "weight": 2,
        "error_type": "account_inactive",
        "error_code": "account_inactive",
        "message": "account_inactive: authentication token is for a deleted user or workspace.",
        "http_status": 401,
        "retryable": False,
    },
]


def build_slack() -> list:
    return _build_tool_log(
        tool_key="slack",
        workflow_steps=SLACK_STEPS,
        success_rate=0.91,
        n_records=180,
        seed=5,
        duration_fn=lambda rng, ok: rng.uniform(0.2, 1.0) if ok else rng.uniform(0.15, 1.2),
        catalog=SLACK_CATALOG,
        id_prefix="slack-exec",
        correlation_fn=lambda rng, i: f"{rng.randint(1000000000, 9999999999)}.{rng.randint(100000, 999999)}",
    )


# --- tool 6: Twilio (Programmable Messaging) ------------------------------------

TWILIO_STEPS = [
    {
        "workflow_name": "Twilio SMS Delivery Gateway",
        "workflow_step": "send_sms",
        "service": "sms-gateway-service",
        "provider": "twilio",
        "endpoint": "/2010-04-01/Accounts/{AccountSid}/Messages.json",
        "http_method": "POST",
        "application_version": "5.1.0",
        "success_status": 201,
    },
]

TWILIO_CATALOG = [
    {
        "weight": 14,
        "error_type": "INVALID_PHONE_NUMBER",
        "error_code": "21211",
        "message": "The 'To' number +1500 is not a valid phone number.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 9,
        "error_type": "UNSUBSCRIBED_RECIPIENT",
        "error_code": "21610",
        "message": "Attempt to send to unsubscribed recipient — the recipient has replied STOP and opted out.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 8,
        "error_type": "RATE_LIMIT",
        "error_code": "20429",
        "message": "Too Many Requests — the account has exceeded the maximum number of requests per second.",
        "http_status": 429,
        "retryable": True,
    },
    {
        "weight": 6,
        "error_type": "UNREACHABLE_HANDSET",
        "error_code": "30003",
        "message": "Unreachable destination handset — the destination handset is switched off or out of coverage.",
        "http_status": 400,
        "retryable": False,
    },
    {
        "weight": 4,
        "error_type": "AUTHENTICATION_ERROR",
        "error_code": "20003",
        "message": "Authentication Error - invalid username or password (Account SID / Auth Token mismatch).",
        "http_status": 401,
        "retryable": False,
    },
]


def build_twilio() -> list:
    return _build_tool_log(
        tool_key="twilio",
        workflow_steps=TWILIO_STEPS,
        success_rate=0.89,
        n_records=210,
        seed=6,
        duration_fn=lambda rng, ok: rng.uniform(0.3, 1.4) if ok else rng.uniform(0.2, 1.8),
        catalog=TWILIO_CATALOG,
        id_prefix="sms-exec",
        correlation_fn=lambda rng, i: "SM" + "".join(rng.choices("0123456789abcdef", k=32)),
    )


FILES = [
    ("n8n_workflow_automation", build_n8n),
    ("whatsapp_business_api", build_whatsapp),
    ("stripe_payments", build_stripe),
    ("openai_ai_pipeline", build_openai),
    ("slack_notifications", build_slack),
    ("twilio_sms_gateway", build_twilio),
]


def generate_all() -> dict:
    return {name: builder() for name, builder in FILES}


# --- serialization ---------------------------------------------------------------


def _to_json_record(row: dict) -> dict:
    is_failure = row["status"] == "failure"
    error = None
    if is_failure:
        error = {
            "type": row["error_type"],
            "message": row["error_message"],
            "code": row["error_code"] or None,
            "stack_summary": row["error_stack_summary"] or None,
        }
    return {
        "execution_id": row["execution_id"],
        "workflow_name": row["workflow_name"],
        "start_time": row["start_time"].strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "status": "failed" if is_failure else "success",
        "duration_seconds": row["duration_seconds"],
        "error": error,
        "environment": row["environment"],
        "workflow_step": row["workflow_step"],
        "service": row["service"],
        "endpoint": row["endpoint"],
        "http_method": row["http_method"],
        "http_status": row["http_status"],
        "correlation_id": row["correlation_id"],
        "retry_count": row["retry_count"],
        "application_version": row["application_version"],
        "provider": row["provider"],
    }


def _to_csv_dataframe(rows: list) -> pd.DataFrame:
    flat = [{**r, "start_time": r["start_time"].strftime("%Y-%m-%d %H:%M:%S")} for r in rows]
    return pd.DataFrame(flat)


def main() -> None:
    out_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    for name, rows in generate_all().items():
        json_records = [_to_json_record(r) for r in rows]
        json_path = os.path.join(out_dir, f"{name}.json")
        with open(json_path, "w") as f:
            json.dump(json_records, f, indent=2)
        failures = sum(1 for r in rows if r["status"] == "failure")
        print(f"Wrote {len(json_records)} records ({failures} failures) to {json_path}")

        csv_path = os.path.join(out_dir, f"{name}.csv")
        _to_csv_dataframe(rows).to_csv(csv_path, index=False)
        print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
