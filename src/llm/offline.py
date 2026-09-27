"""Rule-based fallback investigation generator — used when no LLM provider
key is set, or when a provider call fails. Produces the SAME
ExceptionInvestigation shape an LLM would, built entirely from category-
specific templates applied to the deterministic evidence dict (never an
external call, never invented data). Always labeled "Rule-based
investigation guidance" by the UI layer via `source="offline"`.
"""

from src.llm.base import LLMClient
from src.models import (
    ContainmentAction,
    DiagnosticCommand,
    ExceptionInvestigation,
    Hypothesis,
    InvestigationAction,
    RemediationOption,
    VerificationStep,
)

_HIGH_PRIORITY_LABELS = {"Critical", "High"}


def _evidence_ids(evidence: dict) -> list:
    return [item["evidence_id"] for item in evidence.get("evidence", [])]


def _ids_for(evidence: dict, source_field: str) -> list:
    return [item["evidence_id"] for item in evidence.get("evidence", []) if item["source_field"] == source_field]


def _http_status_clause(ident: dict) -> str:
    """'HTTP 401' when a status is present, or a plain fallback — never the
    literal string 'HTTP None' when a group was classified without one."""
    status = ident.get("http_status")
    return f"HTTP {status}" if status is not None else "an unspecified HTTP status"


def _known_facts(evidence: dict) -> list:
    occ = evidence["occurrence"]
    ident = evidence["exception_identity"]
    facts = [
        f"{occ['occurrence_count']} occurrence(s) of "
        f"'{ident['error_type'] or ident['representative_message']}' recorded in workflow "
        f"'{ident['workflow_name']}', affecting {occ['affected_execution_count']} execution(s).",
        f"Represents {occ['percentage_of_all_failures']}% of all failures and "
        f"{occ['percentage_of_workflow_executions']}% of '{ident['workflow_name']}' executions in this upload.",
        f"Trend classification: {occ['trend']} (first seen {occ['first_occurrence']}, "
        f"last seen {occ['last_occurrence']}).",
    ]
    for item in evidence.get("evidence", []):
        facts.append(
            f"[{item['evidence_id']}] {item['source_field']}: {item['observed_value']} "
            f"({item['occurrence_count']} occurrence(s)) — {item['why_it_matters']}"
        )
    return facts


def _unknowns(evidence: dict) -> list:
    return list(evidence.get("missing_evidence", []))


def _generic_investigation_actions(evidence: dict) -> list:
    ident = evidence["exception_identity"]
    occ_ids = _ids_for(evidence, "occurrence_count")
    return [
        InvestigationAction(
            priority=1,
            action=(
                f"Pull the representative execution and correlation IDs for this group and inspect "
                f"the full request/response trace for '{ident['workflow_step'] or ident['workflow_name']}'."
            ),
            component=ident["service"] or ident["workflow_name"],
            evidence_reference=occ_ids or _evidence_ids(evidence)[:1],
            expected_findings=[
                "A consistent point of failure across the sampled executions.",
                "Whether the failure occurs before, during, or after the call to the dependency/provider.",
            ],
            decision_from_result=(
                "If the failure is consistent across samples, treat it as a systemic issue rather than "
                "per-execution noise; if it varies, broaden the evidence collected before proceeding."
            ),
        )
    ]


def _base_verification_plan(evidence: dict) -> list:
    ident = evidence["exception_identity"]
    trend = evidence["occurrence"]["trend"]
    observation_period = (
        "Through the next deployment cycle, since this is a one-off occurrence"
        if trend == "isolated"
        else "At least one full cycle of this exception's typical recurrence interval (see First seen/Last seen above), to confirm the pattern has genuinely stopped"
    )
    return [
        VerificationStep(
            test=f"Re-run or replay a sample of the affected '{ident['workflow_name']}' executions after applying a fix.",
            expected_result="The exception no longer occurs for the replayed executions.",
            success_metric="Occurrence count for this fingerprint in the affected workflow",
            success_threshold="Zero new occurrences during the observation period",
            observation_period=observation_period,
            rollback_condition="Occurrences continue at the same or higher rate after the change",
        )
    ]


def _containment(evidence: dict, actions: list) -> list:
    """Containment is only appropriate when priority/impact justifies it."""
    if evidence["priority"]["label"] not in _HIGH_PRIORITY_LABELS:
        return []
    return actions


# --- category playbooks -----------------------------------------------------------


def _auth_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    version_ids = _ids_for(evidence, "application_version")
    versions = evidence["occurrence"]["application_versions"]
    version_note = (
        f", which makes a version-specific configuration regression (e.g. OAuth audience/scope) on "
        f"{', '.join(versions)} a plausible cause"
        if version_ids
        else ""
    )
    return {
        "explanation": (
            f"This group represents authentication/authorization failures ('{ident['error_type']}') "
            f"in '{ident['workflow_name']}'{version_note}. Configuration evidence is required for confirmation."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=(
                    "A credential, token, or authorization configuration used by this workflow/service is "
                    "invalid, expired, or scoped incorrectly" + (f" as of version {versions[0]}" if version_ids else "")
                ),
                confidence="medium" if version_ids else "low",
                supporting_evidence_ids=version_ids or _evidence_ids(evidence)[:1],
                contradicting_evidence_ids=[],
                missing_evidence=["Token expiration timestamp", "Identity-provider logs for the affected window"],
                verification_steps=["Compare the token/credential configuration in effect before and after the change."],
            )
        ],
        "containment": [
            ContainmentAction(
                priority=1,
                action="If a specific credential/version is implicated, consider pausing traffic through that path.",
                reason="Prevents repeated authentication failures from continuing to affect executions.",
                risk="Pausing traffic stops legitimate executions too; only do this if failures are near-total.",
            )
        ],
        "remediation": [
            RemediationOption(
                option="Refresh or rotate the credential/token in use",
                appropriate_when="Evidence shows the credential is expired or invalid",
                expected_benefit="Restores successful authentication",
                risks=["Rotating a shared credential can affect other consumers"],
                tradeoffs=["Requires coordinating the rotation with anything else using the same credential"],
                complexity="low",
                change_type="configuration",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Correct the audience/scope configuration for the affected version",
                appropriate_when="Evidence shows a version-specific configuration mismatch",
                expected_benefit="Restores correct authorization without rotating credentials",
                risks=["Misconfiguring scope could grant excess or insufficient access"],
                tradeoffs=["Needs a config diff between versions to apply confidently"],
                complexity="medium",
                change_type="configuration",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add expiration monitoring/alerting for credentials and tokens used by this integration.",
            "Add a contract/config test that fails deployment if authorization configuration changes unexpectedly.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Trace a representative correlation ID through the auth flow",
                command='grep "<correlation_id>" application.log',
                look_for=["The outbound auth request", "The identity provider's response", "Any token/claims returned"],
                interpretation=[
                    "If a token was returned, decode it and compare its audience/scope claim against what this service expects.",
                    "If no token was returned at all, investigate the client credential/configuration instead of the token itself.",
                ],
            )
        ],
    }


def _timeout_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents timeout/latency failures ('{ident['error_type']}') calling "
            f"{ident['provider'] or ident['service'] or 'a dependency'} from '{ident['workflow_name']}'."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=f"{ident['provider'] or ident['service'] or 'The downstream dependency'} is responding slowly or intermittently during the affected window",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["Dependency latency metrics for the affected window", "Provider status/incident history"],
                verification_steps=["Correlate the failure timestamps against the provider's own latency/incident data."],
            )
        ],
        "containment": [
            ContainmentAction(
                priority=1,
                action="If retries are amplifying load on the dependency, temporarily reduce retry aggressiveness.",
                reason="Aggressive retries during a slow dependency can worsen the underlying latency.",
                risk="Reducing retries may increase user-visible failures in the short term.",
            )
        ],
        "remediation": [
            RemediationOption(
                option="Increase the timeout threshold",
                appropriate_when="The dependency reliably succeeds, just slower than the current threshold",
                expected_benefit="Fewer false-positive timeouts",
                risks=["Masks a genuine performance regression", "Increases worst-case execution latency"],
                tradeoffs=["Simple to apply but does not address the underlying slowness"],
                complexity="low",
                change_type="configuration",
                temporary_or_permanent="temporary",
            ),
            RemediationOption(
                option="Add bounded retries with backoff and jitter",
                appropriate_when="Failures are intermittent rather than sustained",
                expected_benefit="Recovers from transient slowness without manual intervention",
                risks=["Non-idempotent operations retried blindly can create duplicates"],
                tradeoffs=["Adds latency to the failing path; needs idempotency protection first"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Move the call to async processing with polling",
                appropriate_when="The workflow does not need an immediate synchronous response",
                expected_benefit="Removes the timeout failure mode entirely",
                risks=["Adds architectural complexity", "Requires a polling/callback mechanism"],
                tradeoffs=["Larger change, but most durable fix"],
                complexity="high",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Add a fallback provider for this call",
                appropriate_when="An alternate provider with equivalent functionality is available",
                expected_benefit="Continues serving executions even when the primary provider is slow",
                risks=["Fallback provider may have different behavior/data guarantees"],
                tradeoffs=["Requires maintaining an additional integration"],
                complexity="high",
                change_type="infrastructure",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add dependency-latency monitoring with alerting thresholds tied to this integration.",
            "Add a provider health check that runs independently of production traffic.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Measure response time to the provider directly",
                command='curl -w "%{time_total}\\n" -o /dev/null -s https://<provider_host><endpoint>',
                look_for=["Response time relative to the configured timeout threshold"],
                interpretation=[
                    "If latency is consistently near or above the threshold, this supports a genuine provider slowdown.",
                    "If latency looks normal, investigate the client-side network path or timeout configuration instead.",
                ],
            )
        ],
    }


def _rate_limit_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents rate-limiting/capacity failures ('{ident['error_type']}', "
            f"{_http_status_clause(ident)}) calling {ident['provider'] or ident['service'] or 'a dependency'}."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="Request volume from this workflow is exceeding the provider's rate limit or quota",
                confidence="medium" if ident["http_status"] == 429 else "low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["Provider-side rate-limit/quota configuration", "Request volume over time for this integration"],
                verification_steps=["Compare request volume against the provider's documented limits for the affected window."],
            )
        ],
        "containment": [
            ContainmentAction(
                priority=1,
                action="Throttle or queue outgoing requests to this provider until the limit resets.",
                reason="Prevents continued 429s from cascading into retry storms.",
                risk="Throttling delays legitimate executions.",
            )
        ],
        "remediation": [
            RemediationOption(
                option="Add client-side rate limiting/backoff aligned to the provider's published limits",
                appropriate_when="Request volume genuinely exceeds the limit",
                expected_benefit="Stays under the limit without dropping requests",
                risks=["Under-tuned backoff can still trigger limits; over-tuned adds latency"],
                tradeoffs=["Requires accurate knowledge of the provider's limit"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Request a quota increase from the provider",
                appropriate_when="Volume is legitimate and expected to keep growing",
                expected_benefit="Removes the limit as a constraint",
                risks=["Provider may deny or delay the increase"],
                tradeoffs=["Depends on an external party's timeline"],
                complexity="low",
                change_type="provider",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add a dashboard/alert tracking request volume against the provider's rate limit.",
            "Add queue-depth or worker-saturation monitoring if requests are batched.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Count requests to this endpoint in the affected window",
                command='grep "<endpoint>" application.log | grep "<provider>" | wc -l',
                look_for=["Request count compared to the provider's published rate limit"],
                interpretation=[
                    "If volume exceeds the published limit, this confirms a capacity issue on this service's side.",
                    "If volume looks normal, the provider's limit may have been lowered — check its account/quota settings.",
                ],
            )
        ],
    }


def _provider_degradation_playbook(evidence: dict) -> dict:
    """Connectivity and dependency, minus DNS/TLS (which get their own
    playbooks). Branches on whether this looks like a genuine server-side
    outage (5xx, or no status at all — e.g. ECONNREFUSED) vs. a per-
    destination reachability rejection (a <500 status such as WhatsApp's
    'recipient unreachable' or Twilio's 'unreachable handset') — the two
    have different likely causes and different fixes, and describing a 400
    as 'server-side failures... an outage' would simply be wrong."""
    ident = evidence["exception_identity"]
    status = ident["http_status"]
    subject = ident["provider"] or ident["service"] or "a downstream dependency"
    is_likely_outage = status is None or status >= 500

    if is_likely_outage:
        return {
            "explanation": (
                f"This group represents server-side failures ({_http_status_clause(ident)}) from "
                f"{subject} while '{ident['workflow_name']}' was calling it."
            ),
            "hypotheses": [
                Hypothesis(
                    rank=1,
                    hypothesis=f"{subject} is experiencing degraded availability or an outage",
                    confidence="low",
                    supporting_evidence_ids=_evidence_ids(evidence)[:2],
                    contradicting_evidence_ids=[],
                    missing_evidence=["Provider response body", "Provider status page/incident history for the affected window"],
                    verification_steps=["Check the provider's status page and response body for the affected time range."],
                )
            ],
            "containment": [
                ContainmentAction(
                    priority=1,
                    action="If the dependency is confirmed down, pause non-essential calls to it.",
                    reason="Reduces load on an already-failing dependency and avoids repeated user-visible failures.",
                    risk="Pausing calls delays legitimate work until the dependency recovers.",
                )
            ],
            "remediation": [
                RemediationOption(
                    option="Add a circuit breaker around this dependency call",
                    appropriate_when="5xx failures are sustained rather than a single transient blip",
                    expected_benefit="Fails fast instead of repeatedly hitting a failing dependency",
                    risks=["Circuit breaker misconfiguration can trip on legitimate load"],
                    tradeoffs=["Adds a new failure mode (open circuit) that must be monitored"],
                    complexity="medium",
                    change_type="code",
                    temporary_or_permanent="permanent",
                ),
                RemediationOption(
                    option="Add a fallback provider or degraded-mode path",
                    appropriate_when="An alternate path exists for this functionality",
                    expected_benefit="Continues serving executions during a provider outage",
                    risks=["Fallback path may have different data/behavior guarantees"],
                    tradeoffs=["Requires maintaining an additional integration"],
                    complexity="high",
                    change_type="infrastructure",
                    temporary_or_permanent="permanent",
                ),
            ],
            "prevention": [
                "Add a provider health check independent of production traffic.",
                "Add alerting on 5xx rate from this dependency.",
            ],
            "commands": [
                DiagnosticCommand(
                    label="Check the provider's own status/incident history",
                    command="curl -s https://status.<provider>.example/api/v2/status.json",
                    look_for=["An active incident overlapping the failure window"],
                    interpretation=[
                        "If an incident is listed, this confirms provider-side degradation.",
                        "If nothing is listed, investigate this service's own request construction or recent changes instead.",
                    ],
                )
            ],
        }

    return {
        "explanation": (
            f"This group represents delivery/reachability failures ('{ident['error_type']}', {_http_status_clause(ident)}) "
            f"reaching a specific destination via {subject} from '{ident['workflow_name']}' — the individual destination "
            f"or recipient could not be reached; this is not necessarily a {subject} outage."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=(
                    f"The specific destinations/recipients in these executions are individually unreachable "
                    f"(invalid, deactivated, or offline) rather than {subject} itself being degraded"
                ),
                confidence="medium",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["Whether these destinations were previously reachable", "Whether the destinations share a common recent change"],
                verification_steps=["Check whether the same destinations fail consistently across attempts, or whether different destinations fail each time."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Validate destination reachability before attempting delivery",
                appropriate_when="Reachability can be checked ahead of time (e.g. a prior successful contact, or a provider validation endpoint)",
                expected_benefit="Avoids sending to destinations already known to be unreachable",
                risks=["A validation check can itself be stale or rate-limited"],
                tradeoffs=["Adds a pre-check step to the send path"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Suppress repeatedly-unreachable destinations instead of retrying them",
                appropriate_when="The same destinations fail across multiple attempts",
                expected_benefit="Stops wasted sends and reduces noise in the exception queue",
                risks=["A destination that becomes reachable again would need to be un-suppressed"],
                tradeoffs=["Requires tracking suppression state per destination"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add a dashboard tracking reachability-failure rate separately from provider-outage rate.",
            "Add a periodic reachability re-check for suppressed destinations.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Check whether the same destination repeats across failures",
                command='grep "<error_code>" application.log | grep -o "\\"to\\":\\"[^\\"]*\\"" | sort | uniq -c | sort -rn',
                look_for=["Whether one destination accounts for most failures, or many different destinations each fail once"],
                interpretation=[
                    "A small number of repeating destinations points to specific bad contacts, not a provider issue.",
                    "Many distinct destinations failing once each is more consistent with a provider-side problem — re-check the 5xx/outage hypothesis.",
                ],
            )
        ],
    }


def _internal_error_playbook(evidence: dict) -> dict:
    """A plain HTTP 500 (or unhandled exception) raised by THIS
    application's own code — not a downstream provider issue. Kept
    distinct from _provider_degradation_playbook so the investigation
    points at the right codebase."""
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents an internal application error ('{ident['error_type'] or 'unhandled exception'}') "
            f"raised by '{ident['service'] or ident['workflow_name']}' itself — not a downstream provider failure."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="An unhandled exception or code defect in this service's own request-handling path is being triggered by a specific input or state",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["Full (untruncated) stack trace", "Deployment timestamp of the code currently running"],
                verification_steps=["Inspect the full stack trace for a sample of failures and compare the triggering input across them."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Add input validation or a null/defensive check at the point in the stack trace that threw",
                appropriate_when="The stack trace consistently points to the same code path",
                expected_benefit="Prevents the specific triggering condition from reaching the faulty code",
                risks=["A defensive check can mask a deeper design issue if applied too broadly"],
                tradeoffs=["Fast to apply; may need a follow-up structural fix"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="temporary",
            ),
            RemediationOption(
                option="Fix the underlying code defect and add a regression test",
                appropriate_when="The root cause in the code is identified",
                expected_benefit="Removes the failure class entirely",
                risks=["Requires a deploy; may need to wait for the next release window"],
                tradeoffs=["Slower, but the durable fix"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add error-rate alerting scoped to this specific code path/endpoint.",
            "Add a regression test that reproduces the triggering input.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Pull the full stack trace for a sample execution",
                command='grep "<execution_id>" application.log',
                look_for=["The complete (untruncated) stack trace and the code path that threw"],
                interpretation=[
                    "If the same code path appears across all sampled executions, this is a deterministic bug, not transient noise.",
                    "If code paths vary, investigate a shared upstream cause (e.g. malformed input) instead.",
                ],
            )
        ],
    }


def _dns_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents DNS resolution failures while '{ident['workflow_name']}' was reaching "
            f"{ident['provider'] or ident['service'] or 'a downstream host'}."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=f"The hostname for {ident['provider'] or ident['service'] or 'the downstream dependency'} is failing to resolve or resolving inconsistently",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["DNS resolver logs for the affected window", "Whether the hostname/DNS record changed recently"],
                verification_steps=["Resolve the hostname from the same network path the failing executions used."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Add DNS-failure-specific retry with a different resolver",
                appropriate_when="Resolution is intermittent rather than fully broken",
                expected_benefit="Recovers from transient resolver issues",
                risks=["Does not help if the DNS record itself is wrong"],
                tradeoffs=["Cheap to apply, treats the symptom"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="temporary",
            ),
            RemediationOption(
                option="Correct or restore the DNS record",
                appropriate_when="The record is confirmed missing, wrong, or recently changed",
                expected_benefit="Directly resolves the failure",
                risks=["May be outside this team's control if DNS is provider-managed"],
                tradeoffs=["Requires access to DNS configuration"],
                complexity="low",
                change_type="infrastructure",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add a DNS health check independent of production traffic.",
            "Add alerting on resolution failures for this hostname.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Resolve the provider hostname from the affected environment",
                command="dig <provider_host>",
                look_for=["Whether the hostname resolves, and to what address"],
                interpretation=[
                    "If resolution fails or is inconsistent, this confirms a DNS issue.",
                    "If resolution succeeds reliably, investigate the network path or provider availability instead.",
                ],
            )
        ],
    }


def _tls_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents TLS/certificate failures while '{ident['workflow_name']}' was connecting to "
            f"{ident['provider'] or ident['service'] or 'a downstream host'}."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="The TLS certificate for this endpoint has expired, been rotated unexpectedly, or the chain is incomplete",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["Certificate expiration/rotation history", "TLS handshake logs for the affected window"],
                verification_steps=["Inspect the certificate chain currently presented by the endpoint."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Renew or correct the TLS certificate",
                appropriate_when="The certificate is confirmed expired or invalid",
                expected_benefit="Directly resolves the failure",
                risks=["May require coordinating a deploy/rotation window"],
                tradeoffs=["Requires access to certificate management for that endpoint"],
                complexity="low",
                change_type="infrastructure",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Update the trusted CA bundle on the calling service",
                appropriate_when="The provider rotated to a new, valid CA the client doesn't yet trust",
                expected_benefit="Restores trust without changing the provider's certificate",
                risks=["Trusting the wrong CA would be a security risk — verify the new CA is legitimate first"],
                tradeoffs=["Requires a deploy of the calling service"],
                complexity="medium",
                change_type="configuration",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add certificate-expiration monitoring for every external endpoint this service depends on.",
            "Add a TLS handshake health check independent of production traffic.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Inspect the certificate chain presented by the provider",
                command="openssl s_client -connect <provider_host>:443 -servername <provider_host> </dev/null 2>/dev/null | openssl x509 -noout -dates -issuer",
                look_for=["Certificate expiration date and issuer chain validity"],
                interpretation=[
                    "If the certificate is expired or the chain is incomplete, this confirms a TLS/certificate issue.",
                    "If the certificate is valid, investigate TLS version/cipher negotiation instead.",
                ],
            )
        ],
    }


def _db_transaction_playbook(evidence: dict) -> dict:
    """Data-integrity failures that are about transaction/lock/consistency
    problems rather than a duplicate-record conflict — kept distinct from
    _duplicate_playbook, which is specifically about 409/duplicate-key
    conflicts and idempotency."""
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents database transaction failures ('{ident['error_type']}') in "
            f"'{ident['workflow_name']}', separate from any duplicate-record conflict."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="A transaction is being blocked or rolled back due to lock contention, a constraint violation, or a partial write",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["Database transaction/lock status for the affected records", "The exact constraint or error raised by the database"],
                verification_steps=["Inspect the database's transaction/lock state for the affected time window."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Reduce transaction scope/lock duration around the affected write",
                appropriate_when="Lock contention is confirmed as the cause",
                expected_benefit="Reduces blocking between concurrent transactions",
                risks=["Splitting a transaction can weaken atomicity guarantees if done carelessly"],
                tradeoffs=["Requires care to preserve correctness"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Add a compensating transaction or reconciliation job for partial writes",
                appropriate_when="A partial write is confirmed to leave inconsistent state",
                expected_benefit="Restores consistency without a manual intervention each time",
                risks=["A poorly designed compensating transaction can itself create inconsistency"],
                tradeoffs=["More engineering effort, but closes the gap durably"],
                complexity="high",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add database transaction-failure monitoring/alerting.",
            "Add a regression test that exercises the concurrent-write scenario that triggered this.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Check for long-held locks or blocked transactions on the affected table",
                command="SELECT * FROM pg_locks WHERE relation = '<table_name>'::regclass;",
                look_for=["Locks or blocked transactions overlapping the failure window"],
                interpretation=[
                    "If locks/blocking are present, this confirms a contention issue.",
                    "If not, investigate a constraint violation or a partial-write bug in the write path instead.",
                ],
            )
        ],
    }


def _schema_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    is_ai = "structured" in evidence["exception_identity"].get("primary_category", "").lower() or ident["error_type"].upper().find("JSON") != -1
    return {
        "explanation": (
            f"This group represents validation/schema failures ('{ident['error_type']}') in '{ident['workflow_name']}'"
            + (" involving model-generated structured output" if is_ai else " on incoming/outgoing data")
            + "."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=(
                    "The producer of this data (an upstream model or system) is emitting a payload that no "
                    "longer matches the expected schema/contract"
                ),
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["The exact payload that failed validation", "Schema/contract version in effect at the time"],
                verification_steps=["Inspect a sample of the raw failing payloads against the current schema definition."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Add stricter output/contract validation with clear rejection reasons at the boundary",
                appropriate_when="The failure is due to malformed data reaching this step",
                expected_benefit="Fails fast with a clear reason instead of propagating bad data",
                risks=["Overly strict validation can reject otherwise-usable data"],
                tradeoffs=["Requires maintaining the schema/contract definition"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Add structured-output controls (e.g. constrained decoding/schema enforcement) at the producer",
                appropriate_when="The producer is an LLM/model generating the structured output",
                expected_benefit="Reduces the rate of schema-violating output at the source",
                risks=["May not be available for all providers/models"],
                tradeoffs=["Addresses the root cause but depends on producer-side support"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add contract/schema tests that run whenever the producer or consumer changes.",
            "Add schema-version checks so a mismatch is caught before it reaches production.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Compare a sample of failing payloads to the current schema",
                command='grep "<execution_id>" application.log | jq .',
                look_for=["Which field(s) are missing, mistyped, or unexpectedly structured"],
                interpretation=[
                    "If the same field fails across samples, the producer is consistently emitting a bad shape — fix at the source.",
                    "If different fields fail each time, the schema itself may be too strict or the producer is unstable.",
                ],
            )
        ],
    }


def _duplicate_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    retry_ids = _ids_for(evidence, "retry_count")
    return {
        "explanation": (
            f"This group represents data-integrity conflicts ('{ident['error_type']}', "
            f"{_http_status_clause(ident)}) in '{ident['workflow_name']}'"
            + (", correlated with retried executions" if retry_ids else "")
            + "."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=(
                    "A retried request is being resubmitted after the original request already succeeded, "
                    "producing a duplicate/unique-key conflict"
                    if retry_ids
                    else "A duplicate or conflicting record is being written without an idempotency check"
                ),
                confidence="medium" if retry_ids else "low",
                supporting_evidence_ids=retry_ids or _evidence_ids(evidence)[:1],
                contradicting_evidence_ids=[],
                missing_evidence=["Whether the original (non-duplicate) request ultimately succeeded", "Database transaction status for the affected records"],
                verification_steps=["Trace the correlation ID of a conflicting execution to confirm whether an earlier attempt already succeeded."],
            )
        ],
        "containment": [
            ContainmentAction(
                priority=1,
                action="Pause automatic retries for this operation until idempotency is confirmed.",
                reason="Blind retries on a non-idempotent write risk creating more duplicate/inconsistent records.",
                risk="Pausing retries may leave some executions unresolved until investigated.",
            )
        ],
        "remediation": [
            RemediationOption(
                option="Add an idempotency key to this write operation",
                appropriate_when="The operation can be safely retried once keyed correctly",
                expected_benefit="Retries become safe; duplicates are prevented at the source",
                risks=["Requires a durable store for idempotency keys"],
                tradeoffs=["Some upfront design work, but removes the whole failure class"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Treat the conflict as success when the existing record matches the intended write",
                appropriate_when="The conflicting record is confirmed to be the same logical event",
                expected_benefit="Avoids surfacing a false failure for an already-completed operation",
                risks=["Incorrectly treating a genuine conflict as success could mask real duplicates"],
                tradeoffs=["Requires a reliable equality check between the new and existing record"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add idempotency controls to every write path that can be retried.",
            "Add a regression test that submits the same request twice and asserts no duplicate is created.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Trace a correlation ID through the write path",
                command='grep "<correlation_id>" application.log',
                look_for=["The initial write attempt", "Its response", "The retried write attempt"],
                interpretation=[
                    "If the initial write actually succeeded, this is a false-failure/idempotency gap, not a real duplicate.",
                    "If the initial write failed too, investigate why the operation isn't idempotent.",
                ],
            )
        ],
    }


def _retry_exhaustion_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    retry_info = evidence["impact"]["retry_info"]
    return {
        "explanation": (
            f"This group represents retry exhaustion in '{ident['workflow_name']}': "
            f"{retry_info.get('executions_with_retries', 0)} execution(s) retried "
            f"(up to {retry_info.get('max_retry_count', '?')}x) and still failed."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="The underlying failure is not transient, so retrying without addressing the cause does not help",
                confidence="medium",
                supporting_evidence_ids=_ids_for(evidence, "retry_count") or _evidence_ids(evidence)[:1],
                contradicting_evidence_ids=[],
                missing_evidence=["Retry logs (attempt-by-attempt timing and responses)", "Whether the operation is idempotency-protected"],
                verification_steps=["Inspect the per-attempt response for a sample of exhausted retries to confirm the failure is consistent, not transient."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Increase retry attempts with exponential backoff and jitter",
                appropriate_when="The underlying cause is confirmed transient (e.g. brief dependency blips)",
                expected_benefit="Recovers from genuinely transient failures without manual intervention",
                risks=["Does nothing if the cause is not transient; can delay failure detection"],
                tradeoffs=["Cheap to apply, but treats the symptom unless paired with root-cause evidence"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="temporary",
            ),
            RemediationOption(
                option="Add a dead-letter/manual-review path after retry exhaustion",
                appropriate_when="Some failures are expected to need human intervention",
                expected_benefit="Prevents silent data loss when retries are exhausted",
                risks=["Requires a process for reviewing the dead-letter queue"],
                tradeoffs=["Operational overhead, but avoids losing failed work"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add alerting specifically on retry-exhaustion rate, separate from first-attempt failure rate.",
            "Add idempotency controls before increasing retry counts, to avoid amplifying side effects.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Inspect per-attempt timing for an exhausted execution",
                command='grep "<execution_id>" application.log | grep -i retry',
                look_for=["Timestamps and responses for each retry attempt"],
                interpretation=[
                    "If every attempt returns the same error, the cause is not transient — retrying more will not help.",
                    "If attempts show varying errors, investigate an unstable dependency instead.",
                ],
            )
        ],
    }


def _configuration_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    version_ids = _ids_for(evidence, "application_version")
    versions = ", ".join(evidence["occurrence"]["application_versions"])
    return {
        "explanation": (
            f"This group represents configuration/deployment-related failures ('{ident['error_type']}') "
            f"in '{ident['workflow_name']}'" + (f", concentrated on version {versions}" if version_ids else "") + "."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="A configuration value or contract changed as part of a recent deployment and was not updated everywhere it's needed",
                confidence="medium" if version_ids else "low",
                supporting_evidence_ids=version_ids or _evidence_ids(evidence)[:1],
                contradicting_evidence_ids=[],
                missing_evidence=["Deployment timestamp", "Configuration diff between the last-known-good and current state"],
                verification_steps=["Compare configuration/environment variables in effect before and after the suspected deployment."],
            )
        ],
        "containment": [
            ContainmentAction(
                priority=1,
                action="If a recent deployment is implicated and impact is severe, consider rolling back.",
                reason="Reverts to a known-good configuration while the root cause is investigated.",
                risk="A rollback can lose other changes shipped in the same deployment.",
            )
        ],
        "remediation": [
            RemediationOption(
                option="Correct the missing/invalid configuration value",
                appropriate_when="A specific config diff is identified as the cause",
                expected_benefit="Directly resolves the failure",
                risks=["Applying the wrong fix without a confirmed diff can mask the real issue"],
                tradeoffs=["Fast once the diff is known, but requires that evidence first"],
                complexity="low",
                change_type="configuration",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Roll back the deployment that introduced the change",
                appropriate_when="The regression started immediately after a specific deployment and impact is high",
                expected_benefit="Quickly restores prior behavior",
                risks=["Loses any other changes in that deployment", "May not be a full fix if the config change was intentional"],
                tradeoffs=["Fast containment, not necessarily a permanent fix"],
                complexity="low",
                change_type="infrastructure",
                temporary_or_permanent="temporary",
            ),
        ],
        "prevention": [
            "Add a deployment check that validates required configuration/environment variables before rollout.",
            "Add a regression test that pins the expected contract for this integration.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Diff configuration between the last-known-good and current deployment",
                command="diff <(cat config.<last_known_good_version>.yaml) <(cat config.<current_version>.yaml)",
                look_for=["Any changed environment variable or config key relevant to this integration"],
                interpretation=[
                    "If a relevant key changed, this strongly supports a configuration regression.",
                    "If nothing relevant changed, investigate the deployment process itself (e.g. a missed environment variable).",
                ],
            )
        ],
    }


def _slow_execution_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    impact = evidence["impact"]
    return {
        "explanation": (
            f"This group represents successful but slow executions in '{ident['workflow_name']}' "
            f"(avg {impact['avg_duration_seconds']}s, P95 {impact['p95_duration_seconds']}s) — "
            "these are performance anomalies, not failures, and should not be treated as exceptions."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=f"{ident['provider'] or ident['service'] or 'A downstream dependency'} or a resource bottleneck is adding latency without causing outright failures",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:1],
                contradicting_evidence_ids=[],
                missing_evidence=["Dependency latency metrics for the affected window", "Infrastructure resource-utilization metrics"],
                verification_steps=["Compare P95 duration for this step against its historical baseline outside the affected window."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Profile the slow step to identify the bottleneck",
                appropriate_when="The cause of the added latency is not yet known",
                expected_benefit="Identifies whether the bottleneck is code, dependency, or infrastructure",
                risks=["Profiling in production can itself add overhead if not done carefully"],
                tradeoffs=["Investigative step, not a fix on its own"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="temporary",
            ),
        ],
        "prevention": [
            "Add P95/P99 duration monitoring with an alert threshold for this workflow step.",
            "Track this as a performance trend separate from failure-rate dashboards.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Compare this step's duration distribution to its historical baseline",
                command='grep "<workflow_step>" application.log | awk \'{print $duration_field}\' | sort -n',
                look_for=["Whether the whole distribution has shifted upward compared to before the affected window"],
                interpretation=[
                    "If the whole distribution shifted, this is a systemic slowdown (dependency or resource contention).",
                    "If only a few outliers exist, investigate per-execution factors (e.g. large payloads) instead.",
                ],
            )
        ],
    }


def _business_rule_playbook(evidence: dict) -> dict:
    """The requested operation was rejected by a business/policy rule the
    downstream system enforces (a declined card, a messaging window that
    has closed, an opted-out recipient, ...) — not a technical fault on
    either side. The fix is almost always upstream: don't attempt the
    operation when the precondition the rule checks for isn't met."""
    ident = evidence["exception_identity"]
    secondary = ident.get("secondary_category") or "a business rule"
    return {
        "explanation": (
            f"This group represents a business-rule rejection ('{ident['error_type']}', {secondary.lower()}) "
            f"in '{ident['workflow_name']}' — the downstream system is enforcing a policy or precondition, "
            f"not reporting a technical fault."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis=(
                    f"The workflow is attempting this operation without first checking the precondition "
                    f"{secondary.lower()} depends on (e.g. eligibility, timing window, or opt-in status)"
                ),
                confidence="medium",
                supporting_evidence_ids=_evidence_ids(evidence)[:2],
                contradicting_evidence_ids=[],
                missing_evidence=["The upstream state (eligibility/opt-in/timing) at the moment the request was made"],
                verification_steps=["Check whether the precondition was true immediately before the request was sent, not just at some earlier point."],
            )
        ],
        "containment": [
            ContainmentAction(
                priority=1,
                action="Stop retrying this operation automatically — the rule will reject it again until the precondition changes.",
                reason="A business-rule rejection is not transient; blind retries waste calls and can look like a retry storm to the provider.",
                risk="Pausing retries means the affected executions stay failed until a human or an upstream check resolves the precondition.",
            )
        ],
        "remediation": [
            RemediationOption(
                option="Add a precondition check before attempting the operation",
                appropriate_when="The precondition is knowable ahead of time (e.g. checking opt-in status or a timing window before sending)",
                expected_benefit="Prevents the doomed request from being sent at all, instead of failing and needing cleanup",
                risks=["The precondition check itself can go stale between check and use"],
                tradeoffs=["Requires a reliable, up-to-date source for the precondition"],
                complexity="medium",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
            RemediationOption(
                option="Route rejected cases to a fallback path or manual queue",
                appropriate_when="The operation cannot be retried but the business still needs it handled",
                expected_benefit="Nothing is silently dropped when the rule rejects it",
                risks=["Adds an operational queue that needs to be monitored and cleared"],
                tradeoffs=["Operational overhead in exchange for not losing the failed case"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add a regression test that exercises this precondition boundary (e.g. the exact moment a window closes or an opt-out takes effect).",
            "Add monitoring on the rejection rate for this rule, separate from technical failure rates.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Trace a rejected execution's correlation ID to see the precondition state just before the request",
                command='grep "<correlation_id>" application.log',
                look_for=["The last known state of the precondition (eligibility/opt-in/timing) before this request was sent"],
                interpretation=[
                    "If the precondition was already false before the request, fix the upstream check that should have caught it.",
                    "If the precondition changed between check and request, look for a race condition or a stale cache.",
                ],
            )
        ],
    }


def _generic_playbook(evidence: dict) -> dict:
    ident = evidence["exception_identity"]
    return {
        "explanation": (
            f"This group represents '{ident['error_type'] or 'an unclassified exception'}' "
            f"in '{ident['workflow_name']}'. Insufficient evidence is available to classify this with high confidence."
        ),
        "hypotheses": [
            Hypothesis(
                rank=1,
                hypothesis="Insufficient evidence is available to propose a specific cause with confidence",
                confidence="low",
                supporting_evidence_ids=_evidence_ids(evidence)[:1],
                contradicting_evidence_ids=[],
                missing_evidence=list(evidence.get("missing_evidence", [])),
                verification_steps=["Collect additional log fields or context for this exception type."],
            )
        ],
        "containment": [],
        "remediation": [
            RemediationOption(
                option="Add structured logging/context to this failure path",
                appropriate_when="The current evidence is insufficient to diagnose the cause",
                expected_benefit="Enables a confident diagnosis on the next occurrence",
                risks=["No immediate remediation of the underlying issue"],
                tradeoffs=["Investigative investment now, faster resolution later"],
                complexity="low",
                change_type="code",
                temporary_or_permanent="permanent",
            ),
        ],
        "prevention": [
            "Add monitoring/alerting for this error type so recurrence is caught quickly.",
        ],
        "commands": [
            DiagnosticCommand(
                label="Search application logs by error code for additional context",
                command='grep "<error_code>" application.log',
                look_for=["Any additional structured fields not captured in this upload"],
                interpretation=["Use any additional context found to re-classify this exception with more confidence."],
            )
        ],
    }


_DUPLICATE_SECONDARY_CATEGORIES = {"Duplicate record", "Unique-key conflict"}


def _select_playbook(evidence: dict):
    """Dispatches on primary_category first, then secondary_category where
    a single primary category covers genuinely distinct investigation
    stories (e.g. Connectivity and dependency covers DNS, TLS, and generic
    provider-unavailable failures, which need different playbooks)."""
    ident = evidence["exception_identity"]
    category = ident["primary_category"]
    secondary = ident.get("secondary_category") or ""

    if category == "Authentication and authorization":
        return _auth_playbook
    if category == "Timeout and latency":
        return _timeout_playbook
    if category == "Rate limiting and capacity":
        return _rate_limit_playbook
    if category == "Connectivity and dependency":
        if secondary == "DNS failure":
            return _dns_playbook
        if secondary == "TLS failure":
            return _tls_playbook
        return _provider_degradation_playbook
    if category in ("Input and validation", "AI and structured-output"):
        return _schema_playbook
    if category == "Data integrity and persistence":
        return _duplicate_playbook if secondary in _DUPLICATE_SECONDARY_CATEGORIES else _db_transaction_playbook
    if category == "Retry and idempotency":
        return _retry_exhaustion_playbook
    if category == "Configuration and deployment":
        return _configuration_playbook
    if category == "Performance anomaly without failure":
        return _slow_execution_playbook
    if category == "Internal application error":
        return _internal_error_playbook
    if category == "Business-rule failure":
        return _business_rule_playbook
    return _generic_playbook


def generate_offline_investigation(evidence: dict) -> ExceptionInvestigation:
    """Rule-based, template-driven substitute for an LLM call. Uses only the
    same deterministic evidence dict an LLM would receive — no external
    call, no invented data. Dispatches to a category-specific playbook so
    the guidance is tailored, not generic boilerplate."""
    playbook = _select_playbook(evidence)(evidence)

    return ExceptionInvestigation(
        exception_explanation=playbook["explanation"],
        known_facts=_known_facts(evidence),
        unknowns=_unknowns(evidence),
        hypotheses=playbook["hypotheses"],
        containment_actions=_containment(evidence, playbook["containment"]),
        investigation_actions=_generic_investigation_actions(evidence),
        remediation_options=playbook["remediation"],
        verification_plan=_base_verification_plan(evidence),
        prevention_actions=playbook["prevention"],
        limitations=[
            "Rule-based investigation guidance — template-generated from the calculated evidence; "
            "no qualitative interpretation was performed. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for "
            "AI-assisted investigation.",
        ],
        diagnostic_commands=playbook.get("commands", []),
        source="offline",
    )


class OfflineLLMClient(LLMClient):
    def analyze(self, evidence: dict) -> ExceptionInvestigation:
        return generate_offline_investigation(evidence)
