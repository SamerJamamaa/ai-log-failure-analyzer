import json
import os

from anthropic import Anthropic

from src.llm.base import PROVIDER_TIMEOUT_SECONDS, SYSTEM_PROMPT, LLMClient, LLMProviderError, parse_exception_investigation
from src.models import ExceptionInvestigation

MODEL = "claude-sonnet-5"


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str = None):
        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )

    def analyze(self, evidence: dict) -> ExceptionInvestigation:
        raw_text = self._call([{"role": "user", "content": json.dumps(evidence)}])

        report = parse_exception_investigation(raw_text, source="llm")
        if report is not None:
            return report

        retry_text = self._call(
            [
                {"role": "user", "content": json.dumps(evidence)},
                {"role": "assistant", "content": raw_text},
                {
                    "role": "user",
                    "content": "That was not valid JSON matching the schema. "
                    "Respond with ONLY the corrected JSON object.",
                },
            ]
        )
        report = parse_exception_investigation(retry_text, source="llm")
        if report is not None:
            return report

        raise LLMProviderError("invalid_structured_output")

    def _call(self, messages: list) -> str:
        """Sends one request to the Anthropic API. Any failure (timeout,
        auth error, rate limit, connection error, ...) is re-raised as
        LLMProviderError carrying only the exception's type name — never
        the raw exception text — so a failure can't leak request details."""
        try:
            message = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            raise LLMProviderError(type(exc).__name__) from exc
        return "".join(block.text for block in message.content if block.type == "text")
