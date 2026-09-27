import json
import os

from openai import OpenAI

from src.llm.base import PROVIDER_TIMEOUT_SECONDS, SYSTEM_PROMPT, LLMClient, LLMProviderError, parse_exception_investigation
from src.models import ExceptionInvestigation

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
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
        """Sends one request to the OpenAI API. Any failure (timeout, auth
        error, rate limit, connection error, ...) is re-raised as
        LLMProviderError carrying only the exception's type name — never
        the raw exception text — so a failure can't leak request details."""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            )
        except Exception as exc:
            raise LLMProviderError(type(exc).__name__) from exc
        return response.choices[0].message.content or ""
