"""Direct HTTPX adapter for the TypeSafe System One endpoint.

The SDK is deliberately not used: every attempt, status, retry delay and timing
has to be visible to the harness, not handled inside a client.

Request and response shapes follow https://docs.typesafe.ai/api (fetched
2026-09-20) exactly. Nothing here is invented. A response missing a documented
field is a recorded parse failure, never a guess.
"""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Any

import httpx

from ..budget import ContextBudget, TruncationRecord, fit_events, longest_question_chars
from ..costs import PriceManifest, recorded_cost_usd
from ..hashing import canonical_json, sha256_obj
from ..redact import redact
from ..retry import ErrorRateGuard, RetryPolicy
from ..schemas import (
    LABEL_ORDER,
    Attempt,
    Decision,
    InferenceView,
    Label,
    Prediction,
    Usage,
)
from ._transport import post_with_retries

VERDICT_KEY = "verdict"
NOUL_KEYS = ("has_success_claim", "tool_evidence_supports_claim", "needs_review")


class MissingCredentialError(RuntimeError):
    pass


class ParseFailureError(RuntimeError):
    pass


def build_state(view: InferenceView, evaluation_rule: str) -> dict[str, Any]:
    """The `state` object sent to Jev. Tool results are data, never instructions."""
    return {
        "goal": view.goal,
        "tool_schema": [entry.model_dump(mode="json") for entry in view.tool_schema],
        "events": [event.model_dump(mode="json", exclude_none=True) for event in view.events],
        "evaluation_rule": evaluation_rule,
    }


def state_envelope_chars(view: InferenceView, evaluation_rule: str) -> int:
    """Length of everything in ``state`` that is not the events.

    The goal, the tool schema and the evaluation rule are paid for before any
    trace content and are never reduced: the goal *is* the question being asked,
    and reducing it would change what the evaluator was asked to judge.
    """
    envelope = {
        "goal": view.goal,
        "tool_schema": [entry.model_dump(mode="json") for entry in view.tool_schema],
        "events": [],
        "evaluation_rule": evaluation_rule,
    }
    return len(canonical_json(envelope))


class JevEvaluator:
    provider = "jev"

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        base_url: str,
        path: str,
        questions: dict[str, Any],
        evaluation_rule: str,
        prices: PriceManifest,
        policy: RetryPolicy | None = None,
        threshold: float = 0.5,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        rng: random.Random | None = None,
        sleep: Any = time.sleep,
        guard: ErrorRateGuard | None = None,
        budget: ContextBudget | None = None,
    ) -> None:
        if not api_key:
            raise MissingCredentialError(
                "TYPESAFE_API_KEY is not set. Export it in your shell; this harness "
                "never accepts a key as a CLI argument and never reads one from a file."
            )
        self._api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.questions = questions
        self.evaluation_rule = evaluation_rule
        self.prices = prices
        self.policy = policy or RetryPolicy()
        self.threshold = threshold
        self.timeout_s = timeout_s
        self._rng = rng or random.Random(0)
        self._sleep = sleep
        self.guard = guard or ErrorRateGuard()
        # Real traces are long enough to overflow the context window. The budget
        # is applied per request and every reduction is recorded, so a run can
        # be read back and told apart from one that sent every trace whole.
        self.budget = budget or ContextBudget()
        self._longest_question_chars = longest_question_chars(questions)
        self.truncations: list[TruncationRecord] = []
        self._truncation_lock = threading.Lock()
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            base_url=self.base_url,
        )

    # -- plumbing -------------------------------------------------------
    @property
    def questions_sha256(self) -> str:
        return sha256_obj(self.questions)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JevEvaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(
        self, body: dict[str, Any]
    ) -> tuple[httpx.Response | None, list[Attempt], str | None]:
        return post_with_retries(
            self._client,
            self.path,
            body,
            self._headers(),
            policy=self.policy,
            rng=self._rng,
            sleep=self._sleep,
        )

    # -- parsing --------------------------------------------------------
    def parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Turn a documented response body into the fields this harness records."""
        if "answers" not in payload:
            raise ParseFailureError("response has no 'answers' object")
        answers = payload["answers"]
        if VERDICT_KEY not in answers:
            raise ParseFailureError(f"response has no {VERDICT_KEY!r} answer")

        verdict = answers[VERDICT_KEY]
        probabilities_raw = verdict.get("probabilities")
        if not isinstance(probabilities_raw, dict):
            raise ParseFailureError("verdict answer has no 'probabilities' map")
        probabilities = {label.value: 0.0 for label in LABEL_ORDER}
        for name, value in probabilities_raw.items():
            if name not in probabilities:
                raise ParseFailureError(f"verdict returned an unknown option {name!r}")
            probabilities[name] = float(value)

        choice = verdict.get("choice")
        if choice not in probabilities:
            raise ParseFailureError(f"verdict 'choice' is {choice!r}, not one of the four labels")

        nouls: dict[str, float | None] = {}
        for key in NOUL_KEYS:
            answer = answers.get(key)
            if answer is None:
                raise ParseFailureError(f"response has no {key!r} answer")
            if "noul" not in answer:
                raise ParseFailureError(f"{key!r} answer has no 'noul' value")
            nouls[key] = float(answer["noul"])

        usage_raw = payload.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
        )

        return {
            "model": str(payload.get("model", self.model_id)),
            "label": Label(choice),
            "probabilities": probabilities,
            # Noul answers carry no confidence; only Choice and Score do.
            # See https://docs.typesafe.ai/confidence -- never synthesised here.
            "confidence": (
                float(verdict["confidence"]) if verdict.get("confidence") is not None else None
            ),
            "nouls": nouls,
            "usage": usage,
        }

    # -- prediction -----------------------------------------------------
    def fit_to_budget(self, view: InferenceView, trace_id: str) -> InferenceView:
        """Reduce a trace to the context budget, recording it if anything changed.

        A trace that already fits is returned untouched and nothing is recorded,
        so a run with an empty truncation file provably sent every trace whole.
        """
        events, record = fit_events(
            trace_id,
            view.events,
            budget=self.budget,
            envelope_chars=state_envelope_chars(view, self.evaluation_rule),
            longest_question_chars=self._longest_question_chars,
        )
        if record is None:
            return view
        with self._truncation_lock:
            self.truncations.append(record)
        return view.model_copy(update={"events": events})

    def predict(self, view: InferenceView, trace_id: str, repeat: int = 0) -> Prediction:
        view = self.fit_to_budget(view, trace_id)
        body = {
            "state": build_state(view, self.evaluation_rule),
            "model": self.model_id,
            "questions": self.questions,
        }
        raw_request = redact(
            {"url": f"{self.base_url}{self.path}", "headers": self._headers(), "body": body}
        )

        wall_start = time.perf_counter()
        response, attempts, transport_error = self._post(body)
        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0

        def failed(error: str, raw_response: dict[str, Any] | None) -> Prediction:
            self.guard.record(failed=True)
            return Prediction(
                trace_id=trace_id,
                provider=self.provider,
                model_id=self.model_id,
                repeat=repeat,
                threshold=self.threshold,
                decision=Decision.error,
                attempts=tuple(attempts),
                end_to_end_latency_ms=elapsed_ms,
                raw_request=raw_request,
                raw_response=raw_response,
                cost_usd=None,
                error=error,
            )

        if response is None:
            return failed(transport_error or "no response", None)

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            return failed(
                f"ParseFailureError: response body is not JSON ({exc})",
                redact({"status": response.status_code, "text": response.text[:4000]}),
            )

        raw_response = redact(
            {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": payload,
            }
        )

        if response.status_code >= 400:
            return failed(transport_error or f"HTTP {response.status_code}", raw_response)

        try:
            parsed = self.parse(payload)
        except ParseFailureError as exc:
            return failed(f"ParseFailureError: {exc}", raw_response)

        self.guard.record(failed=False)
        score = parsed["probabilities"][Label.unsupported_success.value]
        usage = parsed["usage"]

        return Prediction(
            trace_id=trace_id,
            provider=self.provider,
            # The versioned ID the API reports, not the one we asked for.
            model_id=parsed["model"],
            repeat=repeat,
            predicted_label=parsed["label"],
            primary_score=score,
            probabilities=parsed["probabilities"],
            confidence=parsed["confidence"],
            has_success_claim=parsed["nouls"]["has_success_claim"],
            tool_evidence_supports_claim=parsed["nouls"]["tool_evidence_supports_claim"],
            # Kept separate from primary_score by design: Jev does not guarantee
            # arithmetic identities between a Noul and a Choice option.
            needs_review=parsed["nouls"]["needs_review"],
            threshold=self.threshold,
            decision=Decision.flag if score >= self.threshold else Decision.pass_,
            usage=usage,
            attempts=tuple(attempts),
            end_to_end_latency_ms=elapsed_ms,
            raw_request=raw_request,
            raw_response=raw_response,
            cost_usd=recorded_cost_usd(self.prices, self.model_id, usage),
            error=None,
        )

    def smoke(self) -> dict[str, Any]:
        """One tiny non-scored request, used only to prove connectivity."""
        body = {
            "state": {"note": "connectivity check from jev-false-success-eval"},
            "model": self.model_id,
            "questions": {
                "reachable": {
                    "type": "noul",
                    "instructions": "Is this a connectivity check?",
                }
            },
        }
        response, attempts, error = self._post(body)
        if response is None:
            raise RuntimeError(f"smoke request failed: {error}")
        payload: dict[str, Any] = response.json() if response.content else {}
        redacted: dict[str, Any] = redact(
            {
                "status": response.status_code,
                "attempts": [a.model_dump(mode="json") for a in attempts],
                "body": payload,
            }
        )
        return redacted
