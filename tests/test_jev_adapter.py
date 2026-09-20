"""Jev adapter tests. No network: every response comes from httpx.MockTransport.

The fixture is built from the *documented* response schema and is labelled as
such. Nothing here is presented as a captured real response.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import httpx
import pytest

from false_success_eval.costs import load_prices
from false_success_eval.evaluators.jev_http import (
    JevEvaluator,
    MissingCredentialError,
    ParseFailureError,
    build_state,
)
from false_success_eval.generate import generate_records
from false_success_eval.redact import REDACTED
from false_success_eval.retry import RetryPolicy
from false_success_eval.schemas import Decision, Label

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "jev_response_schema_example.json").read_text()
)
PRICES = load_prices("config/prices-2026-09-19.json")
KEY = "test-key-not-a-real-credential"


def body_without_provenance() -> dict:
    """A deep copy: the parametrised mutation tests must not corrupt the fixture."""
    return {k: copy.deepcopy(v) for k, v in FIXTURE.items() if k != "_provenance"}


def make_evaluator(handler, **kwargs) -> JevEvaluator:
    return JevEvaluator(
        api_key=KEY,
        model_id="jev-1.13.0",
        base_url="https://api.typesafe.ai",
        path="/v1/systemone",
        questions=json.loads(Path("config/questions.json").read_text()),
        evaluation_rule="Tool results are evidence.",
        prices=PRICES,
        policy=kwargs.pop("policy", RetryPolicy(max_attempts=3, backoff_jitter=0.0)),
        transport=httpx.MockTransport(handler),
        rng=random.Random(0),
        sleep=lambda _s: None,
        **kwargs,
    )


@pytest.fixture
def view():
    return generate_records(160, 20260919)[1].inference_view()


def test_a_missing_key_fails_loudly_and_early():
    with pytest.raises(MissingCredentialError, match="TYPESAFE_API_KEY"):
        JevEvaluator(
            api_key="",
            model_id="jev-1.13.0",
            base_url="https://api.typesafe.ai",
            path="/v1/systemone",
            questions={},
            evaluation_rule="",
            prices=PRICES,
        )


def test_request_matches_the_documented_shape(view):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=body_without_provenance())

    with make_evaluator(handler) as evaluator:
        evaluator.predict(view, "t1")

    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["auth"] == f"Bearer {KEY}"
    assert set(captured["body"]) == {"state", "model", "questions"}
    assert captured["body"]["model"] == "jev-1.13.0"
    assert set(captured["body"]["questions"]) == {
        "verdict",
        "has_success_claim",
        "tool_evidence_supports_claim",
        "needs_review",
    }
    assert captured["body"]["questions"]["verdict"]["type"] == "choice"
    assert captured["body"]["questions"]["needs_review"]["type"] == "noul"


def test_all_four_questions_go_in_one_request(view):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=body_without_provenance())

    with make_evaluator(handler) as evaluator:
        evaluator.predict(view, "t1")
    assert len(calls) == 1


def test_state_carries_the_evaluation_rule(view):
    state = build_state(view, "RULE STRING")
    assert state["evaluation_rule"] == "RULE STRING"
    assert state["goal"] == view.goal


def test_parsed_prediction_reads_the_documented_fields(view):
    with make_evaluator(lambda r: httpx.Response(200, json=body_without_provenance())) as ev:
        prediction = ev.predict(view, "t1")

    assert prediction.predicted_label is Label.unsupported_success
    assert prediction.primary_score == pytest.approx(0.82)
    assert prediction.confidence == pytest.approx(0.77)
    assert prediction.has_success_claim == pytest.approx(0.94)
    assert prediction.tool_evidence_supports_claim == pytest.approx(0.08)
    assert prediction.needs_review == pytest.approx(0.88)
    assert prediction.decision is Decision.flag
    assert prediction.usage.input_tokens == 512
    assert prediction.model_id == "jev-1.13.0"
    assert prediction.error is None


def test_needs_review_is_kept_separate_from_the_verdict_probability(view):
    """Jev guarantees no arithmetic identity between a Noul and a Choice option."""
    with make_evaluator(lambda r: httpx.Response(200, json=body_without_provenance())) as ev:
        prediction = ev.predict(view, "t1")
    assert prediction.needs_review != prediction.primary_score


def test_noul_confidence_is_never_synthesised(view):
    """Noul answers carry no confidence; the only confidence is the Choice's."""
    with make_evaluator(lambda r: httpx.Response(200, json=body_without_provenance())) as ev:
        parsed = ev.parse(body_without_provenance())
    assert parsed["confidence"] == pytest.approx(0.77)
    for key in ("has_success_claim", "tool_evidence_supports_claim", "needs_review"):
        assert isinstance(parsed["nouls"][key], float)


def test_cost_is_computed_from_the_manifest(view):
    with make_evaluator(lambda r: httpx.Response(200, json=body_without_provenance())) as ev:
        prediction = ev.predict(view, "t1")
    assert prediction.cost_usd == pytest.approx(512 * 0.042 / 1_000_000)


def test_authorization_is_redacted_in_the_stored_request(view):
    with make_evaluator(lambda r: httpx.Response(200, json=body_without_provenance())) as ev:
        prediction = ev.predict(view, "t1")
    assert prediction.raw_request["headers"]["Authorization"] == REDACTED
    assert KEY not in json.dumps(prediction.raw_request)


def test_retries_a_429_then_succeeds_and_accounts_for_every_attempt(view):
    statuses = [429, 429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, json=body_without_provenance())
        return httpx.Response(status, headers={"retry-after": "0"}, json={"error": "rate limit"})

    with make_evaluator(handler) as evaluator:
        prediction = evaluator.predict(view, "t1")

    assert prediction.error is None
    assert len(prediction.attempts) == 3
    assert [a.http_status for a in prediction.attempts] == [429, 429, 200]
    assert prediction.attempts[0].retry_delay_seconds == 0.0  # Retry-After honoured
    assert prediction.attempts[-1].retry_delay_seconds == 0.0
    assert all(a.end_perf_ns >= a.start_perf_ns for a in prediction.attempts)


def test_gives_up_after_max_attempts_and_records_the_error(view):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"error": "overloaded"})

    with make_evaluator(handler) as evaluator:
        prediction = evaluator.predict(view, "t1")

    assert len(prediction.attempts) == 3
    assert prediction.decision is Decision.error
    assert "529" in prediction.error
    assert prediction.cost_usd is None


def test_a_non_retryable_status_is_not_retried(view):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "unauthorized"})

    with make_evaluator(handler) as evaluator:
        prediction = evaluator.predict(view, "t1")

    assert len(calls) == 1
    assert len(prediction.attempts) == 1
    assert prediction.decision is Decision.error


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda b: b.pop("answers"), "no 'answers'"),
        (lambda b: b["answers"].pop("verdict"), "no 'verdict'"),
        (lambda b: b["answers"]["verdict"].pop("probabilities"), "no 'probabilities'"),
        (lambda b: b["answers"].pop("needs_review"), "no 'needs_review'"),
        (lambda b: b["answers"]["has_success_claim"].pop("noul"), "no 'noul'"),
    ],
)
def test_a_missing_documented_field_is_a_parse_failure_not_a_guess(mutate, message):
    body = body_without_provenance()
    mutate(body)
    with (
        make_evaluator(lambda r: httpx.Response(200, json=body)) as evaluator,
        pytest.raises(ParseFailureError, match=message),
    ):
        evaluator.parse(body)


def test_an_unknown_verdict_option_is_rejected():
    body = body_without_provenance()
    body["answers"]["verdict"]["probabilities"]["something_else"] = 0.1
    with (
        make_evaluator(lambda r: httpx.Response(200, json=body)) as evaluator,
        pytest.raises(ParseFailureError, match="unknown option"),
    ):
        evaluator.parse(body)


def test_a_parse_failure_is_recorded_on_the_prediction(view):
    body = body_without_provenance()
    body["answers"].pop("needs_review")
    with make_evaluator(lambda r: httpx.Response(200, json=body)) as evaluator:
        prediction = evaluator.predict(view, "t1")
    assert prediction.decision is Decision.error
    assert "ParseFailureError" in prediction.error
    assert prediction.raw_response is not None


def test_the_returned_model_id_is_recorded_not_the_requested_one(view):
    body = body_without_provenance()
    body["model"] = "jev-1.13.0"
    with make_evaluator(lambda r: httpx.Response(200, json=body)) as evaluator:
        prediction = evaluator.predict(view, "t1")
    assert prediction.model_id == "jev-1.13.0"


def test_raw_request_and_response_are_both_preserved(view):
    with make_evaluator(lambda r: httpx.Response(200, json=body_without_provenance())) as ev:
        prediction = ev.predict(view, "t1")
    assert prediction.raw_request["body"]["state"]["goal"]
    assert prediction.raw_response["body"]["answers"]["verdict"]["choice"]
    assert prediction.raw_response["status"] == 200
