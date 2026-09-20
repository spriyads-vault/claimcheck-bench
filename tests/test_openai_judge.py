"""General-judge adapter tests. No network: every response is a MockTransport.

The fixture is built from OpenAI's own published API description and is labelled
as such. Nothing here is presented as a captured real response, and no model
name in this file is a model this repository chose to run.

The tests that matter most are the parity ones. This arm only means anything if
the general judge and Jev are shown the same trace and asked the same questions,
so that is asserted against the code rather than left to a comment.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import httpx
import pytest

from false_success_eval.budget import ContextBudget
from false_success_eval.costs import load_prices
from false_success_eval.evaluators import general_model
from false_success_eval.evaluators.general_model import (
    GeneralModelConfig,
    GeneralModelEvaluator,
    MissingCredentialError,
    OpenAIChatJudge,
    ParseFailureError,
    build_instructions,
    not_run_reason,
    response_schema,
)
from false_success_eval.evaluators.jev_http import JevEvaluator, build_state
from false_success_eval.generate import generate_records
from false_success_eval.redact import REDACTED
from false_success_eval.retry import RetryPolicy
from false_success_eval.schemas import LABEL_ORDER, Decision, Label

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openai_chat_completion_schema_example.json").read_text()
)
QUESTIONS = json.loads(Path("config/questions.json").read_text())
PRICES = load_prices("config/prices-2026-09-20-2.json")
KEY = "test-key-not-a-real-credential"
MODEL = "a-model-id-from-the-listing"
RULE = "Tool results are evidence."


def body_without_provenance() -> dict:
    return {k: copy.deepcopy(v) for k, v in FIXTURE.items() if k != "_provenance"}


def make_judge(handler, **kwargs) -> OpenAIChatJudge:
    return OpenAIChatJudge(
        api_key=KEY,
        model_id=kwargs.pop("model_id", MODEL),
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        questions=QUESTIONS,
        evaluation_rule=RULE,
        prices=PRICES,
        policy=kwargs.pop("policy", RetryPolicy(max_attempts=3, backoff_jitter=0.0)),
        transport=httpx.MockTransport(handler),
        rng=random.Random(0),
        sleep=lambda _s: None,
        **kwargs,
    )


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=body_without_provenance())


@pytest.fixture
def view():
    return generate_records(160, 20260919)[1].inference_view()


# ---------------------------------------------------------------------------
# Refusal is still the default
# ---------------------------------------------------------------------------


def test_an_unconfigured_vendor_still_records_not_run_rather_than_guessing(view):
    config = GeneralModelConfig.from_mapping("gemini", {"api_key_env": "GEMINI_API_KEY"})
    prediction = GeneralModelEvaluator(config).predict(view, "t-1")
    assert prediction.decision is Decision.not_run
    assert prediction.primary_score is None
    assert "disabled in config" in (prediction.error or "")


def test_an_empty_model_id_is_a_refusal_not_a_default():
    """The whole point of the listing command: no name is ever assumed."""
    reason = not_run_reason(
        GeneralModelConfig.from_mapping(
            "openai",
            {
                "enabled": True,
                "base_url": "https://api.openai.com",
                "path": "/v1/chat/completions",
                "api": "openai_chat_completions",
                "model_id": "",
                "api_key_env": "OPENAI_API_KEY",
            },
        )
    )
    assert reason is not None
    assert "model_id" in reason


def test_an_unverified_request_shape_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    reason = not_run_reason(
        GeneralModelConfig.from_mapping(
            "someone_else",
            {
                "enabled": True,
                "base_url": "https://example.invalid",
                "path": "/v1/chat",
                "model_id": "m",
                "api": "some_api_nobody_verified",
                "api_key_env": "OPENAI_API_KEY",
            },
        )
    )
    assert reason is not None
    assert "no verified request shape" in reason


def test_the_config_this_repository_ships_is_ready_apart_from_the_model_and_the_key():
    """The shipped block must be complete except the two things only a human supplies."""
    import yaml

    raw = yaml.safe_load(Path("config/eval.yaml").read_text())["general_model"]["openai"]
    assert raw["enabled"] is True
    assert raw["base_url"] == "https://api.openai.com"
    assert raw["path"] == "/v1/chat/completions"
    assert raw["api"] == "openai_chat_completions"
    assert raw["api_key_env"] == "OPENAI_API_KEY"
    assert raw["temperature"] == 0.0
    assert raw["model_id"] == "", (
        "a model id must not be committed: it has to be chosen from the account's own "
        "GET /v1/models listing, or the run records a name nobody verified"
    )


def test_no_key_is_a_clear_refusal_not_a_silent_unauthenticated_call():
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY"):
        OpenAIChatJudge(
            api_key="",
            model_id=MODEL,
            base_url="https://api.openai.com",
            path="/v1/chat/completions",
            questions=QUESTIONS,
            evaluation_rule=RULE,
            prices=PRICES,
        )


# ---------------------------------------------------------------------------
# Parity with Jev -- the reason this arm exists at all
# ---------------------------------------------------------------------------


def test_the_judge_is_sent_byte_identical_state_to_jev(view):
    """Same goal, same tool schema, same ordered events, same evaluation rule."""
    judge = make_judge(ok_handler)
    sent = json.loads(judge.build_body(view)["messages"][1]["content"])
    assert sent == build_state(view, RULE)
    judge.close()


def test_the_four_questions_come_from_the_same_frozen_file_jev_is_sent():
    """Not retyped for the judge: generated from questions.json, both of them."""
    schema = response_schema(QUESTIONS)
    assert set(schema["properties"]) == set(QUESTIONS)
    assert schema["required"] == list(QUESTIONS)

    instructions = build_instructions(QUESTIONS, RULE)
    for key, spec in QUESTIONS.items():
        assert key in instructions
        assert spec["instructions"] in instructions
        for description in (spec.get("criteria") or {}).values():
            assert description in instructions


def test_the_verdict_schema_covers_exactly_the_four_labels():
    verdict = response_schema(QUESTIONS)["properties"]["verdict"]
    labels = [label.value for label in LABEL_ORDER]
    assert verdict["properties"]["choice"]["enum"] == labels
    assert verdict["properties"]["probabilities"]["required"] == labels
    assert verdict["properties"]["probabilities"]["additionalProperties"] is False


def test_the_request_is_deterministic_and_strictly_schema_bound(view):
    judge = make_judge(ok_handler)
    body = judge.build_body(view)
    assert body["temperature"] == 0.0
    assert body["model"] == MODEL
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == response_schema(QUESTIONS)
    assert [m["role"] for m in body["messages"]] == ["developer", "user"]
    judge.close()


def test_both_arms_reduce_an_oversized_trace_identically(view):
    """A trace either fits for both lanes or is shortened the same way for both."""
    budget = ContextBudget(total_tokens=2000, state_tokens=1000, chars_per_token=2.5)
    judge = make_judge(ok_handler, budget=budget)
    jev = JevEvaluator(
        api_key=KEY,
        model_id="jev-1.13.0",
        base_url="https://api.typesafe.ai",
        path="/v1/systemone",
        questions=QUESTIONS,
        evaluation_rule=RULE,
        prices=PRICES,
        transport=httpx.MockTransport(ok_handler),
        budget=budget,
    )
    judge_view = judge.fit_to_budget(view, "t-1")
    jev_view = jev.fit_to_budget(view, "t-1")
    assert judge_view.events == jev_view.events
    assert [r.as_dict() for r in judge.truncations] == [r.as_dict() for r in jev.truncations]
    judge.close()
    jev.close()


def test_the_primary_score_is_p_unsupported_success_exactly_as_it_is_for_jev(view):
    judge = make_judge(ok_handler)
    prediction = judge.predict(view, "t-1")
    answer = json.loads(FIXTURE["choices"][0]["message"]["content"])
    assert prediction.primary_score == answer["verdict"]["probabilities"]["unsupported_success"]
    assert prediction.predicted_label is Label.unsupported_success
    assert prediction.decision is Decision.flag
    judge.close()


# ---------------------------------------------------------------------------
# Parsing: a malformed answer is a recorded failure, never a guess
# ---------------------------------------------------------------------------


def test_usage_is_read_from_the_documented_field_names(view):
    judge = make_judge(ok_handler)
    prediction = judge.predict(view, "t-1")
    assert prediction.usage is not None
    assert prediction.usage.input_tokens == FIXTURE["usage"]["prompt_tokens"]
    assert prediction.usage.output_tokens == FIXTURE["usage"]["completion_tokens"]
    judge.close()


def test_the_model_id_recorded_is_the_one_the_api_reports_not_the_one_asked_for(view):
    judge = make_judge(ok_handler)
    prediction = judge.predict(view, "t-1")
    assert prediction.model_id == FIXTURE["model"]
    assert prediction.model_id != MODEL
    judge.close()


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda b: b.pop("choices"), "no 'choices'"),
        (
            lambda b: b["choices"][0]["message"].__setitem__("refusal", "I cannot help"),
            "model refused",
        ),
        (
            lambda b: b["choices"][0].__setitem__("finish_reason", "length"),
            "cut off",
        ),
        (
            lambda b: b["choices"][0]["message"].__setitem__("content", "not json"),
            "not JSON",
        ),
    ],
)
def test_a_malformed_response_is_a_recorded_parse_failure(view, mutate, expected):
    body = body_without_provenance()
    mutate(body)

    judge = make_judge(lambda request: httpx.Response(200, json=body))
    prediction = judge.predict(view, "t-1")
    assert prediction.decision is Decision.error
    assert "ParseFailureError" in (prediction.error or "")
    assert expected in (prediction.error or "")
    assert prediction.primary_score is None
    judge.close()


def test_a_missing_label_in_the_distribution_is_a_failure_not_a_zero():
    judge = make_judge(ok_handler)
    body = body_without_provenance()
    answer = json.loads(body["choices"][0]["message"]["content"])
    answer["verdict"]["probabilities"].pop("no_success_claim")
    body["choices"][0]["message"]["content"] = json.dumps(answer)
    with pytest.raises(ParseFailureError, match="omit"):
        judge.parse(body)
    judge.close()


def test_an_out_of_range_probability_is_a_failure_not_a_clamp():
    """Clamping would turn an out-of-contract answer into a usable score."""
    judge = make_judge(ok_handler)
    body = body_without_provenance()
    answer = json.loads(body["choices"][0]["message"]["content"])
    answer["verdict"]["probabilities"]["unsupported_success"] = 1.4
    body["choices"][0]["message"]["content"] = json.dumps(answer)
    with pytest.raises(ParseFailureError, match=r"outside \[0, 1\]"):
        judge.parse(body)
    judge.close()


def test_an_unknown_verdict_option_is_refused():
    judge = make_judge(ok_handler)
    body = body_without_provenance()
    answer = json.loads(body["choices"][0]["message"]["content"])
    answer["verdict"]["probabilities"]["some_fifth_label"] = 0.0
    body["choices"][0]["message"]["content"] = json.dumps(answer)
    with pytest.raises(ParseFailureError, match="unknown options"):
        judge.parse(body)
    judge.close()


# ---------------------------------------------------------------------------
# Instrumentation: the same as Jev's, or the operational columns mean
# different things in each lane
# ---------------------------------------------------------------------------


def test_a_retryable_status_is_retried_and_every_attempt_is_recorded(view):
    seen = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json=body_without_provenance())

    judge = make_judge(flaky)
    prediction = judge.predict(view, "t-1")
    assert prediction.decision is Decision.flag
    assert len(prediction.attempts) == 3
    assert [a.http_status for a in prediction.attempts] == [429, 429, 200]
    judge.close()


def test_a_persistent_error_is_recorded_not_raised(view):
    judge = make_judge(lambda request: httpx.Response(500, json={"error": "boom"}))
    prediction = judge.predict(view, "t-1")
    assert prediction.decision is Decision.error
    assert prediction.primary_score is None
    assert len(prediction.attempts) == 3
    judge.close()


def test_the_key_never_reaches_a_written_artifact(view, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    judge = make_judge(ok_handler)
    prediction = judge.predict(view, "t-1")
    blob = json.dumps(prediction.model_dump(mode="json"))
    assert KEY not in blob
    assert REDACTED in blob
    judge.close()


# ---------------------------------------------------------------------------
# Cost arithmetic
# ---------------------------------------------------------------------------


def test_an_unpriced_judge_records_tokens_and_a_null_cost_rather_than_a_guess(view):
    """The state this arm ships in until a published rate is on record."""
    judge = make_judge(ok_handler, model_id="OPENAI_GENERAL_JUDGE_UNPRICED")
    prediction = judge.predict(view, "t-1")
    assert prediction.cost_usd is None
    assert prediction.usage is not None
    assert prediction.usage.input_tokens > 0
    assert prediction.usage.output_tokens > 0
    judge.close()


def test_the_judge_is_billed_for_output_as_well_as_input():
    """Jev's output is free; a general model's is not, and both legs are priced."""
    from false_success_eval.costs import (
        MTOK,
        ModelPrice,
        PriceManifest,
        cost_usd,
        project_cost,
    )
    from false_success_eval.schemas import Usage

    manifest = PriceManifest(
        path="test-only-manifest",
        manifest_date="2026-09-20",
        currency="USD",
        models={
            "priced-judge": ModelPrice(
                model_id="priced-judge",
                input_usd_per_mtok=2.0,
                output_usd_per_mtok=8.0,
                verified=True,
                source="test fixture, not a published rate",
            )
        },
    )
    cost = cost_usd(manifest, "priced-judge", Usage(input_tokens=1000, output_tokens=500))
    assert cost == pytest.approx((1000 * 2.0 + 500 * 8.0) / MTOK)

    projection = project_cost(
        manifest,
        "priced-judge",
        request_chars=4000,
        n_requests=2,
        chars_per_token_low=2.5,
        chars_per_token_high=4.0,
        fixed_overhead_tokens=0,
        output_tokens_per_request=100,
    )
    assert projection.output_tokens_total == 200
    expected_output = 200 * 8.0 / MTOK
    assert projection.usd_low == pytest.approx(1000 * 2.0 / MTOK + expected_output)
    assert projection.usd_high == pytest.approx(1600 * 2.0 / MTOK + expected_output)
    assert projection.usd_low < projection.usd_high


def test_an_unpriced_model_projects_no_cost_at_all_rather_than_a_partial_one():
    from false_success_eval.costs import project_cost

    projection = project_cost(
        PRICES,
        "OPENAI_GENERAL_JUDGE_UNPRICED",
        request_chars=10_000,
        n_requests=5,
        chars_per_token_low=2.5,
        chars_per_token_high=4.0,
        fixed_overhead_tokens=290,
        output_tokens_per_request=1200,
    )
    assert projection.usd_low is None
    assert projection.usd_high is None
    # Tokens are still projected: the bill can be reconstructed from a rate
    # added later with a citation.
    assert projection.input_tokens_low > 0
    assert projection.output_tokens_total == 6000


def test_jevs_projection_is_unchanged_by_the_output_leg():
    """Output tokens are free on Jev's manifest, so its band must not move."""
    from false_success_eval.costs import project_cost

    kwargs = {
        "request_chars": 50_000,
        "n_requests": 10,
        "chars_per_token_low": 2.5,
        "chars_per_token_high": 4.0,
        "fixed_overhead_tokens": 290,
    }
    without = project_cost(PRICES, "jev-1.13.0", **kwargs)
    with_output = project_cost(PRICES, "jev-1.13.0", output_tokens_per_request=1200, **kwargs)
    assert without.usd_low == pytest.approx(with_output.usd_low)
    assert without.usd_high == pytest.approx(with_output.usd_high)


# ---------------------------------------------------------------------------
# A model that refuses a decoding parameter: recorded, never hidden
# ---------------------------------------------------------------------------
#
# This is not hypothetical. The first live run of this arm produced 702
# predictions and 702 errors, every one of them:
#
#   HTTP 400 {"error": {"code": "unsupported_value", "param": "temperature",
#     "type": "invalid_request_error", "message": "Unsupported value:
#     'temperature' does not support 0.0 with this model. Only the default (1)
#     value is supported."}}
#
# The fixture below is that body, verbatim from
# runs/appworld/openai-test-20260920T042135Z-6e941d/predictions.jsonl.

TEMPERATURE_REFUSAL = {
    "error": {
        "code": "unsupported_value",
        "message": (
            "Unsupported value: 'temperature' does not support 0.0 with this model. "
            "Only the default (1) value is supported."
        ),
        "param": "temperature",
        "type": "invalid_request_error",
    }
}


def refusing_then_ok(seen: dict) -> object:
    """A model that rejects `temperature` at any value and answers without it."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.setdefault("bodies", []).append(body)
        if "temperature" in body:
            return httpx.Response(400, json=copy.deepcopy(TEMPERATURE_REFUSAL))
        return httpx.Response(200, json=body_without_provenance())

    return handler


def test_the_run_that_failed_702_times_is_reproduced_without_the_fallback(view, monkeypatch):
    """With nothing droppable this is exactly what happened: every trace an error."""
    monkeypatch.setattr(general_model, "DROPPABLE_PARAMS", frozenset())
    judge = make_judge(lambda request: httpx.Response(400, json=TEMPERATURE_REFUSAL))
    prediction = judge.predict(view, "t-1")
    assert prediction.decision is Decision.error
    assert prediction.error == "HTTP 400"
    assert prediction.primary_score is None
    assert judge.deviations == []
    judge.close()


def test_a_refused_temperature_is_dropped_and_the_trace_is_still_scored(view):
    seen: dict = {}
    judge = make_judge(refusing_then_ok(seen))
    prediction = judge.predict(view, "t-1")

    assert prediction.decision is Decision.flag
    assert prediction.primary_score is not None
    # Two rounds: the configured request, then the same request without the
    # parameter the model refused.
    assert len(seen["bodies"]) == 2
    assert seen["bodies"][0]["temperature"] == 0.0
    assert "temperature" not in seen["bodies"][1]
    judge.close()


def test_dropping_temperature_changes_nothing_else_about_the_request(view):
    """The trace, the questions and the strict schema must survive untouched."""
    seen: dict = {}
    judge = make_judge(refusing_then_ok(seen))
    judge.predict(view, "t-1")
    first, second = seen["bodies"]
    assert second == {k: v for k, v in first.items() if k != "temperature"}
    assert second["messages"] == first["messages"]
    assert second["response_format"]["json_schema"]["strict"] is True
    assert second["max_completion_tokens"] == first["max_completion_tokens"]
    judge.close()


def test_the_deviation_is_recorded_with_the_providers_own_words(view):
    judge = make_judge(refusing_then_ok({}))
    judge.predict(view, "t-1")

    (deviation,) = judge.deviations
    assert deviation.parameter == "temperature"
    assert deviation.requested == 0.0
    assert deviation.http_status == 400
    assert deviation.error_code == "unsupported_value"
    assert deviation.error_message == TEMPERATURE_REFUSAL["error"]["message"]
    assert deviation.first_seen_trace_id == "t-1"
    assert deviation.model_id == MODEL
    assert "omitted" in deviation.applied
    judge.close()


def test_every_attempt_of_both_rounds_stays_in_the_record(view):
    """The operational columns must count what the run actually sent."""
    seen: dict = {}
    judge = make_judge(refusing_then_ok(seen))
    prediction = judge.predict(view, "t-1")
    assert [a.http_status for a in prediction.attempts] == [400, 200]
    judge.close()


def test_the_refusal_is_learned_once_not_paid_for_on_every_trace(view):
    seen: dict = {}
    judge = make_judge(refusing_then_ok(seen))
    judge.predict(view, "t-1")
    judge.predict(view, "t-2")
    judge.predict(view, "t-3")

    # One probe, then every later trace goes out correct the first time.
    assert sum(1 for b in seen["bodies"] if "temperature" in b) == 1
    assert len(seen["bodies"]) == 4
    # And it is still one deviation, not three.
    assert len(judge.deviations) == 1
    judge.close()


def test_a_second_trace_hitting_the_refusal_increments_the_count_not_the_list(view):
    """Concurrency means several workers can learn it at once; the record merges."""
    judge = make_judge(refusing_then_ok({}))
    judge._record_deviation("temperature", "unsupported_value", "m", 400, "t-1")
    judge._record_deviation("temperature", "unsupported_value", "m", 400, "t-2")
    (deviation,) = judge.deviations
    assert deviation.occurrences == 2
    assert deviation.first_seen_trace_id == "t-1"
    judge.close()


@pytest.mark.parametrize("param", ["response_format", "messages", "max_completion_tokens"])
def test_a_refusal_of_anything_that_changes_the_question_is_an_error_not_a_fallback(view, param):
    """Relaxing the schema or the inference view would be a different experiment."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("n", 0)
        seen["n"] += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "unsupported_value",
                    "message": f"Unsupported value: {param}",
                    "param": param,
                    "type": "invalid_request_error",
                }
            },
        )

    judge = make_judge(handler)
    prediction = judge.predict(view, "t-1")
    assert prediction.decision is Decision.error
    assert judge.deviations == []
    assert seen["n"] == 1, "a non-droppable refusal must not be retried with a changed body"
    judge.close()


def test_the_body_builder_refuses_to_omit_anything_outside_the_allow_list(view):
    from false_success_eval.evaluators.general_model import build_chat_body

    with pytest.raises(ValueError, match="non-droppable"):
        build_chat_body(
            view,
            model_id=MODEL,
            instructions="i",
            schema=response_schema(QUESTIONS),
            evaluation_rule=RULE,
            temperature=0.0,
            max_output_tokens=1200,
            omit_params=("response_format",),
        )


def test_a_run_with_no_refusal_records_no_deviation(view):
    judge = make_judge(ok_handler)
    judge.predict(view, "t-1")
    assert judge.deviations == []
    assert judge.omitted_params() == frozenset()
    judge.close()


def test_unsupported_parameter_reads_only_what_the_api_said():
    from false_success_eval.evaluators.general_model import unsupported_parameter

    assert unsupported_parameter(TEMPERATURE_REFUSAL) == (
        "temperature",
        "unsupported_value",
        TEMPERATURE_REFUSAL["error"]["message"],
    )
    assert unsupported_parameter({"error": {"code": "rate_limit_exceeded"}}) is None
    assert unsupported_parameter({"choices": []}) is None
    assert unsupported_parameter(None) is None


def test_a_worker_already_in_flight_when_the_refusal_lands_still_retries(view):
    """The concurrency race that cost three traces on the first re-run.

    Four workers send the configured body at once. The first refusal is
    recorded and `temperature` joins the run-level omit set -- but the other
    three are already in flight with it. If the retry loop deferred to the
    run-level set they would each see their parameter as 'already handled',
    skip the retry, and record an error for a request that was never re-sent.
    """
    seen: dict = {}
    judge = make_judge(refusing_then_ok(seen))
    # Exactly the losing state: another worker has already learned it.
    judge._omit_params.add("temperature")

    prediction = judge.predict(view, "t-late")

    assert prediction.decision is Decision.flag, "an in-flight worker must not be dropped"
    assert prediction.primary_score is not None
    judge.close()


def test_the_retry_loop_cannot_spin_on_a_server_that_keeps_refusing(view):
    """Bounded by what this call has dropped, so a repeated refusal terminates."""
    seen: dict = {}

    def always_refuses(request: httpx.Request) -> httpx.Response:
        seen["n"] = seen.get("n", 0) + 1
        return httpx.Response(400, json=copy.deepcopy(TEMPERATURE_REFUSAL))

    judge = make_judge(always_refuses)
    prediction = judge.predict(view, "t-1")
    assert prediction.decision is Decision.error
    # One round with temperature, one without, and then it stops.
    assert seen["n"] == 2
    judge.close()
