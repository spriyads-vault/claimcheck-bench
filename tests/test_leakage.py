"""Split leakage and the InferenceView boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from false_success_eval.evaluators.base import FORBIDDEN_FIELDS, LeakageError, assert_clean_payload
from false_success_eval.evaluators.jev_http import build_state
from false_success_eval.evaluators.rules import RulesEvaluator
from false_success_eval.evaluators.tfidf import FitLeakageError, TfidfEvaluator, render_features
from false_success_eval.generate import generate_records, make_splits
from false_success_eval.hashing import canonical_json
from false_success_eval.schemas import FaultType, InferenceView, Label

RECORDS = 160
SEED = 20260919


@pytest.fixture(scope="module")
def corpus():
    records = generate_records(RECORDS, SEED)
    splits = make_splits(records, SEED, 24, 8, 8, "0" * 64)
    return records, splits


def test_split_family_sets_are_pairwise_disjoint(corpus):
    _, splits = corpus
    dev, validation, test = set(splits.dev), set(splits.validation), set(splits.test)
    assert dev & validation == set()
    assert dev & test == set()
    assert validation & test == set()
    assert len(dev) + len(validation) + len(test) == 40


def test_every_family_lands_in_exactly_one_split(corpus):
    records, splits = corpus
    families = {r.template_family for r in records}
    assert families == set(splits.dev) | set(splits.validation) | set(splits.test)


def test_no_trace_id_crosses_splits(corpus):
    records, splits = corpus
    buckets = {
        name: {r.trace_id for r in records if r.template_family in set(getattr(splits, name))}
        for name in ("dev", "validation", "test")
    }
    assert buckets["dev"] & buckets["validation"] == set()
    assert buckets["dev"] & buckets["test"] == set()
    assert buckets["validation"] & buckets["test"] == set()


# -- the InferenceView boundary ------------------------------------------
def test_inference_view_has_no_ground_truth_fields(corpus):
    records, _ = corpus
    for record in records:
        view = record.inference_view()
        assert set(view.model_dump().keys()) == {"goal", "tool_schema", "events"}
        for field in FORBIDDEN_FIELDS:
            assert not hasattr(view, field)
        assert_clean_payload(view.model_dump(mode="json"))


def test_ground_truth_never_appears_anywhere_in_a_serialised_view(corpus):
    records, _ = corpus
    for record in records:
        blob = canonical_json(record.inference_view().model_dump(mode="json"))
        for field in FORBIDDEN_FIELDS:
            assert f'"{field}"' not in blob
        assert record.label.value not in blob
        assert str(record.oracle) not in blob


def test_every_evaluator_bound_payload_is_clean(corpus):
    """The three payloads that actually reach an evaluator or the wire."""
    records, _ = corpus
    rules = RulesEvaluator()
    for record in records[:40]:
        view = record.inference_view()
        assert_clean_payload(view.model_dump(mode="json"))
        assert_clean_payload(build_state(view, "rule string"))
        assert_clean_payload({"features": render_features(view)})
        prediction = rules.predict(view, record.trace_id)
        assert_clean_payload(prediction.model_dump(mode="json"))


def test_jev_state_carries_only_the_view_plus_the_evaluation_rule(corpus):
    records, _ = corpus
    state = build_state(records[0].inference_view(), "RULE")
    assert set(state) == {"goal", "tool_schema", "events", "evaluation_rule"}
    assert state["evaluation_rule"] == "RULE"


def test_evaluator_inputs_are_independent_of_ground_truth(corpus):
    """The precise invariant, stated structurally rather than by substring.

    Substring checks are the wrong tool here: a tool legitimately named
    `add_label` contains "label", and a tool result legitimately contains
    "error". What matters is that rewriting a record's label, fault type and
    oracle changes nothing an evaluator can see.
    """
    records, _ = corpus
    for record in records:
        rewritten = record.model_copy(
            update={
                "label": Label.no_success_claim
                if record.label is not Label.no_success_claim
                else Label.supported_success,
                "fault_type": FaultType.wrong_parameter
                if record.fault_type is not FaultType.wrong_parameter
                else FaultType.none,
                "oracle": {"totally": "different"},
            }
        )
        assert rewritten.label is not record.label
        assert render_features(rewritten.inference_view()) == render_features(
            record.inference_view()
        )
        assert build_state(rewritten.inference_view(), "R") == build_state(
            record.inference_view(), "R"
        )
        assert rewritten.inference_view() == record.inference_view()


def test_ground_truth_values_never_surface_in_the_feature_text(corpus):
    records, _ = corpus
    oracle_only_keys = (
        "changed_entity",
        "applied_parameter",
        "stale_parameter",
        "claimed_tool",
        "succeeded_tool",
        "confirming_seq",
        "failing_seq",
        "requested_entity",
        "action_attempted",
    )
    for record in records:
        text = render_features(record.inference_view())
        assert record.fault_type.value not in text
        assert record.label.value not in text
        for key in oracle_only_keys:
            assert key not in text


def test_the_guard_actually_catches_leakage():
    with pytest.raises(LeakageError, match="oracle"):
        assert_clean_payload({"events": [{"oracle": {"requested_entity": "CAL-1"}}]})
    with pytest.raises(LeakageError, match="label"):
        assert_clean_payload({"label": "unsupported_success"})


def test_tfidf_refuses_to_fit_on_a_held_out_family(corpus):
    records, splits = corpus
    dev = [r for r in records if r.template_family in set(splits.dev)]
    validation = [r for r in records if r.template_family in set(splits.validation)]
    test = [r for r in records if r.template_family in set(splits.test)]
    evaluator = TfidfEvaluator()
    with pytest.raises(FitLeakageError, match="must never be fit on"):
        evaluator.fit(dev + test[:1], validation, frozenset(splits.test))


def test_inference_view_is_frozen(corpus):
    records, _ = corpus
    view: InferenceView = records[0].inference_view()
    with pytest.raises(ValidationError):
        view.goal = "mutated"
