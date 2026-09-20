"""Context budget: the reduction rule, and the accounting that makes it visible.

The property under test throughout is not "the trace got smaller" but "nothing
was lost silently". A reduction has to fit the budget, keep the two ends, leave
a marker the evaluator can see, and produce a record the run writes down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from false_success_eval.budget import (
    ELISION_TOOL,
    ContextBudget,
    fit_events,
    load_budget,
    longest_question_chars,
)
from false_success_eval.hashing import canonical_json
from false_success_eval.schemas import Event, EventType


def _events(n: int, payload_chars: int = 40) -> tuple[Event, ...]:
    out: list[Event] = []
    for i in range(n):
        out.append(
            Event(
                seq=len(out),
                type=EventType.assistant_tool_call,
                tool="execute_python",
                arguments={"code": f"step_{i}(" + "x" * payload_chars + ")"},
            )
        )
        out.append(
            Event(
                seq=len(out),
                type=EventType.tool_result,
                tool="execute_python",
                status="returned",
                data={"output": f"result_{i} " + "y" * payload_chars},
            )
        )
    out.append(Event(seq=len(out), type=EventType.assistant_message, text="complete_task()"))
    return tuple(out)


def _chars(events) -> int:
    return sum(len(canonical_json(e.model_dump(mode="json", exclude_none=True))) for e in events)


def test_a_trace_that_fits_is_returned_untouched_and_unrecorded():
    events = _events(3)
    budget = ContextBudget(state_tokens=32_000, chars_per_token=2.5)
    fitted, record = fit_events(
        "t1", events, budget=budget, envelope_chars=100, longest_question_chars=200
    )
    assert fitted == events
    assert record is None, "a trace that fitted must not appear in the truncation account"


def test_an_oversized_trace_is_reduced_to_within_the_budget():
    events = _events(300, payload_chars=200)
    budget = ContextBudget(state_tokens=2_000, chars_per_token=2.5)
    fitted, record = fit_events(
        "t2", events, budget=budget, envelope_chars=100, longest_question_chars=100
    )
    assert record is not None
    assert record.fits, "the reduction must actually get the trace inside the budget"
    assert 100 + _chars(fitted) <= record.budget_chars
    assert len(fitted) < len(events)


def test_the_reduction_keeps_both_ends_and_eats_the_middle():
    """The goal and the closing claim are the question being asked."""
    events = _events(200, payload_chars=200)
    budget = ContextBudget(state_tokens=2_000, chars_per_token=2.5, keep_head_events=4)
    fitted, record = fit_events(
        "t3", events, budget=budget, envelope_chars=100, longest_question_chars=100
    )
    assert record is not None
    # The first events survive.
    assert "step_0" in json.dumps(fitted[0].model_dump(mode="json"))
    # The closing message survives, and it is still the closing message.
    assert fitted[-1].type is EventType.assistant_message
    assert fitted[-1].text == "complete_task()"


def test_the_reduction_leaves_a_marker_the_evaluator_can_see():
    """A shortened trace must not look like a complete one."""
    events = _events(200, payload_chars=200)
    budget = ContextBudget(state_tokens=2_000, chars_per_token=2.5)
    fitted, record = fit_events(
        "t4", events, budget=budget, envelope_chars=100, longest_question_chars=100
    )
    assert record is not None
    markers = [e for e in fitted if e.tool == ELISION_TOOL]
    assert len(markers) == 1
    assert markers[0].data is not None
    assert markers[0].data["dropped_events"] == record.dropped_events
    assert "context budget" in markers[0].data["note"]


def test_an_oversized_payload_is_shortened_before_any_event_is_dropped():
    """Trimming one huge tool result is cheaper than losing a whole step."""
    events = (
        Event(seq=0, type=EventType.assistant_tool_call, tool="t", arguments={"code": "a"}),
        Event(
            seq=1,
            type=EventType.tool_result,
            tool="t",
            status="returned",
            data={"output": "z" * 50_000},
        ),
        Event(seq=2, type=EventType.assistant_message, text="done"),
    )
    budget = ContextBudget(state_tokens=4_000, chars_per_token=2.5, max_result_chars=1_000)
    fitted, record = fit_events(
        "t5", events, budget=budget, envelope_chars=50, longest_question_chars=50
    )
    assert record is not None
    assert record.shortened_payloads == 1
    assert record.dropped_events == 0, "no event needed dropping once the payload shrank"
    assert len(fitted) == len(events)
    assert fitted[1].data is not None and "__elided__" in fitted[1].data


def test_sequence_numbers_are_renumbered_contiguously_after_a_reduction():
    events = _events(200, payload_chars=200)
    budget = ContextBudget(state_tokens=2_000, chars_per_token=2.5)
    fitted, record = fit_events(
        "t6", events, budget=budget, envelope_chars=100, longest_question_chars=100
    )
    assert record is not None
    assert [e.seq for e in fitted] == list(range(len(fitted)))


def test_the_record_accounts_for_every_dropped_event():
    events = _events(200, payload_chars=200)
    budget = ContextBudget(state_tokens=2_000, chars_per_token=2.5)
    fitted, record = fit_events(
        "t7", events, budget=budget, envelope_chars=100, longest_question_chars=100
    )
    assert record is not None
    # kept + dropped must reconcile against the original, allowing for the one
    # marker event the reduction inserts.
    markers = sum(1 for e in fitted if e.tool == ELISION_TOOL)
    assert record.kept_events + record.dropped_events - markers == record.original_events
    assert record.original_events == len(events)
    assert record.kept_events == len(fitted)


def test_the_longest_question_is_charged_against_the_state_budget():
    budget = ContextBudget(state_tokens=1_000, chars_per_token=2.5)
    small = budget.state_char_budget(0)
    large = budget.state_char_budget(1_200)
    assert large < small


def test_longest_question_chars_measures_the_largest_question():
    questions = {"a": {"type": "noul"}, "b": {"type": "choice", "criteria": {"x": "y" * 500}}}
    assert longest_question_chars(questions) > 500
    assert longest_question_chars({}) == 0


def test_budget_is_loaded_from_config_and_falls_back_to_the_documented_defaults():
    loaded = load_budget({"total_tokens": 64000, "state_tokens": 32000})
    assert loaded.total_tokens == 64000
    assert loaded.state_tokens == 32000
    assert loaded.chars_per_token == ContextBudget().chars_per_token
    assert load_budget(None) == ContextBudget()


def test_the_real_config_declares_the_documented_jev_budget():
    """64k total, 32k for state plus the longest question."""
    import yaml

    from false_success_eval.runner import load_config

    raw = yaml.safe_load(Path("config/eval.yaml").read_text(encoding="utf-8"))
    assert raw["context_budget"]["total_tokens"] == 64_000
    assert raw["context_budget"]["state_tokens"] == 32_000
    budget = load_budget(load_config("config/eval.yaml").section("context_budget"))
    assert budget.state_tokens == 32_000


@pytest.mark.parametrize("n_events", [8, 40, 400])
def test_reduction_is_deterministic(n_events: int):
    events = _events(n_events, payload_chars=300)
    budget = ContextBudget(state_tokens=2_000, chars_per_token=2.5)
    first = fit_events("t", events, budget=budget, envelope_chars=80, longest_question_chars=80)
    second = fit_events("t", events, budget=budget, envelope_chars=80, longest_question_chars=80)
    assert first[0] == second[0]
    assert first[1] == second[1]
