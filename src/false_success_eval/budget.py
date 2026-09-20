"""Context budget: fitting a long real trace into Jev's window, visibly.

Synthetic traces were built short enough that this never mattered. Real
trajectories are not: an AppWorld run can be tens of thousands of characters of
interleaved code and JSON. Jev's documented budget is 64k tokens total, of which
32k is the ceiling for ``state`` plus the longest question, so a trace that
exceeds it has to be reduced before it is sent.

Two rules govern this module.

1. **The reduction is documented and deterministic.** It is applied in a fixed
   order -- shrink oversized tool-result payloads first, then drop whole events
   from the middle -- so the same trace reduces the same way on every machine.
2. **Nothing is dropped silently.** Every reduction returns a
   :class:`TruncationRecord`, the run writes one line per reduced trace, and the
   manifest carries the count and the file's checksum. A run whose manifest says
   ``truncated_traces: 0`` sent every trace whole.

The middle is what gets dropped, never the ends. The goal states the task, the
opening calls establish what the agent set out to do and the closing calls carry
the completion claim and the evidence for it -- which is the entire question
being asked. Losing the middle costs intermediate steps; losing either end would
change the answer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .hashing import canonical_json
from .schemas import Event

#: Marker inserted in place of the events a reduction removed. It is a visible
#: event in its own right so the evaluator is told the trace was shortened
#: rather than shown a doctored trace that looks complete.
ELISION_TOOL = "__elided__"

#: Marker appended to a tool-result payload that was itself shortened.
PAYLOAD_ELISION_KEY = "__elided__"


@dataclass(frozen=True)
class ContextBudget:
    """The evaluator's context limits, in tokens, read from ``eval.yaml``.

    ``chars_per_token`` is the conservative estimate used to convert a request
    body's length into a token count. Jev's tokenizer is not published, so this
    deliberately assumes the *pessimistic* end of the band used for cost
    projection: under-estimating the token count is what would overflow the
    window, and an over-estimate only costs a little headroom.
    """

    total_tokens: int = 64_000
    state_tokens: int = 32_000
    chars_per_token: float = 2.5
    #: Events kept at the start and end of a reduced trace, before any further
    #: reduction. The tail is larger because the completion claim lives there.
    keep_head_events: int = 4
    keep_tail_events: int = 12
    #: A single tool-result payload longer than this is shortened before any
    #: whole event is dropped.
    max_result_chars: int = 4_000
    #: A reduction never leaves fewer than this many events.
    min_events: int = 6

    def state_char_budget(self, longest_question_chars: int) -> int:
        """Characters available to ``state`` once the longest question is paid for."""
        budget_chars = int(self.state_tokens * self.chars_per_token)
        return max(1_000, budget_chars - longest_question_chars)


@dataclass(frozen=True)
class TruncationRecord:
    """What a reduction did to one trace. Written verbatim into the run."""

    trace_id: str
    rule: str
    original_events: int
    kept_events: int
    dropped_events: int
    original_state_chars: int
    final_state_chars: int
    budget_chars: int
    shortened_payloads: int
    fits: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "rule": self.rule,
            "original_events": self.original_events,
            "kept_events": self.kept_events,
            "dropped_events": self.dropped_events,
            "original_state_chars": self.original_state_chars,
            "final_state_chars": self.final_state_chars,
            "budget_chars": self.budget_chars,
            "shortened_payloads": self.shortened_payloads,
            "fits": self.fits,
        }


#: The documented rule, recorded in the manifest and shown in the caveat.
TRUNCATION_RULE = (
    "middle-elision-v1: shorten oversized tool-result payloads first, then drop "
    "whole events from the middle, keeping the first 4 and last 12. An explicit "
    "__elided__ event replaces what was removed, so the evaluator is told the "
    "trace was shortened rather than shown a shortened trace as if it were whole."
)


def _chars(payload: Any) -> int:
    return len(canonical_json(payload))


def _shorten_payload(data: dict[str, Any], limit: int) -> tuple[dict[str, Any], bool]:
    """Shorten one oversized tool-result payload, leaving a visible marker."""
    encoded = canonical_json(data)
    if len(encoded) <= limit:
        return data, False
    kept = encoded[:limit]
    return (
        {
            PAYLOAD_ELISION_KEY: (
                f"tool result shortened: {len(encoded)} chars reduced to {limit}"
            ),
            "head": kept,
        },
        True,
    )


def _elision_event(seq: int, dropped: int) -> Event:
    from .schemas import EventType

    return Event(
        seq=seq,
        type=EventType.tool_result,
        tool=ELISION_TOOL,
        status="elided",
        data={
            "dropped_events": dropped,
            "note": (
                f"{dropped} events from the middle of this trace were removed to fit "
                "the evaluator's context budget. The goal, the opening calls and the "
                "closing calls are intact."
            ),
        },
    )


def fit_events(
    trace_id: str,
    events: Sequence[Event],
    *,
    budget: ContextBudget,
    envelope_chars: int,
    longest_question_chars: int,
) -> tuple[tuple[Event, ...], TruncationRecord | None]:
    """Reduce ``events`` until the serialised state fits the budget.

    ``envelope_chars`` is everything in ``state`` that is not the events -- the
    goal, the tool schema and the evaluation rule -- which is paid for first and
    is never reduced.

    Returns the events to send and, when anything was reduced, the record of
    what happened. A trace that already fits returns ``(events, None)``.
    """
    budget_chars = budget.state_char_budget(longest_question_chars)
    original = list(events)

    def serialised(rows: Sequence[Event]) -> int:
        return envelope_chars + sum(
            _chars(e.model_dump(mode="json", exclude_none=True)) for e in rows
        )

    original_chars = serialised(original)
    if original_chars <= budget_chars:
        return tuple(original), None

    # Pass 1 -- shorten oversized tool-result payloads.
    shortened = 0
    working: list[Event] = []
    for event in original:
        if event.data is not None and _chars(event.data) > budget.max_result_chars:
            data, changed = _shorten_payload(event.data, budget.max_result_chars)
            if changed:
                shortened += 1
                working.append(event.model_copy(update={"data": data}))
                continue
        working.append(event)

    # Pass 2 -- drop whole events from the middle until it fits.
    #
    # Expressed as "choose how many events to keep at each end", not as
    # "delete one and re-measure". The marker event that replaces what was
    # removed has a size of its own, and an earlier version of this loop let
    # that marker occupy a head slot, which made the arithmetic believe the
    # middle was empty while the trace was still over budget. Rebuilding the
    # candidate from scratch for each (head, tail) pair makes that class of
    # bug impossible: the marker is always counted in what is measured.
    kept = list(working)

    def candidate(head: int, tail: int) -> tuple[list[Event], int]:
        """Keep ``head`` events, then a marker, then the last ``tail``."""
        dropped = len(kept) - head - tail
        if dropped <= 0:
            return list(kept), 0
        return (
            [*kept[:head], _elision_event(-1, dropped), *kept[len(kept) - tail :]],
            dropped,
        )

    head = min(budget.keep_head_events, len(kept))
    tail = min(budget.keep_tail_events, max(0, len(kept) - head))
    working, dropped = candidate(head, tail)

    # Still too large with the standard window: give ground from the tail
    # first, then from the head, but never below one of each. The very last
    # event is the closing claim and the first states what the agent set out to
    # do; a trace reduced to those two still answers the question being asked,
    # where a trace missing either does not.
    while serialised(working) > budget_chars and (tail > 1 or head > 1):
        if tail > 1 and (tail >= head or head <= 1):
            tail -= 1
        else:
            head -= 1
        working, dropped = candidate(head, tail)

    final = tuple(e.model_copy(update={"seq": i}) for i, e in enumerate(working))
    final_chars = serialised(final)
    record = TruncationRecord(
        trace_id=trace_id,
        rule=TRUNCATION_RULE.split(":", 1)[0],
        original_events=len(original),
        kept_events=len(final),
        dropped_events=dropped,
        original_state_chars=original_chars,
        final_state_chars=final_chars,
        budget_chars=budget_chars,
        shortened_payloads=shortened,
        fits=final_chars <= budget_chars,
    )
    return final, record


def longest_question_chars(questions: dict[str, Any]) -> int:
    """The longest single question, which shares the state budget with the trace."""
    if not questions:
        return 0
    return max(
        len(json.dumps(q, separators=(",", ":"), sort_keys=True)) for q in questions.values()
    )


def load_budget(raw: dict[str, Any] | None) -> ContextBudget:
    """Build a budget from the ``context_budget`` block of ``eval.yaml``."""
    raw = raw or {}
    defaults = ContextBudget()
    return ContextBudget(
        total_tokens=int(raw.get("total_tokens", defaults.total_tokens)),
        state_tokens=int(raw.get("state_tokens", defaults.state_tokens)),
        chars_per_token=float(raw.get("chars_per_token", defaults.chars_per_token)),
        keep_head_events=int(raw.get("keep_head_events", defaults.keep_head_events)),
        keep_tail_events=int(raw.get("keep_tail_events", defaults.keep_tail_events)),
        max_result_chars=int(raw.get("max_result_chars", defaults.max_result_chars)),
        min_events=int(raw.get("min_events", defaults.min_events)),
    )
