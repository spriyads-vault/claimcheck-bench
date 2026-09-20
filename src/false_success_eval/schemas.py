"""Frozen Pydantic v2 models.

The central invariant of this harness lives here: :class:`InferenceView` is the
only object an evaluator is ever given. ``label``, ``fault_type`` and ``oracle``
exist on :class:`TraceRecord` and are unreachable from the view.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Label(str, Enum):
    supported_success = "supported_success"
    unsupported_success = "unsupported_success"
    reported_failure_or_uncertainty = "reported_failure_or_uncertainty"
    no_success_claim = "no_success_claim"


class FaultType(str, Enum):
    none = "none"
    explicit_error = "explicit_error"
    timeout_null_result = "timeout_null_result"
    wrong_entity = "wrong_entity"
    wrong_parameter = "wrong_parameter"
    stale_evidence = "stale_evidence"
    attempt_without_confirmation = "attempt_without_confirmation"
    irrelevant_success = "irrelevant_success"
    valid_retry_recovery = "valid_retry_recovery"
    noop_positive = "noop_positive"


class EventType(str, Enum):
    assistant_tool_call = "assistant_tool_call"
    tool_result = "tool_result"
    assistant_message = "assistant_message"


LABEL_ORDER: tuple[Label, ...] = (
    Label.supported_success,
    Label.unsupported_success,
    Label.reported_failure_or_uncertainty,
    Label.no_success_claim,
)

POSITIVE_LABEL = Label.unsupported_success


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolSchemaEntry(Frozen):
    name: str
    required: tuple[str, ...]


class Event(Frozen):
    seq: int
    type: EventType
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    status: str | None = None
    data: dict[str, Any] | None = None
    text: str | None = None


class InferenceView(Frozen):
    """Everything an evaluator may see. Nothing else.

    Constructed only by :meth:`TraceRecord.inference_view`.
    """

    goal: str
    tool_schema: tuple[ToolSchemaEntry, ...]
    events: tuple[Event, ...]

    def final_assistant_text(self) -> str:
        for event in reversed(self.events):
            if event.type is EventType.assistant_message and event.text:
                return event.text
        return ""


class TraceRecord(Frozen):
    trace_id: str
    domain: str
    template_family: str
    goal: str
    tool_schema: tuple[ToolSchemaEntry, ...]
    events: tuple[Event, ...]
    label: Label
    fault_type: FaultType
    oracle: dict[str, Any]

    def inference_view(self) -> InferenceView:
        return InferenceView(
            goal=self.goal,
            tool_schema=self.tool_schema,
            events=self.events,
        )


class Attempt(Frozen):
    attempt_number: int
    http_status: int | None = None
    error_class: str | None = None
    retry_delay_seconds: float = 0.0
    start_perf_ns: int = 0
    end_perf_ns: int = 0
    latency_ms: float = 0.0


class Usage(Frozen):
    input_tokens: int = 0
    output_tokens: int = 0


class Decision(str, Enum):
    flag = "flag"
    pass_ = "pass"
    error = "error"
    not_run = "not_run"


class Prediction(Frozen):
    trace_id: str
    provider: str
    model_id: str
    repeat: int = 0
    predicted_label: Label | None = None
    primary_score: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    has_success_claim: float | None = None
    tool_evidence_supports_claim: float | None = None
    needs_review: float | None = None
    threshold: float = 0.5
    decision: Decision = Decision.error
    usage: Usage | None = None
    attempts: tuple[Attempt, ...] = ()
    end_to_end_latency_ms: float = 0.0
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    cost_usd: float | None = None
    error: str | None = None


class Splits(Frozen):
    seed: int
    records: int
    dev: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    dataset_sha256: str

    def families_for(self, split: str) -> tuple[str, ...]:
        mapping = {"dev": self.dev, "validation": self.validation, "test": self.test}
        if split not in mapping:
            raise KeyError(f"unknown split {split!r}; expected one of {sorted(mapping)}")
        return mapping[split]


class RunManifest(Frozen):
    run_id: str
    created_utc: str
    provider: str
    model_id: str
    split: str
    repeats: int
    concurrency: int
    threshold: float
    n_traces: int
    n_predictions: int
    dataset_path: str
    dataset_sha256: str
    splits_sha256: str
    questions_sha256: str
    eval_config_sha256: str
    prices_path: str
    prices_sha256: str
    predictions_path: str
    predictions_sha256: str
    attempts_path: str
    attempts_sha256: str
    git_commit: str | None
    python_version: str
    package_version: str
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float | None
    error_count: int
    retry_count: int
    parse_failure_count: int
