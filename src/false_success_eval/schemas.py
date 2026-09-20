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
    # Amendment A6. The goal names its target by description, never by an ID or
    # value that appears verbatim in the evidence; the agent acts on a plausible
    # but wrong target and the tool confirms success for it. Nothing a string or
    # field comparison can compute resolves this -- only reading the goal
    # against the evidence does. See semantic.py.
    semantic_target_mismatch = "semantic_target_mismatch"
    # The matched negative control for the above: same indirection, same
    # candidate listing, right target.
    semantic_target_match = "semantic_target_match"

    # -- real, ingested corpora ------------------------------------------
    # A real trajectory has no synthetic fault taxonomy: its ground truth is a
    # single programmatic pass/fail from the benchmark's own database, so the
    # fault is named for what the ground truth says and nothing more. The
    # difficulty suffix is benchmark metadata, not an inference from the text,
    # which is what keeps the per-fault panel informative without inventing a
    # taxonomy the source data does not carry.
    real_unsupported_completion_easy = "real_unsupported_completion_easy"
    real_unsupported_completion_medium = "real_unsupported_completion_medium"
    real_unsupported_completion_hard = "real_unsupported_completion_hard"
    real_supported_completion = "real_supported_completion"
    real_reported_failure = "real_reported_failure"
    real_no_completion_claim = "real_no_completion_claim"


#: Fault types produced only by the synthetic generator.
SYNTHETIC_FAULTS: frozenset[str] = frozenset(
    {
        "none",
        "explicit_error",
        "timeout_null_result",
        "wrong_entity",
        "wrong_parameter",
        "stale_evidence",
        "attempt_without_confirmation",
        "irrelevant_success",
        "valid_retry_recovery",
        "noop_positive",
        "semantic_target_mismatch",
        "semantic_target_match",
    }
)

#: Fault types produced only by an ingested real corpus.
REAL_FAULTS: frozenset[str] = frozenset(
    {
        "real_unsupported_completion_easy",
        "real_unsupported_completion_medium",
        "real_unsupported_completion_hard",
        "real_supported_completion",
        "real_reported_failure",
        "real_no_completion_claim",
    }
)


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
    # -- context-budget accounting --------------------------------------
    # Real traces are long. When one does not fit the evaluator's context
    # budget it is reduced by the documented rule in ``budget.py`` and the
    # reduction is recorded here. Content is never silently dropped: a run
    # whose manifest says ``truncated_traces: 0`` sent every trace whole.
    truncated_traces: int = 0
    truncation_path: str = ""
    truncation_sha256: str = ""
    # -- provider deviations --------------------------------------------
    # A request parameter the model refused, which this harness dropped so the
    # run could proceed. It is a departure from the frozen config, so it is
    # counted here and written out in full beside the run rather than left in
    # the prose. ``deviation_count: 0`` is the positive statement that this run
    # sent exactly what config/eval.yaml specifies.
    deviation_count: int = 0
    deviations_path: str = ""
    deviations_sha256: str = ""


# ---------------------------------------------------------------------------
# Dataset provenance
# ---------------------------------------------------------------------------


class DatasetKind(str, Enum):
    """Where a dataset's traces and labels came from.

    This is the field every honesty statement keys off. A caveat is generated
    from the loaded run rather than written into the page, so a synthetic run
    keeps its construction-label warning and a real run does not inherit it.
    """

    synthetic = "synthetic"
    real = "real"


class SourceLicence(Frozen):
    """A licence verified against its primary source before any ingestion.

    Nothing is ingested on a guess. ``verified_utc`` and ``verified_from`` say
    when this was checked and against what, so a reader can re-check it rather
    than take the harness's word for it.
    """

    spdx: str
    name: str
    url: str
    permits_this_use: bool
    conditions: tuple[str, ...] = ()
    attribution: str = ""
    verified_utc: str = ""
    verified_from: str = ""


class LabelRule(Frozen):
    """The deterministic map from source ground truth to this harness's labels.

    Labels are derived from the benchmark's own programmatic state, never from
    the assistant's wording. ``claim_field`` names the structured field that
    carries the completion claim and ``truth_field`` the field that carries the
    ground truth; both are machine-readable in the source.
    """

    name: str
    claim_field: str
    truth_field: str
    description: str
    mapping: tuple[tuple[str, str], ...]
    text_independent: bool = True


class DatasetProvenance(Frozen):
    """Everything a caveat needs to state about where a dataset came from."""

    dataset_id: str
    kind: DatasetKind
    source_name: str
    source_url: str
    source_version: str = ""
    source_sha256: str = ""
    retrieved_utc: str = ""
    licence: SourceLicence | None = None
    label_rule: LabelRule | None = None
    split_unit: str = "template_family"
    split_unit_description: str = ""
    split_sha256: str = ""
    n_records: int = 0
    label_counts: dict[str, int] = Field(default_factory=dict)
    fault_counts: dict[str, int] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
