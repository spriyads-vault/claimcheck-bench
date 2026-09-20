"""The evaluator protocol and the leakage guard that backs it."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..schemas import InferenceView, Prediction

#: Ground truth. None of these may reach an evaluator, in any nesting.
FORBIDDEN_FIELDS: tuple[str, ...] = ("label", "fault_type", "oracle")


class LeakageError(AssertionError):
    """Raised when ground truth is found inside an evaluator-bound payload."""


def assert_clean_payload(payload: Any, path: str = "$") -> None:
    """Recursively assert no ground-truth field appears in ``payload``."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key in FORBIDDEN_FIELDS:
                raise LeakageError(f"ground-truth field {key!r} present at {path}")
            assert_clean_payload(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_clean_payload(item, f"{path}[{i}]")


@runtime_checkable
class Evaluator(Protocol):
    """Anything that turns an InferenceView into a Prediction."""

    provider: str
    model_id: str

    def predict(self, view: InferenceView, trace_id: str, repeat: int = 0) -> Prediction: ...
