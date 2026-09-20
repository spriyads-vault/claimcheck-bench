"""TF-IDF + logistic regression baseline.

Fitting discipline, enforced in code rather than by convention:

* The classifier is fit on the **dev** families only.
* Probabilities are calibrated on the **validation** families.
* :meth:`fit` refuses any record whose family is held out, so the test split
  cannot be touched even by accident.

Features are built from the InferenceView alone, exactly as at inference time.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ..hashing import canonical_json
from ..schemas import (
    LABEL_ORDER,
    Decision,
    EventType,
    InferenceView,
    Label,
    Prediction,
    TraceRecord,
    Usage,
)


class FitLeakageError(AssertionError):
    """Raised when a held-out family reaches the fitting routine."""


def render_features(view: InferenceView) -> str:
    """Flatten an InferenceView to the text the vectoriser sees."""
    parts = [f"goal: {view.goal}"]
    parts.append("tools: " + " ".join(entry.name for entry in view.tool_schema))
    for event in view.events:
        if event.type is EventType.assistant_tool_call:
            parts.append(f"call {event.tool} args {canonical_json(event.arguments)}")
        elif event.type is EventType.tool_result:
            parts.append(
                f"result {event.tool} status {event.status} data {canonical_json(event.data)}"
            )
        else:
            parts.append(f"assistant {event.text}")
    return "\n".join(parts)


def _build_pipeline() -> Any:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                lowercase=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                                lowercase=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=2.0, random_state=0, solver="lbfgs"),
            ),
        ]
    )


class TfidfEvaluator:
    provider = "tfidf"
    model_id = "tfidf-logreg-v1"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._pipeline: Any = None
        self._calibrated: Any = None
        self._classes: tuple[str, ...] = ()

    def fit(
        self,
        dev: Sequence[TraceRecord],
        validation: Sequence[TraceRecord],
        held_out_families: frozenset[str],
    ) -> None:
        for record in list(dev) + list(validation):
            if record.template_family in held_out_families:
                raise FitLeakageError(
                    f"held-out family {record.template_family!r} reached fit(); "
                    "the test split must never be fit on"
                )
        if not dev or not validation:
            raise ValueError("both a dev and a validation set are required")

        x_dev = [render_features(r.inference_view()) for r in dev]
        y_dev = [r.label.value for r in dev]
        pipeline = _build_pipeline()
        pipeline.fit(x_dev, y_dev)

        x_val = [render_features(r.inference_view()) for r in validation]
        y_val = [r.label.value for r in validation]
        # sklearn >= 1.6 replaced cv="prefit" with an explicitly frozen estimator.
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
        calibrated.fit(x_val, y_val)

        self._pipeline = pipeline
        self._calibrated = calibrated
        self._classes = tuple(str(c) for c in calibrated.classes_)

    def predict(self, view: InferenceView, trace_id: str, repeat: int = 0) -> Prediction:
        if self._calibrated is None:
            raise RuntimeError("TfidfEvaluator.predict called before fit()")
        start = time.perf_counter()
        proba = np.asarray(self._calibrated.predict_proba([render_features(view)])[0], dtype=float)
        probabilities = {label.value: 0.0 for label in LABEL_ORDER}
        for name, value in zip(self._classes, proba, strict=True):
            probabilities[name] = float(value)

        score = probabilities[Label.unsupported_success.value]
        best = max(probabilities.items(), key=lambda kv: kv[1])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return Prediction(
            trace_id=trace_id,
            provider=self.provider,
            model_id=self.model_id,
            repeat=repeat,
            predicted_label=Label(best),
            primary_score=score,
            probabilities=probabilities,
            confidence=float(max(probabilities.values())),
            has_success_claim=None,
            tool_evidence_supports_claim=None,
            needs_review=None,
            threshold=self.threshold,
            decision=Decision.flag if score >= self.threshold else Decision.pass_,
            usage=Usage(input_tokens=0, output_tokens=0),
            attempts=(),
            end_to_end_latency_ms=elapsed_ms,
            raw_request=None,
            raw_response=None,
            cost_usd=0.0,
            error=None,
        )
