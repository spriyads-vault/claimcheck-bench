"""TF-IDF + gradient-boosted trees: the bar to beat, not a straw man.

This is the baseline the paper reports beating every LLM judge it tested --
AUROC about 0.825 on tau2-bench and 0.953 on AppWorld, against no judge
configuration above 0.65. A comparison against a weak free baseline would prove
nothing, so this one is built to the published recipe: bigram word TF-IDF over
the serialised trajectory, gradient-boosted trees on top, fit on the train split
alone and calibrated on validation.

Fitting discipline is enforced in code rather than by convention, exactly as in
:mod:`.tfidf`:

* trees are fit on the **dev** families only;
* probabilities are calibrated on the **validation** families;
* :meth:`fit` refuses any record whose family is held out, so the test split
  cannot be touched even by accident.

Features come from the InferenceView alone, identically at fit and at inference.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from ..schemas import (
    LABEL_ORDER,
    Decision,
    InferenceView,
    Label,
    Prediction,
    TraceRecord,
    Usage,
)
from .tfidf import FitLeakageError, render_features


def _booster(random_state: int) -> Any:
    """The gradient-booster, preferring the library the published baseline used.

    XGBoost is what the paper ran, and it takes the sparse TF-IDF matrix
    directly. When it is unavailable this falls back to scikit-learn's
    histogram booster on a densified matrix, which is the same family of model
    but not the same implementation -- so the fallback is recorded in the model
    id rather than passed off as the published baseline.
    """
    try:
        from xgboost import XGBClassifier
    except Exception:
        # Deliberately broad. XGBoost fails to load for reasons that are not
        # ImportError -- a missing OpenMP runtime on macOS raises XGBoostError
        # from a dlopen deep inside the package. Any failure to obtain the
        # published implementation is the same situation from here: use the
        # fallback and say in the model id that that is what happened.
        from sklearn.ensemble import HistGradientBoostingClassifier

        return "sklearn-hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=random_state
        )
    return "xgboost", XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
    )


class _DenseIfNeeded:
    """Densify only for an estimator that cannot take a sparse matrix."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def fit(self, x: Any, y: Any = None) -> _DenseIfNeeded:
        return self

    def transform(self, x: Any) -> Any:
        if self.enabled and hasattr(x, "toarray"):
            return np.asarray(x.toarray())
        return x

    def fit_transform(self, x: Any, y: Any = None) -> Any:
        return self.transform(x)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def set_params(self, **params: Any) -> _DenseIfNeeded:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TfidfGbmEvaluator:
    """TF-IDF features with a gradient-boosted classifier."""

    provider = "tfidf_gbm"

    def __init__(self, threshold: float = 0.5, random_state: int = 0) -> None:
        self.threshold = threshold
        self.random_state = random_state
        self._calibrated: Any = None
        self._classes: tuple[str, ...] = ()
        self._label_index: dict[str, int] = {}
        self.model_id = "tfidf-gbm-v1"

    def _build_pipeline(self) -> tuple[str, Any]:
        backend, estimator = _booster(self.random_state)
        pipeline = Pipeline(
            [
                (
                    "features",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        lowercase=True,
                        max_features=60_000,
                    ),
                ),
                ("dense", _DenseIfNeeded(enabled=backend != "xgboost")),
                ("clf", estimator),
            ]
        )
        return backend, pipeline

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

        backend, pipeline = self._build_pipeline()
        self.model_id = f"tfidf-gbm-{backend}-v1"

        x_dev = [render_features(r.inference_view()) for r in dev]
        y_dev_labels = [r.label.value for r in dev]
        # XGBoost needs contiguous integer classes, so the label<->index map is
        # built here and inverted at predict time rather than relying on the
        # estimator to round-trip strings.
        classes = sorted(set(y_dev_labels))
        self._label_index = {name: i for i, name in enumerate(classes)}
        y_dev = np.array([self._label_index[name] for name in y_dev_labels], dtype=int)
        pipeline.fit(x_dev, y_dev)

        x_val = [render_features(r.inference_view()) for r in validation]
        y_val_labels = [r.label.value for r in validation]
        unseen = sorted(set(y_val_labels) - set(self._label_index))
        if unseen:
            raise ValueError(
                f"validation contains labels absent from dev: {unseen}. Calibration "
                "cannot introduce a class the model was never fit on."
            )
        y_val = np.array([self._label_index[name] for name in y_val_labels], dtype=int)

        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
        calibrated.fit(x_val, y_val)
        self._calibrated = calibrated
        index_to_label = {i: name for name, i in self._label_index.items()}
        self._classes = tuple(index_to_label[int(c)] for c in calibrated.classes_)

    def predict(self, view: InferenceView, trace_id: str, repeat: int = 0) -> Prediction:
        if self._calibrated is None:
            raise RuntimeError("TfidfGbmEvaluator.predict called before fit()")
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
