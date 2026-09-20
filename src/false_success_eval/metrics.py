"""Metric computation.

Primary task is binary: ``unsupported_success`` against everything else, scored
on ``primary_score``. The four-way label is used only for macro-F1, the
per-fault matrix and selective accuracy.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .schemas import LABEL_ORDER, Label, Prediction, TraceRecord


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lower": self.lower, "upper": self.upper}


def binary_targets(records: Sequence[TraceRecord]) -> np.ndarray:
    return np.array([1 if r.label is Label.unsupported_success else 0 for r in records], dtype=int)


def auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def prf_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "threshold": threshold,
    }


def recall_at_budget(y: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """Recall when the top ``budget`` fraction by score is sent to review.

    Ties at the cut-off are handled by expectation rather than by an arbitrary
    sort order, which matters because the rules baseline emits discrete scores.
    """
    n = len(y)
    positives = int(y.sum())
    if n == 0 or positives == 0:
        return float("nan")
    k = max(1, math.ceil(budget * n))
    if k >= n:
        return 1.0
    order = np.argsort(-scores, kind="stable")
    cutoff = scores[order[k - 1]]
    above = scores > cutoff
    tied = scores == cutoff
    n_above = int(above.sum())
    tp_above = int(y[above].sum())
    n_tied = int(tied.sum())
    tp_tied = int(y[tied].sum())
    slots = max(0, k - n_above)
    expected_tied = tp_tied * (slots / n_tied) if n_tied else 0.0
    return float((tp_above + expected_tied) / positives)


def recall_at_budget_ceiling(y: np.ndarray, budget: float) -> float:
    """The most recall any ranker can reach at this budget on this split.

    ``budget * n / positives``, capped at 1. When a provider sits on this
    ceiling the metric is saturated and cannot distinguish providers -- which is
    what happens on a label-balanced dataset with a small review budget.
    """
    n = len(y)
    positives = int(y.sum())
    if n == 0 or positives == 0:
        return float("nan")
    return float(min(1.0, max(1, math.ceil(budget * n)) / positives))


def brier(y: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - y) ** 2))


def ece_equal_frequency(y: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-frequency bins."""
    n = len(y)
    if n == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    total = 0.0
    edges = np.linspace(0, n, bins + 1).astype(int)
    for start, stop in pairwise(edges):
        if stop <= start:
            continue
        idx = order[start:stop]
        confidence = float(np.mean(scores[idx]))
        accuracy = float(np.mean(y[idx]))
        total += (stop - start) / n * abs(accuracy - confidence)
    return total


def macro_f1(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    return float(
        f1_score(
            list(true_labels),
            list(predicted_labels),
            labels=[label.value for label in LABEL_ORDER],
            average="macro",
            zero_division=0,
        )
    )


def per_fault_matrix(
    records: Sequence[TraceRecord], predictions: Sequence[Prediction], threshold: float
) -> dict[str, dict[str, Any]]:
    by_id = {p.trace_id: p for p in predictions}
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        prediction = by_id.get(record.trace_id)
        if prediction is None or prediction.primary_score is None:
            continue
        key = f"{record.label.value}/{record.fault_type.value}"
        entry = out.setdefault(
            key,
            {
                "label": record.label.value,
                "fault_type": record.fault_type.value,
                "n": 0,
                "flagged": 0,
                "mean_score": 0.0,
            },
        )
        entry["n"] += 1
        entry["flagged"] += int(prediction.primary_score >= threshold)
        entry["mean_score"] += prediction.primary_score
    for entry in out.values():
        if entry["n"]:
            entry["mean_score"] /= entry["n"]
            entry["flag_rate"] = entry["flagged"] / entry["n"]
    return out


def paired_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: str,
    resamples: int,
    seed: int,
    budget: float = 0.05,
) -> Interval:
    """Paired bootstrap CI for ``metric(a) - metric(b)`` over resampled traces."""

    def compute(yy: np.ndarray, ss: np.ndarray) -> float:
        if metric == "auprc":
            return auprc(yy, ss)
        if metric == "recall_at_budget":
            return recall_at_budget(yy, ss, budget)
        raise ValueError(f"unknown metric {metric!r}")

    point = compute(y, scores_a) - compute(y, scores_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(resamples, dtype=float)
    filled = 0
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        delta = compute(yy, scores_a[idx]) - compute(yy, scores_b[idx])
        if not math.isnan(delta):
            deltas[filled] = delta
            filled += 1
    if filled == 0:
        return Interval(point=point, lower=float("nan"), upper=float("nan"))
    sample = deltas[:filled]
    return Interval(
        point=point,
        lower=float(np.percentile(sample, 2.5)),
        upper=float(np.percentile(sample, 97.5)),
    )


def selective_accuracy(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    confidences: Sequence[float | None],
    fractions: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.40),
) -> list[dict[str, float]]:
    """Accuracy on what remains as the least-confident cases are escalated."""
    usable = [
        (c, t, p)
        for c, t, p in zip(confidences, true_labels, predicted_labels, strict=True)
        if c is not None
    ]
    if not usable:
        return []
    usable.sort(key=lambda row: row[0])
    n = len(usable)
    out = []
    for fraction in fractions:
        drop = round(fraction * n)
        kept = usable[drop:]
        if not kept:
            continue
        correct = sum(1 for _, t, p in kept if t == p)
        out.append(
            {
                "escalated_fraction": fraction,
                "kept": len(kept),
                "accuracy": correct / len(kept),
            }
        )
    return out


def repeat_stability(predictions: Sequence[Prediction]) -> dict[str, float]:
    """Label agreement and score variance across repeats of the same trace."""
    by_trace: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_trace[prediction.trace_id].append(prediction)
    agreements: list[float] = []
    variances: list[float] = []
    repeats = 0
    for group in by_trace.values():
        repeats = max(repeats, len(group))
        if len(group) < 2:
            continue
        labels = [p.predicted_label.value for p in group if p.predicted_label is not None]
        if labels:
            modal = Counter(labels).most_common(1)[0][1]
            agreements.append(modal / len(labels))
        scores = [p.primary_score for p in group if p.primary_score is not None]
        if len(scores) >= 2:
            variances.append(float(np.var(np.array(scores, dtype=float), ddof=0)))
    return {
        "max_repeats": float(repeats),
        "mean_label_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "mean_score_variance": float(np.mean(variances)) if variances else float("nan"),
    }


def latency_summary(predictions: Sequence[Prediction]) -> dict[str, float]:
    values = np.array(
        [p.end_to_end_latency_ms for p in predictions if p.end_to_end_latency_ms > 0], dtype=float
    )
    if values.size == 0:
        return {
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "mean_ms": float("nan"),
            "total_s": 0.0,
            "throughput_per_s": float("nan"),
        }
    total_s = float(values.sum()) / 1000.0
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "total_s": total_s,
        "throughput_per_s": (len(values) / total_s) if total_s > 0 else float("nan"),
    }


def operational_summary(predictions: Sequence[Prediction]) -> dict[str, Any]:
    errors = [p for p in predictions if p.error is not None]
    parse_failures = [p for p in errors if "ParseFailure" in (p.error or "")]
    retries = sum(max(0, len(p.attempts) - 1) for p in predictions)
    input_tokens = sum(p.usage.input_tokens for p in predictions if p.usage)
    output_tokens = sum(p.usage.output_tokens for p in predictions if p.usage)
    costs = [p.cost_usd for p in predictions if p.cost_usd is not None]
    return {
        "n_predictions": len(predictions),
        "error_count": len(errors),
        "error_rate": len(errors) / len(predictions) if predictions else 0.0,
        "parse_failure_count": len(parse_failures),
        "retry_count": retries,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cost_usd": float(sum(costs)) if costs else None,
        "cost_unpriced_predictions": len(predictions) - len(costs),
    }


def prior_shift_precision(tpr: float, fpr: float, prevalence: float) -> float:
    """Projected precision at an assumed production prevalence.

    A projection from observed TPR/FPR, not an observed result.
    """
    numerator = prevalence * tpr
    denominator = numerator + (1.0 - prevalence) * fpr
    return float(numerator / denominator) if denominator > 0 else float("nan")


def prevalence(records: Sequence[TraceRecord]) -> dict[str, float]:
    counts = Counter(r.label.value for r in records)
    total = len(records) or 1
    return {label.value: counts.get(label.value, 0) / total for label in LABEL_ORDER}
