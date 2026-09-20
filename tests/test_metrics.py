from __future__ import annotations

import numpy as np
import pytest

from false_success_eval import metrics
from false_success_eval.schemas import Attempt, Decision, Label, Prediction, Usage


def test_auprc_is_one_for_a_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    assert metrics.auprc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auroc_is_half_for_a_constant_score():
    y = np.array([0, 1, 0, 1])
    assert metrics.auroc(y, np.array([0.5] * 4)) == pytest.approx(0.5)


def test_prf_counts_match_a_hand_worked_example():
    y = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.8, 0.1, 0.7])
    result = metrics.prf_at_threshold(y, scores, 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["fpr"] == pytest.approx(0.5)


def test_recall_at_budget_picks_the_top_slice():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.99, 0.98, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert metrics.recall_at_budget(y, scores, 0.2) == pytest.approx(1.0)
    assert metrics.recall_at_budget(y, scores, 0.1) == pytest.approx(0.5)


def test_recall_at_budget_handles_ties_by_expectation():
    """Discrete scores must not be resolved by an arbitrary sort order."""
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    # One slot out of four tied rows: expect a quarter of the single positive.
    assert metrics.recall_at_budget(y, scores, 0.25) == pytest.approx(0.25)


def test_brier_is_zero_for_perfect_confident_predictions():
    y = np.array([1, 0, 1, 0])
    assert metrics.brier(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_ece_is_zero_for_a_perfectly_calibrated_split():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.array([0.0] * 50 + [1.0] * 50)
    assert metrics.ece_equal_frequency(y, scores, bins=2) == pytest.approx(0.0)


def test_ece_detects_systematic_overconfidence():
    y = np.zeros(100, dtype=int)
    scores = np.full(100, 0.9)
    assert metrics.ece_equal_frequency(y, scores, bins=10) == pytest.approx(0.9)


def test_macro_f1_is_one_for_a_perfect_four_class_prediction():
    labels = [label.value for label in Label]
    assert metrics.macro_f1(labels, labels) == pytest.approx(1.0)


def test_paired_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 40)
    good = np.where(y == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100))
    poor = rng.uniform(0, 1, 100)
    interval = metrics.paired_bootstrap(y, good, poor, "auprc", 400, seed=1)
    assert interval.point > 0
    assert interval.lower <= interval.point <= interval.upper


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed():
    y = np.array([0, 1] * 30)
    a = np.linspace(0, 1, 60)
    b = np.linspace(1, 0, 60)
    first = metrics.paired_bootstrap(y, a, b, "auprc", 200, seed=7)
    second = metrics.paired_bootstrap(y, a, b, "auprc", 200, seed=7)
    assert (first.point, first.lower, first.upper) == (second.point, second.lower, second.upper)


def test_prior_shift_projection_matches_bayes():
    # TPR 0.8, FPR 0.1, prevalence 0.05 -> 0.04 / (0.04 + 0.095)
    assert metrics.prior_shift_precision(0.8, 0.1, 0.05) == pytest.approx(0.04 / 0.135)


def test_prior_shift_precision_falls_as_prevalence_falls():
    high = metrics.prior_shift_precision(0.9, 0.05, 0.10)
    low = metrics.prior_shift_precision(0.9, 0.05, 0.01)
    assert high > low


def _prediction(**kwargs) -> Prediction:
    base = {
        "trace_id": "t1",
        "provider": "p",
        "model_id": "m",
        "predicted_label": Label.unsupported_success,
        "primary_score": 0.7,
        "usage": Usage(input_tokens=10, output_tokens=2),
        "cost_usd": 0.001,
        "end_to_end_latency_ms": 5.0,
    }
    base.update(kwargs)
    return Prediction(**base)


def test_repeat_stability_reports_agreement_and_variance():
    predictions = [
        _prediction(repeat=0, primary_score=0.7),
        _prediction(repeat=1, primary_score=0.7),
        _prediction(repeat=2, primary_score=0.7, predicted_label=Label.supported_success),
    ]
    stability = metrics.repeat_stability(predictions)
    assert stability["mean_label_agreement"] == pytest.approx(2 / 3)
    assert stability["mean_score_variance"] == pytest.approx(0.0)


def test_operational_summary_counts_errors_retries_and_tokens():
    predictions = [
        _prediction(trace_id="a"),
        _prediction(trace_id="b", error="ParseFailure: nope", cost_usd=None),
        _prediction(trace_id="c", attempts=(Attempt(attempt_number=1), Attempt(attempt_number=2))),
    ]
    summary = metrics.operational_summary(predictions)
    assert summary["error_count"] == 1
    assert summary["parse_failure_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["total_input_tokens"] == 30
    assert summary["cost_unpriced_predictions"] == 1


def test_latency_percentiles_are_ordered():
    predictions = [
        _prediction(trace_id=str(i), end_to_end_latency_ms=float(i + 1)) for i in range(100)
    ]
    latency = metrics.latency_summary(predictions)
    assert latency["p50_ms"] <= latency["p95_ms"] <= latency["p99_ms"]
    assert latency["throughput_per_s"] > 0


def test_selective_accuracy_improves_as_low_confidence_cases_are_escalated():
    true_labels = ["a"] * 10
    predicted = ["a"] * 5 + ["b"] * 5
    confidences = [0.9] * 5 + [0.1] * 5
    curve = metrics.selective_accuracy(true_labels, predicted, confidences, fractions=(0.0, 0.5))
    assert curve[0]["accuracy"] == pytest.approx(0.5)
    assert curve[1]["accuracy"] == pytest.approx(1.0)


def test_selective_accuracy_skips_providers_with_no_confidence():
    assert metrics.selective_accuracy(["a"], ["a"], [None]) == []


def test_decision_enum_round_trips():
    assert Decision("flag") is Decision.flag
    assert Decision("pass") is Decision.pass_
