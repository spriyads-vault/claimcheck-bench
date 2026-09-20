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


def test_a_prediction_that_does_not_join_to_the_dataset_is_refused():
    """Scoring the remainder of a broken join is how a biased subset gets reported."""
    import pytest

    from false_success_eval.report import evaluate_run
    from false_success_eval.runner import load_config
    from false_success_eval.schemas import (
        Decision,
        FaultType,
        InferenceView,
        Label,
        Prediction,
        TraceRecord,
    )

    record = TraceRecord(
        trace_id="known/1",
        domain="d",
        template_family="f",
        goal="g",
        tool_schema=(),
        events=(),
        label=Label.unsupported_success,
        fault_type=FaultType.none,
        oracle={},
    )
    assert isinstance(record.inference_view(), InferenceView)
    predictions = [
        Prediction(
            trace_id=trace_id,
            provider="p",
            model_id="m",
            primary_score=0.5,
            decision=Decision.flag,
        )
        for trace_id in ("known/1", "vanished-by-redaction/2")
    ]
    with pytest.raises(RuntimeError, match=r"is not\s+in the dataset"):
        evaluate_run([record], predictions, 0.5, load_config("config/eval.yaml"))


# ---------------------------------------------------------------------------
# the gate against the strong free baseline
# ---------------------------------------------------------------------------
#
# The preregistered useful-signal gate is frozen against the deterministic
# rules baseline. Once a strong free baseline exists, clearing that gate stops
# meaning very much -- a provider can beat hand-written rules comfortably and
# still lose to TF-IDF with boosting. The second gate is what answers the
# question actually asked, and it is read off the interval, not the point.


def _gate_summary() -> dict:
    return {
        "precision": 0.9,
        "recall": 0.9,
        "ece": 0.05,
        "operational": {"error_rate": 0.0, "parse_failure_count": 0},
    }


def _comparisons(point: float, lower: float, upper: float) -> dict:
    return {
        "tfidf_gbm": {
            "baseline": "tfidf_gbm",
            "auprc_delta": {"point": point, "lower": lower, "upper": upper},
            "recall_delta_at_5%": {"point": 0.0, "lower": -0.1, "upper": 0.1},
        }
    }


def test_losing_to_the_strong_baseline_fails_the_gate_and_is_named_as_a_loss():
    from false_success_eval.report import evaluate_gates
    from false_success_eval.runner import load_config

    gates = evaluate_gates(
        _gate_summary(),
        _comparisons(-0.1142, -0.1450, -0.0848),
        load_config("config/eval.yaml"),
        n_records=3507,
        provider="jev",
    )
    strong = gates["beats_strong_baseline"]
    assert strong["reported"] and strong["passed"] is False
    assert strong["loses"] is True, "an interval wholly below zero is a loss, not a tie"


def test_an_interval_straddling_zero_is_neither_a_win_nor_a_loss():
    from false_success_eval.report import evaluate_gates
    from false_success_eval.runner import load_config

    strong = evaluate_gates(
        _gate_summary(),
        _comparisons(0.01, -0.04, 0.06),
        load_config("config/eval.yaml"),
        n_records=3507,
        provider="jev",
    )["beats_strong_baseline"]
    assert strong["passed"] is False and strong["loses"] is False


def test_a_positive_point_estimate_alone_does_not_pass_the_gate():
    """The whole interval has to clear zero, not just the central estimate."""
    from false_success_eval.report import evaluate_gates
    from false_success_eval.runner import load_config

    strong = evaluate_gates(
        _gate_summary(),
        _comparisons(0.08, -0.01, 0.17),
        load_config("config/eval.yaml"),
        n_records=3507,
        provider="jev",
    )["beats_strong_baseline"]
    assert strong["passed"] is False


def test_clearing_the_frozen_gate_does_not_imply_clearing_the_strong_one():
    """The exact situation on the real corpus: beats rules, loses to tfidf_gbm."""
    from false_success_eval.report import evaluate_gates
    from false_success_eval.runner import load_config

    comparisons = _comparisons(-0.1142, -0.1450, -0.0848)
    comparisons["rules"] = {
        "baseline": "rules",
        "auprc_delta": {"point": 0.2858, "lower": 0.2526, "upper": 0.3198},
        "recall_delta_at_5%": {"point": 0.3, "lower": 0.2, "upper": 0.4},
    }
    gates = evaluate_gates(
        _gate_summary(),
        comparisons,
        load_config("config/eval.yaml"),
        n_records=3507,
        provider="jev",
    )
    assert gates["useful_signal"]["passed"] is True
    assert gates["beats_strong_baseline"]["passed"] is False
    assert "does not say" in gates["useful_signal"]["note"]


def test_the_strong_baseline_itself_gets_no_self_comparison():
    from false_success_eval.report import evaluate_gates
    from false_success_eval.runner import load_config

    strong = evaluate_gates(
        _gate_summary(), {}, load_config("config/eval.yaml"), 3507, "tfidf_gbm"
    )["beats_strong_baseline"]
    assert strong["reported"] is False
    assert "strong baseline itself" in strong["note"]


def test_a_calibration_only_failure_still_fails_the_deployment_gate():
    """Regression: numpy.bool_ serialises to the truthy string "False".

    Every criterion below passes except calibration. If the criteria are not
    coerced to real booleans, ``all()`` sees the string "False", reads it as
    truthy, and the gate reports PASS on a run that missed its ECE bar.
    """
    import numpy as np

    from false_success_eval.report import evaluate_gates
    from false_success_eval.runner import load_config

    summary = {
        "precision": np.float64(0.95),
        "recall": np.float64(0.90),
        "ece": np.float64(0.42),  # the only failure
        "operational": {"error_rate": 0.0, "parse_failure_count": 0},
    }
    deployment = evaluate_gates(summary, {}, load_config("config/eval.yaml"), 3507, "jev")[
        "deployment_candidate"
    ]
    assert deployment["ece_ok"] is False, "must be a bool, not numpy.bool_ or a string"
    assert deployment["passed"] is False


# ---------------------------------------------------------------------------
# Pairing two arms that did not score the same traces
# ---------------------------------------------------------------------------


def _summary(trace_ids, y, scores):
    import numpy as np

    return {
        "_trace_ids": list(trace_ids),
        "_y": np.array(y, dtype=float),
        "_scores": np.array(scores, dtype=float),
    }


def test_two_arms_are_paired_on_the_traces_they_both_scored():
    """One arm failing on a trace must not shift every later pair by one row."""
    from false_success_eval.report import pair_on_shared_traces

    # The judge failed on "b"; the baseline scored everything.
    mine = _summary(["a", "c", "d"], [1, 0, 1], [0.9, 0.1, 0.8])
    theirs = _summary(["a", "b", "c", "d"], [1, 1, 0, 1], [0.2, 0.7, 0.3, 0.4])

    y, scores, baseline_scores, dropped = pair_on_shared_traces(mine, theirs)

    assert list(y) == [1, 0, 1]
    assert list(scores) == [0.9, 0.1, 0.8]
    # "c" must be matched against the baseline's "c" (0.3), not its "b" (0.7).
    assert list(baseline_scores) == [0.2, 0.3, 0.4]
    assert dropped == ["b"]


def test_a_comparison_reports_how_many_traces_it_could_pair():
    from false_success_eval.report import compare
    from false_success_eval.runner import load_config

    config = load_config("config/eval.yaml")
    config.raw["metrics"]["bootstrap_resamples"] = 50
    mine = _summary(["a", "c", "d"], [1, 0, 1], [0.9, 0.1, 0.8])
    theirs = _summary(["a", "b", "c", "d"], [1, 1, 0, 1], [0.2, 0.7, 0.3, 0.4])

    entry = compare(mine, theirs, config, "rules")
    assert entry["n_paired"] == 3
    assert entry["n_unpaired"] == 1
    assert entry["unpaired_trace_ids"] == ["b"]


def test_arms_that_scored_the_same_traces_pair_completely():
    from false_success_eval.report import pair_on_shared_traces

    mine = _summary(["a", "b"], [1, 0], [0.9, 0.1])
    theirs = _summary(["a", "b"], [1, 0], [0.2, 0.3])
    y, _scores, _baseline_scores, dropped = pair_on_shared_traces(mine, theirs)
    assert len(y) == 2
    assert dropped == []
