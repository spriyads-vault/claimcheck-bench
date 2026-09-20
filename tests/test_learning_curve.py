"""The low-label learning curve: subsampling, task-disjointness, crossover.

Three properties carry the whole result and each is tested on its own:

* **determinism** -- a size and a seed name one subset, whatever else has drawn
  random numbers first. Without it the curve is not reproducible and neither is
  the crossover read off it.
* **task-disjointness** -- rows are only ever reached through their task, and
  the evaluation split's tasks are never reachable. Row-level sampling would
  put sibling trajectories of the same AppWorld task on both sides of the split
  and inflate every point on the curve.
* **the crossover calculation** -- including the two ways it is allowed to
  answer "never", and the refusal to call a single lucky point a crossing.

A fourth block asserts the thing the whole step was conditioned on: nothing
here can spend money.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from false_success_eval import learning_curve as lc
from false_success_eval.schemas import (
    Decision,
    Event,
    EventType,
    FaultType,
    Label,
    Prediction,
    ToolSchemaEntry,
    TraceRecord,
    Usage,
)

# The four labels in a fixed order, so a fixture's prevalence is written down
# rather than implied.
LABELS = (
    Label.unsupported_success,
    Label.no_success_claim,
    Label.reported_failure_or_uncertainty,
    Label.supported_success,
)


def make_record(family: str, index: int, label: Label) -> TraceRecord:
    return TraceRecord(
        trace_id=f"{family}_{index}",
        domain="test",
        template_family=family,
        goal=f"goal for {family}",
        tool_schema=(ToolSchemaEntry(name="do_thing", required=("target",)),),
        events=(
            Event(seq=0, type=EventType.assistant_tool_call, tool="do_thing", arguments={"a": 1}),
            Event(seq=1, type=EventType.tool_result, tool="do_thing", status="ok", data={"b": 2}),
            Event(seq=2, type=EventType.assistant_message, text=f"done {label.value}"),
        ),
        label=label,
        fault_type=FaultType.none,
        oracle={},
    )


def make_pool(n_families: int = 20, per_family: int = 16) -> tuple[TraceRecord, ...]:
    """A pool whose label mix is the same in every task.

    Deliberately homogeneous across tasks: it makes the stratified quota the
    only thing that decides the label mix of a sample, so a test about the
    quota is not really a test about which task got drawn.
    """
    records: list[TraceRecord] = []
    for f in range(n_families):
        family = f"fam{f:03d}"
        for i in range(per_family):
            # 8 / 4 / 2 / 2 of 16 -- a lopsided but realistic mix.
            label = (
                LABELS[0] if i < 8 else LABELS[1] if i < 12 else LABELS[2] if i < 14 else LABELS[3]
            )
            records.append(make_record(family, i, label))
    return tuple(records)


def family_set(sample: lc.Subsample) -> set[str]:
    return {r.template_family for r in sample.records}


# -- determinism ---------------------------------------------------------


def test_the_same_size_and_seed_draw_the_same_subset():
    pool = make_pool()
    first = lc.subsample(pool, 50, seed=3)
    second = lc.subsample(pool, 50, seed=3)
    assert [r.trace_id for r in first.records] == [r.trace_id for r in second.records]
    assert first.tasks == second.tasks
    assert first.label_counts == second.label_counts


def test_the_draw_does_not_depend_on_the_global_random_state():
    """A seed must name a subset, not a position in someone else's stream."""
    pool = make_pool()
    baseline = [r.trace_id for r in lc.subsample(pool, 50, seed=3).records]
    np.random.default_rng(999).random(1000)
    np.random.seed(12345)
    np.random.random(1000)
    assert [r.trace_id for r in lc.subsample(pool, 50, seed=3).records] == baseline


def test_different_seeds_draw_different_subsets():
    pool = make_pool()
    draws = {tuple(r.trace_id for r in lc.subsample(pool, 50, seed=s).records) for s in range(5)}
    assert len(draws) > 1, "five seeds produced one subset; the seed is not being used"


def test_full_size_is_the_pool_in_order_whatever_the_seed():
    pool = make_pool()
    for seed in (0, 7):
        drawn = lc.subsample(pool, None, seed=seed)
        assert [r.trace_id for r in drawn.records] == [r.trace_id for r in pool]
        assert drawn.size is None


def test_a_sample_is_drawn_in_dataset_order_not_grouped_by_label():
    """A booster handed rows sorted by class is a different experiment."""
    pool = make_pool()
    drawn = lc.subsample(pool, 60, seed=1)
    order = {r.trace_id: i for i, r in enumerate(pool)}
    positions = [order[r.trace_id] for r in drawn.records]
    assert positions == sorted(positions)
    labels = [r.label.value for r in drawn.records]
    assert len(set(labels)) > 1
    assert labels != sorted(labels), "the sample came out grouped by label"


# -- stratification ------------------------------------------------------


def test_the_quota_sums_to_the_requested_size():
    pool = make_pool()
    for size in (10, 25, 50, 100, 200, 400):
        quota = lc.label_quota(pool, size)
        assert sum(quota.values()) == size


def test_every_present_label_gets_at_least_one_example():
    pool = make_pool()
    quota = lc.label_quota(pool, 10)
    assert set(quota) == {label.value for label in LABELS}
    assert min(quota.values()) >= 1


def test_the_quota_tracks_prevalence_at_a_workable_size():
    pool = make_pool()
    quota = lc.label_quota(pool, 160)
    # The pool is 8/4/2/2 of every 16, so 160 examples should come out 80/40/20/20.
    assert quota[LABELS[0].value] == 80
    assert quota[LABELS[1].value] == 40
    assert quota[LABELS[2].value] == 20
    assert quota[LABELS[3].value] == 20


def test_a_sample_honours_its_quota():
    pool = make_pool()
    drawn = lc.subsample(pool, 100, seed=2)
    assert drawn.n == 100
    assert drawn.label_counts == lc.label_quota(pool, 100)


def test_a_size_below_the_label_count_is_refused_rather_than_truncated():
    pool = make_pool()
    with pytest.raises(ValueError, match="one of each"):
        lc.label_quota(pool, 3)


def test_a_size_beyond_the_pool_is_refused():
    pool = make_pool(n_families=2, per_family=4)
    with pytest.raises(ValueError, match="pool holds"):
        lc.subsample(pool, 100, seed=0)


# -- task-disjointness ---------------------------------------------------


def test_rows_only_arrive_through_the_tasks_the_sample_names():
    pool = make_pool()
    drawn = lc.subsample(pool, 80, seed=4)
    assert family_set(drawn) == set(drawn.tasks)
    assert set(drawn.tasks) <= set(lc.task_ids(pool))


def test_a_small_budget_comes_from_a_handful_of_whole_tasks():
    """Task-first sampling means a small budget is concentrated, not scattered.

    Row-level sampling of 10 rows from 20 tasks would touch about 10 tasks. Task
    -level sampling touches as few as the quota allows, and that difference is
    the whole point: it is what keeps sibling trajectories of one task together.
    """
    pool = make_pool(n_families=20, per_family=16)
    drawn = lc.subsample(pool, 10, seed=0)
    assert len(drawn.tasks) <= 3


def test_admitted_tasks_grow_with_the_budget_for_one_seed():
    """Bigger budgets extend the same task order rather than reshuffling it."""
    pool = make_pool()
    seen = [set(lc.subsample(pool, size, seed=1).tasks) for size in (10, 25, 50, 100, 200)]
    for smaller, larger in pairwise(seen):
        assert smaller <= larger


def test_a_curve_refuses_a_train_split_that_shares_a_task_with_the_split_scored(tmp_path):
    from false_success_eval.runner import EvalConfig
    from false_success_eval.schemas import Splits

    pool = make_pool(n_families=6, per_family=16)
    families = list(lc.task_ids(pool))
    splits = Splits(
        seed=0,
        records=len(pool),
        dev=tuple(families[:4]),
        validation=(families[4],),
        # Deliberately overlapping: fam000 is in dev as well.
        test=(families[5], families[0]),
        dataset_sha256="0" * 64,
    )
    config = EvalConfig(path=str(tmp_path / "eval.yaml"), raw={"thresholds": {"default": 0.5}})
    with pytest.raises(ValueError, match="shares"):
        lc.run_curve(config=config, records=pool, splits=splits, sizes=(10,), seeds=(0,))


def test_the_evaluators_own_leakage_check_still_fires_on_a_held_out_task():
    """Belt and braces: the curve's check is not the only one standing."""
    from false_success_eval.evaluators.tfidf import FitLeakageError

    pool = make_pool(n_families=4, per_family=16)
    families = lc.task_ids(pool)
    with pytest.raises(FitLeakageError):
        lc.fit_and_score(
            "tfidf",
            train=pool,
            validation=pool,
            test=pool,
            held_out_families=frozenset({families[0]}),
            threshold=0.5,
        )


# -- the crossover -------------------------------------------------------


def curve_rows(values, lowers=None, provider="tfidf_gbm"):
    """Aggregated rows shaped exactly as :func:`lc.aggregate` returns them."""
    sizes = [10, 25, 50, 100, 200]
    lowers = lowers if lowers is not None else [v - 0.05 for v in values]
    return [
        {
            "provider": provider,
            "size_label": str(size),
            "requested_size": size,
            "n_train": size,
            "n_train_tasks_mean": 2.0,
            "n_calibration": 700,
            "n_test": 702,
            "seeds": [0, 1, 2, 3, 4],
            "model_id": "test",
            "auprc": {
                "mean": mean,
                "sd": 0.01,
                "lower": lower,
                "upper": mean + 0.05,
                "min": lower,
                "max": mean + 0.05,
                "n": 5,
            },
        }
        for size, mean, lower in zip(sizes, values, lowers, strict=True)
    ]


def test_the_crossover_is_the_first_size_that_stays_above_the_line():
    rows = curve_rows([0.70, 0.75, 0.88, 0.91, 0.95])
    result = lc.crossover(rows, 0.80, "jev")
    assert result["mean"]["n_train"] == 50
    assert result["mean"]["value"] == pytest.approx(0.88)
    assert result["mean"]["margin"] == pytest.approx(0.08)
    assert result["reference_leads_anywhere"] is True
    assert result["beats_at_smallest"] is False
    assert "needs about 50 labelled examples to beat jev" in result["statement"]


def test_a_single_lucky_point_is_not_reported_as_a_crossing():
    """Up at 25, back down at 50: the curve has not crossed over."""
    rows = curve_rows([0.70, 0.85, 0.78, 0.91, 0.95])
    result = lc.crossover(rows, 0.80, "jev")
    assert result["mean"]["n_train"] == 100
    assert result["mean"]["value"] == pytest.approx(0.91)


def test_a_classifier_that_wins_from_the_smallest_budget_says_so():
    rows = curve_rows([0.86, 0.90, 0.93, 0.94, 0.96])
    result = lc.crossover(rows, 0.80, "jev")
    assert result["beats_at_smallest"] is True
    assert result["mean"]["n_train"] == 10
    assert result["reference_leads_anywhere"] is False
    assert result["statement"] == (
        "tfidf_gbm beats jev even at the smallest size (10 labelled examples)"
    )


def test_a_classifier_that_never_wins_says_that_too():
    rows = curve_rows([0.20, 0.30, 0.40, 0.50, 0.60])
    result = lc.crossover(rows, 0.80, "jev")
    assert result["mean"] is None
    assert result["lower_bound"] is None
    assert result["reference_leads_anywhere"] is True
    assert "never beats jev" in result["statement"]
    assert "up to 200 labelled examples" in result["statement"]


def test_the_lower_bound_crosses_no_earlier_than_the_mean():
    rows = curve_rows([0.78, 0.83, 0.88, 0.91, 0.95], lowers=[0.70, 0.76, 0.79, 0.86, 0.92])
    result = lc.crossover(rows, 0.80, "jev")
    assert result["mean"]["n_train"] == 25
    assert result["lower_bound"]["n_train"] == 100
    assert result["lower_bound"]["n_train"] >= result["mean"]["n_train"]


def test_a_lower_bound_that_never_clears_is_reported_as_never():
    rows = curve_rows([0.85, 0.88, 0.90, 0.92, 0.95], lowers=[0.10] * 5)
    result = lc.crossover(rows, 0.80, "jev")
    assert result["mean"]["n_train"] == 10
    assert result["lower_bound"] is None


def test_a_nan_point_never_counts_as_a_crossing():
    rows = curve_rows([0.85, 0.88, 0.90, 0.92, 0.95])
    rows[0]["auprc"]["mean"] = float("nan")
    result = lc.crossover(rows, 0.80, "jev")
    assert result["mean"]["n_train"] == 25
    assert result["beats_at_smallest"] is False


def test_an_empty_curve_states_that_rather_than_a_crossover():
    result = lc.crossover([], 0.80, "jev")
    assert result["mean"] is None
    assert result["reference_leads_anywhere"] is False
    assert "no curve points" in result["statement"]


def test_the_crossover_is_recomputed_from_stored_points_not_read_back():
    """The report re-derives the crossing against its own zero-shot numbers."""
    points = [
        lc.CurvePoint(
            provider="tfidf",
            model_id="tfidf-logreg-v1",
            size_label=str(size),
            requested_size=size,
            seed=seed,
            n_train=size,
            n_train_tasks=2,
            n_calibration=700,
            n_test=702,
            train_label_counts={},
            threshold=0.5,
            auprc=value + seed * 0.001,
            auroc=value,
            recall=value,
            precision=value,
            f1=value,
        )
        for size, value in ((10, 0.70), (50, 0.85), (200, 0.95))
        for seed in range(5)
    ]
    document = {"points": [p.as_dict() for p in points], "curves": {}, "crossovers": {}}
    strict = lc.rebuild(document, {"jev": {"auprc": 0.90}})
    lenient = lc.rebuild(document, {"jev": {"auprc": 0.60}})
    assert strict["crossovers"]["tfidf"]["jev"]["mean"]["n_train"] == 200
    assert lenient["crossovers"]["tfidf"]["jev"]["mean"]["n_train"] == 10


# -- the interval --------------------------------------------------------


def test_the_interval_over_seeds_uses_the_t_quantile_not_the_normal_one():
    values = [0.90, 0.92, 0.88, 0.91, 0.89]
    summary = lc.summarise(values)
    array = np.array(values)
    expected_half = 2.776 * array.std(ddof=1) / np.sqrt(5)
    assert summary["mean"] == pytest.approx(array.mean())
    assert summary["upper"] - summary["mean"] == pytest.approx(expected_half, rel=1e-6)
    assert summary["n"] == 5


def test_a_single_seed_reports_its_own_value_as_the_whole_interval():
    summary = lc.summarise([0.91])
    assert summary["lower"] == summary["upper"] == pytest.approx(0.91)
    assert summary["sd"] == 0.0


def test_t_critical_falls_back_to_the_normal_quantile_for_a_large_sample():
    assert lc.t_critical(4) == pytest.approx(2.776)
    assert lc.t_critical(500) == pytest.approx(1.96)


# -- no paid calls -------------------------------------------------------


def test_a_paid_provider_is_refused_before_anything_is_fitted():
    with pytest.raises(lc.PaidProviderError, match="must be offline"):
        lc.assert_offline(["tfidf", "jev"])
    with pytest.raises(lc.PaidProviderError):
        lc.assert_offline(["openai"])


def test_the_free_classifiers_pass_the_offline_check():
    lc.assert_offline(lc.CURVE_PROVIDERS)


def test_run_curve_refuses_a_paid_provider(tmp_path):
    from false_success_eval.runner import EvalConfig
    from false_success_eval.schemas import Splits

    pool = make_pool(n_families=6, per_family=16)
    families = list(lc.task_ids(pool))
    splits = Splits(
        seed=0,
        records=len(pool),
        dev=tuple(families[:4]),
        validation=(families[4],),
        test=(families[5],),
        dataset_sha256="0" * 64,
    )
    config = EvalConfig(path=str(tmp_path / "eval.yaml"), raw={"thresholds": {"default": 0.5}})
    with pytest.raises(lc.PaidProviderError):
        lc.run_curve(
            config=config, records=pool, splits=splits, providers=("jev",), sizes=(10,), seeds=(0,)
        )


def priced_prediction(cost: float | None, tokens: int = 0) -> Prediction:
    return Prediction(
        trace_id="t1",
        provider="tfidf",
        model_id="tfidf-logreg-v1",
        repeat=0,
        predicted_label=Label.unsupported_success,
        primary_score=0.9,
        probabilities={},
        confidence=0.9,
        has_success_claim=None,
        tool_evidence_supports_claim=None,
        needs_review=None,
        threshold=0.5,
        decision=Decision.flag,
        usage=Usage(input_tokens=tokens, output_tokens=0),
        attempts=(),
        end_to_end_latency_ms=1.0,
        raw_request=None,
        raw_response=None,
        cost_usd=cost,
        error=None,
    )


def test_a_costed_prediction_stops_the_curve():
    with pytest.raises(lc.PaidProviderError, match="recorded a cost"):
        lc.assert_no_spend([priced_prediction(0.002)])


def test_metered_tokens_stop_the_curve_even_when_the_cost_is_zero():
    with pytest.raises(lc.PaidProviderError, match="metered tokens"):
        lc.assert_no_spend([priced_prediction(0.0, tokens=1200)])


def test_a_free_prediction_passes():
    lc.assert_no_spend([priced_prediction(0.0)])


def test_the_curve_document_records_that_nothing_was_spent(tmp_path):
    from false_success_eval.runner import EvalConfig

    config_path = tmp_path / "eval.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text("{}", encoding="utf-8")
    document = lc.curve_document(
        points=[],
        zero_shot={"jev": {"auprc": 0.85, "trains": False}},
        split="test",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_sha256="a" * 64,
        splits_path=splits_path,
        config=EvalConfig(path=str(config_path), raw={}),
        sizes=(10, None),
        seeds=(0,),
        curve_id="curve-test-x",
    )
    assert document["paid_calls"] == 0
    assert document["total_cost_usd"] == 0.0
    assert document["sizes"] == ["10", "full"]
    assert "N counts the labels" in document["caveats"]["calibration"]


# -- end to end on a small pool ------------------------------------------


def test_a_small_real_sweep_produces_one_point_per_size_and_seed(tmp_path):
    """The whole loop, on a pool small enough to fit in a unit test."""
    from false_success_eval.runner import EvalConfig
    from false_success_eval.schemas import Splits

    pool = make_pool(n_families=12, per_family=16)
    families = list(lc.task_ids(pool))
    splits = Splits(
        seed=0,
        records=len(pool),
        dev=tuple(families[:8]),
        validation=tuple(families[8:10]),
        test=tuple(families[10:]),
        dataset_sha256="0" * 64,
    )
    config = EvalConfig(path=str(tmp_path / "eval.yaml"), raw={"thresholds": {"default": 0.5}})
    points = lc.run_curve(
        config=config,
        records=pool,
        splits=splits,
        providers=("tfidf",),
        sizes=(10, 20, None),
        seeds=(0, 1),
    )
    # Two seeds at each finite size, one at full.
    assert [p.size_label for p in points] == ["10", "10", "20", "20", "full"]
    assert all(p.n_calibration == 32 for p in points)
    assert all(p.n_test == 32 for p in points)

    curves = lc.aggregate(points)
    assert [row["size_label"] for row in curves["tfidf"]] == ["10", "20", "full"]
    assert curves["tfidf"][0]["auprc"]["n"] == 2
    assert curves["tfidf"][-1]["auprc"]["n"] == 1
    assert curves["tfidf"][-1]["auprc"]["lower"] == curves["tfidf"][-1]["auprc"]["upper"]


def test_a_written_curve_is_found_again_only_for_its_own_dataset(tmp_path):
    runs_root = tmp_path / "runs"
    document = {
        "curve_id": "curve-test-1",
        "split": "test",
        "dataset_sha256": "a" * 64,
        "points": [],
        "curves": {},
    }
    lc.write_curve(document, lc.curve_root(runs_root) / "curve-test-1")
    assert lc.load_curve(runs_root, "a" * 64, "test") is not None
    assert lc.load_curve(runs_root, "b" * 64, "test") is None
    assert lc.load_curve(runs_root, "a" * 64, "validation") is None


def test_a_curve_directory_is_not_mistaken_for_a_provider_run(tmp_path):
    """The run globs must not reach into the curve namespace."""
    from false_success_eval.dashboard.aggregate import discover_dashboard_runs
    from false_success_eval.runner import discover_runs

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    lc.write_curve({"split": "test", "points": [], "curves": {}}, lc.curve_root(runs_root) / "c1")
    assert discover_runs(runs_root) == []
    assert discover_dashboard_runs(runs_root) == []
