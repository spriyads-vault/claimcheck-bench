"""Low-label learning curve: how many labels the free classifier needs.

The comparison this harness reports at full size is lopsided -- the free
TF-IDF baselines beat both paid arms on AUPRC. That leaves one honest opening
for a zero-shot paid judge: it needs **no labels at all**, and a classifier
needs some. So the question this module answers is where that advantage ends.

The method:

* the train (``dev``) split is subsampled to a budget of ``N`` labelled
  examples, **task-disjoint** -- whole AppWorld tasks are admitted in a seeded
  order and rows are only ever drawn from admitted tasks, so no task is half in
  the sample and half out, and the test tasks are unreachable by construction;
* the sample is **stratified**: the four labels keep the train split's own
  prevalence, by largest remainder, with at least one of each label that exists
  so a classifier is never fit on a subset that cannot express a class;
* five seeds per size, so the curve carries a spread rather than one lucky draw;
* every model is scored on the **frozen test split** -- the same split, the same
  records and the same metric functions the report uses;
* Jev and the general judge do not train. They are overlaid as flat lines at the
  AUPRC their **already-recorded** test-split predictions produce. This module
  never constructs a paid evaluator and never opens a socket; see
  :func:`assert_offline`.

What ``N`` counts, stated plainly because the headline depends on it: ``N`` is
the number of labelled examples the classifier is **fit** on. Probability
calibration still runs on the whole validation split, exactly as it does for the
full-size runs, because changing the fitting recipe between the curve and the
reported run would make the two incomparable. Calibration is a monotone
per-class transform and AUPRC is a ranking metric, so its effect on this curve
is small -- but it is not zero, and those labels are real. Every artifact this
module writes records ``n_calibration`` beside ``n_train`` so the stricter
accounting is always one column away, and :func:`curve_document` carries the
caveat into the report and the dashboard.
"""

from __future__ import annotations

import json
import secrets
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, metrics
from .evaluators.tfidf import TfidfEvaluator
from .evaluators.tfidf_gbm import TfidfGbmEvaluator
from .hashing import canonical_json, sha256_file
from .runner import (
    FREE_BASELINES,
    GENERAL_JUDGE_PROVIDER,
    OFFLINE_PROVIDERS,
    PAID_PROVIDERS,
    PRIMARY_PAID_PROVIDER,
    EvalConfig,
    select_split,
)
from .schemas import LABEL_ORDER, Label, Prediction, Splits, TraceRecord

#: The label budgets the curve is measured at. ``None`` is the whole train
#: split, which is what the reported full-size runs use -- so the last point of
#: the curve is a reproduction of an already-recorded number, and disagreeing
#: with it is a defect rather than a finding.
TRAIN_SIZES: tuple[int | None, ...] = (10, 25, 50, 100, 200, 400, None)

#: Five seeds per size. Small enough to run in minutes, enough to show a spread.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: The providers that have a learning curve at all. Both are free and local.
#: ``rules`` is excluded on purpose: it is hand-written and does not train, so a
#: curve for it would be a flat line drawn from no labels.
CURVE_PROVIDERS: tuple[str, ...] = ("tfidf", "tfidf_gbm")

#: The zero-shot arms overlaid as flat lines. Neither is ever *run* here.
ZERO_SHOT_PROVIDERS: tuple[str, ...] = (PRIMARY_PAID_PROVIDER, GENERAL_JUDGE_PROVIDER)

#: The directory, under a corpus's runs root, that curve artifacts live in. It
#: is nested one level deeper than a run directory precisely so the run
#: discovery globs (``*/manifest.json``, ``*/progress.json``) cannot pick a
#: curve up and try to read it as a provider run.
CURVE_DIRNAME = "_learning_curve"

#: Two-sided 97.5% Student-t quantiles by degrees of freedom, so a confidence
#: interval over a handful of seeds does not need scipy (not a dependency) and
#: does not quietly use the normal quantile at n=5, where it is 30% too narrow.
#: Values from the standard table; df beyond the table falls back to 1.96.
_T_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


class PaidProviderError(RuntimeError):
    """Raised when a paid provider is asked to appear in a learning curve."""


def assert_offline(providers: Sequence[str]) -> None:
    """Refuse a paid provider before anything is fitted.

    The learning curve is a free-classifier experiment by definition: a
    zero-shot judge has no training size to sweep. Asking for one would mean
    thousands of billed requests, so it is refused here rather than guarded by
    a comment.
    """
    offending = [p for p in providers if p not in OFFLINE_PROVIDERS]
    if offending:
        raise PaidProviderError(
            f"learning curve providers must be offline; {offending} are not. "
            f"Offline providers: {sorted(OFFLINE_PROVIDERS)}. The paid arms "
            f"({sorted(PAID_PROVIDERS)}) do not train, so they are overlaid from "
            "their recorded test-split predictions and never re-run."
        )


def assert_no_spend(predictions: Sequence[Prediction]) -> None:
    """Refuse to record a curve whose predictions cost anything.

    A belt-and-braces check on the output rather than the input: if a future
    evaluator ever started billing, the curve would stop rather than publish a
    number whose provenance it had mis-stated.
    """
    for prediction in predictions:
        cost = prediction.cost_usd
        if cost:
            raise PaidProviderError(
                f"{prediction.provider} recorded a cost of {cost} on "
                f"{prediction.trace_id}; a learning curve must be free."
            )
        if prediction.usage and (prediction.usage.input_tokens or prediction.usage.output_tokens):
            raise PaidProviderError(
                f"{prediction.provider} recorded metered tokens on {prediction.trace_id}; "
                "a learning curve must be free."
            )


# -- subsampling ---------------------------------------------------------


def task_ids(records: Sequence[TraceRecord]) -> tuple[str, ...]:
    """The distinct task ids in a record set, in a stable order.

    ``template_family`` is the AppWorld task id. Sorting first means the seeded
    shuffle below starts from an order that does not depend on how the dataset
    file happened to be written.
    """
    return tuple(sorted({r.template_family for r in records}))


def label_quota(pool: Sequence[TraceRecord], size: int) -> dict[str, int]:
    """How many of each label a sample of ``size`` should contain.

    The train split's own prevalence, allocated by largest remainder, with a
    floor of one for every label present. The floor is not cosmetic: a subset
    missing a class produces a pipeline whose ``predict_proba`` has fewer
    columns than the calibration set has classes, which fails deep inside
    scikit-learn with an error that says nothing about the real cause.
    """
    counts = Counter(r.label.value for r in pool)
    present = [label.value for label in LABEL_ORDER if counts.get(label.value)]
    if not present:
        raise ValueError("the pool contains no labelled records")
    if size < len(present):
        raise ValueError(
            f"a sample of {size} cannot hold one of each of the {len(present)} labels "
            f"present in the pool; the smallest usable size is {len(present)}"
        )
    total = sum(counts[name] for name in present)
    exact = {name: size * counts[name] / total for name in present}
    quota = {name: max(1, int(exact[name])) for name in present}

    # Largest remainder, then repair the distortion the floor introduced. Both
    # loops move one record at a time and always pick by the same key, so the
    # result is a deterministic function of (pool, size).
    while sum(quota.values()) < size:
        name = max(present, key=lambda n: (exact[n] - quota[n], counts[n], n))
        quota[name] += 1
    while sum(quota.values()) > size:
        name = max(
            (n for n in present if quota[n] > 1),
            key=lambda n: (quota[n] - exact[n], counts[n], n),
        )
        quota[name] -= 1
    return quota


@dataclass(frozen=True)
class Subsample:
    """One drawn training subset, and how it was drawn."""

    records: tuple[TraceRecord, ...]
    size: int | None
    seed: int
    tasks: tuple[str, ...]
    label_counts: dict[str, int]

    @property
    def n(self) -> int:
        return len(self.records)


def subsample(records: Sequence[TraceRecord], size: int | None, seed: int) -> Subsample:
    """Draw ``size`` labelled examples task-disjointly and stratified by label.

    ``size=None`` returns the pool whole, in its original order, so the last
    point of a curve is exactly the fit the reported full-size run made.

    The draw is task-first and never row-first:

    1. the distinct task ids are shuffled with ``seed``;
    2. tasks are admitted in that order, each contributing **all** of its rows
       to the per-label buckets, until every label's bucket can meet its quota;
    3. the quota is taken off the front of each bucket.

    Step 2 is what "task-disjoint" means here: a row can only be sampled by
    admitting its whole task, so two sizes or two seeds never split one task's
    rows between a fitted set and an unseen one. The test tasks are not in the
    pool at all, which is checked by the caller and again by the evaluator's own
    ``fit``.
    """
    if size is None:
        pool = tuple(records)
        return Subsample(
            records=pool,
            size=None,
            seed=seed,
            tasks=task_ids(pool),
            label_counts=dict(Counter(r.label.value for r in pool)),
        )
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if size > len(records):
        raise ValueError(
            f"asked for {size} examples but the pool holds {len(records)}; "
            "pass size=None for the whole split rather than an oversized number"
        )

    quota = label_quota(records, size)
    families = list(task_ids(records))
    # A dedicated generator seeded only by `seed`, so the same seed and size
    # give the same subset whatever else has drawn random numbers first.
    np.random.default_rng(seed).shuffle(families)

    by_family: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        by_family[record.template_family].append(record)

    buckets: dict[str, list[TraceRecord]] = defaultdict(list)
    for family in families:
        for record in by_family[family]:
            buckets[record.label.value].append(record)
        if all(len(buckets[name]) >= need for name, need in quota.items()):
            break
    else:
        short = {
            name: (len(buckets[name]), need)
            for name, need in quota.items()
            if len(buckets[name]) < need
        }
        raise ValueError(
            f"the pool cannot supply a stratified sample of {size}: {short} "
            "(have, needed) after admitting every task"
        )

    drawn: list[TraceRecord] = []
    for label in LABEL_ORDER:
        need = quota.get(label.value, 0)
        drawn.extend(buckets[label.value][:need])
    # Ordered by the source dataset, not by label, so the fitted rows are not
    # handed to a booster grouped by class.
    order = {record.trace_id: i for i, record in enumerate(records)}
    drawn.sort(key=lambda r: order[r.trace_id])
    used = tuple(sorted({r.template_family for r in drawn}))
    return Subsample(
        records=tuple(drawn),
        size=size,
        seed=seed,
        tasks=used,
        label_counts=dict(Counter(r.label.value for r in drawn)),
    )


# -- one point on the curve ---------------------------------------------


@dataclass(frozen=True)
class CurvePoint:
    """One (provider, size, seed) result, scored on the frozen test split."""

    provider: str
    model_id: str
    size_label: str
    requested_size: int | None
    seed: int
    n_train: int
    n_train_tasks: int
    n_calibration: int
    n_test: int
    train_label_counts: dict[str, int]
    threshold: float
    auprc: float
    auroc: float
    recall: float
    precision: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def size_label(size: int | None) -> str:
    return "full" if size is None else str(size)


def _build(provider: str, threshold: float) -> TfidfEvaluator | TfidfGbmEvaluator:
    if provider == "tfidf":
        return TfidfEvaluator(threshold=threshold)
    if provider == "tfidf_gbm":
        return TfidfGbmEvaluator(threshold=threshold)
    raise PaidProviderError(
        f"{provider!r} has no learning curve. Curve providers: {list(CURVE_PROVIDERS)}"
    )


def fit_and_score(
    provider: str,
    *,
    train: Sequence[TraceRecord],
    validation: Sequence[TraceRecord],
    test: Sequence[TraceRecord],
    held_out_families: frozenset[str],
    threshold: float,
) -> tuple[dict[str, float], str, list[Prediction]]:
    """Fit on ``train``, calibrate on ``validation``, score on ``test``.

    The evaluator classes are used unchanged, including their own leakage
    check: a held-out family reaching ``fit`` raises rather than being dropped.
    Metrics come from :mod:`.metrics`, the same functions the report calls, so a
    curve point and a reported run cannot disagree about what AUPRC means.
    """
    evaluator = _build(provider, threshold)
    evaluator.fit(train, validation, held_out_families)
    predictions = [
        evaluator.predict(record.inference_view(), record.trace_id, 0) for record in test
    ]
    assert_no_spend(predictions)

    by_id = {p.trace_id: p for p in predictions}
    usable = [r for r in test if by_id[r.trace_id].primary_score is not None]
    y = metrics.binary_targets(usable)
    scores = np.array([by_id[r.trace_id].primary_score for r in usable], dtype=float)
    prf = metrics.prf_at_threshold(y, scores, threshold)
    summary = {
        "auprc": metrics.auprc(y, scores),
        "auroc": metrics.auroc(y, scores),
        "recall": prf["recall"],
        "precision": prf["precision"],
        "f1": prf["f1"],
        "n_scored": float(len(usable)),
    }
    return summary, str(evaluator.model_id), predictions


def run_curve(
    *,
    config: EvalConfig,
    records: Sequence[TraceRecord],
    splits: Splits,
    split: str = "test",
    providers: Sequence[str] = CURVE_PROVIDERS,
    sizes: Sequence[int | None] = TRAIN_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    on_point: Any = None,
) -> list[CurvePoint]:
    """Every (provider, size, seed) point, scored on ``split``.

    ``on_point`` is called with each :class:`CurvePoint` as it lands, so a long
    sweep writes its artifact as it goes rather than only at the end.
    """
    assert_offline(providers)
    train = select_split(records, splits, "dev")
    validation = select_split(records, splits, "validation")
    test = select_split(records, splits, split)
    if not train or not validation or not test:
        raise ValueError(
            f"the curve needs a non-empty dev, validation and {split!r} split; "
            f"got {len(train)}, {len(validation)}, {len(test)}"
        )
    held_out = frozenset(splits.families_for(split))
    leaked = held_out & {r.template_family for r in train}
    if leaked:
        raise ValueError(
            f"the train split shares {len(leaked)} task(s) with {split!r}: {sorted(leaked)[:3]}"
        )

    points: list[CurvePoint] = []
    for provider in providers:
        threshold = config.threshold_for(provider)
        for size in sizes:
            # At full size every seed draws the identical subset, so the band
            # collapses to a point there by construction. Fitting it five times
            # would burn minutes to reproduce one number, so it is fitted once
            # and recorded against the first seed.
            point_seeds = (seeds[0],) if size is None else tuple(seeds)
            for seed in point_seeds:
                drawn = subsample(train, size, seed)
                summary, model_id, _ = fit_and_score(
                    provider,
                    train=drawn.records,
                    validation=validation,
                    test=test,
                    held_out_families=held_out,
                    threshold=threshold,
                )
                point = CurvePoint(
                    provider=provider,
                    model_id=model_id,
                    size_label=size_label(size),
                    requested_size=size,
                    seed=seed,
                    n_train=drawn.n,
                    n_train_tasks=len(drawn.tasks),
                    n_calibration=len(validation),
                    n_test=int(summary["n_scored"]),
                    train_label_counts=drawn.label_counts,
                    threshold=threshold,
                    auprc=summary["auprc"],
                    auroc=summary["auroc"],
                    recall=summary["recall"],
                    precision=summary["precision"],
                    f1=summary["f1"],
                )
                points.append(point)
                if on_point is not None:
                    on_point(point)
    return points


# -- aggregation ---------------------------------------------------------


def t_critical(df: int) -> float:
    """Two-sided 97.5% Student-t quantile, from the table above."""
    if df <= 0:
        return float("nan")
    if df in _T_975:
        return _T_975[df]
    smaller = [k for k in _T_975 if k < df]
    if df > max(_T_975):
        return 1.96
    return _T_975[max(smaller)] if smaller else _T_975[min(_T_975)]


def summarise(values: Sequence[float]) -> dict[str, float]:
    """Mean, spread and a 95% interval for the mean over seeds.

    The interval is a Student-t interval on the seed-to-seed spread. It states
    how much the *subsample draw* moves the result, and nothing else -- in
    particular it is not a confidence interval for the test-split estimate
    itself, which has its own sampling error that this band does not contain. A
    single-seed point (the full-size fit) has no spread and reports its own
    value as the whole interval.
    """
    clean = [float(v) for v in values if not np.isnan(v)]
    if not clean:
        nan = float("nan")
        return {"mean": nan, "sd": nan, "lower": nan, "upper": nan, "min": nan, "max": nan, "n": 0}
    array = np.array(clean, dtype=float)
    mean = float(array.mean())
    if array.size == 1:
        return {
            "mean": mean,
            "sd": 0.0,
            "lower": mean,
            "upper": mean,
            "min": mean,
            "max": mean,
            "n": 1,
        }
    sd = float(array.std(ddof=1))
    half = t_critical(array.size - 1) * sd / float(np.sqrt(array.size))
    return {
        "mean": mean,
        "sd": sd,
        "lower": mean - half,
        "upper": mean + half,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def aggregate(points: Sequence[CurvePoint]) -> dict[str, list[dict[str, Any]]]:
    """Per provider, one row per size, ordered by the number of labels used."""
    grouped: dict[tuple[str, str], list[CurvePoint]] = defaultdict(list)
    for point in points:
        grouped[(point.provider, point.size_label)].append(point)

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (provider, label), group in grouped.items():
        row: dict[str, Any] = {
            "provider": provider,
            "size_label": label,
            "requested_size": group[0].requested_size,
            "n_train": int(np.mean([p.n_train for p in group])),
            "n_train_tasks_mean": float(np.mean([p.n_train_tasks for p in group])),
            "n_calibration": group[0].n_calibration,
            "n_test": group[0].n_test,
            "seeds": sorted(p.seed for p in group),
            "model_id": group[0].model_id,
        }
        for metric in ("auprc", "auroc", "recall", "precision", "f1"):
            row[metric] = summarise([getattr(p, metric) for p in group])
        out[provider].append(row)
    for provider in out:
        out[provider].sort(key=lambda r: r["n_train"])
    return dict(out)


# -- the crossover -------------------------------------------------------


def crossover(
    rows: Sequence[dict[str, Any]],
    reference_auprc: float,
    reference_name: str,
    metric: str = "auprc",
) -> dict[str, Any]:
    """Where a classifier curve first passes a flat zero-shot line.

    Two answers, because they say different things:

    * ``mean`` -- the smallest size whose **mean** over seeds exceeds the
      reference. The everyday reading: a typical draw of this many labels wins.
    * ``lower_bound`` -- the smallest size whose **lower 95% bound** exceeds it.
      The conservative reading: it wins whichever draw you happened to get.

    ``first exceeds`` is taken literally and is not allowed to be a fluke of one
    size: a size only counts as the crossover if it and **every larger size**
    stay above the line. A curve that pops above the reference once and falls
    back has not crossed over, and saying it had would be the most flattering
    possible reading of noise.
    """
    ordered = sorted(rows, key=lambda r: r["n_train"])
    if not ordered:
        return {
            "reference": reference_name,
            "reference_auprc": reference_auprc,
            "metric": metric,
            "mean": None,
            "lower_bound": None,
            "beats_at_smallest": False,
            "reference_leads_anywhere": False,
            "statement": f"no curve points, so nothing can be said about {reference_name}",
        }

    def first_sustained(key: str) -> dict[str, Any] | None:
        for index, row in enumerate(ordered):
            value = row[metric][key]
            if np.isnan(value) or value <= reference_auprc:
                continue
            if all(
                not np.isnan(later[metric][key]) and later[metric][key] > reference_auprc
                for later in ordered[index + 1 :]
            ):
                return {
                    "size_label": row["size_label"],
                    "n_train": row["n_train"],
                    "n_calibration": row["n_calibration"],
                    "value": float(value),
                    "margin": float(value - reference_auprc),
                }
        return None

    mean_cross = first_sustained("mean")
    lower_cross = first_sustained("lower")
    smallest = ordered[0]
    beats_at_smallest = bool(
        not np.isnan(smallest[metric]["mean"]) and smallest[metric]["mean"] > reference_auprc
    )
    leads_anywhere = any(
        not np.isnan(row[metric]["mean"]) and row[metric]["mean"] < reference_auprc
        for row in ordered
    )

    provider = ordered[0]["provider"]
    if mean_cross is None:
        statement = (
            f"{provider} never beats {reference_name} on mean {metric} at any size tested, "
            f"up to {ordered[-1]['n_train']} labelled examples"
        )
    elif beats_at_smallest and mean_cross["n_train"] == smallest["n_train"]:
        statement = (
            f"{provider} beats {reference_name} even at the smallest size "
            f"({smallest['n_train']} labelled examples)"
        )
    else:
        statement = (
            f"{provider} needs about {mean_cross['n_train']} labelled examples "
            f"to beat {reference_name}"
        )

    return {
        "reference": reference_name,
        "reference_auprc": reference_auprc,
        "metric": metric,
        "mean": mean_cross,
        "lower_bound": lower_cross,
        "beats_at_smallest": beats_at_smallest,
        "reference_leads_anywhere": leads_anywhere,
        "statement": statement,
    }


# -- the artifact --------------------------------------------------------


CALIBRATION_CAVEAT = (
    "N counts the labels the classifier is FIT on. Probability calibration runs "
    "on the whole validation split at every size, exactly as it does for the "
    "reported full-size runs, so the total labels touched is N + n_calibration. "
    "Calibration is a monotone per-class transform and AUPRC is a ranking "
    "metric, so its influence on this curve is small -- but it is not zero."
)

BAND_CAVEAT = (
    "The band is a Student-t 95% interval on the mean over seeds. It shows how "
    "much the subsample draw moves the result. It is not a confidence interval "
    "for the test-split estimate, and the flat zero-shot lines are point "
    "estimates with their own unshown uncertainty."
)


def curve_document(
    *,
    points: Sequence[CurvePoint],
    zero_shot: dict[str, dict[str, Any]],
    split: str,
    dataset_path: Path,
    dataset_sha256: str,
    splits_path: Path,
    config: EvalConfig,
    sizes: Sequence[int | None] = TRAIN_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    curve_id: str = "",
) -> dict[str, Any]:
    """The whole curve as one serialisable document.

    ``zero_shot`` maps a paid provider to ``{"auprc", "recall", "model_id",
    "run_id"}`` read from its **recorded** run. Nothing in here calls a paid
    provider; the flat lines are quotations of numbers that already exist.
    """
    curves = aggregate(points)
    crossovers: dict[str, dict[str, Any]] = {}
    for provider, rows in curves.items():
        crossovers[provider] = {
            name: crossover(rows, float(arm["auprc"]), name)
            for name, arm in zero_shot.items()
            if arm.get("auprc") is not None and not np.isnan(float(arm["auprc"]))
        }

    return {
        "curve_id": curve_id,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "package_version": __version__,
        "split": split,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "splits_path": str(splits_path),
        "splits_sha256": sha256_file(splits_path),
        "eval_config_sha256": config.sha256,
        "sizes": [size_label(s) for s in sizes],
        "seeds": list(seeds),
        "providers": sorted(curves),
        "curves": curves,
        "zero_shot": zero_shot,
        "crossovers": crossovers,
        "points": [p.as_dict() for p in points],
        "paid_calls": 0,
        "total_cost_usd": 0.0,
        "caveats": {"calibration": CALIBRATION_CAVEAT, "band": BAND_CAVEAT},
        "publication_restriction": (
            "Local artifacts only. TypeSafe's Master Customer Agreement restricts "
            "publication of benchmark or performance results."
        ),
    }


def new_curve_id(split: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"curve-{split}-{stamp}-{secrets.token_hex(3)}"


def curve_root(runs_root: Path) -> Path:
    return runs_root / CURVE_DIRNAME


def write_curve(document: dict[str, Any], curve_dir: Path) -> Path:
    """Write ``curve.json`` into a fresh curve directory."""
    curve_dir.mkdir(parents=True, exist_ok=True)
    path = curve_dir / "curve.json"
    path.write_text(canonical_json(document) + "\n", encoding="utf-8")
    return path


def discover_curves(runs_root: Path) -> list[Path]:
    """Every curve directory under a runs root, newest id last."""
    root = curve_root(runs_root)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/curve.json"))


def load_curve(
    runs_root: Path, dataset_sha256: str | None = None, split: str = "test"
) -> dict[str, Any] | None:
    """The newest curve for a split, or ``None``.

    A curve computed against a different dataset is skipped rather than shown:
    the same rule the report applies to stale provider runs, for the same
    reason.
    """
    for curve_dir in reversed(discover_curves(runs_root)):
        try:
            document = json.loads((curve_dir / "curve.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if document.get("split") != split:
            continue
        if dataset_sha256 and document.get("dataset_sha256") != dataset_sha256:
            continue
        document["curve_dir"] = str(curve_dir)
        return dict(document)
    return None


def points_from_document(document: dict[str, Any]) -> list[CurvePoint]:
    """Rehydrate the raw points a stored curve carries."""
    fields = set(CurvePoint.__dataclass_fields__)
    return [
        CurvePoint(**{k: v for k, v in row.items() if k in fields}) for row in document["points"]
    ]


def zero_shot_from_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The flat lines, read from already-computed report summaries.

    These are quotations. ``summaries`` comes from scoring predictions that are
    already on disk, so drawing a zero-shot line costs nothing and sends
    nothing.
    """
    out: dict[str, dict[str, Any]] = {}
    for provider in ZERO_SHOT_PROVIDERS:
        summary = summaries.get(provider)
        if summary is None:
            continue
        out[provider] = {
            "auprc": summary.get("auprc"),
            "recall": summary.get("recall"),
            "model_id": summary.get("model_id"),
            "run_id": summary.get("run_id"),
            "n_scored": summary.get("n_scored"),
            "trains": False,
        }
    return out


def zero_shot_from_runs(
    *,
    runs_root: Path,
    records: Sequence[TraceRecord],
    dataset_sha256: str,
    split: str,
    config: EvalConfig,
) -> dict[str, dict[str, Any]]:
    """Score the recorded paid runs for this split, without re-running them.

    Reads ``predictions.jsonl`` off disk and hands it to the report's own
    ``evaluate_run``. No evaluator is constructed, so no key is read and no
    request is made -- the flat lines are arithmetic over a file.
    """
    from .report import evaluate_run
    from .runner import discover_runs, load_run

    newest: dict[str, tuple[Any, list[Prediction]]] = {}
    for run_dir in discover_runs(runs_root):
        try:
            manifest, predictions = load_run(run_dir)
        except (OSError, ValueError):
            continue
        if manifest.provider not in ZERO_SHOT_PROVIDERS or manifest.split != split:
            continue
        if manifest.dataset_sha256 != dataset_sha256:
            continue
        current = newest.get(manifest.provider)
        if current is None or manifest.created_utc >= current[0].created_utc:
            newest[manifest.provider] = (manifest, predictions)

    split_records = list(records)
    summaries: dict[str, dict[str, Any]] = {}
    for provider, (manifest, predictions) in newest.items():
        summary = evaluate_run(split_records, predictions, manifest.threshold, config)
        summary["model_id"] = manifest.model_id
        summary["run_id"] = manifest.run_id
        summaries[provider] = summary
    return zero_shot_from_summaries(summaries)


def rebuild(document: dict[str, Any], zero_shot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recompute the curves and crossovers of a stored document.

    The report calls this with the zero-shot AUPRCs **it** just computed, so a
    flat line on the learning-curve panel is the same number the rest of the
    report states for that provider. A stored crossover is never read back and
    quoted; it is always recomputed from the stored points.
    """
    points = points_from_document(document)
    curves = aggregate(points)
    crossovers = {
        provider: {
            name: crossover(rows, float(arm["auprc"]), name)
            for name, arm in zero_shot.items()
            if arm.get("auprc") is not None and not np.isnan(float(arm["auprc"]))
        }
        for provider, rows in curves.items()
    }
    return {
        **document,
        "curves": curves,
        "zero_shot": zero_shot,
        "crossovers": crossovers,
    }


__all__ = [
    "BAND_CAVEAT",
    "CALIBRATION_CAVEAT",
    "CURVE_PROVIDERS",
    "DEFAULT_SEEDS",
    "FREE_BASELINES",
    "TRAIN_SIZES",
    "CurvePoint",
    "Label",
    "PaidProviderError",
    "Subsample",
    "aggregate",
    "assert_no_spend",
    "assert_offline",
    "crossover",
    "curve_document",
    "curve_root",
    "discover_curves",
    "fit_and_score",
    "label_quota",
    "load_curve",
    "new_curve_id",
    "points_from_document",
    "rebuild",
    "run_curve",
    "size_label",
    "subsample",
    "summarise",
    "t_critical",
    "task_ids",
    "write_curve",
    "zero_shot_from_runs",
    "zero_shot_from_summaries",
]
