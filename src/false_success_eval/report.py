"""Report building: tables, plots, results.json and a Markdown write-up."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from . import metrics
from .runner import EvalConfig, discover_runs, load_run
from .schemas import Prediction, TraceRecord


def _pct_key(label: str) -> float:
    """Sort '1%', '5%', '10%', '20%' numerically rather than lexically."""
    return float(label.rstrip("%"))


BASELINE = "rules"
SECOND_BASELINE = "tfidf"
#: The strong free baseline -- TF-IDF features with a gradient-boosted
#: classifier. This is the published recipe that beats every LLM judge it was
#: measured against, so it is the bar a paid provider has to clear.
STRONG_BASELINE = "tfidf_gbm"
BASELINES = (BASELINE, SECOND_BASELINE, STRONG_BASELINE)

#: The typed model, and the general LLM judge it is measured against. Both are
#: paid, both see the same inference view and are asked the same four questions
#: at temperature 0, so the difference between them is the model rather than the
#: prompt, the data or the scoring.
PRIMARY = "jev"
GENERAL_JUDGE = "openai"

#: Every provider a comparison may be stated against. The free baselines answer
#: "did the paid lane buy anything at all"; the general judge answers the
#: separate question of whether the *typed* model beats a general one.
COMPARISON_TARGETS = (*BASELINES, GENERAL_JUDGE)
STAGE_MIN_RECORDS = 1000


def _point_predictions(predictions: Sequence[Prediction]) -> dict[str, Prediction]:
    """One prediction per trace (repeat 0). Shared with the dashboard."""
    return metrics.point_predictions(predictions)


def evaluate_run(
    records: Sequence[TraceRecord],
    predictions: Sequence[Prediction],
    threshold: float,
    config: EvalConfig,
) -> dict[str, Any]:
    by_id = _point_predictions(predictions)
    usable = [
        r for r in records if r.trace_id in by_id and by_id[r.trace_id].primary_score is not None
    ]

    # A prediction whose trace_id is not in the dataset cannot be scored, and
    # dropping it quietly is how a biased subset gets reported as the whole
    # split. This is not hypothetical: over-eager redaction once rewrote 117 of
    # 702 real trace ids and every metric was computed on what was left,
    # without a word anywhere. Refuse instead.
    known = {r.trace_id for r in records}
    unmatched = sorted(t for t in by_id if t not in known)
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} of {len(by_id)} predictions carry a trace_id that is not "
            f"in the dataset, so they cannot be scored. Refusing to report a metric "
            f"computed on the remainder. First few: {unmatched[:3]}"
        )
    y = metrics.binary_targets(usable)
    scores = np.array([by_id[r.trace_id].primary_score for r in usable], dtype=float)

    mcfg = config.section("metrics")
    budgets = [float(b) for b in mcfg.get("review_budgets", [0.01, 0.05, 0.10, 0.20])]
    prf = metrics.prf_at_threshold(y, scores, threshold)

    true_labels = [r.label.value for r in usable]
    predicted_labels: list[str] = []
    for record in usable:
        label = by_id[record.trace_id].predicted_label
        predicted_labels.append(label.value if label is not None else "")
    confidences = [by_id[r.trace_id].confidence for r in usable]

    return {
        "n_scored": len(usable),
        # Which traces this provider actually scored. A provider that failed on
        # a trace has no score for it, and a paired comparison against one that
        # did must be computed on the traces they share -- not on two vectors of
        # different lengths that happen to line up positionally.
        "_trace_ids": [r.trace_id for r in usable],
        "prevalence": metrics.prevalence(usable),
        "auprc": metrics.auprc(y, scores),
        "auroc": metrics.auroc(y, scores),
        "threshold": threshold,
        **{k: v for k, v in prf.items() if k != "threshold"},
        "recall_at_budget": {f"{b:.0%}": metrics.recall_at_budget(y, scores, b) for b in budgets},
        # With 25% prevalence a 5% review budget caps recall at 0.20 for *any*
        # ranker, perfect or not. Carry the ceiling so saturation is visible.
        "recall_at_budget_ceiling": {
            f"{b:.0%}": metrics.recall_at_budget_ceiling(y, b) for b in budgets
        },
        "macro_f1": metrics.macro_f1(true_labels, predicted_labels),
        "brier": metrics.brier(y, scores),
        "ece": metrics.ece_equal_frequency(y, scores, int(mcfg.get("ece_bins", 10))),
        "selective_accuracy": metrics.selective_accuracy(
            true_labels, predicted_labels, confidences
        ),
        "per_fault": metrics.per_fault_matrix(usable, list(by_id.values()), threshold),
        "repeat_stability": metrics.repeat_stability(predictions),
        "latency": metrics.latency_summary(predictions),
        "operational": metrics.operational_summary(predictions),
        "_y": y,
        "_scores": scores,
    }


def provider_intervals(
    summaries: dict[str, dict[str, Any]], config: EvalConfig
) -> dict[str, dict[str, Any]]:
    """Each provider's own 95% interval on AUPRC and AUROC.

    Separate from :func:`compare`, which intervals a *difference*. A paired
    interval on a difference is narrower than either arm's own interval, so the
    two are never interchangeable: the headline chart shows how precisely each
    provider is known, and the verdict shows whether one beats another.

    Computed here rather than in :func:`evaluate_run` because it costs a
    10,000-resample bootstrap per provider, and ``evaluate_run`` is on the
    dashboard's per-snapshot path. Both the report and the page call this, once,
    and cache it.
    """
    mcfg = config.section("metrics")
    resamples = int(mcfg.get("bootstrap_resamples", 10000))
    seed = int(mcfg.get("bootstrap_seed", 0))
    out: dict[str, dict[str, Any]] = {}
    for provider, summary in summaries.items():
        y, scores = summary["_y"], summary["_scores"]
        out[provider] = {
            "auprc": metrics.bootstrap_interval(y, scores, "auprc", resamples, seed).as_dict(),
            "auroc": metrics.bootstrap_interval(y, scores, "auroc", resamples, seed).as_dict(),
            "resamples": resamples,
            "n_scored": int(summary["n_scored"]),
        }
    return out


def pair_on_shared_traces(
    summary: dict[str, Any], baseline: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align two arms on the traces they *both* scored.

    A paired bootstrap needs pairs. Two arms do not necessarily score the same
    traces: one can fail on a trace the other answered, and then the two score
    vectors are different lengths and their positions mean different things.
    Zipping them anyway would compare each arm against the wrong row of the
    other, quietly, for every row after the first gap.

    Returns the shared labels, the two aligned score vectors, and the ids that
    had to be dropped, so the caller can report the loss rather than absorb it.
    """
    mine = {t: i for i, t in enumerate(summary["_trace_ids"])}
    theirs = {t: i for i, t in enumerate(baseline["_trace_ids"])}
    shared = [t for t in summary["_trace_ids"] if t in theirs]
    dropped = sorted(set(mine) ^ set(theirs))
    rows = np.array([mine[t] for t in shared], dtype=int)
    cols = np.array([theirs[t] for t in shared], dtype=int)
    if not shared:
        empty = np.array([], dtype=float)
        return empty, empty, empty, dropped
    return (
        np.asarray(summary["_y"])[rows],
        np.asarray(summary["_scores"])[rows],
        np.asarray(baseline["_scores"])[cols],
        dropped,
    )


def compare(
    summary: dict[str, Any],
    baseline: dict[str, Any],
    config: EvalConfig,
    baseline_name: str,
) -> dict[str, Any]:
    mcfg = config.section("metrics")
    resamples = int(mcfg.get("bootstrap_resamples", 10000))
    seed = int(mcfg.get("bootstrap_seed", 20260919))
    budget = float(config.section("gates").get("useful_signal_budget", 0.05))
    y, scores, baseline_scores, dropped = pair_on_shared_traces(summary, baseline)
    auprc_ci = metrics.paired_bootstrap(y, scores, baseline_scores, "auprc", resamples, seed)
    recall_ci = metrics.paired_bootstrap(
        y, scores, baseline_scores, "recall_at_budget", resamples, seed, budget=budget
    )
    return {
        "baseline": baseline_name,
        "resamples": resamples,
        # The comparison is over the traces both arms scored. When that is fewer
        # than either arm's own n_scored, the interval below is on a smaller
        # sample than the headline table, and the report says so.
        "n_paired": len(y),
        "n_unpaired": len(dropped),
        "unpaired_trace_ids": dropped[:10],
        "auprc_delta": auprc_ci.as_dict(),
        f"recall_delta_at_{budget:.0%}": recall_ci.as_dict(),
    }


def _useful_signal_note(stage_ok: bool, n_records: int, provider: str) -> str:
    if stage_ok:
        return (
            f"Reported from the scored run. Preregistered against {BASELINE!r}, the "
            f"deterministic baseline, and frozen there. Clearing it says the provider "
            f"beats hand-written rules; it does not say the provider beats the strong "
            f"free baseline. Read 'beats_strong_baseline' for that."
        )
    if provider == BASELINE:
        return (
            f"NOT APPLICABLE. {BASELINE!r} is the baseline the gate is defined against; "
            "there is nothing for it to improve over."
        )
    return (
        f"NOT REPORTED. Stage-1 pipeline-proof run of {n_records} records; the "
        f"useful-signal gate is only read from a run of at least {STAGE_MIN_RECORDS}."
    )


def evaluate_gates(
    summary: dict[str, Any],
    comparisons: dict[str, Any],
    config: EvalConfig,
    n_records: int,
    provider: str,
) -> dict[str, Any]:
    gates = config.section("gates")
    stage_ok = n_records >= STAGE_MIN_RECORDS and provider != BASELINE
    useful: dict[str, Any] = {
        "reported": stage_ok,
        "note": _useful_signal_note(stage_ok, n_records, provider),
    }
    if stage_ok and BASELINE in comparisons:
        auprc_lower = comparisons[BASELINE]["auprc_delta"]["lower"]
        budget_key = next(k for k in comparisons[BASELINE] if k.startswith("recall_delta_at_"))
        recall_point = comparisons[BASELINE][budget_key]["point"] * 100.0
        useful["auprc_lower_bound"] = auprc_lower
        useful["recall_delta_points"] = recall_point
        useful["passed"] = bool(
            auprc_lower > float(gates.get("useful_signal_auprc_delta_lower_bound", 0.05))
            or recall_point >= float(gates.get("useful_signal_recall_delta_points", 10.0))
        )

    strong: dict[str, Any] = {
        "baseline": STRONG_BASELINE,
        "note": (
            f"Not preregistered. Added when {STRONG_BASELINE!r} was built, and reported "
            f"alongside the frozen gate rather than in place of it. The verdict is taken "
            f"from the interval, not the point estimate: a paid provider only counts as "
            f"beating the strong free baseline when the whole interval is above zero."
        ),
    }
    if STRONG_BASELINE in comparisons:
        delta = comparisons[STRONG_BASELINE]["auprc_delta"]
        strong["reported"] = True
        strong["auprc_delta"] = delta
        strong["passed"] = bool(delta["lower"] > 0.0)
        strong["loses"] = bool(delta["upper"] < 0.0)
    else:
        strong["reported"] = False
        strong["note"] = (
            f"NOT APPLICABLE. {provider!r} has no comparison against {STRONG_BASELINE!r} "
            f"in this report."
            if provider != STRONG_BASELINE
            else f"NOT APPLICABLE. {STRONG_BASELINE!r} is the strong baseline itself."
        )

    # bool() around every criterion is load-bearing, not decoration. These
    # comparisons run against numpy scalars, so they yield numpy.bool_, which
    # serialises to the *string* "False" -- and a non-empty string is truthy, so
    # ``all()`` would have passed a run that failed only on calibration.
    deployment = {
        "precision_ok": bool(
            summary["precision"] >= float(gates.get("deployment_precision", 0.90))
        ),
        "recall_ok": bool(summary["recall"] >= float(gates.get("deployment_recall", 0.80))),
        "ece_ok": bool(summary["ece"] <= float(gates.get("deployment_max_ece", 0.10))),
        "error_rate_ok": bool(
            summary["operational"]["error_rate"]
            < float(gates.get("deployment_max_api_error_rate", 0.01))
        ),
        "parse_failures_ok": bool(
            summary["operational"]["parse_failure_count"]
            <= int(gates.get("deployment_max_parse_failures", 0))
        ),
    }
    deployment["passed"] = all(deployment.values())
    return {
        "useful_signal": useful,
        "beats_strong_baseline": strong,
        "deployment_candidate": deployment,
    }


# -- plots --------------------------------------------------------------
def plot_pr_curves(runs: dict[str, dict[str, Any]], out: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    for provider, summary in sorted(runs.items()):
        y, scores = summary["_y"], summary["_scores"]
        if len(set(y.tolist())) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, scores)
        plt.plot(recall, precision, label=f"{provider} (AUPRC {summary['auprc']:.3f})")
    baseline_rate = float(np.mean(next(iter(runs.values()))["_y"])) if runs else 0.0
    plt.axhline(baseline_rate, ls="--", c="grey", lw=1, label=f"prevalence {baseline_rate:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Unsupported-success detection")
    plt.ylim(0, 1.02)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_reliability(runs: dict[str, dict[str, Any]], out: Path, bins: int = 10) -> None:
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    for provider, summary in sorted(runs.items()):
        y, scores = summary["_y"], summary["_scores"]
        order = np.argsort(scores, kind="stable")
        edges = np.linspace(0, len(y), bins + 1).astype(int)
        xs, ys = [], []
        for start, stop in pairwise(edges):
            if stop <= start:
                continue
            idx = order[start:stop]
            xs.append(float(np.mean(scores[idx])))
            ys.append(float(np.mean(y[idx])))
        plt.plot(xs, ys, marker="o", ms=4, label=f"{provider} (ECE {summary['ece']:.3f})")
    plt.xlabel("Mean predicted P(unsupported_success)")
    plt.ylabel("Observed frequency")
    plt.title("Reliability, equal-frequency bins")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


#: The curve line colours, kept in step with the dashboard's palette so the
#: report plot and the page panel are recognisably the same figure.
_CURVE_COLOURS = {"tfidf": "#9497E6", "tfidf_gbm": "#6366F1"}
_ZERO_SHOT_COLOURS = {"jev": "#111827", "openai": "#D97706"}


def plot_learning_curve(curve: dict[str, Any] | None, out: Path) -> bool:
    """Classifier curves with their seed band, and the flat zero-shot lines.

    Returns whether a figure was written. Nothing is drawn when no curve has
    been swept: an empty axis labelled "learning curve" would read as a result.
    """
    if not curve or not curve.get("curves"):
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for provider, rows in sorted(curve["curves"].items()):
        xs = [row["n_train"] for row in rows]
        means = [row["auprc"]["mean"] for row in rows]
        lower = [row["auprc"]["lower"] for row in rows]
        upper = [row["auprc"]["upper"] for row in rows]
        colour = _CURVE_COLOURS.get(provider, "#6366F1")
        ax.fill_between(xs, lower, upper, color=colour, alpha=0.18, linewidth=0)
        ax.plot(xs, means, marker="o", ms=4, color=colour, label=f"{provider} (mean of seeds)")

    for provider, arm in sorted((curve.get("zero_shot") or {}).items()):
        value = arm.get("auprc")
        if value is None:
            continue
        colour = _ZERO_SHOT_COLOURS.get(provider, "grey")
        ax.axhline(
            float(value),
            ls="--",
            lw=1.3,
            color=colour,
            label=f"{provider} zero-shot ({float(value):.3f})",
        )

    # The crossovers, marked where they happen. Only the mean crossing is drawn;
    # the lower-bound crossing is stated in the table beneath the figure. Several
    # crossings land on the same budget -- two classifiers passing the same arm
    # at 25 labels, say -- so they are grouped by budget and written as one
    # stacked label per rule. Annotating each one at its own curve point put four
    # captions on top of each other.
    at_budget: dict[int, list[tuple[str, str]]] = {}
    for provider, references in sorted((curve.get("crossovers") or {}).items()):
        for reference, result in sorted(references.items()):
            mark = result.get("mean")
            if mark:
                at_budget.setdefault(int(mark["n_train"]), []).append((provider, reference))
    # Stacked vertically as well as grouped, because two budgets an inch apart on
    # a log axis would otherwise have their captions run into each other.
    for index, (budget, crossings) in enumerate(sorted(at_budget.items())):
        colours = {_CURVE_COLOURS.get(p, "grey") for p, _ in crossings}
        rule_colour = colours.pop() if len(colours) == 1 else "#6B7185"
        ax.axvline(budget, ls=":", lw=1, color=rule_colour)
        ax.annotate(
            f"{budget} labels\n" + "\n".join(f"{p} > {r}" for p, r in crossings),
            xy=(budget, 0.03 + 0.13 * index),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            va="bottom",
            color=rule_colour,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Labelled training examples (task-disjoint, stratified)")
    ax.set_ylabel("AUPRC on the frozen test split")
    ax.set_title("Low-label learning curve against the zero-shot arms")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_fault_heatmap(runs: dict[str, dict[str, Any]], out: Path) -> None:
    keys = sorted({k for s in runs.values() for k in s["per_fault"]})
    providers = sorted(runs)
    if not keys or not providers:
        return
    matrix = np.array(
        [
            [runs[p]["per_fault"].get(k, {}).get("flag_rate", np.nan) for k in keys]
            for p in providers
        ]
    )
    plt.figure(figsize=(max(7, len(keys) * 0.55), 1.6 + 0.5 * len(providers)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="rocket_r",
        xticklabels=keys,
        yticklabels=providers,
        cbar_kws={"label": "flag rate"},
    )
    plt.title("Flag rate by true label / fault type")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# -- audit --------------------------------------------------------------
def audit_section(key_path: Path, blind_path: Path) -> dict[str, Any]:
    """Join the audit key with whatever a human filled into the blinded sheet."""
    if not key_path.exists():
        return {"status": "no_audit_key", "note": f"{key_path} not found."}
    key: dict[str, dict[str, str]] = {}
    with key_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key[row["trace_id"]] = row

    # The stratified sample. Every (label, fault) cell contributes, so the rarer
    # subclasses -- semantic_target_mismatch above all -- are guaranteed to be in
    # front of the auditor rather than sampled in by luck.
    sample = {tid for tid, row in key.items() if (row.get("in_sample") or "").upper() == "TRUE"}
    strata: dict[str, int] = {}
    for tid in sample:
        strata[key[tid].get("audit_stratum") or "unknown"] = (
            strata.get(key[tid].get("audit_stratum") or "unknown", 0) + 1
        )

    if not blind_path.exists():
        return {
            "status": "not_performed",
            "n_records": len(key),
            "note": (
                f"{blind_path} not found. The blinded audit sheet is produced by "
                "`jev-eval generate`; the audit itself has not been carried out."
            ),
            "n_sample": len(sample),
            "sample_strata": strata,
        }

    filled: list[dict[str, str]] = []
    auditors: set[str] = set()
    with blind_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("auditor_label") or "").strip():
                filled.append(row)
                if (row.get("auditor_id") or "").strip():
                    auditors.add(row["auditor_id"].strip())

    if not filled:
        return {
            "status": "not_performed",
            "n_records": len(key),
            "n_audited": 0,
            "note": (
                "The blinded audit sheet exists but no auditor_label cells are filled in. "
                "No agreement figure can be reported, and none is invented here."
            ),
            "n_sample": len(sample),
            "sample_strata": strata,
        }

    corrections = []
    agree = 0
    for row in filled:
        truth = key.get(row["trace_id"])
        if truth is None:
            continue
        auditor_label = row["auditor_label"].strip()
        if auditor_label == truth["label"]:
            agree += 1
        else:
            corrections.append(
                {
                    "trace_id": row["trace_id"],
                    "template_family": truth["template_family"],
                    "construction_label": truth["label"],
                    "construction_fault_type": truth["fault_type"],
                    "auditor_label": auditor_label,
                    "auditor_fault_type": (row.get("auditor_fault_type") or "").strip(),
                    "auditor_note": (row.get("auditor_note") or "").strip(),
                    "auditor_id": (row.get("auditor_id") or "").strip(),
                }
            )

    n_auditors = len(auditors)
    audited_ids = {row["trace_id"] for row in filled}
    in_sample_audited = audited_ids & sample
    by_stratum_audited: dict[str, int] = {}
    for tid in in_sample_audited:
        stratum = key[tid].get("audit_stratum") or "unknown"
        by_stratum_audited[stratum] = by_stratum_audited.get(stratum, 0) + 1
    unaudited_strata = sorted(set(strata) - set(by_stratum_audited))
    return {
        "status": "performed",
        "n_records": len(key),
        "n_audited": len(filled),
        "coverage": len(filled) / len(key) if key else 0.0,
        "n_sample": len(sample),
        "sample_strata": strata,
        "n_sample_audited": len(in_sample_audited),
        "sample_coverage": len(in_sample_audited) / len(sample) if sample else 0.0,
        "sample_strata_audited": by_stratum_audited,
        "strata_with_no_audited_trace": unaudited_strata,
        "agreement_with_construction_labels": agree / len(filled) if filled else float("nan"),
        "n_corrections": len(corrections),
        "corrections": corrections,
        "n_auditors": n_auditors,
        "inter_rater_reliability": None,
        "irr_note": (
            "Single auditor: inter-rater reliability is not defined and is not reported."
            if n_auditors <= 1
            else f"{n_auditors} auditors recorded. Inter-rater reliability is not computed by "
            "this harness; compute it separately if you need it."
        ),
    }


# -- markdown -----------------------------------------------------------
def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return format(value, spec)
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def _money(usd: float | None, gbp: float | None) -> str:
    """USD beside GBP, or an explicit 'unavailable' -- never a zero standing in."""
    if usd is None or gbp is None:
        return "unavailable"
    return f"${usd:.4f} / £{gbp:.4f}"


def _render_general_judge(results: dict[str, Any]) -> str:
    """The typed model against the general judge, rendered either way it lands."""
    block = results.get("general_judge") or {}
    if not block:
        return ""
    primary, judge = block["primary"], block["general_judge"]
    out: list[str] = ["## Typed model against general judge", ""]

    if not block.get("reported"):
        out += [f"**{block['verdict_text']}**", "", block["note"], ""]
        return "\n".join(out)

    headline = {
        "primary_wins": f"**{primary} beats the general judge.**",
        "judge_wins": f"**{primary} does not beat the general judge.**",
        "no_difference": "**No difference established.**",
        "not_computable": "**Not computable on this split.**",
    }.get(block["verdict"], "**Comparison unavailable.**")
    out += [f"{headline} {block['verdict_text']}", ""]

    arms = block["arms"]
    rows = []
    for name in (primary, judge):
        arm = arms.get(name)
        if arm is None:
            continue
        rows.append(
            [
                f"`{name}`",
                f"`{arm['model_id']}`",
                _fmt(arm["auprc"]),
                _fmt(arm["auroc"]),
                _fmt(arm["precision"]),
                _fmt(arm["recall"]),
                _fmt(arm["ece"]),
                _money(arm["cost_per_1000_traces_usd"], arm["cost_per_1000_traces_gbp"]),
            ]
        )
    out += [
        _table(
            [
                "arm",
                "model",
                "AUPRC",
                "AUROC",
                "precision",
                "recall",
                "ECE",
                "cost / 1000 traces",
            ],
            rows,
        ),
        "",
    ]

    auprc = block.get("auprc_delta") or {}
    recall = block.get("recall_delta") or {}
    if auprc:
        out.append(
            f"- ΔAUPRC ({primary} minus {judge}): **{_fmt(auprc.get('point'), '+.4f')}** "
            f"[{_fmt(auprc.get('lower'), '+.4f')}, {_fmt(auprc.get('upper'), '+.4f')}]"
        )
    if recall:
        budget = str(block.get("recall_delta_key", "")).replace("recall_delta_at_", "")
        out.append(
            f"- Δrecall@{budget} ({primary} minus {judge}): "
            f"**{_fmt(recall.get('point'), '+.4f')}** "
            f"[{_fmt(recall.get('lower'), '+.4f')}, {_fmt(recall.get('upper'), '+.4f')}]"
        )
    out.append(
        f"- Paired bootstrap over {block.get('resamples')} resamples, 95% percentile "
        "interval. The verdict is read from the interval, not the point estimate."
    )
    unpaired = int(block.get("n_unpaired", 0) or 0)
    if unpaired:
        out.append(
            f"- Computed on the **{block.get('n_paired')} traces both arms scored**. "
            f"{unpaired} trace(s) were scored by one arm and not the other and are "
            "excluded from the interval (they remain in each arm's own row above). "
            "A paired interval needs pairs; the unpaired traces are dropped from the "
            "comparison rather than matched against the wrong row."
        )

    deviations = block.get("deviations") or []
    if deviations:
        out += [
            "",
            "### Deviations from the frozen request",
            "",
            "These arms did not send exactly what `config/eval.yaml` specifies. The "
            "provider refused a parameter and the harness dropped it rather than "
            "abandoning the arm. Read every figure above with this in mind.",
            "",
            _table(
                ["arm", "model", "parameter", "requested", "applied", "the provider's words"],
                [
                    [
                        f"`{row.get('provider', '')}`",
                        f"`{row.get('run_model_id') or row.get('model_id')}`",
                        f"`{row['parameter']}`",
                        f"`{row['requested']!r}`",
                        str(row["applied"]),
                        f"HTTP {row['http_status']} `{row['error_code']}`: {row['error_message']}",
                    ]
                    for row in deviations
                ],
            ),
            "",
        ]

    unpriced = [name for name, arm in arms.items() if not arm.get("cost_available")]
    if unpriced:
        out.append(
            f"- Cost is **unavailable** for {', '.join(f'`{n}`' for n in unpriced)}: no "
            "verified published rate is on record in the dated price manifest, so tokens "
            "are recorded and money is not. No rate was guessed."
        )
    out += ["", block["note"], ""]
    return "\n".join(out)


def _render_learning_curve(results: dict[str, Any]) -> str:
    """The low-label section: where a free classifier overtakes a zero-shot arm.

    Absent rather than empty when no curve has been swept. A section that said
    "no crossover" because nobody had run the sweep would be indistinguishable
    from one that said it because there is none.
    """
    curve = results.get("learning_curve")
    if not curve or not curve.get("curves"):
        return ""

    out: list[str] = []
    a = out.append
    a("## Low-label learning curve")
    a("")
    a(
        "The full-size comparison is lopsided: the free classifiers win. The one "
        "advantage a zero-shot judge keeps is that it needs **no labels at all**. "
        "This section is where that advantage ends."
    )
    a("")
    a(
        "- Subsets are drawn from the train split alone, which shares no task with "
        f"**{curve['split']}**. The draw is task-first (whole tasks admitted in a seeded "
        "order, never individual rows) and stratified to the train split's own label "
        "prevalence."
    )
    a(f"- Seeds per size: {len(curve.get('seeds') or [])}. Sizes: {', '.join(curve['sizes'])}.")
    a(
        f"- Every model is scored on the frozen **{curve['split']}** split, with the metric "
        "functions this report uses everywhere else."
    )
    a(f"- Paid calls made to build this curve: **{curve.get('paid_calls', 0)}**.")
    a("")
    a("![Learning curve](learning_curve.png)")
    a("")

    a("### Crossover")
    a("")
    rows: list[list[str]] = []
    for provider in sorted(curve["crossovers"]):
        for reference, result in sorted(curve["crossovers"][provider].items()):
            mean_cross = result["mean"]
            lower_cross = result["lower_bound"]
            rows.append(
                [
                    provider,
                    reference,
                    _fmt(result["reference_auprc"]),
                    f"{mean_cross['n_train']}" if mean_cross else "never",
                    f"{lower_cross['n_train']}" if lower_cross else "never",
                    "yes" if result["reference_leads_anywhere"] else "no",
                ]
            )
    a(
        _table(
            [
                "classifier",
                "zero-shot arm",
                "its AUPRC",
                "labels for mean to win",
                "labels for lower bound to win",
                "arm leads anywhere",
            ],
            rows,
        )
    )
    a("")
    for provider in sorted(curve["crossovers"]):
        for _reference, result in sorted(curve["crossovers"][provider].items()):
            a(f"- {result['statement']}.")
    a("")
    a(
        "A size only counts as the crossover when it **and every larger size** stay above "
        "the line, so one lucky point does not get reported as a crossing."
    )
    a("")

    a("### Curve")
    a("")
    curve_rows: list[list[str]] = []
    for provider in sorted(curve["curves"]):
        for row in curve["curves"][provider]:
            curve_rows.append(
                [
                    provider,
                    row["size_label"],
                    str(row["n_train"]),
                    f"{row['n_train_tasks_mean']:.1f}",
                    _fmt(row["auprc"]["mean"]),
                    _fmt(row["auprc"]["lower"]),
                    _fmt(row["auprc"]["upper"]),
                    _fmt(row["recall"]["mean"]),
                    str(row["auprc"]["n"]),
                ]
            )
    a(
        _table(
            [
                "classifier",
                "size",
                "labels fit on",
                "tasks (mean)",
                "AUPRC mean",
                "lower 95%",
                "upper 95%",
                "recall mean",
                "seeds",
            ],
            curve_rows,
        )
    )
    a("")
    caveats = curve.get("caveats") or {}
    if caveats.get("calibration"):
        a(f"**What N counts.** {caveats['calibration']}")
        a("")
    if caveats.get("band"):
        a(f"**What the band is.** {caveats['band']}")
        a("")
    return "\n".join(out)


def render_markdown(results: dict[str, Any]) -> str:
    providers = sorted(results["runs"])
    lines: list[str] = []
    a = lines.append

    a("# False-success detection: offline comparison")
    a("")
    a("> **Not for publication.** TypeSafe's Master Customer Agreement restricts publication of")
    a("> benchmark or performance results. These artifacts are local only.")
    a("")
    a(
        f"- Dataset: `{results['dataset_path']}` ({results['n_records']} records, "
        f"sha256 `{results['dataset_sha256'][:16]}...`)"
    )
    a(f"- Split scored: **{results['split']}** ({results['n_scored']} traces)")
    a(f"- Stage: **{results['stage']}**")
    a(f"- Bootstrap: {results['bootstrap_resamples']} paired resamples, 95% percentile intervals")
    if results.get("stale_runs_skipped"):
        a(
            f"- Skipped {len(results['stale_runs_skipped'])} run(s) scored against a different "
            "dataset: " + ", ".join(f"`{s}`" for s in results["stale_runs_skipped"])
        )
    a("")

    a(_render_general_judge(results))

    curve_section = _render_learning_curve(results)
    if curve_section:
        a(curve_section)

    a("## Primary metrics (unsupported_success vs rest)")
    a("")

    def _interval(provider: str, metric: str) -> str:
        """The metric with its own 95% interval, never the bare point estimate."""
        entry = (results.get("intervals") or {}).get(provider, {}).get(metric)
        point = _fmt(results["runs"][provider][metric])
        if not entry or entry.get("lower") is None:
            return point
        return f"{point} [{_fmt(entry['lower'], '.3f')}, {_fmt(entry['upper'], '.3f')}]"

    a(
        _table(
            [
                "provider",
                "model",
                "AUPRC [95% CI]",
                "AUROC [95% CI]",
                "precision",
                "recall",
                "F1",
                "thr",
                "macro-F1",
            ],
            [
                [
                    p,
                    results["runs"][p]["model_id"],
                    _interval(p, "auprc"),
                    _interval(p, "auroc"),
                    _fmt(results["runs"][p]["precision"]),
                    _fmt(results["runs"][p]["recall"]),
                    _fmt(results["runs"][p]["f1"]),
                    _fmt(results["runs"][p]["threshold"], ".2f"),
                    _fmt(results["runs"][p]["macro_f1"]),
                ]
                for p in providers
            ],
        )
    )
    a("")
    a(
        "Intervals are 95% percentile bootstrap over resampled traces, "
        f"{results.get('bootstrap_resamples', 0):,} resamples, computed per provider. "
        "They are **not** the paired intervals used for the verdicts below: a paired "
        "interval on a difference is narrower than either arm's own interval."
    )
    a("")

    a("## Confusion at the frozen threshold")
    a("")
    a(
        _table(
            ["provider", "TP", "FP", "FN", "TN", "TPR", "FPR"],
            [
                [
                    p,
                    str(results["runs"][p]["tp"]),
                    str(results["runs"][p]["fp"]),
                    str(results["runs"][p]["fn"]),
                    str(results["runs"][p]["tn"]),
                    _fmt(results["runs"][p]["tpr"]),
                    _fmt(results["runs"][p]["fpr"]),
                ]
                for p in providers
            ],
        )
    )
    a("")

    budgets = (
        sorted(results["runs"][providers[0]]["recall_at_budget"], key=_pct_key) if providers else []
    )
    a("## Recall at review budget")
    a("")
    a(
        _table(
            ["provider", *budgets],
            [
                [p, *[_fmt(results["runs"][p]["recall_at_budget"][b]) for b in budgets]]
                for p in providers
            ]
            + [
                [
                    "_ceiling (any ranker)_",
                    *[
                        _fmt(results["runs"][providers[0]]["recall_at_budget_ceiling"][b])
                        for b in budgets
                    ],
                ]
            ],
        )
    )
    a("")
    a("The ceiling row is `budget x n / positives`: the most recall *any* ranker can reach")
    a("at that budget on this split. Where a provider equals the ceiling, the metric is")
    a("saturated and cannot separate providers - see Limitations.")
    a("")

    a("## Calibration and stability")
    a("")
    a(
        _table(
            ["provider", "Brier", "ECE (10 eq-freq bins)", "label agreement", "score variance"],
            [
                [
                    p,
                    _fmt(results["runs"][p]["brier"]),
                    _fmt(results["runs"][p]["ece"]),
                    _fmt(results["runs"][p]["repeat_stability"]["mean_label_agreement"]),
                    _fmt(results["runs"][p]["repeat_stability"]["mean_score_variance"], ".6f"),
                ]
                for p in providers
            ],
        )
    )
    a("")

    if results["comparisons"]:
        a("## Paired differences (95% bootstrap CI)")
        a("")
        rows = []
        for provider, against in sorted(results["comparisons"].items()):
            for baseline, comparison in sorted(against.items()):
                budget_key = next(k for k in comparison if k.startswith("recall_delta_at_"))
                rows.append(
                    [
                        provider,
                        baseline,
                        f"{_fmt(comparison['auprc_delta']['point'])} "
                        f"[{_fmt(comparison['auprc_delta']['lower'])}, "
                        f"{_fmt(comparison['auprc_delta']['upper'])}]",
                        f"{_fmt(comparison[budget_key]['point'])} "
                        f"[{_fmt(comparison[budget_key]['lower'])}, "
                        f"{_fmt(comparison[budget_key]['upper'])}]",
                    ]
                )
        a(_table(["provider", "vs", "ΔAUPRC [95% CI]", "Δrecall@budget [95% CI]"], rows))
        a("")

    a("## Operational")
    a("")
    a(
        _table(
            [
                "provider",
                "n",
                "errors",
                "parse failures",
                "retries",
                "in tok",
                "out tok",
                "cost USD",
                "p50 ms",
                "p95 ms",
                "p99 ms",
            ],
            [
                [
                    p,
                    str(results["runs"][p]["operational"]["n_predictions"]),
                    str(results["runs"][p]["operational"]["error_count"]),
                    str(results["runs"][p]["operational"]["parse_failure_count"]),
                    str(results["runs"][p]["operational"]["retry_count"]),
                    str(results["runs"][p]["operational"]["total_input_tokens"]),
                    str(results["runs"][p]["operational"]["total_output_tokens"]),
                    _fmt(results["runs"][p]["operational"]["total_cost_usd"], ".6f"),
                    _fmt(results["runs"][p]["latency"]["p50_ms"], ".3f"),
                    _fmt(results["runs"][p]["latency"]["p95_ms"], ".3f"),
                    _fmt(results["runs"][p]["latency"]["p99_ms"], ".3f"),
                ]
                for p in providers
            ],
        )
    )
    a("")

    a("## Flag rate by fault type")
    a("")
    keys = sorted({k for p in providers for k in results["runs"][p]["per_fault"]})
    a(
        _table(
            ["label / fault", *providers, "n"],
            [
                [
                    k,
                    *[
                        _fmt(results["runs"][p]["per_fault"].get(k, {}).get("flag_rate"), ".2f")
                        for p in providers
                    ],
                    str(
                        next(
                            (
                                results["runs"][p]["per_fault"][k]["n"]
                                for p in providers
                                if k in results["runs"][p]["per_fault"]
                            ),
                            0,
                        )
                    ),
                ]
                for k in keys
            ],
        )
    )
    a("")

    a("## The semantic subclass and its matched control")
    a("")
    a(
        "`semantic_target_mismatch` is the one positive fault with no exact-match handle: "
        "the goal names its target only by description, and the write succeeds and confirms "
        "the target the agent chose. `semantic_target_match` is its matched negative control "
        "- same goal shape, same candidate listing, right target."
    )
    a("")
    a(
        "The two flag rates are the measurement. A checker that reads the goal against the "
        "evidence separates them; one that keys on surface shape flags both at the same rate "
        "and its apparent recall on the subclass is worth nothing."
    )
    a("")
    mismatch_key = "unsupported_success/semantic_target_mismatch"
    control_key = "supported_success/semantic_target_match"
    rows = []
    for p in providers:
        per_fault = results["runs"][p]["per_fault"]
        fault = per_fault.get(mismatch_key, {})
        control = per_fault.get(control_key, {})
        fault_rate = fault.get("flag_rate")
        control_rate = control.get("flag_rate")
        separation = (
            fault_rate - control_rate
            if fault_rate is not None and control_rate is not None
            else None
        )
        rows.append(
            [
                p,
                f"{_fmt(fault_rate, '.2f')} (n={fault.get('n', 0)})",
                f"{_fmt(control_rate, '.2f')} (n={control.get('n', 0)})",
                _fmt(separation, "+.2f"),
                _fmt(fault.get("mean_score"), ".3f"),
                _fmt(control.get("mean_score"), ".3f"),
            ]
        )
    a(
        _table(
            [
                "provider",
                "flag rate on the fault",
                "flag rate on its control",
                "separation",
                "mean score (fault)",
                "mean score (control)",
            ],
            rows,
        )
    )
    a("")
    a(
        "**Separation** is the difference of the two. Zero means the provider cannot tell the "
        "fault from its control at all, whatever its flag rate on either."
    )
    a("")

    a("## Projected precision under prior shift")
    a("")
    a("These are **projections** from the observed TPR and FPR at an assumed production")
    a("prevalence. They are not observed results.")
    a("")
    degenerate = [p for p in providers if results["runs"][p]["fpr"] == 0.0]
    if degenerate:
        a(f"⚠ {', '.join(degenerate)} recorded **zero** false positives on this split, so the")
        a("projection collapses to 1.0 at every prevalence. That is arithmetic, not evidence:")
        a("an unobserved FPR is not a zero FPR. Read these rows as 'too few negatives to")
        a("estimate an FPR', and widen the test split before relying on them.")
        a("")
    prevs = sorted(results["runs"][providers[0]]["prior_shift"], key=_pct_key) if providers else []
    a(
        _table(
            ["provider", *[f"precision @ {p} prevalence" for p in prevs]],
            [[p, *[_fmt(results["runs"][p]["prior_shift"][k]) for k in prevs]] for p in providers],
        )
    )
    a("")

    a("## Decision gates")
    a("")
    for provider in providers:
        gates = results["gates"].get(provider)
        if not gates:
            continue
        a(f"**{provider}**")
        a("")
        useful = gates["useful_signal"]
        if useful.get("reported"):
            a(
                f"- Useful signal: **{'PASS' if useful.get('passed') else 'FAIL'}** "
                f"(ΔAUPRC lower bound {_fmt(useful.get('auprc_lower_bound'))}, "
                f"Δrecall {_fmt(useful.get('recall_delta_points'), '.1f')} points)"
            )
        else:
            a(f"- Useful signal: _{useful['note']}_")
        strong = gates["beats_strong_baseline"]
        if strong.get("reported"):
            delta = strong["auprc_delta"]
            if strong["passed"]:
                verdict = "PASS"
            elif strong.get("loses"):
                verdict = "FAIL (loses to it)"
            else:
                verdict = "FAIL (no difference established)"
            a(
                f"- Beats the strong free baseline ({STRONG_BASELINE}): **{verdict}** "
                f"(ΔAUPRC {_fmt(delta['point'])}, 95% CI "
                f"[{_fmt(delta['lower'])}, {_fmt(delta['upper'])}])"
            )
        else:
            a(f"- Beats the strong free baseline: _{strong['note']}_")
        deployment = gates["deployment_candidate"]
        failed = [
            name.removesuffix("_ok")
            for name, ok in deployment.items()
            if name != "passed" and not ok
        ]
        verdict = "PASS" if deployment["passed"] else "FAIL"
        detail = "all criteria met" if not failed else "failed on: " + ", ".join(failed)
        a(f"- Deployment candidate: **{verdict}** ({detail})")
        a("")

    audit = results["audit"]
    a("## Blinded human audit")
    a("")
    if audit["status"] == "performed":
        a(f"- Records in audit key: {audit['n_records']}")
        a(f"- Traces audited: {audit['n_audited']} ({audit['coverage']:.1%} coverage)")
        a(
            f"- Stratified sample: {audit['n_sample']} traces across "
            f"{len(audit['sample_strata'])} (label / fault) cells; "
            f"{audit['n_sample_audited']} audited "
            f"({audit['sample_coverage']:.1%} of the sample)"
        )
        if audit["strata_with_no_audited_trace"]:
            a(
                "- **Cells with no audited trace:** "
                + ", ".join(f"`{s}`" for s in audit["strata_with_no_audited_trace"])
                + ". Their construction labels rest on the generator alone."
            )
        a(
            f"- Agreement with construction labels: "
            f"**{audit['agreement_with_construction_labels']:.1%}**"
        )
        a(f"- Corrections recorded: {audit['n_corrections']}")
        a(f"- Auditors: {audit['n_auditors']}. {audit['irr_note']}")
        a("")
        if audit["corrections"]:
            a(
                _table(
                    ["trace_id", "family", "constructed", "auditor", "note"],
                    [
                        [
                            c["trace_id"],
                            c["template_family"],
                            c["construction_label"],
                            c["auditor_label"],
                            c["auditor_note"] or "-",
                        ]
                        for c in audit["corrections"]
                    ],
                )
            )
        else:
            a("No corrections: the auditor agreed with every construction label they reviewed.")
    else:
        a(f"- Status: **{audit['status']}**")
        a(f"- {audit['note']}")
        if audit.get("n_sample"):
            a(
                f"- The sheet marks a stratified sample of {audit['n_sample']} traces "
                f"(`in_sample=TRUE`) across {len(audit['sample_strata'])} (label / fault) cells, "
                "including every `semantic_target_mismatch` cell. That sample is the minimum a "
                "human needs to fill in."
            )
        a("")
        a("Until this is filled in, every label in this report rests on the generator's")
        a("construction rules alone and has not been independently checked by a human.")
    a("")

    caveats = results.get("caveats") or {}
    if caveats:
        a("## Caveats for this run")
        a("")
        a("Generated from the loaded dataset and run, not written into this template: a")
        a("synthetic run and a real run do not carry each other's caveats.")
        a("")
        for item in caveats.get("items", []):
            a(f"- **{item['title']}.** {item['body']}")
        a("")

    external = results.get("external_reference") or {}
    if external.get("citation"):
        a("## External context, not produced here")
        a("")
        a(external.get("note", ""))
        a("")
        a(f"> {external['citation']}")
        a("")
        figures = external.get("figures", {})
        a(
            f"That paper reports a TF-IDF detector at AUROC {figures.get('tfidf_auroc_appworld')} "
            f"on AppWorld and {figures.get('tfidf_auroc_tau2')} on tau2-bench, against a best "
            f"LLM-judge AUROC of {figures.get('best_judge_auroc_appworld')} and "
            f"{figures.get('best_judge_auroc_tau2')}. **These are the paper's numbers.** This "
            "harness did not reproduce them and nothing here confirms them."
        )
        a("")

    if (results.get("dataset") or {}).get("kind") == "real":
        return "\n".join(
            lines + ["## Plots", ""] + [f"![{n}]({n})" for n in results["plots"]] + [""]
        )

    a("## Limitations")
    a("")
    a("Read the tables above against these four constraints. They are properties of the")
    a("design, known before the test split was opened, and recorded in preregistration.md.")
    a("")
    a("1. **The rules baseline is near its ceiling by construction.** The generator builds")
    a("   faults that code *can* compute exactly: a status field, an entity mismatch, a")
    a("   parameter mismatch, a zero-row update. A deterministic checker should score close")
    a("   to perfect here, and does. A near-perfect rules score is evidence about the")
    a("   dataset, not evidence that false-success detection is a solved problem. What this")
    a("   harness can measure is whether a model *matches* a strong checker on faults that")
    a("   are cleanly computable; it cannot measure performance on the messy, partially")
    a("   observable traces where a deterministic checker would actually fail.")
    a("")
    a("2. **Recall at review budget saturates.** The dataset is label-balanced, so")
    a("   prevalence is 25%. A 5% review budget can therefore never exceed 20% recall, for")
    a("   any ranker. Where providers sit on the ceiling row, that metric carries no signal")
    a("   and the paired interval around it will be degenerate. The AUPRC comparison is the")
    a("   one to read; the prior-shift projections show what these operating points would")
    a("   mean at a realistic production prevalence.")
    a("")
    a("3. **ECE penalises an uncalibrated ranking score.** The rules score is a weighted")
    a("   sum of evidence checks, designed to rank, not to be read as a probability. Its")
    a("   ECE is therefore poor even when its ranking is perfect. Treat a rules ECE as a")
    a("   statement about the score's scale, not its discrimination.")
    a("")
    a("4. **Synthetic data, single generator.** Every trace comes from 40 templates written")
    a("   by one author. Agreement between a model and this dataset is agreement with one")
    a("   person's construction rules until the blinded audit above is filled in.")
    a("")
    a("## Plots")
    a("")
    for name in results["plots"]:
        a(f"![{name}]({name})")
    a("")
    return "\n".join(lines)


def _truncation_summary(
    latest: dict[str, tuple[Path, Any, list[Prediction]]],
) -> dict[str, Any] | None:
    """Truncation accounting for the caveat, read off the run that could truncate.

    Only a context-bounded provider shortens anything; the free baselines read
    the whole trace. Returning ``None`` when no run wrote an account is
    deliberate -- the caveat then says nothing about truncation, which is not
    the same claim as "nothing was truncated".
    """
    from .budget import TRUNCATION_RULE
    from .runner import TRUNCATION_FILE

    for provider in (PRIMARY, GENERAL_JUDGE):
        entry = latest.get(provider)
        if entry is None:
            continue
        run_dir, _manifest, predictions = entry
        path = run_dir / TRUNCATION_FILE
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return {
            "provider": provider,
            "truncated_traces": len(rows),
            "n_traces": len(predictions),
            "dropped_events": sum(int(r.get("dropped_events", 0)) for r in rows),
            "shortened_payloads": sum(int(r.get("shortened_payloads", 0)) for r in rows),
            "rule": rows[0].get("rule", "") if rows else TRUNCATION_RULE,
            "did_not_fit": sum(1 for r in rows if not r.get("fits", True)),
        }
    return None


def read_deviations(run_dir: Path) -> list[dict[str, Any]]:
    """The parameters a provider refused in this run, read off its own account."""
    from .runner import DEVIATION_FILE

    path = run_dir / DEVIATION_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _deviation_summary(
    latest: dict[str, tuple[Path, Any, list[Prediction]]],
) -> dict[str, Any] | None:
    """Every departure from the frozen request shape, across the paid arms.

    Returns ``None`` only when no run wrote an account at all -- which is not
    the same claim as "nothing deviated", and the caveat is worded accordingly.
    """
    found = False
    rows: list[dict[str, Any]] = []
    for provider in (PRIMARY, GENERAL_JUDGE):
        entry = latest.get(provider)
        if entry is None:
            continue
        run_dir, manifest, _predictions = entry
        if not (run_dir / "deviations.jsonl").exists():
            continue
        found = True
        for row in read_deviations(run_dir):
            rows.append({**row, "provider": provider, "run_model_id": manifest.model_id})
    if not found:
        return None
    return {"deviations": rows}


def _deviation_note(deviations: Sequence[dict[str, Any]]) -> str:
    """One sentence per refused parameter, in the model's own words."""
    parts = []
    for row in deviations:
        parts.append(
            f"{row.get('run_model_id') or row.get('model_id')} refused "
            f"{row['parameter']}={row['requested']!r} (HTTP {row['http_status']} "
            f'{row["error_code"]}: "{row["error_message"]}"), so it was '
            f"{row['applied']}."
        )
    return " ".join(parts)


def general_judge_headline(
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    config: EvalConfig,
) -> dict[str, Any]:
    """Jev against the general LLM judge: AUPRC, recall, cost, and the interval.

    This is its own block rather than one more row in the comparison table
    because it answers a different question from the free baselines. Those ask
    whether a paid lane bought anything at all. This asks whether the *typed*
    model beats a general one that was shown the same trace and asked the same
    four questions -- and what each answer costs.

    The verdict is read from the interval, never the point estimate, exactly as
    the strong-baseline verdict is. Both arms are reported whichever way it
    goes, and a missing arm is reported as missing rather than as a win.
    """
    fx = float(config.raw["fx_rate_gbp_usd"])
    present = [p for p in (PRIMARY, GENERAL_JUDGE) if p in summaries]
    deviations = [
        row for provider in present for row in (summaries[provider].get("deviations") or [])
    ]
    # "At temperature 0" is a claim about what was sent, so it is read off the
    # runs rather than written here. A model that refused the setting gets the
    # deviation stated in the same sentence, not in a footnote below it.
    temperature_note = (
        "at temperature 0"
        if not any(row["parameter"] == "temperature" for row in deviations)
        else "at the decoding settings each provider would accept"
    )
    note = (
        "Both arms see the same InferenceView, are asked the same four questions "
        f"from the same frozen questions.json {temperature_note}, are scored on "
        "P(unsupported_success) from the verdict, and run zero-shot on the same "
        "task-disjoint test split with no training. The one asymmetry that was "
        "designed in: Jev's probabilities come from the API, while the general "
        "judge's are self-reported inside its own JSON answer."
    )
    if deviations:
        note += " Deviation from the frozen request: " + _deviation_note(deviations)
    base: dict[str, Any] = {
        "primary": PRIMARY,
        "general_judge": GENERAL_JUDGE,
        "fx_rate_gbp_usd": fx,
        "fx_rate_date": str(config.raw["fx_rate_date"]),
        "deviations": deviations,
        "note": note,
    }
    if len(present) < 2:
        missing = [p for p in (PRIMARY, GENERAL_JUDGE) if p not in summaries]
        return {
            **base,
            "reported": False,
            "arms": {p: _judge_arm(summaries[p], fx) for p in present},
            "verdict": "not_run",
            "verdict_text": (
                f"No comparison: {', '.join(missing)} has no run on this split. "
                "Run it and rebuild the report."
            ),
        }

    entry = (comparisons.get(PRIMARY) or {}).get(GENERAL_JUDGE) or {}
    auprc = entry.get("auprc_delta") or {}
    recall_key = next((k for k in entry if k.startswith("recall_delta_at_")), "")
    recall = entry.get(recall_key) or {}
    lower, upper = auprc.get("lower"), auprc.get("upper")

    if lower is None or upper is None or lower != lower or upper != upper:
        verdict, text = (
            "not_computable",
            (
                "The paired bootstrap produced no usable resamples, so no interval can be "
                "stated and no winner is declared."
            ),
        )
    elif lower > 0:
        verdict, text = (
            "primary_wins",
            (
                f"Jev beats the general judge on AUPRC by {auprc['point']:+.3f} "
                f"(95% CI [{lower:+.3f}, {upper:+.3f}], entirely above zero)."
            ),
        )
    elif upper < 0:
        verdict, text = (
            "judge_wins",
            (
                f"Jev is beaten by the general judge on AUPRC by {auprc['point']:+.3f} "
                f"(95% CI [{lower:+.3f}, {upper:+.3f}], entirely below zero)."
            ),
        )
    else:
        verdict, text = (
            "no_difference",
            (
                f"No difference established. The AUPRC gap is {auprc['point']:+.3f} but the "
                f"95% CI is [{lower:+.3f}, {upper:+.3f}], which spans zero. On this split "
                "the data does not show the typed model beating the general judge, and does "
                "not show the reverse either."
            ),
        )

    return {
        **base,
        "reported": True,
        "arms": {p: _judge_arm(summaries[p], fx) for p in present},
        "metric": "auprc",
        "auprc_delta": auprc,
        "recall_delta_key": recall_key,
        "recall_delta": recall,
        "resamples": entry.get("resamples"),
        "n_paired": entry.get("n_paired"),
        "n_unpaired": entry.get("n_unpaired", 0),
        "verdict": verdict,
        "verdict_text": text,
        "primary_beats_judge": verdict == "primary_wins",
    }


def _judge_arm(summary: dict[str, Any], fx: float) -> dict[str, Any]:
    """One arm's headline figures and what 1000 traces of it costs.

    Cost per 1000 traces is the mean cost of one *scored* request times 1000 --
    one pass over 1000 traces, not this run's repeats. An arm whose model has no
    verified rate reports tokens and ``None`` for cost rather than a guess, and
    the report prints that as "unavailable".
    """
    operational = summary["operational"]
    n_priced = operational["n_predictions"] - operational["cost_unpriced_predictions"]
    total = operational["total_cost_usd"]
    per_request = (total / n_priced) if total is not None and n_priced else None
    return {
        "model_id": summary.get("model_id", ""),
        "run_id": summary.get("run_id", ""),
        "auprc": summary["auprc"],
        "auroc": summary["auroc"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "ece": summary["ece"],
        "n_scored": summary["n_scored"],
        "total_cost_usd": total,
        "n_priced_requests": n_priced,
        "n_unpriced_requests": operational["cost_unpriced_predictions"],
        "input_tokens": operational["total_input_tokens"],
        "output_tokens": operational["total_output_tokens"],
        "cost_per_1000_traces_usd": (per_request * 1000) if per_request is not None else None,
        "cost_per_1000_traces_gbp": (per_request * 1000 * fx if per_request is not None else None),
        "cost_available": total is not None,
    }


def _headline_ci(comparisons: dict[str, Any]) -> dict[str, Any] | None:
    """The interval on Jev against the strong free baseline, if both were run."""
    against = comparisons.get("jev") or {}
    entry = against.get(STRONG_BASELINE) or against.get(BASELINE)
    if not isinstance(entry, dict):
        return None
    interval = entry.get("auprc_delta") or entry.get("delta_auprc") or entry.get("ci")
    return interval if isinstance(interval, dict) else None


def build_report(
    runs_root: Path,
    out_dir: Path,
    records: Sequence[TraceRecord],
    splits_path: Path,
    dataset_path: Path,
    dataset_sha256: str,
    config: EvalConfig,
    split: str,
    audit_key: Path,
    audit_blind: Path,
    n_records_total: int,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    latest: dict[str, tuple[Path, Any, list[Prediction]]] = {}
    stale: list[str] = []
    for run_dir in discover_runs(runs_root):
        manifest, predictions = load_run(run_dir)
        if manifest.split != split:
            continue
        if manifest.dataset_sha256 != dataset_sha256:
            # A run scored against a different dataset. Regenerating the corpus
            # (amendment A6 did) leaves these behind, and "most recent run per
            # provider" would happily mix them with current ones if the records
            # they were scored on were never checked.
            stale.append(f"{run_dir} (dataset sha256 {manifest.dataset_sha256[:16]}...)")
            continue
        current = latest.get(manifest.provider)
        if current is None or manifest.created_utc >= current[1].created_utc:
            latest[manifest.provider] = (run_dir, manifest, predictions)

    if not latest:
        raise RuntimeError(
            f"no runs for split {split!r} under {runs_root} matching dataset sha256 "
            f"{dataset_sha256[:16]}..."
            + (f"; {len(stale)} run(s) were skipped as stale: {stale}" if stale else "")
        )

    summaries: dict[str, dict[str, Any]] = {}
    for provider, (run_dir, manifest, predictions) in latest.items():
        summary = evaluate_run(records, predictions, manifest.threshold, config)
        summary["model_id"] = manifest.model_id
        summary["run_id"] = manifest.run_id
        summary["run_dir"] = str(run_dir)
        summary["deviations"] = [
            {**row, "provider": provider, "run_model_id": manifest.model_id}
            for row in read_deviations(run_dir)
        ]
        summary["prior_shift"] = {
            f"{p:.0%}": metrics.prior_shift_precision(summary["tpr"], summary["fpr"], float(p))
            for p in config.section("metrics").get("production_prevalences", [0.01, 0.05, 0.10])
        }
        summaries[provider] = summary

    comparisons: dict[str, dict[str, Any]] = {}

    for provider, summary in summaries.items():
        against: dict[str, Any] = {}
        for baseline in COMPARISON_TARGETS:
            if baseline == provider or baseline not in summaries:
                continue
            against[baseline] = compare(summary, summaries[baseline], config, baseline)
        if against:
            comparisons[provider] = against

    from .caveats import build_caveats
    from .dashboard.aggregate import audit_status
    from .ingest.pipeline import load_provenance

    provenance = load_provenance(provenance_path) if provenance_path else None

    gates = {
        provider: evaluate_gates(
            summary, comparisons.get(provider, {}), config, n_records_total, provider
        )
        for provider, summary in summaries.items()
    }
    general_judge = general_judge_headline(summaries, comparisons, config)
    # Each provider's own interval, for the headline chart's error bars. The
    # report and the dashboard read the same numbers from the same function.
    intervals = provider_intervals(summaries, config)

    # The low-label learning curve, if one has been swept for this dataset and
    # split. The stored crossovers are never quoted: they are recomputed here
    # against the AUPRCs this report just measured, so the panel and the tables
    # above it cannot state a different number for the same zero-shot arm.
    from . import learning_curve as lc

    stored_curve = lc.load_curve(runs_root, dataset_sha256, split)
    curve = (
        lc.rebuild(stored_curve, lc.zero_shot_from_summaries(summaries)) if stored_curve else None
    )

    plot_pr_curves(summaries, out_dir / "pr_curve.png")
    plot_reliability(summaries, out_dir / "reliability.png")
    plot_fault_heatmap(summaries, out_dir / "fault_heatmap.png")
    curve_plotted = plot_learning_curve(curve, out_dir / "learning_curve.png")

    stage = (
        f"scored (>= {STAGE_MIN_RECORDS} records)"
        if n_records_total >= STAGE_MIN_RECORDS
        else f"stage-1 pipeline proof ({n_records_total} records)"
    )

    n_scored = int(next(iter(summaries.values()))["n_scored"])
    n_positives = int(np.sum(next(iter(summaries.values()))["_y"]))

    results: dict[str, Any] = {
        "split": split,
        "stage": stage,
        "n_records": n_records_total,
        "n_scored": n_scored,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "splits_path": str(splits_path),
        "bootstrap_resamples": int(config.section("metrics").get("bootstrap_resamples", 10000)),
        "runs": {
            p: {k: v for k, v in s.items() if not k.startswith("_")} for p, s in summaries.items()
        },
        "comparisons": comparisons,
        "intervals": intervals,
        "gates": gates,
        # Typed model against general judge, stated from the interval.
        "general_judge": general_judge,
        "audit": audit_section(audit_key, audit_blind),
        # Generated from this dataset's provenance, exactly as the dashboard
        # does it, so the report and the page cannot say different things about
        # the same run.
        "caveats": build_caveats(
            provenance=provenance,
            audit=audit_status(audit_blind),
            headline_ci=_headline_ci(comparisons),
            n_positives=n_positives,
            n_traces=n_scored,
            truncation=_truncation_summary(latest),
            deviation=_deviation_summary(latest),
        ).as_dict(),
        "dataset": {
            "id": provenance.dataset_id if provenance else "",
            "kind": provenance.kind.value if provenance else "synthetic",
        },
        "external_reference": config.raw.get("external_reference") or {},
        "stale_runs_skipped": stale,
        # The curve carries every raw point, so a reader can recompute the
        # crossover without re-fitting anything.
        "learning_curve": curve,
        "plots": ["pr_curve.png", "reliability.png", "fault_heatmap.png"]
        + (["learning_curve.png"] if curve_plotted else []),
        "publication_restriction": (
            "Local artifacts only. TypeSafe's Master Customer Agreement restricts publication "
            "of benchmark or performance results."
        ),
    }

    # Re-attach for markdown, then drop the numpy arrays before serialising.
    for provider, summary in summaries.items():
        results["runs"][provider]["prior_shift"] = summary["prior_shift"]

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(results), encoding="utf-8")
    return results
