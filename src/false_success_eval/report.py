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
STAGE_MIN_RECORDS = 1000


def _point_predictions(predictions: Sequence[Prediction]) -> dict[str, Prediction]:
    """One prediction per trace (repeat 0) for point metrics."""
    out: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.repeat == 0:
            out[prediction.trace_id] = prediction
    if not out:
        for prediction in predictions:
            out.setdefault(prediction.trace_id, prediction)
    return out


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


def compare(
    y: np.ndarray,
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    config: EvalConfig,
    baseline_name: str,
) -> dict[str, Any]:
    mcfg = config.section("metrics")
    resamples = int(mcfg.get("bootstrap_resamples", 10000))
    seed = int(mcfg.get("bootstrap_seed", 20260919))
    budget = float(config.section("gates").get("useful_signal_budget", 0.05))
    auprc_ci = metrics.paired_bootstrap(y, scores, baseline_scores, "auprc", resamples, seed)
    recall_ci = metrics.paired_bootstrap(
        y, scores, baseline_scores, "recall_at_budget", resamples, seed, budget=budget
    )
    return {
        "baseline": baseline_name,
        "resamples": resamples,
        "auprc_delta": auprc_ci.as_dict(),
        f"recall_delta_at_{budget:.0%}": recall_ci.as_dict(),
    }


def _useful_signal_note(stage_ok: bool, n_records: int, provider: str) -> str:
    if stage_ok:
        return "Reported from the scored run."
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

    deployment = {
        "precision_ok": summary["precision"] >= float(gates.get("deployment_precision", 0.90)),
        "recall_ok": summary["recall"] >= float(gates.get("deployment_recall", 0.80)),
        "ece_ok": summary["ece"] <= float(gates.get("deployment_max_ece", 0.10)),
        "error_rate_ok": summary["operational"]["error_rate"]
        < float(gates.get("deployment_max_api_error_rate", 0.01)),
        "parse_failures_ok": summary["operational"]["parse_failure_count"]
        <= int(gates.get("deployment_max_parse_failures", 0)),
    }
    deployment["passed"] = all(deployment.values())
    return {"useful_signal": useful, "deployment_candidate": deployment}


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

    if not blind_path.exists():
        return {
            "status": "not_performed",
            "n_records": len(key),
            "note": (
                f"{blind_path} not found. The blinded audit sheet is produced by "
                "`jev-eval generate`; the audit itself has not been carried out."
            ),
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
    return {
        "status": "performed",
        "n_records": len(key),
        "n_audited": len(filled),
        "coverage": len(filled) / len(key) if key else 0.0,
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
    a("")

    a("## Primary metrics (unsupported_success vs rest)")
    a("")
    a(
        _table(
            ["provider", "model", "AUPRC", "AUROC", "precision", "recall", "F1", "thr", "macro-F1"],
            [
                [
                    p,
                    results["runs"][p]["model_id"],
                    _fmt(results["runs"][p]["auprc"]),
                    _fmt(results["runs"][p]["auroc"]),
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
        a("")
        a("Until this is filled in, every label in this report rests on the generator's")
        a("construction rules alone and has not been independently checked by a human.")
    a("")

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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    latest: dict[str, tuple[Path, Any, list[Prediction]]] = {}
    for run_dir in discover_runs(runs_root):
        manifest, predictions = load_run(run_dir)
        if manifest.split != split:
            continue
        current = latest.get(manifest.provider)
        if current is None or manifest.created_utc >= current[1].created_utc:
            latest[manifest.provider] = (run_dir, manifest, predictions)

    if not latest:
        raise RuntimeError(f"no runs for split {split!r} under {runs_root}")

    summaries: dict[str, dict[str, Any]] = {}
    for provider, (run_dir, manifest, predictions) in latest.items():
        summary = evaluate_run(records, predictions, manifest.threshold, config)
        summary["model_id"] = manifest.model_id
        summary["run_id"] = manifest.run_id
        summary["run_dir"] = str(run_dir)
        summary["prior_shift"] = {
            f"{p:.0%}": metrics.prior_shift_precision(summary["tpr"], summary["fpr"], float(p))
            for p in config.section("metrics").get("production_prevalences", [0.01, 0.05, 0.10])
        }
        summaries[provider] = summary

    comparisons: dict[str, dict[str, Any]] = {}
    for provider, summary in summaries.items():
        against: dict[str, Any] = {}
        for baseline in (BASELINE, SECOND_BASELINE):
            if baseline == provider or baseline not in summaries:
                continue
            against[baseline] = compare(
                summary["_y"],
                summary["_scores"],
                summaries[baseline]["_scores"],
                config,
                baseline,
            )
        if against:
            comparisons[provider] = against

    gates = {
        provider: evaluate_gates(
            summary, comparisons.get(provider, {}), config, n_records_total, provider
        )
        for provider, summary in summaries.items()
    }

    plot_pr_curves(summaries, out_dir / "pr_curve.png")
    plot_reliability(summaries, out_dir / "reliability.png")
    plot_fault_heatmap(summaries, out_dir / "fault_heatmap.png")

    stage = (
        f"scored (>= {STAGE_MIN_RECORDS} records)"
        if n_records_total >= STAGE_MIN_RECORDS
        else f"stage-1 pipeline proof ({n_records_total} records)"
    )

    results: dict[str, Any] = {
        "split": split,
        "stage": stage,
        "n_records": n_records_total,
        "n_scored": next(iter(summaries.values()))["n_scored"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "splits_path": str(splits_path),
        "bootstrap_resamples": int(config.section("metrics").get("bootstrap_resamples", 10000)),
        "runs": {
            p: {k: v for k, v in s.items() if not k.startswith("_")} for p, s in summaries.items()
        },
        "comparisons": comparisons,
        "gates": gates,
        "audit": audit_section(audit_key, audit_blind),
        "plots": ["pr_curve.png", "reliability.png", "fault_heatmap.png"],
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
