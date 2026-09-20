"""Aggregation for the dashboard: with-Jev against without-Jev, honestly.

Everything the page shows is computed here from run artifacts on disk and the
dated price manifest. Two rules govern this module:

1. **Nothing raw reaches the browser.** A :class:`Prediction` carries
   ``raw_request`` and ``raw_response``. Those are redacted on disk, but they are
   still request and response bodies and they have no business in a web page, so
   :func:`public_row` builds the browser's view field by field rather than
   dumping the model.
2. **Nothing is smoothed.** The panel showing where Jev adds nothing is built by
   the same code path as the panel showing where it wins. A group cannot appear
   in one and be quietly dropped from the other.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import metrics
from ..caveats import build_caveats
from ..costs import PriceManifest, load_prices
from ..ingest.pipeline import load_provenance
from ..runner import (
    DatasetPaths,
    EvalConfig,
    load_run,
    load_splits,
    read_progress,
    select_split,
)
from ..schemas import POSITIVE_LABEL, DatasetProvenance, FaultType, Prediction, TraceRecord

FOOTER_NOTICE = (
    "Internal only. Performance figures are covered by TypeSafe's MCA and must not be "
    "published without written permission."
)

JEV_PROVIDER = "jev"
FREE_GUARD = "rules"
SECOND_FREE_BASELINE = "tfidf"

#: The strong free baseline: TF-IDF features with a gradient-boosted classifier,
#: the published recipe that beats every LLM judge in the literature it was
#: measured against. This is the bar. A paid provider that does not clear it has
#: not bought anything, and the page says so in as many words.
STRONG_FREE_BASELINE = "tfidf_gbm"

#: Every free lane, weakest first. Order matters only for display.
FREE_BASELINES = (FREE_GUARD, SECOND_FREE_BASELINE, STRONG_FREE_BASELINE)

#: The other paid lane: a general LLM judge shown the same InferenceView and
#: asked the same four questions at temperature 0. It is not a free baseline and
#: is never folded into the free union -- it is the arm that answers the separate
#: question of whether the *typed* model beats a general one, and what each costs.
GENERAL_JUDGE = "openai"
GENERAL_JUDGE_LABEL = "General LLM judge"

#: How each provider is presented: a display name, the *role* its colour
#: encodes, and the group it is read in. Colour is by role, never by rank --
#: nothing on this page is coloured by who won, and the two TF-IDF arms share
#: one hue in two steps because they are one role, not two ranks.
PROVIDER_META: dict[str, dict[str, Any]] = {
    STRONG_FREE_BASELINE: {
        "label": "TF-IDF + boosting",
        "short": "TF-IDF+GBM",
        "role": "trained",
        "group": "Trained on labels",
        "trains": True,
        "paid": False,
    },
    SECOND_FREE_BASELINE: {
        "label": "TF-IDF + logistic regression",
        "short": "TF-IDF",
        "role": "trained-alt",
        "group": "Trained on labels",
        "trains": True,
        "paid": False,
    },
    JEV_PROVIDER: {
        "label": "Jev, typed detector",
        "short": "Jev",
        "role": "typed",
        "group": "Zero-shot, no training data",
        "trains": False,
        "paid": True,
    },
    GENERAL_JUDGE: {
        "label": "General LLM judge",
        "short": "LLM judge",
        "role": "general",
        "group": "Zero-shot, no training data",
        "trains": False,
        "paid": True,
    },
    FREE_GUARD: {
        "label": "Hand-written rules",
        "short": "Rules",
        "role": "heuristic",
        "group": "Zero-shot, no training data",
        "trains": False,
        "paid": False,
    },
}

#: Reading order for every figure: strongest free classifier first, then the
#: paid arms, then the rule checker. Fixed rather than sorted by score, so a
#: chart does not silently reorder itself when a number moves.
PROVIDER_ORDER: tuple[str, ...] = (
    STRONG_FREE_BASELINE,
    SECOND_FREE_BASELINE,
    JEV_PROVIDER,
    GENERAL_JUDGE,
    FREE_GUARD,
)

#: The one positive fault no string or field comparison resolves. It is the
#: reason this evaluation exists, so the UI focuses on it -- and it is also the
#: smallest cell, so every figure about it is reported with its n attached.
FOCUS_FAULT = FaultType.semantic_target_mismatch.value

#: Its matched negative control. Same goal shape, same candidate listing, right
#: target. A provider that flags both at the same rate has separated nothing.
CONTROL_FAULT = FaultType.semantic_target_match.value


#: Every figure declares which table column holds which field, so a generic
#: reader -- the browser's table view, the PNG export, and the parity test --
#: can round-trip a chart back to its numbers without knowing the figure's kind.
#:
#: ``key`` is the provider id, carried beside the display label so an exported
#: CSV can be joined back to a run directory. A table that identified a series
#: only by its prose name would be unreadable by anything but a person.
FIELD_INDEX = {"series": 0, "key": 1, "x": 2, "y": 3, "lower": 4, "upper": 5}

#: Table columns before the per-figure extras. The first two are identity, the
#: next four are the point and its interval.
LEADING_COLUMNS = 2


def _fmt_value(value: float | None, digits: int = 3) -> str:
    """One formatter for every table cell, so a chart and its table round-trip."""
    return (
        "—"
        if value is None or (isinstance(value, float) and np.isnan(value))
        else f"{value:.{digits}f}"
    )


def build_figure(
    *,
    figure_id: str,
    kind: str,
    title: str,
    caption: str,
    value_label: str,
    x_label: str,
    series: Sequence[dict[str, Any]],
    domain: Sequence[float] | None = None,
    digits: int = 3,
    references: Sequence[dict[str, Any]] = (),
    markers: Sequence[dict[str, Any]] = (),
    extra_columns: Sequence[str] = (),
    table_note: str = "",
    x_scale: str = "linear",
) -> dict[str, Any]:
    """A chart and its table view, built from one list of points.

    Every figure on this page goes through here. The table is **derived** from
    the same ``series`` the chart draws, in the same order, with the same
    formatter -- so the two cannot drift, and `tests/test_dashboard_design.py`
    re-derives the rows independently and fails if they ever do.

    That matters beyond tidiness: identity on this page is never colour alone.
    A reader who cannot separate two hues, or who is reading a screenshot in
    grayscale, gets the same numbers from the table behind every chart.
    """
    rows: list[list[str]] = []
    for entry in series:
        for point in entry["points"]:
            row = [
                entry["label"],
                str(entry["key"]),
                str(point["x"]),
                _fmt_value(point.get("y"), digits),
                _fmt_value(point.get("lower"), digits),
                _fmt_value(point.get("upper"), digits),
            ]
            row.extend(str(point.get("extra", {}).get(name, "—")) for name in extra_columns)
            rows.append(row)
    return {
        "id": figure_id,
        "kind": kind,
        "title": title,
        "caption": caption,
        "value_label": value_label,
        "x_label": x_label,
        "x_scale": x_scale,
        "domain": list(domain) if domain else None,
        "digits": digits,
        "series": list(series),
        "references": list(references),
        "markers": list(markers),
        "table": {
            "columns": [
                "Series",
                "Key",
                x_label,
                value_label,
                "Lower 95%",
                "Upper 95%",
                *extra_columns,
            ],
            "leading_columns": LEADING_COLUMNS,
            "rows": rows,
            "field_index": dict(FIELD_INDEX),
            "note": table_note,
        },
    }


@dataclass(frozen=True)
class RunRef:
    """A run directory the dashboard can open, finished or still in flight."""

    run_id: str
    path: Path
    provider: str
    model_id: str
    split: str
    concurrency: int
    repeats: int
    threshold: float
    status: str
    n_expected: int
    n_written: int
    created_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "provider": self.provider,
            "model_id": self.model_id,
            "split": self.split,
            "concurrency": self.concurrency,
            "repeats": self.repeats,
            "threshold": self.threshold,
            "status": self.status,
            "n_expected": self.n_expected,
            "n_written": self.n_written,
            "created_utc": self.created_utc,
        }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def describe_run(run_dir: Path) -> RunRef | None:
    """Describe a run directory from whichever of its two headers exists.

    A finished run has ``manifest.json``. A run still in flight has only
    ``progress.json``. A run written before ``progress.json`` existed has only
    the manifest. All three are readable.
    """
    manifest_path = run_dir / "manifest.json"
    progress = read_progress(run_dir)

    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = str(progress.get("status")) if progress else "complete"
        n_written = int(progress["n_written"]) if progress else int(raw["n_predictions"])
        return RunRef(
            run_id=str(raw["run_id"]),
            path=run_dir,
            provider=str(raw["provider"]),
            model_id=str(raw["model_id"]),
            split=str(raw["split"]),
            concurrency=int(raw["concurrency"]),
            repeats=int(raw["repeats"]),
            threshold=float(raw["threshold"]),
            status=status,
            n_expected=int(raw["n_predictions"]),
            n_written=n_written,
            created_utc=str(raw["created_utc"]),
        )

    if progress is None:
        return None
    return RunRef(
        run_id=str(progress["run_id"]),
        path=run_dir,
        provider=str(progress["provider"]),
        model_id=str(progress["model_id"]),
        split=str(progress["split"]),
        concurrency=int(progress["concurrency"]),
        repeats=int(progress["repeats"]),
        threshold=float(progress["threshold"]),
        status=str(progress["status"]),
        n_expected=int(progress["n_expected"]),
        n_written=int(progress.get("n_written", _count_lines(run_dir / "predictions.jsonl"))),
        created_utc=str(progress["started_utc"]),
    )


def discover_dashboard_runs(runs_root: Path) -> list[RunRef]:
    """Every run under a root, in-flight ones included, newest first."""
    seen: dict[Path, RunRef] = {}
    for pattern in ("*/manifest.json", "*/progress.json"):
        for path in runs_root.glob(pattern):
            run_dir = path.parent
            if run_dir in seen:
                continue
            ref = describe_run(run_dir)
            if ref is not None:
                seen[run_dir] = ref
    return sorted(seen.values(), key=lambda r: (r.created_utc, r.run_id), reverse=True)


def read_predictions_from(path: Path, start_line: int = 0) -> tuple[list[Prediction], int]:
    """Read whole JSONL lines from ``start_line`` on.

    A partial final line is a writer mid-flush, not a corrupt file: it is left
    unread and the offset is not advanced past it, so the next poll picks it up
    complete.
    """
    if not path.exists():
        return [], start_line
    out: list[Prediction] = []
    line_number = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.endswith("\n"):
                break
            line_number += 1
            if line_number <= start_line:
                continue
            stripped = raw.strip()
            if not stripped:
                continue
            out.append(Prediction.model_validate(json.loads(stripped)))
    return out, start_line + len(out)


def public_row(prediction: Prediction, record: TraceRecord | None) -> dict[str, Any]:
    """The only shape of a prediction that is ever sent to the browser.

    Built field by field on purpose. ``raw_request`` and ``raw_response`` are
    never included, at any verbosity, for any view.
    """
    return {
        "trace_id": prediction.trace_id,
        "repeat": prediction.repeat,
        "provider": prediction.provider,
        "model_id": prediction.model_id,
        "score": prediction.primary_score,
        "decision": prediction.decision.value,
        "predicted_label": (
            prediction.predicted_label.value if prediction.predicted_label else None
        ),
        "needs_review": prediction.needs_review,
        "latency_ms": prediction.end_to_end_latency_ms,
        "input_tokens": prediction.usage.input_tokens if prediction.usage else 0,
        "cost_usd": prediction.cost_usd,
        "error": prediction.error,
        "label": record.label.value if record else None,
        "fault_type": record.fault_type.value if record else None,
        "domain": record.domain if record else None,
    }


#: Shown while :meth:`DashboardData.ensure_comparison` has not run yet. It is
#: deliberately not the "not computable" block: "still working" and "these runs
#: do not overlap" are different claims and must not render the same.
_PENDING_COMPARISON: dict[str, Any] = {
    "baseline": STRONG_FREE_BASELINE,
    "baseline_label": "TF-IDF + gradient boosting",
    "computable": False,
    "pending": True,
    "verdict": "pending",
    "verdict_text": (
        "Comparing against the strong free baseline. The paired bootstrap is still "
        "running; the verdict appears here when it finishes."
    ),
}


#: The Jev-against-general-judge panel before its bootstrap has finished. Kept
#: distinct from "not computable" for the same reason the strong-baseline one is.
_PENDING_JUDGE: dict[str, Any] = {
    "rival": GENERAL_JUDGE,
    "rival_label": GENERAL_JUDGE_LABEL,
    "computable": False,
    "pending": True,
    "verdict": "pending",
    "verdict_text": (
        "Comparing against the general LLM judge. The paired bootstrap is still "
        "running; the verdict appears here when it finishes."
    ),
}


def _rate(flagged: int, n: int) -> float | None:
    """A rate over nothing is None, never 0.0. Zero is a finding; empty is not."""
    return flagged / n if n else None


def _flag_counts(
    points: dict[str, Prediction], trace_ids: Sequence[str], threshold: float
) -> tuple[int, int]:
    """(flagged, scored) over the traces a provider has actually scored."""
    flagged = 0
    scored = 0
    for trace_id in trace_ids:
        prediction = points.get(trace_id)
        if prediction is None or prediction.primary_score is None:
            continue
        scored += 1
        flagged += int(prediction.primary_score >= threshold)
    return flagged, scored


def _lane(
    points: dict[str, Prediction], trace_ids: Sequence[str], threshold: float
) -> dict[str, Any]:
    flagged, scored = _flag_counts(points, trace_ids, threshold)
    return {"flagged": flagged, "scored": scored, "rate": _rate(flagged, scored)}


def _union_points(
    lanes: Sequence[tuple[dict[str, Prediction], float]], trace_ids: Sequence[str]
) -> dict[str, Any]:
    """The strongest thing the free tier can do: flag if *any* free guard flags.

    Carried so the headline cannot be accused of being measured against the
    weaker of the two free baselines.
    """
    flagged = 0
    scored = 0
    for trace_id in trace_ids:
        seen = False
        hit = False
        for points, threshold in lanes:
            prediction = points.get(trace_id)
            if prediction is None or prediction.primary_score is None:
                continue
            seen = True
            hit = hit or prediction.primary_score >= threshold
        if seen:
            scored += 1
            flagged += int(hit)
    return {"flagged": flagged, "scored": scored, "rate": _rate(flagged, scored)}


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _parse_utc(stamp: str) -> float | None:
    from datetime import datetime

    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except ValueError:
        return None


def audit_status(audit_blind: Path) -> dict[str, Any]:
    """Whether a human has checked any label yet. Reported, never assumed."""
    if not audit_blind.exists():
        return {
            "status": "sheet_missing",
            "filled": 0,
            "in_sample": 0,
            "text": (
                "No blinded audit sheet exists for this dataset. Every label is a "
                "construction label from the generator."
            ),
        }
    import csv

    filled = 0
    in_sample = 0
    with audit_blind.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("in_sample", "")).strip().upper() in {"TRUE", "1", "YES"}:
                in_sample += 1
            if str(row.get("auditor_label", "")).strip():
                filled += 1
    if filled == 0:
        return {
            "status": "not_performed",
            "filled": 0,
            "in_sample": in_sample,
            "text": (
                f"Labels are construction labels from the generator. The blinded human "
                f"audit is pending: {in_sample} traces are marked for audit and 0 have "
                f"been filled in. No figure on this page has been independently checked."
            ),
        }
    return {
        "status": "partial" if filled < in_sample else "performed",
        "filled": filled,
        "in_sample": in_sample,
        "text": (
            f"Blinded audit {filled} of {in_sample} sampled traces filled in. Figures "
            f"for unaudited cells still rest on construction labels."
        ),
    }


class DashboardData:
    """Reads run artifacts for one stage and builds the snapshot the page renders."""

    def __init__(
        self,
        *,
        config: EvalConfig,
        paths: DatasetPaths,
        records: Sequence[TraceRecord],
        runs_root: Path,
        prices: PriceManifest | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.runs_root = runs_root
        self.all_records = list(records)
        self.splits = load_splits(paths.splits)
        self.prices = prices if prices is not None else load_prices(config.raw["prices"])
        self._baseline_cache: dict[tuple[str, str], dict[str, Prediction]] = {}
        # A finished run's own predictions, and the things derived from them
        # that cost real time to compute. Both are properties of the run, not
        # of how far a replay has scrolled, so they are computed once.
        self._complete_cache: dict[str, dict[str, Prediction]] = {}
        self._comparison_cache: dict[str, dict[str, Any]] = {}
        self._judge_cache: dict[str, dict[str, Any]] = {}
        self._reference_cache: dict[str, dict[str, Any]] = {}
        # The learning curve is a property of the split, not of the selected
        # run, so it is cached by split and survives a run change.
        self._curve_cache: dict[str, dict[str, Any]] = {}
        # Per-provider bootstrap intervals: one 10,000-resample pass per
        # provider, a property of the finished runs, computed once per split.
        self._interval_cache: dict[str, dict[str, dict[str, Any]]] = {}
        # The provenance sidecar decides every honesty statement on the page.
        # A dataset written before provenance existed has none, and the caveat
        # builder falls back to the synthetic wording -- which is the safe
        # direction: it over-warns rather than under-warns.
        self.provenance: DatasetProvenance | None = load_provenance(paths.provenance)

    # -- lookups --------------------------------------------------------
    def runs(self) -> list[RunRef]:
        return discover_dashboard_runs(self.runs_root)

    def find_run(self, run_id_or_path: str | None) -> RunRef | None:
        """Resolve a run by id, by directory path, or default to the newest Jev run."""
        available = self.runs()
        if run_id_or_path:
            candidate = Path(run_id_or_path)
            for ref in available:
                if ref.run_id == run_id_or_path or ref.path == candidate:
                    return ref
            if (candidate / "predictions.jsonl").exists():
                return describe_run(candidate)
            return None
        for ref in available:
            if ref.provider == JEV_PROVIDER:
                return ref
        return available[0] if available else None

    def split_records(self, split: str) -> list[TraceRecord]:
        return list(select_split(self.all_records, self.splits, split))

    def baseline_points(self, provider: str, split: str) -> dict[str, Prediction]:
        """Point predictions for a free baseline on this split, newest run wins."""
        key = (provider, split)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        chosen: dict[str, Prediction] = {}
        for ref in self.runs():
            if ref.provider != provider or ref.split != split or ref.status != "complete":
                continue
            try:
                _, predictions = load_run(ref.path)
            except (OSError, ValueError):
                continue
            chosen = metrics.point_predictions(predictions)
            break
        self._baseline_cache[key] = chosen
        return chosen

    def complete_points(self, ref: RunRef) -> dict[str, Prediction]:
        """Every point prediction the selected run wrote, read from disk.

        The streamed set is whatever has arrived so far; this is the run. The
        distinction matters because the expensive statistics below are
        properties of the finished run and must not move as a replay scrolls.
        """
        if ref.run_id in self._complete_cache:
            return self._complete_cache[ref.run_id]
        try:
            predictions, _ = read_predictions_from(ref.path / "predictions.jsonl")
        except (OSError, ValueError):
            predictions = []
        points = metrics.point_predictions(predictions)
        self._complete_cache[ref.run_id] = points
        return points

    def reference_metrics(self, ref: RunRef) -> dict[str, Any]:
        """Per-provider metrics from the metrics module, for this split.

        This is the same call `jev-eval report` makes, on the same records and
        the same threshold, so the page cannot quote a different recall or
        AUPRC than the report does for the identical run. The page reads these
        rather than deriving its own.
        """
        if ref.run_id in self._reference_cache:
            return self._reference_cache[ref.run_id]
        from ..report import evaluate_run

        split_records = self.split_records(ref.split)
        providers: dict[str, list[Prediction]] = {}
        for run in self.runs():
            if run.split != ref.split or run.status != "complete":
                continue
            if run.provider in providers:
                continue
            try:
                _, predictions = load_run(run.path)
            except (OSError, ValueError):
                continue
            providers[run.provider] = list(predictions)

        out: dict[str, Any] = {}
        for provider, predictions in providers.items():
            threshold = (
                ref.threshold if provider == ref.provider else self.config.threshold_for(provider)
            )
            try:
                summary = evaluate_run(split_records, predictions, threshold, self.config)
            except (RuntimeError, ValueError) as exc:
                # A provider whose predictions cannot be scored is reported as
                # unscorable, never silently omitted or shown as zero.
                out[provider] = {"scorable": False, "reason": str(exc)}
                continue
            out[provider] = {
                "scorable": True,
                "threshold": threshold,
                "n_scored": summary["n_scored"],
                "caught": summary["tp"],
                "positives": summary["tp"] + summary["fn"],
                "recall": summary["recall"],
                "precision": summary["precision"],
                "auprc": summary["auprc"],
                "auroc": summary["auroc"],
                "ece": summary["ece"],
            }
        self._reference_cache[ref.run_id] = out
        return out

    def learning_curve(self, ref: RunRef) -> dict[str, Any]:
        """The low-label curve for this split, or a panel that says why not.

        The classifier curves come off disk; the two flat zero-shot lines are
        taken from :meth:`reference_metrics`, which is the metrics module's own
        answer for the runs on this split. So the flat line on this panel is the
        same AUPRC the report prints for that arm -- the page never derives a
        second one.
        """
        from .. import learning_curve as lc

        key = f"curve:{ref.split}"
        cached = self._curve_cache.get(key)
        if cached is not None:
            return cached

        reference = self.reference_metrics(ref)
        zero_shot = {
            provider: {
                "auprc": entry.get("auprc"),
                "recall": entry.get("recall"),
                "model_id": (entry.get("model_id") or provider),
                "trains": False,
            }
            for provider in lc.ZERO_SHOT_PROVIDERS
            if (entry := reference.get(provider)) and entry.get("scorable")
        }
        # `reference_metrics` does not carry model ids, so they are filled from
        # the run list rather than left as the provider name.
        for run in self.runs():
            if run.provider in zero_shot and run.split == ref.split:
                zero_shot[run.provider]["model_id"] = run.model_id

        dataset_sha = self.splits.dataset_sha256
        stored = lc.load_curve(self.runs_root, dataset_sha, ref.split)
        if stored is None:
            block: dict[str, Any] = {
                "available": False,
                "reason": (
                    f"No learning curve has been swept for split {ref.split!r} on this "
                    "dataset. Run `jev-eval learning-curve` -- it is free and offline."
                ),
                "curves": {},
                "zero_shot": zero_shot,
                "crossovers": {},
                "caveats": {"calibration": lc.CALIBRATION_CAVEAT, "band": lc.BAND_CAVEAT},
            }
        else:
            rebuilt = lc.rebuild(stored, zero_shot)
            block = {
                "available": True,
                "reason": "",
                "curve_id": rebuilt.get("curve_id", ""),
                "curve_dir": rebuilt.get("curve_dir", ""),
                "split": rebuilt.get("split", ref.split),
                "sizes": rebuilt.get("sizes", []),
                "seeds": rebuilt.get("seeds", []),
                "paid_calls": rebuilt.get("paid_calls", 0),
                "curves": rebuilt["curves"],
                "zero_shot": rebuilt["zero_shot"],
                "crossovers": rebuilt["crossovers"],
                "caveats": rebuilt.get("caveats", {}),
            }
        self._curve_cache[key] = block
        return block

    def finding(self, ref: RunRef) -> dict[str, Any]:
        """The one-sentence finding, assembled clause by clause from the runs.

        Nothing here is a written sentence with numbers dropped into it. Each
        clause is emitted only if the interval that supports it is on the right
        side of zero, so a result that moves changes the sentence rather than
        leaving a stale claim on the page. Two rules are enforced structurally:

        * the trained-classifier clause is **last and unconditional** whenever a
          trained arm beats the typed one, so the sentence can never end on the
          paid arm's win;
        * no clause may say the typed arm is the best detector. The strongest
          thing it can say is that it beats the other zero-shot arm.
        """
        reference = self.reference_metrics(ref)
        comparison = self._comparison_cache.get(ref.run_id) or {}
        judge = self._judge_cache.get(ref.run_id) or {}
        curve = self.learning_curve(ref)

        clauses: list[str] = []
        claims: list[dict[str, Any]] = []

        judge_ci = judge.get("ci") or {}
        beats_judge = bool(judge.get("computable")) and (judge_ci.get("lower") or 0.0) > 0.0
        if beats_judge:
            clauses.append(
                "Jev detects false success better than a general LLM judge on the same traces"
            )
            claims.append(
                {
                    "text": "Jev beats the general LLM judge on AUPRC",
                    "detail": judge.get("verdict_text", ""),
                    "stance": "for",
                }
            )

        cost = self._judge_cost_block(ref)
        arms = cost.get("arms") or {}
        jev_cost = (arms.get(JEV_PROVIDER) or {}).get("cost_per_1000_traces_usd")
        judge_cost = (arms.get(GENERAL_JUDGE) or {}).get("cost_per_1000_traces_usd")
        if jev_cost is not None and judge_cost is not None and jev_cost < judge_cost:
            clauses.append("and costs less per trace")
            claims.append(
                {
                    "text": "Jev costs less per trace than the judge",
                    "detail": f"${jev_cost:.4f} against ${judge_cost:.4f} per 1000 traces.",
                    "stance": "for",
                }
            )
        elif judge_cost is None:
            claims.append(
                {
                    "text": "The two arms' costs cannot be compared",
                    "detail": (
                        "The general judge's model has no verified rate in the dated price "
                        "manifest, so its cost is unavailable rather than zero. No cost "
                        "comparison is claimed."
                    ),
                    "stance": "caveat",
                }
            )

        # The clause the page exists to keep in view.
        strong = reference.get(STRONG_FREE_BASELINE) or {}
        typed = reference.get(JEV_PROVIDER) or {}
        baseline_ci = comparison.get("ci") or {}
        beaten_by_trained = bool(comparison.get("computable")) and (
            baseline_ci.get("upper") is not None and baseline_ci["upper"] < 0.0
        )
        if beaten_by_trained:
            clauses.append("— and a TF-IDF classifier trained on labels beats them both")
            claims.append(
                {
                    "text": "A trained TF-IDF classifier beats every paid arm",
                    "detail": comparison.get("verdict_text", ""),
                    "stance": "against",
                }
            )

        sentence = (
            (", ".join(clauses[:1]) + (" " + " ".join(clauses[1:]) if len(clauses) > 1 else ""))
            if clauses
            else "No comparison on this split is established yet."
        )
        if clauses:
            sentence = f"On real AppWorld traces, {sentence}."

        sub = ""
        crossings = (curve.get("crossovers") or {}).get(STRONG_FREE_BASELINE, {})
        against_typed = crossings.get(JEV_PROVIDER) or {}
        mark = against_typed.get("mean")
        if mark and against_typed.get("reference_leads_anywhere"):
            sub = (
                f"Jev leads until you have about {mark['n_train']} labelled examples; "
                "past that, the classifier is ahead at every budget tested."
            )
        elif mark and not against_typed.get("reference_leads_anywhere"):
            sub = (
                "The classifier is ahead at every label budget tested, including the "
                "smallest. There is no low-label window where Jev leads."
            )
        elif curve.get("available"):
            sub = (
                "Across every label budget tested the classifier never overtakes Jev, so "
                "the low-label advantage does not close on this corpus."
            )

        return {
            "sentence": sentence,
            "sub": sub,
            "claims": claims,
            "figures": {
                "trained_auprc": strong.get("auprc"),
                "typed_auprc": typed.get("auprc"),
                "judge_auprc": (reference.get(GENERAL_JUDGE) or {}).get("auprc"),
            },
            "pending": not comparison or comparison.get("verdict") == "pending",
        }

    def provider_intervals(self, ref: RunRef) -> dict[str, dict[str, Any]]:
        """Each provider's own 95% bootstrap interval on AUPRC and AUROC.

        Delegated to :func:`report.provider_intervals` so the error bars on this
        page are the same numbers `jev-eval report` prints, from the same
        resample count and the same seed. Cached per split: it is a
        10,000-resample bootstrap per provider and a property of the finished
        runs, not of how far a replay has scrolled.
        """
        key = f"intervals:{ref.split}"
        cached = self._interval_cache.get(key)
        if cached is not None:
            return cached
        from ..report import evaluate_run, provider_intervals

        split_records = self.split_records(ref.split)
        summaries: dict[str, dict[str, Any]] = {}
        for run in self.runs():
            if run.split != ref.split or run.status != "complete" or run.provider in summaries:
                continue
            try:
                _, predictions = load_run(run.path)
            except (OSError, ValueError):
                continue
            threshold = self.config.threshold_for(run.provider)
            try:
                summaries[run.provider] = evaluate_run(
                    split_records, predictions, threshold, self.config
                )
            except (RuntimeError, ValueError):
                continue
        out = provider_intervals(summaries, self.config) if summaries else {}
        self._interval_cache[key] = out
        return out

    def figures(self, ref: RunRef) -> dict[str, Any]:
        """Every chart on the page, each with the table it round-trips to."""
        reference = self.reference_metrics(ref)
        intervals = self.provider_intervals(ref)
        return {
            "headline_auprc": self._headline_figure(reference, intervals),
            "learning_curve": self._learning_curve_figure(ref),
            "per_fault": self._per_fault_figure(ref),
            "cost": self._cost_figure(ref),
            "latency": self._latency_figure(ref),
        }

    def _headline_figure(
        self, reference: dict[str, Any], intervals: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """AUPRC per provider with its own interval, grouped by whether it trains.

        Ordered by the fixed reading order rather than by score, and grouped by
        training regime rather than by result, because the thing worth seeing is
        that the group which trains is above the group which does not -- not a
        ranking with a winner at the top.
        """
        series: list[dict[str, Any]] = []
        for provider in PROVIDER_ORDER:
            entry = reference.get(provider)
            if not entry or not entry.get("scorable"):
                continue
            meta = PROVIDER_META[provider]
            interval = (intervals.get(provider) or {}).get("auprc") or {}
            series.append(
                {
                    "key": provider,
                    "label": meta["label"],
                    "short": meta["short"],
                    "role": meta["role"],
                    "group": meta["group"],
                    "trains": meta["trains"],
                    "paid": meta["paid"],
                    "points": [
                        {
                            "x": meta["short"],
                            "y": entry.get("auprc"),
                            "lower": interval.get("lower"),
                            "upper": interval.get("upper"),
                            "extra": {
                                "n scored": entry.get("n_scored"),
                                "Recall": _fmt_value(entry.get("recall")),
                                "Precision": _fmt_value(entry.get("precision")),
                            },
                        }
                    ],
                }
            )
        resamples = next(iter(intervals.values()), {}).get("resamples", 0) if intervals else 0
        return build_figure(
            figure_id="fig-headline-auprc",
            kind="interval-dots",
            title="Detection quality, with its uncertainty",
            caption=(
                "Area under the precision-recall curve for unsupported_success against "
                "everything else, on the frozen test split. Bars are 95% percentile "
                f"bootstrap intervals over resampled traces, {resamples:,} resamples. "
                "Providers are grouped by whether they train on labels, and ordered by a "
                "fixed reading order rather than by score."
            ),
            value_label="AUPRC",
            x_label="Detector",
            series=series,
            domain=[0.0, 1.0],
            extra_columns=["n scored", "Recall", "Precision"],
            table_note=(
                "Recall and precision are at each provider's frozen threshold of 0.5. "
                "They are counts at one operating point; AUPRC is threshold-free."
            ),
        )

    def _learning_curve_figure(self, ref: RunRef) -> dict[str, Any]:
        """The label-budget sweep, as a banded line against two flat references."""
        curve = self.learning_curve(ref)
        series: list[dict[str, Any]] = []
        for provider, rows in sorted((curve.get("curves") or {}).items()):
            meta = PROVIDER_META.get(provider, {})
            series.append(
                {
                    "key": provider,
                    "label": meta.get("label", provider),
                    "short": meta.get("short", provider),
                    "role": meta.get("role", "trained"),
                    "group": "Trained on labels",
                    "trains": True,
                    "paid": False,
                    "points": [
                        {
                            "x": row["n_train"],
                            "y": row["auprc"]["mean"],
                            "lower": row["auprc"]["lower"],
                            "upper": row["auprc"]["upper"],
                            "extra": {
                                "Seeds": row["auprc"]["n"],
                                "Tasks": f"{row['n_train_tasks_mean']:.1f}",
                            },
                        }
                        for row in rows
                    ],
                }
            )
        references = [
            {
                "key": provider,
                "label": PROVIDER_META.get(provider, {}).get("label", provider),
                "short": PROVIDER_META.get(provider, {}).get("short", provider),
                "role": PROVIDER_META.get(provider, {}).get("role", "typed"),
                "value": arm.get("auprc"),
            }
            for provider, arm in sorted((curve.get("zero_shot") or {}).items())
            if arm.get("auprc") is not None
        ]
        by_budget: dict[int, list[str]] = {}
        for provider, refs in sorted((curve.get("crossovers") or {}).items()):
            for name, result in sorted(refs.items()):
                mark = result.get("mean")
                if mark:
                    short = PROVIDER_META.get(provider, {}).get("short", provider)
                    against = PROVIDER_META.get(name, {}).get("short", name)
                    by_budget.setdefault(int(mark["n_train"]), []).append(
                        f"{short} passes {against}"
                    )
        markers = [
            {"x": budget, "label": f"{budget} labels", "lines": lines}
            for budget, lines in sorted(by_budget.items())
        ]
        return {
            **build_figure(
                figure_id="fig-learning-curve",
                kind="curve-band",
                title="How many labels the classifier needs",
                caption=(
                    "Classifier AUPRC against the number of labelled training examples, "
                    "drawn task-disjointly and stratified, five seeds per budget. The band "
                    "is a Student-t 95% interval over seeds. The dashed rules are the two "
                    "zero-shot arms: they do not train, so they are flat, and they are "
                    "quoted from their recorded runs rather than re-run."
                ),
                value_label="AUPRC",
                x_label="Labelled training examples",
                series=series,
                references=references,
                markers=markers,
                x_scale="log",
                extra_columns=["Seeds", "Tasks"],
                table_note=(curve.get("caveats") or {}).get("calibration", ""),
            ),
            "available": curve.get("available", False),
            "reason": curve.get("reason", ""),
            "crossovers": curve.get("crossovers", {}),
            "caveats": curve.get("caveats", {}),
            "paid_calls": curve.get("paid_calls", 0),
        }

    def _per_fault_figure(self, ref: RunRef) -> dict[str, Any]:
        """Detection rate per fault group, one small panel per group.

        Small multiples rather than one crowded chart: every panel shares the
        0-100% scale, so a reader compares across groups by position alone. Each
        bar carries a Wilson interval, because a rate on 56 traces and a rate on
        374 are not the same claim.
        """
        split_records = self.split_records(ref.split)
        positives: dict[str, list[str]] = {}
        for record in split_records:
            if record.label is POSITIVE_LABEL:
                positives.setdefault(record.fault_type.value, []).append(record.trace_id)

        panels = [
            {"key": fault, "label": fault.replace("real_", "").replace("_", " "), "n": len(ids)}
            for fault, ids in sorted(positives.items())
        ]
        series: list[dict[str, Any]] = []
        for provider in PROVIDER_ORDER:
            points_by_id = self.baseline_points(provider, ref.split)
            if provider == ref.provider:
                points_by_id = self.complete_points(ref)
            if not points_by_id:
                continue
            threshold = self.config.threshold_for(provider)
            meta = PROVIDER_META[provider]
            points: list[dict[str, Any]] = []
            for fault, ids in sorted(positives.items()):
                lane = _lane(points_by_id, ids, threshold)
                scored = int(lane["scored"])
                flagged = int(lane["flagged"])
                interval = metrics.wilson_interval(flagged, scored)
                points.append(
                    {
                        "x": fault.replace("real_", "").replace("_", " "),
                        "panel": fault,
                        "y": None if scored == 0 else interval.point,
                        "lower": None if scored == 0 else interval.lower,
                        "upper": None if scored == 0 else interval.upper,
                        "extra": {"Caught": f"{flagged}/{scored}" if scored else "not scored"},
                    }
                )
            series.append(
                {
                    "key": provider,
                    "label": meta["label"],
                    "short": meta["short"],
                    "role": meta["role"],
                    "group": meta["group"],
                    "trains": meta["trains"],
                    "paid": meta["paid"],
                    "points": points,
                }
            )
        return {
            **build_figure(
                figure_id="fig-per-fault",
                kind="small-multiples",
                title="Detection by fault group",
                caption=(
                    "Share of the false successes in each group that each detector flags at "
                    "its frozen threshold. One panel per group, all on the same 0-100% "
                    "scale. Bars are 95% Wilson intervals; a group with few traces gets a "
                    "wide one, which is the point. A missing bar means that lane did not "
                    "score the group, which is not a zero."
                ),
                value_label="Detection rate",
                x_label="Fault group",
                series=series,
                domain=[0.0, 1.0],
                extra_columns=["Caught"],
            ),
            "panels": panels,
        }

    def _cost_figure(self, ref: RunRef) -> dict[str, Any]:
        """What one pass over 1000 traces costs, per paid arm.

        Unavailable is rendered as unavailable. A model with no verified rate in
        the dated price manifest gets a null and a reason, never a zero and
        never a guess -- a reader cannot tell "free" from "nobody priced it" if
        both come out as 0.00.
        """
        judge = self._judge_cost_block(ref)
        fx = float(self.config.raw["fx_rate_gbp_usd"])
        series: list[dict[str, Any]] = []
        for provider, arm in sorted((judge.get("arms") or {}).items()):
            meta = PROVIDER_META.get(provider)
            if meta is None:
                continue
            usd = arm.get("cost_per_1000_traces_usd")
            series.append(
                {
                    "key": provider,
                    "label": meta["label"],
                    "short": meta["short"],
                    "role": meta["role"],
                    "group": meta["group"],
                    "trains": meta["trains"],
                    "paid": True,
                    "model_id": arm.get("model_id"),
                    "available": bool(arm.get("cost_available")),
                    "unavailable_reason": (
                        ""
                        if arm.get("cost_available")
                        else "no verified rate in the dated price manifest"
                    ),
                    "points": [
                        {
                            "x": meta["short"],
                            "y": usd,
                            "lower": usd,
                            "upper": usd,
                            "extra": {
                                "GBP": "—" if usd is None else f"£{usd * fx:.4f}",
                                "Input tokens": f"{int(arm.get('input_tokens') or 0):,}",
                                "Output tokens": f"{int(arm.get('output_tokens') or 0):,}",
                                "Model": arm.get("model_id") or "—",
                            },
                        }
                    ],
                }
            )
        # The free lanes, stated rather than omitted. Their marginal cost is
        # zero on this manifest, which is a fact about metered calls and not a
        # claim that they are free to build.
        for provider in (STRONG_FREE_BASELINE, SECOND_FREE_BASELINE, FREE_GUARD):
            meta = PROVIDER_META[provider]
            series.append(
                {
                    "key": provider,
                    "label": meta["label"],
                    "short": meta["short"],
                    "role": meta["role"],
                    "group": meta["group"],
                    "trains": meta["trains"],
                    "paid": False,
                    "available": True,
                    "unavailable_reason": "",
                    "points": [
                        {
                            "x": meta["short"],
                            "y": 0.0,
                            "lower": 0.0,
                            "upper": 0.0,
                            "extra": {
                                "GBP": "£0.0000",
                                "Input tokens": "0",
                                "Output tokens": "0",
                                "Model": "local",
                            },
                        }
                    ],
                }
            )
        return {
            **build_figure(
                figure_id="fig-cost",
                kind="interval-dots",
                title="Cost of one pass over 1000 traces",
                caption=(
                    "Measured from the tokens each API actually returned, priced through "
                    f"the dated manifest {self.prices.manifest_date}. GBP at a fixed, "
                    f"operator-supplied rate of {fx} recorded in the config, never fetched. "
                    "An arm whose model has no verified rate is shown as unavailable."
                ),
                value_label="USD per 1000 traces",
                x_label="Detector",
                series=series,
                digits=4,
                extra_columns=["GBP", "Input tokens", "Output tokens", "Model"],
                table_note=(
                    "A recorded cost, not a projection: the mean cost of one request times "
                    "1000. The local lanes make no metered call, so their marginal cost is "
                    "zero on this manifest. That is not a claim that they are free to build."
                ),
            ),
            "fx_rate_gbp_usd": fx,
            "fx_rate_date": str(self.config.raw["fx_rate_date"]),
            "prices_path": self.prices.path,
            "prices_date": self.prices.manifest_date,
        }

    def _latency_figure(self, ref: RunRef) -> dict[str, Any]:
        """Measured end-to-end latency percentiles, per concurrency arm.

        A separate chart from cost on purpose: milliseconds and dollars are
        different measures, and a second y-axis would invite a comparison
        neither number supports.
        """
        speed = self._speed_block(ref, list(self.complete_points(ref).values()))
        arms = [arm for arm in speed.get("arms", []) if arm.get("n")]
        series: list[dict[str, Any]] = []
        # Every line here measures the SAME arm, so every line wears that arm's
        # role colour. A percentile is a rank, and this page does not colour by
        # rank: p50 and p99 are told apart by their own label and by the dash
        # pattern, never by hue. Borrowing the trained/typed/general hues for
        # them would make one blue mean two different things across figures.
        role = PROVIDER_META.get(ref.provider, PROVIDER_META[JEV_PROVIDER])["role"]
        for label, key, dash in (
            ("p50", "p50_ms", ""),
            ("p95", "p95_ms", "6 3"),
            ("p99", "p99_ms", "2 4"),
        ):
            series.append(
                {
                    "key": label,
                    "label": f"{label} latency",
                    "short": label,
                    "role": role,
                    "dash": dash,
                    "group": "Measured percentiles",
                    "trains": False,
                    "paid": True,
                    "points": [
                        {
                            "x": arm["concurrency"],
                            "y": arm[key],
                            "lower": None,
                            "upper": None,
                            "extra": {
                                "n": arm["n"],
                                "req/s": _fmt_value(arm.get("throughput_per_s"), 2),
                            },
                        }
                        for arm in arms
                    ],
                }
            )
        return build_figure(
            figure_id="fig-latency",
            kind="lines",
            title="Latency of the typed arm, by concurrency",
            caption=(
                "End-to-end wall-clock latency per request. These are measured order "
                "statistics of the run, not estimates of a population, so they carry no "
                "interval: there is nothing here to be uncertain about beyond the run "
                "itself."
            ),
            value_label="Milliseconds",
            x_label="Concurrency",
            series=series,
            extra_columns=["n", "req/s"],
            table_note=speed.get("note", ""),
        )

    def sibling_jev_runs(self, split: str) -> list[RunRef]:
        """Every Jev run on this split: the concurrency sweep, arm by arm."""
        return sorted(
            (r for r in self.runs() if r.provider == JEV_PROVIDER and r.split == split),
            key=lambda r: (r.concurrency, r.created_utc),
        )

    # -- the snapshot ---------------------------------------------------
    def snapshot(self, ref: RunRef, predictions: Sequence[Prediction]) -> dict[str, Any]:
        split_records = self.split_records(ref.split)
        jev_points = metrics.point_predictions(predictions)
        threshold = ref.threshold

        rules_points = self.baseline_points(FREE_GUARD, ref.split)
        tfidf_points = self.baseline_points(SECOND_FREE_BASELINE, ref.split)
        strong_points = self.baseline_points(STRONG_FREE_BASELINE, ref.split)
        # The general judge is read the same way as any other lane, but it is
        # deliberately NOT part of `free_lanes`: it costs money, so folding it
        # into the free union would credit the free tier with a paid result.
        judge_points = self.baseline_points(GENERAL_JUDGE, ref.split)
        rules_threshold = self.config.threshold_for(FREE_GUARD)
        tfidf_threshold = self.config.threshold_for(SECOND_FREE_BASELINE)
        strong_threshold = self.config.threshold_for(STRONG_FREE_BASELINE)
        judge_threshold = self.config.threshold_for(GENERAL_JUDGE)
        free_lanes = [
            (rules_points, rules_threshold),
            (tfidf_points, tfidf_threshold),
            (strong_points, strong_threshold),
        ]

        positives = [r for r in split_records if r.label is POSITIVE_LABEL]
        positive_ids = [r.trace_id for r in positives]

        jev_positive = _lane(jev_points, positive_ids, threshold)
        rules_positive = _lane(rules_points, positive_ids, rules_threshold)
        tfidf_positive = _lane(tfidf_points, positive_ids, tfidf_threshold)
        strong_positive = _lane(strong_points, positive_ids, strong_threshold)
        judge_positive = _lane(judge_points, positive_ids, judge_threshold)
        union_positive = _union_points(free_lanes, positive_ids)

        # The headline: positives Jev flags that the *strong* free baseline
        # does not. Measuring against the weak rule checker would flatter Jev,
        # so the strong baseline is the one the headline is stated against and
        # the weaker lanes are kept beside it rather than instead of it.
        missed_by_strong = [
            trace_id
            for trace_id in positive_ids
            if (p := strong_points.get(trace_id)) is not None
            and p.primary_score is not None
            and p.primary_score < strong_threshold
        ]
        missed_by_rules = [
            trace_id
            for trace_id in positive_ids
            if (p := rules_points.get(trace_id)) is not None
            and p.primary_score is not None
            and p.primary_score < rules_threshold
        ]
        missed_by_union = [
            trace_id
            for trace_id in positive_ids
            if all(
                (p := lane.get(trace_id)) is None
                or p.primary_score is None
                or p.primary_score < lane_threshold
                for lane, lane_threshold in free_lanes
            )
        ]
        jev_only_vs_rules = _lane(jev_points, missed_by_rules, threshold)
        jev_only_vs_strong = _lane(jev_points, missed_by_strong, threshold)
        jev_only_vs_union = _lane(jev_points, missed_by_union, threshold)

        detection = self._detection_groups(
            split_records,
            jev_points,
            rules_points,
            tfidf_points,
            strong_points,
            judge_points,
            free_lanes,
            threshold,
            rules_threshold,
            tfidf_threshold,
            strong_threshold,
            judge_threshold,
        )
        # The comparison is a property of the finished run, not of how much of
        # it has streamed in. Computing it from the partial set made every
        # snapshot pay for two 10,000-resample paired bootstraps -- about 30
        # seconds each -- so the stream never got past its first, empty
        # snapshot and the page showed Jev catching nothing while the
        # baselines, read whole from disk, looked fine. It is computed once
        # from the complete run and cached.
        comparison = self._comparison_cache.get(ref.run_id) or _PENDING_COMPARISON
        judge_comparison = self._judge_cache.get(ref.run_id) or _PENDING_JUDGE
        cost = self._cost_block(ref, predictions, split_records)
        return {
            "generated_utc": _now(),
            "footer": FOOTER_NOTICE,
            "run": {
                **ref.as_dict(),
                # The selected run drives the "with" lane. If it is not a Jev run
                # the page must not label it as one, so the flag is carried here
                # rather than inferred in the browser.
                "is_jev_run": ref.provider == JEV_PROVIDER,
                # The loaded record count, not the config's declared one: a real
                # corpus declares no count in eval.yaml and printed "0 records".
                "records": len(self.all_records),
                "runs_root": str(self.runs_root),
                "prices_path": self.prices.path,
                "prices_date": self.prices.manifest_date,
            },
            "kpi": {
                "traces_total": len(split_records),
                "traces_evaluated": len({p.trace_id for p in predictions}),
                "predictions_written": len(predictions),
                "predictions_expected": ref.n_expected,
                "positives_total": len(positives),
                "caught_with_jev": jev_positive["flagged"],
                "caught_with_jev_scored": jev_positive["scored"],
                "caught_without_jev": rules_positive["flagged"],
                "caught_without_jev_strong": strong_positive["flagged"],
                "caught_without_jev_union": union_positive["flagged"],
                "caught_by_general_judge": judge_positive["flagged"],
                "caught_by_general_judge_scored": judge_positive["scored"],
                "headline_jev_only_vs_free_guard": jev_only_vs_rules["flagged"],
                "headline_jev_only_vs_strong": jev_only_vs_strong["flagged"],
                "headline_jev_only_vs_free_union": jev_only_vs_union["flagged"],
                "free_guard_misses": len(missed_by_rules),
                "strong_baseline_misses": len(missed_by_strong),
                "free_union_misses": len(missed_by_union),
                "spend_usd": cost["spend_usd"],
                "spend_gbp": cost["spend_gbp"],
            },
            "lanes": {
                "jev": jev_positive,
                "rules": rules_positive,
                "tfidf": tfidf_positive,
                "tfidf_gbm": strong_positive,
                "free_union": union_positive,
                GENERAL_JUDGE: judge_positive,
            },
            "comparison": comparison,
            # The second headline: typed model against general judge. A separate
            # block from `comparison` because it answers a separate question,
            # and it renders at the same size whichever way it lands.
            "general_judge": judge_comparison,
            # How many labels the free classifier needs before it overtakes the
            # zero-shot arms. The one question the full-size tables cannot
            # answer, and the only place a zero-shot judge can still be ahead.
            "learning_curve": self.learning_curve(ref),
            # Every chart, each carrying the table it round-trips to. The page
            # renders from these and derives nothing of its own.
            "figures": self.figures(ref),
            # The one-sentence finding, assembled from the intervals rather than
            # written down. See DashboardData.finding.
            "finding": self.finding(ref),
            "providers": {
                key: {**meta, "order": PROVIDER_ORDER.index(key)}
                for key, meta in PROVIDER_META.items()
            },
            "detection": detection["groups"],
            "no_added_value": detection["no_added_value"],
            "per_fault": detection["per_fault"],
            "cost": cost,
            "speed": self._speed_block(ref, predictions),
            "errors": {
                "count": sum(1 for p in predictions if p.error is not None),
                "parse_failures": sum(
                    1 for p in predictions if p.error and "ParseFailure" in p.error
                ),
                "retries": sum(max(0, len(p.attempts) - 1) for p in predictions),
                "recent": [
                    {"trace_id": p.trace_id, "error": p.error}
                    for p in predictions
                    if p.error is not None
                ][-8:],
            },
            "honesty": self._honesty_block(split_records, ref, jev_points, threshold),
            "caveats": self._caveats_block(
                ref, comparison, split_records, self.complete_points(ref)
            ),
            "dataset": {
                "id": self.paths.dataset_id,
                "kind": self.paths.kind,
                "provenance": (
                    self.provenance.model_dump(mode="json") if self.provenance else None
                ),
            },
            # The metrics module's own numbers for every provider with a
            # complete run on this split. The page reads its headline recall,
            # AUPRC and caught counts from here so they match `jev-eval
            # report` exactly; the streamed lanes above are the progressive
            # view and are only equal to these once a replay has finished.
            "reference": self.reference_metrics(ref),
            "external_reference": self.config.raw.get("external_reference") or {},
            "baseline_availability": {
                "rules": len(rules_points),
                "tfidf": len(tfidf_points),
                "tfidf_gbm": len(strong_points),
                GENERAL_JUDGE: len(judge_points),
            },
        }

    # -- panels ---------------------------------------------------------
    def ensure_comparison(self, ref: RunRef) -> dict[str, Any]:
        """Compute and cache the paired-bootstrap comparison for this run.

        Separated from :meth:`snapshot` because it is the one genuinely slow
        thing on the page: two 10,000-resample paired bootstraps, about thirty
        seconds on a 702-trace split. Callers run it once, off the event loop,
        and every snapshot afterwards reads the cache. Until it lands the page
        shows a `pending` verdict, which is a different statement from "these
        two runs do not overlap enough to compare".
        """
        if ref.run_id not in self._comparison_cache:
            self._comparison_cache[ref.run_id] = self._comparison_block(
                self.split_records(ref.split),
                self.complete_points(ref),
                self.baseline_points(STRONG_FREE_BASELINE, ref.split),
                ref,
            )
        if ref.run_id not in self._judge_cache:
            self._judge_cache[ref.run_id] = self._judge_block(ref)
        return self._comparison_cache[ref.run_id]

    def _judge_block(self, ref: RunRef) -> dict[str, Any]:
        """Jev against the general LLM judge, on the traces both actually scored.

        A third paired bootstrap, so this is deliberately computed inside
        :meth:`ensure_comparison` off the event loop with the others rather than
        on every snapshot. It is skipped entirely when the selected run is not
        the Jev run: comparing the general judge against itself would be a
        nonsense the page must not render as a result.
        """
        split_records = self.split_records(ref.split)
        judge_points = self.baseline_points(GENERAL_JUDGE, ref.split)
        judge_threshold = self.config.threshold_for(GENERAL_JUDGE)

        base: dict[str, Any] = {
            "rival": GENERAL_JUDGE,
            "rival_label": GENERAL_JUDGE_LABEL,
            "jev_run": ref.run_id,
            "note": (
                "Both arms see the same InferenceView and are asked the same four "
                "questions from the same frozen questions.json at temperature 0, on the "
                "same task-disjoint test split, zero-shot. The scores are "
                "P(unsupported_success) from each verdict. The one asymmetry: Jev's "
                "probabilities come from the API, the general judge's are self-reported "
                "inside its own JSON answer."
            ),
        }
        if ref.provider != JEV_PROVIDER:
            return {
                **base,
                "computable": False,
                "verdict": "not_applicable",
                "verdict_text": (
                    f"The selected run is {ref.provider}, not Jev. This panel compares the "
                    "Jev run against the general judge; select a Jev run to see it."
                ),
            }
        if not judge_points:
            return {
                **base,
                "computable": False,
                "verdict": "not_run",
                "verdict_text": (
                    "No general-judge run on this split. Run "
                    f"`jev-eval run --provider {GENERAL_JUDGE} --split {ref.split}` "
                    "and reload."
                ),
            }

        jev_points = self.complete_points(ref)
        paired = [
            (record, jev_points.get(record.trace_id), judge_points.get(record.trace_id))
            for record in split_records
        ]
        usable = [
            (record, a, b)
            for record, a, b in paired
            if a is not None
            and b is not None
            and a.primary_score is not None
            and b.primary_score is not None
        ]
        n_positives = sum(1 for record, _, _ in usable if record.label is POSITIVE_LABEL)
        base |= {"n_paired": len(usable), "n_positives": n_positives}
        if len(usable) < 2 or n_positives == 0 or n_positives == len(usable):
            return {
                **base,
                "computable": False,
                "verdict": "not_computable",
                "verdict_text": (
                    "Jev and the general judge have not both scored enough of this split "
                    "for a comparison."
                ),
            }

        metrics_cfg = self.config.section("metrics")
        resamples = int(metrics_cfg.get("bootstrap_resamples", 10000))
        seed = int(metrics_cfg.get("bootstrap_seed", 20260919))
        budget = float(self.config.section("gates").get("useful_signal_budget", 0.05))

        y = metrics.binary_targets([record for record, _, _ in usable])
        jev_scores = np.array([float(a.primary_score or 0.0) for _, a, _ in usable])
        judge_scores = np.array([float(b.primary_score or 0.0) for _, _, b in usable])

        interval = metrics.paired_bootstrap(y, jev_scores, judge_scores, "auprc", resamples, seed)
        recall_interval = metrics.paired_bootstrap(
            y, jev_scores, judge_scores, "recall_at_budget", resamples, seed, budget=budget
        )
        lower, upper = interval.lower, interval.upper

        if lower != lower or upper != upper:  # NaN
            verdict, text = (
                "not_computable",
                ("The paired bootstrap produced no usable resamples on this split."),
            )
        elif lower > 0:
            verdict, text = (
                "jev_wins",
                (
                    f"Jev beats the general judge on AUPRC by {interval.point:+.3f} "
                    f"(95% CI [{lower:+.3f}, {upper:+.3f}], entirely above zero)."
                ),
            )
        elif upper < 0:
            verdict, text = (
                "judge_wins",
                (
                    f"Jev is beaten by the general judge on AUPRC by {interval.point:+.3f} "
                    f"(95% CI [{lower:+.3f}, {upper:+.3f}], entirely below zero). The general "
                    "model is the better detector on this data."
                ),
            )
        else:
            verdict, text = (
                "no_difference",
                (
                    f"No difference established. The AUPRC gap is {interval.point:+.3f} but "
                    f"the 95% CI is [{lower:+.3f}, {upper:+.3f}], which spans zero. On this "
                    "split the data does not show the typed model beating the general judge, "
                    "and does not show the reverse either."
                ),
            )

        jev_prf = metrics.prf_at_threshold(y, jev_scores, ref.threshold)
        judge_prf = metrics.prf_at_threshold(y, judge_scores, judge_threshold)
        return {
            **base,
            "computable": True,
            "metric": "auprc",
            "jev_auprc": metrics.auprc(y, jev_scores),
            "judge_auprc": metrics.auprc(y, judge_scores),
            "jev_auroc": metrics.auroc(y, jev_scores),
            "judge_auroc": metrics.auroc(y, judge_scores),
            "jev_recall": jev_prf["recall"],
            "judge_recall": judge_prf["recall"],
            "jev_precision": jev_prf["precision"],
            "judge_precision": judge_prf["precision"],
            "delta_auprc": interval.point,
            "ci": interval.as_dict(),
            "delta_recall_at_budget": recall_interval.point,
            "recall_ci": recall_interval.as_dict(),
            "recall_budget": budget,
            "resamples": resamples,
            "verdict": verdict,
            "verdict_text": text,
            "jev_beats_judge": verdict == "jev_wins",
            "cost": self._judge_cost_block(ref),
        }

    def _judge_cost_block(self, ref: RunRef) -> dict[str, Any]:
        """Cost per 1000 traces for each paid arm, side by side.

        An arm whose model has no verified published rate reports its tokens and
        ``None`` for cost. The page prints that as "unavailable", never as zero:
        a lane that looks free because nobody priced it would be the most
        expensive kind of wrong answer this dashboard could give.
        """
        fx = float(self.config.raw["fx_rate_gbp_usd"])
        out: dict[str, Any] = {"fx_rate_gbp_usd": fx, "arms": {}}
        for provider in (JEV_PROVIDER, GENERAL_JUDGE):
            run = next(
                (
                    r
                    for r in self.runs()
                    if r.provider == provider and r.split == ref.split and r.status == "complete"
                ),
                None,
            )
            if run is None:
                out["arms"][provider] = {"available": False}
                continue
            try:
                manifest, predictions = load_run(run.path)
            except (OSError, ValueError):
                out["arms"][provider] = {"available": False}
                continue
            priced = [p.cost_usd for p in predictions if p.cost_usd is not None]
            per_request = (sum(priced) / len(priced)) if priced else None
            out["arms"][provider] = {
                "available": True,
                "run_id": manifest.run_id,
                "model_id": manifest.model_id,
                "n_predictions": manifest.n_predictions,
                "n_priced": len(priced),
                "input_tokens": manifest.total_input_tokens,
                "output_tokens": manifest.total_output_tokens,
                "total_cost_usd": manifest.total_cost_usd,
                "cost_available": manifest.total_cost_usd is not None,
                "cost_per_1000_traces_usd": (
                    per_request * 1000 if per_request is not None else None
                ),
                "cost_per_1000_traces_gbp": (
                    per_request * 1000 * fx if per_request is not None else None
                ),
            }
        return out

    def _comparison_block(
        self,
        split_records: Sequence[TraceRecord],
        jev_points: dict[str, Prediction],
        strong_points: dict[str, Prediction],
        ref: RunRef,
    ) -> dict[str, Any]:
        """Did Jev beat the strong free baseline? Answered, not implied.

        A negative result renders exactly as loudly as a positive one. The
        verdict is driven by the *interval*, not the point estimate: an interval
        that straddles zero means this data settles nothing, and saying "Jev
        wins" off a point estimate inside such an interval would be the whole
        failure mode this harness exists to measure.
        """
        metrics_cfg = self.config.section("metrics")
        resamples = int(metrics_cfg.get("bootstrap_resamples", 10000))
        seed = int(metrics_cfg.get("bootstrap_seed", 20260919))

        paired = [
            (record, jev_points.get(record.trace_id), strong_points.get(record.trace_id))
            for record in split_records
        ]
        usable = [
            (record, jev, strong)
            for record, jev, strong in paired
            if jev is not None
            and strong is not None
            and jev.primary_score is not None
            and strong.primary_score is not None
        ]
        n_positives = sum(1 for record, _, _ in usable if record.label is POSITIVE_LABEL)
        base: dict[str, Any] = {
            "baseline": STRONG_FREE_BASELINE,
            "baseline_label": "TF-IDF + gradient boosting",
            "n_paired": len(usable),
            "n_positives": n_positives,
            "jev_run": ref.run_id,
        }
        if len(usable) < 2 or n_positives == 0 or n_positives == len(usable):
            return {
                **base,
                "computable": False,
                "verdict": "not_computable",
                "verdict_text": (
                    "Jev and the strong baseline have not both scored enough of this "
                    "split for a comparison. Run "
                    f"`jev-eval run --provider {STRONG_FREE_BASELINE} --split {ref.split}` "
                    "and a Jev run over the same split."
                ),
            }

        y = metrics.binary_targets([record for record, _, _ in usable])
        jev_scores = np.array([float(j.primary_score or 0.0) for _, j, _ in usable])
        base_scores = np.array([float(b.primary_score or 0.0) for _, _, b in usable])

        interval = metrics.paired_bootstrap(y, jev_scores, base_scores, "auprc", resamples, seed)
        auroc_interval = metrics.paired_bootstrap(
            y, jev_scores, base_scores, "auroc", resamples, seed
        )
        jev_auprc = metrics.auprc(y, jev_scores)
        base_auprc = metrics.auprc(y, base_scores)
        jev_auroc = metrics.auroc(y, jev_scores)
        base_auroc = metrics.auroc(y, base_scores)

        lower, upper = interval.lower, interval.upper
        if lower != lower or upper != upper:  # NaN
            verdict, text = (
                "not_computable",
                ("The paired bootstrap produced no usable resamples on this split."),
            )
        elif lower > 0:
            verdict, text = (
                "jev_wins",
                (
                    f"Jev beats the strong free baseline on AUPRC by {interval.point:+.3f} "
                    f"(95% CI [{lower:+.3f}, {upper:+.3f}], entirely above zero)."
                ),
            )
        elif upper < 0:
            verdict, text = (
                "baseline_wins",
                (
                    f"Jev is beaten by the strong free baseline on AUPRC by "
                    f"{interval.point:+.3f} (95% CI [{lower:+.3f}, {upper:+.3f}], entirely "
                    "below zero). The free baseline is the better detector on this data."
                ),
            )
        else:
            verdict, text = (
                "no_difference",
                (
                    f"No difference established. The AUPRC gap is {interval.point:+.3f} but "
                    f"the 95% CI is [{lower:+.3f}, {upper:+.3f}], which spans zero. On this "
                    "split this data does not show Jev beating the free baseline, and does "
                    "not show the reverse either."
                ),
            )

        return {
            **base,
            "computable": True,
            "metric": "auprc",
            "jev_auprc": jev_auprc,
            "baseline_auprc": base_auprc,
            "delta_auprc": interval.point,
            "ci": interval.as_dict(),
            "jev_auroc": jev_auroc,
            "baseline_auroc": base_auroc,
            "delta_auroc": auroc_interval.point,
            "auroc_ci": auroc_interval.as_dict(),
            "resamples": resamples,
            "verdict": verdict,
            "verdict_text": text,
            "beats_baseline": verdict == "jev_wins",
        }

    def _caveats_block(
        self,
        ref: RunRef,
        comparison: dict[str, Any],
        split_records: Sequence[TraceRecord],
        jev_points: dict[str, Prediction],
    ) -> dict[str, Any]:
        """Caveats generated from this run and this dataset, never hard-coded."""
        truncation = _truncation_summary(ref.path)
        return build_caveats(
            provenance=self.provenance,
            audit=audit_status(self.paths.audit_blind),
            headline_ci=comparison.get("ci"),
            n_positives=int(comparison.get("n_positives", 0) or 0),
            n_traces=int(comparison.get("n_paired", 0) or 0),
            truncation=truncation,
            deviation=self._deviation_summary(ref),
        ).as_dict()

    def _deviation_summary(self, ref: RunRef) -> dict[str, Any] | None:
        """Every request parameter a paid arm on this split had refused.

        Read from each run's own deviation file, so the page cannot disagree
        with `jev-eval report` about what was sent.
        """
        found = False
        rows: list[dict[str, Any]] = []
        candidates = [ref] + [
            run
            for run in self.runs()
            if run.provider in (JEV_PROVIDER, GENERAL_JUDGE)
            and run.split == ref.split
            and run.status == "complete"
            and run.run_id != ref.run_id
        ]
        seen: set[str] = set()
        for run in candidates:
            if run.provider in seen:
                continue
            path = run.path / "deviations.jsonl"
            if not path.exists():
                continue
            seen.add(run.provider)
            found = True
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows.append({**row, "provider": run.provider, "run_model_id": run.model_id})
        return {"deviations": rows} if found else None

    def _detection_groups(
        self,
        split_records: Sequence[TraceRecord],
        jev_points: dict[str, Prediction],
        rules_points: dict[str, Prediction],
        tfidf_points: dict[str, Prediction],
        strong_points: dict[str, Prediction],
        judge_points: dict[str, Prediction],
        free_lanes: Sequence[tuple[dict[str, Prediction], float]],
        threshold: float,
        rules_threshold: float,
        tfidf_threshold: float,
        strong_threshold: float,
        judge_threshold: float,
    ) -> dict[str, Any]:
        """Paired bars per fault group, plus the two honesty panels built from them.

        ``no_added_value`` is derived from the same rows as ``groups``. It is not
        a separate curated list, so a group cannot win in one panel and vanish
        from the other.
        """
        by_fault: dict[str, list[TraceRecord]] = {}
        by_cell: dict[str, list[TraceRecord]] = {}
        for record in split_records:
            key = f"{record.label.value}/{record.fault_type.value}"
            by_cell.setdefault(key, []).append(record)
            if record.label is POSITIVE_LABEL:
                by_fault.setdefault(record.fault_type.value, []).append(record)

        groups: list[dict[str, Any]] = []
        no_added_value: list[dict[str, Any]] = []
        for fault, records in sorted(by_fault.items()):
            ids = [r.trace_id for r in records]
            jev = _lane(jev_points, ids, threshold)
            rules = _lane(rules_points, ids, rules_threshold)
            tfidf = _lane(tfidf_points, ids, tfidf_threshold)
            strong = _lane(strong_points, ids, strong_threshold)
            judge = _lane(judge_points, ids, judge_threshold)
            union = _union_points(free_lanes, ids)
            jev_costs = [
                p.cost_usd
                for p in jev_points.values()
                if p.trace_id in set(ids) and p.cost_usd is not None
            ]
            jev_latencies = [
                p.end_to_end_latency_ms for p in jev_points.values() if p.trace_id in set(ids)
            ]
            row = {
                "fault": fault,
                "n": len(records),
                "jev": jev,
                "rules": rules,
                "tfidf": tfidf,
                "tfidf_gbm": strong,
                GENERAL_JUDGE: judge,
                "free_union": union,
                # The gap that answers the new question: typed model against
                # general judge, on the same traces at each lane's frozen
                # threshold. None when either lane has not scored the group --
                # an unscored lane is not a zero.
                "gap_vs_general_judge": (
                    None
                    if jev["rate"] is None or judge["rate"] is None
                    else jev["rate"] - judge["rate"]
                ),
                "gap_vs_rules": (
                    None
                    if jev["rate"] is None or rules["rate"] is None
                    else jev["rate"] - rules["rate"]
                ),
                # The gap that decides whether Jev bought anything. Measured
                # against the strong baseline, not the weak one.
                "gap_vs_strong": (
                    None
                    if jev["rate"] is None or strong["rate"] is None
                    else jev["rate"] - strong["rate"]
                ),
                "rule_computable": fault != FOCUS_FAULT,
                "focus": fault == FOCUS_FAULT,
                "jev_cost_usd": sum(jev_costs) if jev_costs else 0.0,
                "jev_mean_latency_ms": (
                    sum(jev_latencies) / len(jev_latencies) if jev_latencies else None
                ),
            }
            groups.append(row)
            # Where a free lane already catches everything, or already matches
            # Jev, the paid call buys nothing but a bill and a wait. Same row,
            # same code path, shown plainly.
            caught_free = (rules["rate"] == 1.0 and rules["scored"] > 0) or (
                strong["rate"] == 1.0 and strong["scored"] > 0
            )
            gap_vs_strong = (
                None
                if jev["rate"] is None or strong["rate"] is None
                else jev["rate"] - strong["rate"]
            )
            no_better = gap_vs_strong is not None and gap_vs_strong <= 0.0 and strong["scored"] > 0
            if caught_free or no_better:
                no_added_value.append(row)

        per_fault: list[dict[str, Any]] = []
        for cell, records in sorted(by_cell.items()):
            ids = [r.trace_id for r in records]
            jev = _lane(jev_points, ids, threshold)
            rules = _lane(rules_points, ids, rules_threshold)
            tfidf = _lane(tfidf_points, ids, tfidf_threshold)
            strong = _lane(strong_points, ids, strong_threshold)
            judge = _lane(judge_points, ids, judge_threshold)
            label, fault = cell.split("/", 1)
            per_fault.append(
                {
                    "cell": cell,
                    "label": label,
                    "fault": fault,
                    "n": len(records),
                    "is_positive": label == POSITIVE_LABEL.value,
                    "without_jev_rate": rules["rate"],
                    "without_jev_tfidf_rate": tfidf["rate"],
                    "without_jev_strong_rate": strong["rate"],
                    "general_judge_rate": judge["rate"],
                    "general_judge_scored": judge["scored"],
                    "with_jev_rate": jev["rate"],
                    "with_jev_scored": jev["scored"],
                    "gap": (
                        None
                        if jev["rate"] is None or rules["rate"] is None
                        else jev["rate"] - rules["rate"]
                    ),
                    "gap_vs_strong": (
                        None
                        if jev["rate"] is None or strong["rate"] is None
                        else jev["rate"] - strong["rate"]
                    ),
                    "gap_vs_general_judge": (
                        None
                        if jev["rate"] is None or judge["rate"] is None
                        else jev["rate"] - judge["rate"]
                    ),
                }
            )

        return {"groups": groups, "no_added_value": no_added_value, "per_fault": per_fault}

    def _cost_block(
        self,
        ref: RunRef,
        predictions: Sequence[Prediction],
        split_records: Sequence[TraceRecord],
    ) -> dict[str, Any]:
        fx = float(self.config.raw["fx_rate_gbp_usd"])
        fx_date = str(self.config.raw["fx_rate_date"])
        fx_source = str(self.config.raw.get("fx_rate_source", ""))

        priced = [p.cost_usd for p in predictions if p.cost_usd is not None]
        spend = float(sum(priced))
        n_priced = len(priced)
        per_request = spend / n_priced if n_priced else None

        sweep_spend = 0.0
        sweep_requests = 0
        for sibling in self.sibling_jev_runs(ref.split):
            if sibling.run_id == ref.run_id:
                sweep_spend += spend
                sweep_requests += n_priced
                continue
            try:
                manifest, _ = load_run(sibling.path)
            except (OSError, ValueError):
                continue
            if manifest.total_cost_usd is not None:
                sweep_spend += manifest.total_cost_usd
                sweep_requests += manifest.n_predictions

        rate = (
            self.prices.get(ref.model_id).input_usd_per_mtok
            if _priceable(self.prices, ref.model_id)
            else None
        )

        return {
            "currency": "USD",
            "fx_rate_gbp_usd": fx,
            "fx_rate_date": fx_date,
            "fx_rate_source": fx_source,
            "spend_usd": spend,
            "spend_gbp": spend * fx,
            "sweep_spend_usd": sweep_spend,
            "sweep_spend_gbp": sweep_spend * fx,
            "sweep_requests": sweep_requests,
            "n_priced_requests": n_priced,
            "n_unpriced_requests": len(predictions) - n_priced,
            "cost_per_request_usd": per_request,
            "cost_per_1000_traces_usd": per_request * 1000 if per_request is not None else None,
            "cost_per_1000_traces_gbp": (
                per_request * 1000 * fx if per_request is not None else None
            ),
            "input_tokens": sum(p.usage.input_tokens for p in predictions if p.usage),
            "output_tokens": sum(p.usage.output_tokens for p in predictions if p.usage),
            "rate_usd_per_mtok": rate,
            "prices_path": self.prices.path,
            "prices_date": self.prices.manifest_date,
            "traces_in_split": len(split_records),
            "without_jev_usd": 0.0,
            "without_jev_note": (
                "The without-Jev lane runs locally: a deterministic rule checker and a "
                "TF-IDF classifier. Neither makes a metered call, so its marginal cost "
                "is zero on this manifest. It is not free to build or maintain."
            ),
            "projection_scale_traces": int(
                (self.config.raw.get("cost_reporting") or {}).get(
                    "projection_scale_traces", 1_000_000
                )
            ),
            "per_1000_note": (
                "Cost per 1000 traces is the mean cost of one request times 1000: one "
                "pass over 1000 traces, not this run's repeats."
            ),
        }

    def _speed_block(self, ref: RunRef, predictions: Sequence[Prediction]) -> dict[str, Any]:
        arms: list[dict[str, Any]] = []
        for sibling in self.sibling_jev_runs(ref.split):
            if sibling.run_id == ref.run_id:
                latencies = [p.end_to_end_latency_ms for p in predictions]
                n = len(predictions)
                progress = read_progress(sibling.path)
            else:
                try:
                    _, sibling_predictions = load_run(sibling.path)
                except (OSError, ValueError):
                    continue
                latencies = [p.end_to_end_latency_ms for p in sibling_predictions]
                n = len(sibling_predictions)
                progress = read_progress(sibling.path)

            wall_s: float | None = None
            if progress and progress.get("status") == "complete":
                start = _parse_utc(str(progress.get("started_utc", "")))
                end = _parse_utc(str(progress.get("updated_utc", "")))
                if start is not None and end is not None and end > start:
                    wall_s = end - start
            arms.append(
                {
                    "run_id": sibling.run_id,
                    "concurrency": sibling.concurrency,
                    "n": n,
                    "is_current": sibling.run_id == ref.run_id,
                    "p50_ms": _percentile(latencies, 0.50),
                    "p95_ms": _percentile(latencies, 0.95),
                    "p99_ms": _percentile(latencies, 0.99),
                    "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                    "wall_s": wall_s,
                    "throughput_per_s": (n / wall_s) if wall_s else None,
                    "status": sibling.status,
                }
            )
        current = [p.end_to_end_latency_ms for p in predictions]
        return {
            "arms": arms,
            "current": {
                "p50_ms": _percentile(current, 0.50),
                "p95_ms": _percentile(current, 0.95),
                "p99_ms": _percentile(current, 0.99),
                "n": len(current),
            },
            "note": (
                "Throughput is measured wall-clock over a completed arm: predictions "
                "divided by the seconds between the run starting and finishing. An arm "
                "still in flight shows no throughput rather than a partial one."
            ),
        }

    def _honesty_block(
        self,
        split_records: Sequence[TraceRecord],
        ref: RunRef,
        jev_points: dict[str, Prediction],
        threshold: float,
    ) -> dict[str, Any]:
        focus = [r.trace_id for r in split_records if r.fault_type.value == FOCUS_FAULT]
        control = [r.trace_id for r in split_records if r.fault_type.value == CONTROL_FAULT]
        jev_focus = _lane(jev_points, focus, threshold)
        jev_control = _lane(jev_points, control, threshold)
        rules_points = self.baseline_points(FREE_GUARD, ref.split)
        rules_threshold = self.config.threshold_for(FREE_GUARD)
        separation = (
            None
            if jev_focus["rate"] is None or jev_control["rate"] is None
            else jev_focus["rate"] - jev_control["rate"]
        )
        return {
            "labels": audit_status(self.paths.audit_blind),
            "semantic_subclass": {
                "fault": FOCUS_FAULT,
                "control": CONTROL_FAULT,
                "fault_n": len(focus),
                "control_n": len(control),
                "jev_fault": jev_focus,
                "jev_control": jev_control,
                "rules_fault": _lane(rules_points, focus, rules_threshold),
                "rules_control": _lane(rules_points, control, rules_threshold),
                "separation": separation,
                "text": (
                    f"The semantic subclass is {len(focus)} mismatch and {len(control)} "
                    f"matched-control traces in this split. Every figure about it rests "
                    f"on that small a sample; read the separation, not the flag rate "
                    f"alone, and treat the interval around it as wide."
                ),
            },
            "threshold": threshold,
            "threshold_note": (
                "Thresholds were frozen on the validation split before the test split was opened."
            ),
            "repeats_note": (
                "Point figures use one pass per trace. Repeats measure stability, not "
                "accuracy, and are not averaged into a detection rate."
            ),
        }


def _truncation_summary(run_dir: Path) -> dict[str, Any] | None:
    """Read a run's own account of what it had to shorten.

    Returns ``None`` for a run written before truncation accounting existed,
    which is different from a run that shortened nothing: the caveat says
    nothing at all rather than claiming everything fitted.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if "truncated_traces" not in manifest:
        return None
    truncated = int(manifest.get("truncated_traces", 0) or 0)
    dropped_events = 0
    shortened_payloads = 0
    did_not_fit = 0
    rule = ""
    path = Path(str(manifest.get("truncation_path", "")))
    if truncated and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            dropped_events += int(row.get("dropped_events", 0) or 0)
            shortened_payloads += int(row.get("shortened_payloads", 0) or 0)
            rule = rule or str(row.get("rule", ""))
            if not row.get("fits", True):
                did_not_fit += 1
    return {
        "truncated_traces": truncated,
        "n_traces": int(manifest.get("n_traces", 0) or 0),
        "dropped_events": dropped_events,
        "shortened_payloads": shortened_payloads,
        "rule": rule,
        "did_not_fit": did_not_fit,
    }


def _priceable(prices: PriceManifest, model_id: str) -> bool:
    try:
        price = prices.get(model_id)
    except Exception:
        return False
    return price.verified and price.input_usd_per_mtok is not None


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
