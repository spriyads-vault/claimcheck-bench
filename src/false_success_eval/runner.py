"""Run orchestration: config, evaluator construction, append-only run artifacts."""

from __future__ import annotations

import concurrent.futures
import json
import platform
import secrets
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .costs import PriceManifest, load_prices, recorded_cost_usd
from .evaluators.base import assert_clean_payload
from .evaluators.rules import RulesEvaluator
from .evaluators.tfidf import TfidfEvaluator
from .evaluators.tfidf_gbm import TfidfGbmEvaluator
from .hashing import canonical_json, git_commit, sha256_file, sha256_obj
from .redact import contains_secret, redact
from .retry import ErrorRateGuard, RetryPolicy
from .schemas import Prediction, RunManifest, Splits, TraceRecord

#: Providers that run locally: no key, no network, no metered cost.
OFFLINE_PROVIDERS = frozenset({"rules", "tfidf", "tfidf_gbm"})

#: Providers that spend money. Each needs a key from the environment, a
#: pre-spend projection and ``--confirm-paid`` before a single request goes out.
#: ``openai`` is the general-judge arm: a general LLM asked the same four
#: questions on the same inference view, so the comparison is typed-model
#: against general-model rather than paid against free.
PAID_PROVIDERS = frozenset({"jev", "openai"})

#: The paid arm the headline is stated for, and the paid arm it is stated
#: against. Kept here so the report and the dashboard cannot disagree.
PRIMARY_PAID_PROVIDER = "jev"
GENERAL_JUDGE_PROVIDER = "openai"

#: The free baselines Jev is measured against. ``tfidf_gbm`` is the strong
#: one -- the published recipe that beats every LLM judge in the literature --
#: and it is the bar a paid provider has to clear to have bought anything.
FREE_BASELINES = ("rules", "tfidf", "tfidf_gbm")


@dataclass(frozen=True)
class EvalConfig:
    path: str
    raw: dict[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name) or {}
        if not isinstance(value, dict):
            raise TypeError(f"config section {name!r} is not a mapping")
        return value

    def retry_policy(self) -> RetryPolicy:
        r = self.section("retry")
        return RetryPolicy(
            max_attempts=int(r.get("max_attempts", 3)),
            backoff_initial_s=float(r.get("backoff_initial_s", 0.5)),
            backoff_max_s=float(r.get("backoff_max_s", 5.0)),
            backoff_jitter=float(r.get("backoff_jitter", 0.25)),
            retry_statuses=frozenset(int(s) for s in r.get("retry_statuses", [429, 529])),
            respect_retry_after=bool(r.get("respect_retry_after", True)),
            total_timeout_s=float(r.get("total_timeout_s", 30.0)),
        )

    def guard(self) -> ErrorRateGuard:
        r = self.section("retry")
        return ErrorRateGuard(
            max_error_rate=float(r.get("abort_error_rate", 0.05)),
            min_samples=int(r.get("abort_min_samples", 20)),
        )

    def threshold_for(self, provider: str) -> float:
        thresholds = self.section("thresholds")
        return float(thresholds.get(provider, thresholds.get("default", 0.5)))


def load_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} did not parse to a mapping")
    return EvalConfig(path=str(path), raw=raw)


@dataclass(frozen=True)
class DatasetPaths:
    """Artifact paths for one dataset, synthetic stage or ingested corpus.

    Every dataset owns a disjoint set of paths, so a synthetic run and a real
    run can never be mixed into one report by accident: the runs root differs
    as well as the dataset file.
    """

    records: int
    dataset: Path
    splits: Path
    hashes: Path
    audit: Path
    audit_blind: Path
    runs_root: Path
    dataset_id: str = ""
    kind: str = "synthetic"
    provenance: Path = Path("")

    @property
    def is_real(self) -> bool:
        return self.kind == "real"


def available_corpora(config: EvalConfig) -> dict[str, dict[str, Any]]:
    """The ingested corpora declared in ``eval.yaml``, by id."""
    section = config.raw.get("corpora") or {}
    if not isinstance(section, dict):
        raise TypeError("config section 'corpora' is not a mapping")
    return {str(k): dict(v or {}) for k, v in section.items()}


def resolve_dataset_paths(
    config: EvalConfig,
    records: int | None = None,
    corpus: str | None = None,
) -> DatasetPaths:
    """Resolve the artifact paths for one dataset.

    With ``corpus`` set, the paths come from that corpus's block under
    ``corpora`` in the config. Otherwise ``{records}`` is substituted into the
    synthetic ``dataset`` block, so each synthetic stage owns its own artifacts.
    """
    if corpus:
        corpora = available_corpora(config)
        if corpus not in corpora:
            raise KeyError(
                f"unknown corpus {corpus!r}. Declared corpora: "
                f"{sorted(corpora) or 'none'}. Add a block under 'corpora' in the "
                "config, then run `jev-eval ingest`."
            )
        block = corpora[corpus]
        root = Path(str(block.get("root", f"data/{corpus}")))
        return DatasetPaths(
            records=int(block.get("records", 0)),
            dataset=root / "dataset.jsonl",
            splits=root / "splits.json",
            hashes=root / "hashes.json",
            audit=root / "audit.csv",
            audit_blind=root / "audit_blind.csv",
            runs_root=Path(str(block.get("runs_root", f"runs/{corpus}"))),
            dataset_id=corpus,
            kind=str(block.get("kind", "real")),
            provenance=root / "provenance.json",
        )

    section = config.section("dataset")
    count = records if records is not None else int(section["records"])

    def path(key: str) -> Path:
        return Path(str(section[key]).format(records=count))

    dataset = path("output")
    return DatasetPaths(
        records=count,
        dataset=dataset,
        splits=path("splits"),
        hashes=path("hashes"),
        audit=path("audit"),
        audit_blind=path("audit_blind"),
        runs_root=path("runs_root"),
        dataset_id=f"synthetic-{count}",
        kind="synthetic",
        provenance=dataset.with_name(f"provenance-{count}.json"),
    )


def load_splits(path: str | Path) -> Splits:
    return Splits.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def select_split(
    records: Sequence[TraceRecord], splits: Splits, split: str
) -> tuple[TraceRecord, ...]:
    families = frozenset(splits.families_for(split))
    return tuple(r for r in records if r.template_family in families)


def new_run_id(provider: str, split: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{provider}-{split}-{stamp}-{secrets.token_hex(3)}"


def _write_jsonl_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write append-only. Exclusive create means a prior run is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    return sha256_file(path)


def build_offline_evaluator(
    provider: str,
    config: EvalConfig,
    records: Sequence[TraceRecord],
    splits: Splits,
) -> RulesEvaluator | TfidfEvaluator | TfidfGbmEvaluator:
    """Construct a local evaluator, fitting it on dev and calibrating on validation."""
    threshold = config.threshold_for(provider)
    if provider == "rules":
        return RulesEvaluator(threshold=threshold)
    if provider in {"tfidf", "tfidf_gbm"}:
        fitted: TfidfEvaluator | TfidfGbmEvaluator = (
            TfidfEvaluator(threshold=threshold)
            if provider == "tfidf"
            else TfidfGbmEvaluator(threshold=threshold)
        )
        dev = select_split(records, splits, "dev")
        validation = select_split(records, splits, "validation")
        fitted.fit(dev, validation, frozenset(splits.test))
        return fitted
    raise ValueError(f"{provider!r} is not an offline provider")


def run_predictions(
    evaluator: Any,
    records: Sequence[TraceRecord],
    repeats: int,
    concurrency: int,
    on_result: Callable[[Prediction], object] | None = None,
) -> list[Prediction]:
    """Run the evaluator over every trace, ``repeats`` times each.

    The evaluator is handed an InferenceView and nothing else. The payload is
    checked for ground-truth fields before it leaves this function.

    ``on_result`` is called once per prediction, **as it lands** rather than at
    the end. That is what makes the run artifact append-only as it is built, and
    what the live dashboard tails. Under concurrency the callback is invoked from
    the completing worker thread, so an implementation must be thread-safe.
    """
    jobs: list[tuple[TraceRecord, int]] = [
        (record, repeat) for repeat in range(repeats) for record in records
    ]

    def run_one(job: tuple[TraceRecord, int]) -> Prediction:
        record, repeat = job
        view = record.inference_view()
        assert_clean_payload(view.model_dump(mode="json"), path=f"$.view[{record.trace_id}]")
        prediction: Prediction = evaluator.predict(view, record.trace_id, repeat)
        return prediction

    results: list[Prediction] = []
    if concurrency <= 1:
        for job in jobs:
            prediction = run_one(job)
            if on_result is not None:
                on_result(prediction)
            results.append(prediction)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            prediction = future.result()
            if on_result is not None:
                on_result(prediction)
            results.append(prediction)
    return results


def finalise_costs(predictions: Sequence[Prediction], prices: PriceManifest) -> list[Prediction]:
    """Recompute every cost from the dated manifest, so one code path owns it."""
    return [
        prediction.model_copy(
            update={"cost_usd": recorded_cost_usd(prices, prediction.model_id, prediction.usage)}
        )
        for prediction in predictions
    ]


PROGRESS_FILE = "progress.json"

#: Append-only account of every trace the evaluator had to shorten. Written on
#: finalise and checksummed in the manifest. An empty file is a positive
#: statement: this run sent every trace whole.
TRUNCATION_FILE = "truncation.jsonl"

#: Append-only account of every request parameter a provider refused and this
#: harness therefore dropped. Written on finalise and checksummed in the
#: manifest, exactly like the truncation account. An empty file is a positive
#: statement: this run sent the request the frozen config describes.
DEVIATION_FILE = "deviations.jsonl"


class RunWriter:
    """An append-only run directory, written *as the run happens*.

    Predictions are appended and flushed the moment each one lands, so an
    in-progress run is a complete, readable artifact at every instant and can be
    tailed by the live dashboard. Both JSONL files are opened with exclusive
    create, so a prior run is still never overwritten.

    ``progress.json`` sits beside them. It is the one mutable file in the
    directory and exists purely so a reader can tell a run that is still going
    from one that stopped: the append-only files themselves cannot say which.
    It is not part of the verified artifact set and no metric is read from it.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        provider: str,
        model_id: str,
        split: str,
        repeats: int,
        concurrency: int,
        threshold: float,
        n_traces: int,
        n_expected: int,
        prices: PriceManifest,
        truncation_source: Any | None = None,
        deviation_source: Any | None = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        self.run_id = run_id
        self.provider = provider
        self.model_id = model_id
        self.split = split
        self.repeats = repeats
        self.concurrency = concurrency
        self.threshold = threshold
        self.n_traces = n_traces
        self.n_expected = n_expected
        self.prices = prices
        # The evaluator, when it is one that reduces oversized traces. Its
        # records are drained on finalise into truncation.jsonl, so a run
        # carries its own account of what was shortened and by how much.
        self.truncation_source = truncation_source
        self.truncation_path = run_dir / TRUNCATION_FILE
        # The evaluator, when it is one that can be forced off the frozen
        # request shape by the provider. Defaults to the truncation source
        # because it is the same object in every current caller.
        self.deviation_source = (
            deviation_source if deviation_source is not None else truncation_source
        )
        self.deviations_path = run_dir / DEVIATION_FILE
        self.started_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        self.predictions_path = run_dir / "predictions.jsonl"
        self.attempts_path = run_dir / "attempts.jsonl"
        self._predictions_handle = self.predictions_path.open("x", encoding="utf-8", newline="\n")
        self._attempts_handle = self.attempts_path.open("x", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._written: list[Prediction] = []
        self._write_progress("running")

    # -- progress -------------------------------------------------------
    def _progress_payload(self, status: str, error: str | None = None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "split": self.split,
            "repeats": self.repeats,
            "concurrency": self.concurrency,
            "threshold": self.threshold,
            "n_traces": self.n_traces,
            "n_expected": self.n_expected,
            "n_written": len(self._written),
            "status": status,
            "started_utc": self.started_utc,
            "updated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": error,
        }

    def _write_progress(self, status: str, error: str | None = None) -> None:
        """Rewrite atomically, so a tailing reader never sees a half-written file."""
        payload = self._progress_payload(status, error)
        tmp = self.run_dir / (PROGRESS_FILE + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.run_dir / PROGRESS_FILE)

    # -- appending ------------------------------------------------------
    def append(self, prediction: Prediction) -> Prediction:
        """Price, redact and append one prediction. Thread-safe.

        Cost is recomputed here from the dated price manifest, so the streaming
        path and the batch path share the one code path that owns a rate.
        """
        priced = prediction.model_copy(
            update={
                "cost_usd": recorded_cost_usd(self.prices, prediction.model_id, prediction.usage)
            }
        )
        row = redact(priced.model_dump(mode="json"))
        if contains_secret(row):
            raise RuntimeError("a live secret survived redaction; refusing to write this run")
        attempt_rows = [
            redact(
                {
                    "trace_id": priced.trace_id,
                    "repeat": priced.repeat,
                    **attempt.model_dump(mode="json"),
                }
            )
            for attempt in priced.attempts
        ]
        with self._lock:
            self._predictions_handle.write(canonical_json(row) + "\n")
            self._predictions_handle.flush()
            for attempt_row in attempt_rows:
                self._attempts_handle.write(canonical_json(attempt_row) + "\n")
            self._attempts_handle.flush()
            self._written.append(priced)
            self._write_progress("running")
        return priced

    @property
    def predictions(self) -> list[Prediction]:
        with self._lock:
            return list(self._written)

    # -- finishing ------------------------------------------------------
    def abort(self, error: str) -> None:
        self._close_handles()
        self._write_progress("failed", error=error)

    def _close_handles(self) -> None:
        for handle in (self._predictions_handle, self._attempts_handle):
            if not handle.closed:
                handle.flush()
                handle.close()

    def _write_truncations(self) -> tuple[int, str]:
        """Write the truncation account. Always written, even when empty."""
        records = list(getattr(self.truncation_source, "truncations", []) or [])
        with self.truncation_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(records, key=lambda r: r.trace_id):
                handle.write(canonical_json(record.as_dict()) + "\n")
        return len(records), sha256_file(self.truncation_path)

    def _write_deviations(self) -> tuple[int, str]:
        """Write the deviation account. Always written, even when empty."""
        records = list(getattr(self.deviation_source, "deviations", []) or [])
        with self.deviations_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(records, key=lambda r: r.parameter):
                handle.write(canonical_json(redact(record.as_dict())) + "\n")
        return len(records), sha256_file(self.deviations_path)

    def deviation_summary(self) -> dict[str, Any]:
        """What the caveat needs to say about departures from the frozen request."""
        records = list(getattr(self.deviation_source, "deviations", []) or [])
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "deviations": [r.as_dict() for r in records],
        }

    def truncation_summary(self) -> dict[str, Any]:
        """What the caveat needs to say about reductions in this run."""
        records = list(getattr(self.truncation_source, "truncations", []) or [])
        return {
            "truncated_traces": len(records),
            "n_traces": self.n_traces,
            "dropped_events": sum(r.dropped_events for r in records),
            "shortened_payloads": sum(r.shortened_payloads for r in records),
            "rule": records[0].rule if records else "",
            "did_not_fit": sum(1 for r in records if not r.fits),
        }

    def finalise(
        self,
        *,
        config: EvalConfig,
        splits_path: Path,
        dataset_path: Path,
        questions: dict[str, Any] | None,
        repo_root: Path,
    ) -> RunManifest:
        self._close_handles()
        truncated_traces, truncation_sha = self._write_truncations()
        deviation_count, deviations_sha = self._write_deviations()
        predictions = self.predictions
        costs = [p.cost_usd for p in predictions if p.cost_usd is not None]
        errors = [p for p in predictions if p.error is not None]

        manifest = RunManifest(
            run_id=self.run_id,
            created_utc=self.started_utc,
            provider=self.provider,
            model_id=self.model_id,
            split=self.split,
            repeats=self.repeats,
            concurrency=self.concurrency,
            threshold=self.threshold,
            n_traces=self.n_traces,
            n_predictions=len(predictions),
            dataset_path=str(dataset_path),
            dataset_sha256=sha256_file(dataset_path),
            splits_sha256=sha256_file(splits_path),
            questions_sha256=sha256_obj(questions) if questions is not None else "",
            eval_config_sha256=config.sha256,
            prices_path=self.prices.path,
            prices_sha256=sha256_file(self.prices.path),
            predictions_path=str(self.predictions_path),
            predictions_sha256=sha256_file(self.predictions_path),
            attempts_path=str(self.attempts_path),
            attempts_sha256=sha256_file(self.attempts_path),
            git_commit=git_commit(repo_root),
            python_version=platform.python_version(),
            package_version=__version__,
            total_input_tokens=sum(p.usage.input_tokens for p in predictions if p.usage),
            total_output_tokens=sum(p.usage.output_tokens for p in predictions if p.usage),
            total_cost_usd=float(sum(costs)) if len(costs) == len(predictions) else None,
            error_count=len(errors),
            retry_count=sum(max(0, len(p.attempts) - 1) for p in predictions),
            parse_failure_count=sum(1 for p in errors if "ParseFailure" in (p.error or "")),
            truncated_traces=truncated_traces,
            truncation_path=str(self.truncation_path),
            truncation_sha256=truncation_sha,
            deviation_count=deviation_count,
            deviations_path=str(self.deviations_path),
            deviations_sha256=deviations_sha,
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_progress("complete")
        return manifest


def read_progress(run_dir: Path) -> dict[str, Any] | None:
    """The mutable status file beside a run, or None for a run written before it existed."""
    path = run_dir / PROGRESS_FILE
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def write_run(
    *,
    run_dir: Path,
    run_id: str,
    provider: str,
    model_id: str,
    split: str,
    repeats: int,
    concurrency: int,
    threshold: float,
    records: Sequence[TraceRecord],
    predictions: Sequence[Prediction],
    config: EvalConfig,
    splits_path: Path,
    dataset_path: Path,
    questions: dict[str, Any] | None,
    prices: PriceManifest,
    repo_root: Path,
) -> RunManifest:
    run_dir.mkdir(parents=True, exist_ok=False)

    prediction_rows = [redact(p.model_dump(mode="json")) for p in predictions]
    attempt_rows = [
        redact({"trace_id": p.trace_id, "repeat": p.repeat, **attempt.model_dump(mode="json")})
        for p in predictions
        for attempt in p.attempts
    ]
    for row in prediction_rows:
        if contains_secret(row):
            raise RuntimeError("a live secret survived redaction; refusing to write this run")

    predictions_path = run_dir / "predictions.jsonl"
    attempts_path = run_dir / "attempts.jsonl"
    predictions_sha = _write_jsonl_exclusive(predictions_path, prediction_rows)
    attempts_sha = _write_jsonl_exclusive(attempts_path, attempt_rows)

    costs = [p.cost_usd for p in predictions if p.cost_usd is not None]
    errors = [p for p in predictions if p.error is not None]

    manifest = RunManifest(
        run_id=run_id,
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        provider=provider,
        model_id=model_id,
        split=split,
        repeats=repeats,
        concurrency=concurrency,
        threshold=threshold,
        n_traces=len(records),
        n_predictions=len(predictions),
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        splits_sha256=sha256_file(splits_path),
        questions_sha256=sha256_obj(questions) if questions is not None else "",
        eval_config_sha256=config.sha256,
        prices_path=prices.path,
        prices_sha256=sha256_file(prices.path),
        predictions_path=str(predictions_path),
        predictions_sha256=predictions_sha,
        attempts_path=str(attempts_path),
        attempts_sha256=attempts_sha,
        git_commit=git_commit(repo_root),
        python_version=platform.python_version(),
        package_version=__version__,
        total_input_tokens=sum(p.usage.input_tokens for p in predictions if p.usage),
        total_output_tokens=sum(p.usage.output_tokens for p in predictions if p.usage),
        total_cost_usd=float(sum(costs)) if len(costs) == len(predictions) else None,
        error_count=len(errors),
        retry_count=sum(max(0, len(p.attempts) - 1) for p in predictions),
        parse_failure_count=sum(1 for p in errors if "ParseFailure" in (p.error or "")),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_run(run_dir: Path) -> tuple[RunManifest, list[Prediction]]:
    manifest = RunManifest.model_validate(
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    predictions: list[Prediction] = []
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                predictions.append(Prediction.model_validate(json.loads(line)))
    return manifest, predictions


def discover_runs(runs_root: Path) -> list[Path]:
    return sorted(p.parent for p in runs_root.glob("*/manifest.json"))


def verify_run(run_dir: Path, archived: bool = False) -> list[str]:
    """Recompute checksums and cost arithmetic. Returns a list of problems.

    ``archived=True`` verifies the run against **its own manifest alone** and
    skips the comparison against whatever dataset currently sits at the
    manifest's path. An archived run is kept precisely because the dataset it
    was scored on has since been replaced (see ``runs/_archive/README.md``);
    reporting that replacement as a defect every time would make the check
    useless for the runs it applies to. Everything else -- predictions and
    attempts checksums, the price manifest, the cost arithmetic -- is still
    verified in full, and the manifest's own ``dataset_sha256`` is left exactly
    as recorded. It is never the default: for a live run a dataset that has
    moved under it is a real problem.
    """
    problems: list[str] = []
    manifest, predictions = load_run(run_dir)

    for label, path, expected in (
        ("predictions", run_dir / "predictions.jsonl", manifest.predictions_sha256),
        ("attempts", run_dir / "attempts.jsonl", manifest.attempts_sha256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{label} sha256 mismatch: manifest {expected}, file {actual}")

    prices_path = Path(manifest.prices_path)
    if not prices_path.exists():
        problems.append(f"price manifest {prices_path} is missing; cost cannot be verified")
        return problems
    actual_prices_sha = sha256_file(prices_path)
    if actual_prices_sha != manifest.prices_sha256:
        problems.append(
            f"price manifest sha256 mismatch: manifest {manifest.prices_sha256}, "
            f"file {actual_prices_sha}"
        )

    dataset_path = Path(manifest.dataset_path)
    if not archived and dataset_path.exists():
        actual_dataset = sha256_file(dataset_path)
        if actual_dataset != manifest.dataset_sha256:
            problems.append(
                f"dataset sha256 mismatch: manifest {manifest.dataset_sha256}, "
                f"file {actual_dataset}"
            )

    # Cost is recomputed from the dated manifest, never from a literal in code.
    prices = load_prices(prices_path)
    recomputed_total = 0.0
    all_priced = True
    for prediction in predictions:
        expected_cost = recorded_cost_usd(prices, prediction.model_id, prediction.usage)
        if expected_cost is None:
            all_priced = False
            if prediction.cost_usd is not None:
                problems.append(
                    f"{prediction.trace_id}: cost recorded as {prediction.cost_usd} but the "
                    f"manifest has no verified rate for {prediction.model_id}"
                )
            continue
        recomputed_total += expected_cost
        if prediction.cost_usd is None or abs(prediction.cost_usd - expected_cost) > 1e-12:
            problems.append(
                f"{prediction.trace_id}: cost {prediction.cost_usd} != recomputed "
                f"{expected_cost} from {prices_path}"
            )

    if all_priced:
        if manifest.total_cost_usd is None:
            problems.append("manifest total_cost_usd is null although every prediction is priced")
        elif abs(manifest.total_cost_usd - recomputed_total) > 1e-9:
            problems.append(
                f"manifest total_cost_usd {manifest.total_cost_usd} != recomputed "
                f"{recomputed_total}"
            )

    if manifest.truncation_path:
        # Resolved from the run directory, not from the manifest's recorded
        # path: that path is relative to wherever the run was launched, and
        # verification routinely happens from somewhere else. Predictions and
        # attempts are resolved the same way, just above.
        truncation_file = run_dir / TRUNCATION_FILE
        if not truncation_file.exists():
            problems.append(
                f"manifest names a truncation account at {truncation_file} but the "
                "file is missing; what was shortened cannot be verified"
            )
        else:
            actual = sha256_file(truncation_file)
            if actual != manifest.truncation_sha256:
                problems.append(
                    f"truncation sha256 mismatch: manifest {manifest.truncation_sha256}, "
                    f"file {actual}"
                )
            else:
                rows = sum(1 for line in truncation_file.read_text().splitlines() if line.strip())
                if rows != manifest.truncated_traces:
                    problems.append(
                        f"truncated_traces {manifest.truncated_traces} != {rows} rows in "
                        f"{truncation_file}"
                    )

    if manifest.deviations_path:
        # Same resolution rule as the truncation account, for the same reason.
        deviations_file = run_dir / DEVIATION_FILE
        if not deviations_file.exists():
            problems.append(
                f"manifest names a deviation account at {deviations_file} but the "
                "file is missing; what the run departed from cannot be verified"
            )
        else:
            actual = sha256_file(deviations_file)
            if actual != manifest.deviations_sha256:
                problems.append(
                    f"deviations sha256 mismatch: manifest {manifest.deviations_sha256}, "
                    f"file {actual}"
                )
            else:
                rows = sum(1 for line in deviations_file.read_text().splitlines() if line.strip())
                if rows != manifest.deviation_count:
                    problems.append(
                        f"deviation_count {manifest.deviation_count} != {rows} rows in "
                        f"{deviations_file}"
                    )

    recomputed_tokens = sum(p.usage.input_tokens for p in predictions if p.usage)
    if recomputed_tokens != manifest.total_input_tokens:
        problems.append(
            f"total_input_tokens {manifest.total_input_tokens} != recomputed {recomputed_tokens}"
        )
    if len(predictions) != manifest.n_predictions:
        problems.append(
            f"n_predictions {manifest.n_predictions} != {len(predictions)} rows on disk"
        )
    return problems


def python_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
