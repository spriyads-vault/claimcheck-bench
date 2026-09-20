"""Run orchestration: config, evaluator construction, append-only run artifacts."""

from __future__ import annotations

import concurrent.futures
import json
import platform
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .costs import PriceManifest, cost_usd, load_prices
from .evaluators.base import assert_clean_payload
from .evaluators.rules import RulesEvaluator
from .evaluators.tfidf import TfidfEvaluator
from .hashing import canonical_json, git_commit, sha256_file, sha256_obj
from .redact import contains_secret, redact
from .retry import ErrorRateGuard, RetryPolicy
from .schemas import Prediction, RunManifest, Splits, TraceRecord

OFFLINE_PROVIDERS = frozenset({"rules", "tfidf"})


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
    """Artifact paths for one dataset stage, resolved from the record count."""

    records: int
    dataset: Path
    splits: Path
    hashes: Path
    audit: Path
    audit_blind: Path
    runs_root: Path


def resolve_dataset_paths(config: EvalConfig, records: int | None = None) -> DatasetPaths:
    """Substitute ``{records}`` so each stage owns its own artifacts."""
    section = config.section("dataset")
    count = records if records is not None else int(section["records"])

    def path(key: str) -> Path:
        return Path(str(section[key]).format(records=count))

    return DatasetPaths(
        records=count,
        dataset=path("output"),
        splits=path("splits"),
        hashes=path("hashes"),
        audit=path("audit"),
        audit_blind=path("audit_blind"),
        runs_root=path("runs_root"),
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
) -> RulesEvaluator | TfidfEvaluator:
    threshold = config.threshold_for(provider)
    if provider == "rules":
        return RulesEvaluator(threshold=threshold)
    if provider == "tfidf":
        evaluator = TfidfEvaluator(threshold=threshold)
        dev = select_split(records, splits, "dev")
        validation = select_split(records, splits, "validation")
        evaluator.fit(dev, validation, frozenset(splits.test))
        return evaluator
    raise ValueError(f"{provider!r} is not an offline provider")


def run_predictions(
    evaluator: Any,
    records: Sequence[TraceRecord],
    repeats: int,
    concurrency: int,
) -> list[Prediction]:
    """Run the evaluator over every trace, ``repeats`` times each.

    The evaluator is handed an InferenceView and nothing else. The payload is
    checked for ground-truth fields before it leaves this function.
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

    if concurrency <= 1:
        return [run_one(job) for job in jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(run_one, jobs))


def finalise_costs(predictions: Sequence[Prediction], prices: PriceManifest) -> list[Prediction]:
    """Recompute every cost from the dated manifest, so one code path owns it."""
    return [
        prediction.model_copy(
            update={"cost_usd": cost_usd(prices, prediction.model_id, prediction.usage)}
        )
        for prediction in predictions
    ]


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


def verify_run(run_dir: Path) -> list[str]:
    """Recompute checksums and cost arithmetic. Returns a list of problems."""
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
    if dataset_path.exists():
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
        expected_cost = cost_usd(prices, prediction.model_id, prediction.usage)
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
