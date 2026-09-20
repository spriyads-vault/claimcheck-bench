"""`jev-eval` command line.

Dry-run discipline: any command that spends money or touches the network prints
exactly what it will do and requires explicit confirmation before it runs.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .costs import load_prices
from .generate import (
    generate_records,
    load_dataset,
    make_splits,
    write_audit_files,
    write_dataset,
)
from .hashing import sha256_file
from .runner import (
    OFFLINE_PROVIDERS,
    build_offline_evaluator,
    finalise_costs,
    load_config,
    load_splits,
    new_run_id,
    resolve_dataset_paths,
    run_predictions,
    select_split,
    verify_run,
    write_run,
)
from .schemas import LABEL_ORDER, FaultType

app = typer.Typer(
    add_completion=False,
    help="Offline-first evaluation harness for false-success detection in agent traces.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "config/eval.yaml"


def _echo(message: str) -> None:
    typer.echo(message)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the harness version."""
    _echo(__version__)


@app.command()
def generate(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to eval.yaml."),
    seed: int | None = typer.Option(None, help="Override the config seed."),
    records: int | None = typer.Option(None, help="Override the record count."),
) -> None:
    """Build the deterministic dataset, splits, hashes and audit sheets."""
    cfg = load_config(config)
    split_cfg = cfg.section("splits")
    seed_value = seed if seed is not None else cfg.seed
    paths = resolve_dataset_paths(cfg, records)

    _echo(f"Generating {paths.records} records with seed {seed_value} ...")
    built = generate_records(paths.records, seed_value)

    dataset_sha = write_dataset(built, paths.dataset)
    splits = make_splits(
        built,
        seed_value,
        int(split_cfg["dev_families"]),
        int(split_cfg["validation_families"]),
        int(split_cfg["test_families"]),
        dataset_sha,
    )
    paths.splits.write_text(
        json.dumps(splits.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_audit_files(built, paths.audit, paths.audit_blind)

    hashes = {
        paths.dataset.name: dataset_sha,
        paths.splits.name: sha256_file(paths.splits),
        paths.audit.name: sha256_file(paths.audit),
        paths.audit_blind.name: sha256_file(paths.audit_blind),
        "seed": seed_value,
        "records": paths.records,
    }
    paths.hashes.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _echo(f"  dataset      {paths.dataset}  sha256 {dataset_sha}")
    _echo(
        f"  splits       {paths.splits}  dev={len(splits.dev)} "
        f"validation={len(splits.validation)} test={len(splits.test)} families"
    )
    _echo(f"  audit key    {paths.audit}")
    _echo(f"  audit sheet  {paths.audit_blind}  (blinded; fill auditor_label in)")
    _echo("")
    _echo("Next: the blinded audit is a required step before the report is complete.")


@app.command("validate-data")
def validate_data(
    config: str = typer.Option(DEFAULT_CONFIG),
    records: int | None = typer.Option(None, help="Which stage's dataset to validate."),
    dataset: str | None = typer.Option(
        None, "--input", help="Dataset path override (defaults to the stage's dataset)."
    ),
) -> None:
    """Check schema, label balance, fault coverage and split leakage."""
    cfg = load_config(config)
    paths = resolve_dataset_paths(cfg, records)
    dataset_path = Path(dataset) if dataset else paths.dataset
    if not dataset_path.exists():
        _fail(f"{dataset_path} not found. Run `jev-eval generate` first.")

    built = load_dataset(dataset_path)
    splits = load_splits(paths.splits)
    problems: list[str] = []

    actual_sha = sha256_file(dataset_path)
    if actual_sha != splits.dataset_sha256:
        problems.append(
            f"{paths.splits} records dataset sha256 {splits.dataset_sha256} but the file is "
            f"{actual_sha}; regenerate."
        )

    counts = Counter(r.label.value for r in built)
    target = len(built) // len(LABEL_ORDER)
    for label in LABEL_ORDER:
        if counts.get(label.value, 0) != target:
            problems.append(
                f"label {label.value}: {counts.get(label.value, 0)} records, expected {target}"
            )

    seen_faults = {r.fault_type for r in built}
    for fault in FaultType:
        if fault not in seen_faults:
            problems.append(f"fault type never generated: {fault.value}")

    dev, validation, test = set(splits.dev), set(splits.validation), set(splits.test)
    for a_name, a, b_name, b in (
        ("dev", dev, "validation", validation),
        ("dev", dev, "test", test),
        ("validation", validation, "test", test),
    ):
        overlap = a & b
        if overlap:
            problems.append(f"family leakage between {a_name} and {b_name}: {sorted(overlap)}")

    families = {r.template_family for r in built}
    uncovered = families - (dev | validation | test)
    if uncovered:
        problems.append(f"families in no split: {sorted(uncovered)}")

    per_family: dict[str, set[str]] = {}
    for record in built:
        per_family.setdefault(record.template_family, set()).add(record.label.value)
    homogeneous = sorted(f for f, labels in per_family.items() if len(labels) < 2)
    if homogeneous:
        problems.append(
            f"label-homogeneous families (a family split would become a label split): {homogeneous}"
        )

    ids = [r.trace_id for r in built]
    if len(set(ids)) != len(ids):
        problems.append("duplicate trace_id values")

    if problems:
        for problem in problems:
            typer.secho(f"  FAIL  {problem}", fg=typer.colors.RED, err=True)
        _fail(f"{len(problems)} validation problem(s).")

    _echo(f"OK  {len(built)} records, {len(families)} families, {target} per label.")
    _echo(f"OK  splits disjoint: dev={len(dev)} validation={len(validation)} test={len(test)}")
    _echo(f"OK  all {len(list(FaultType))} fault types present, every family spans >1 label.")
    _echo(f"OK  dataset sha256 {actual_sha}")


def _build_jev(cfg: Any, threshold: float) -> Any:
    from .evaluators.jev_http import JevEvaluator

    jev_cfg = cfg.section("jev")
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        _fail(
            "TYPESAFE_API_KEY is not set.\n"
            "  Export it in your shell:  export TYPESAFE_API_KEY=...\n"
            "  This harness never accepts a key as a CLI argument and never reads one "
            "from a committed file."
        )
    questions = json.loads(Path(jev_cfg["questions"]).read_text(encoding="utf-8"))
    return JevEvaluator(
        api_key=api_key,
        model_id=str(jev_cfg["model"]),
        base_url=str(jev_cfg["base_url"]),
        path=str(jev_cfg["path"]),
        questions=questions,
        evaluation_rule=str(jev_cfg["evaluation_rule"]),
        prices=load_prices(cfg.raw["prices"]),
        policy=cfg.retry_policy(),
        threshold=threshold,
        timeout_s=float(jev_cfg.get("request_timeout_s", 60.0)),
        guard=cfg.guard(),
    )


@app.command()
def smoke(
    config: str = typer.Option(DEFAULT_CONFIG),
    provider: str = typer.Option("jev", help="Only 'jev' is supported."),
    model: str | None = typer.Option(None, help="Model ID override."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Send ONE tiny non-scored request to prove connectivity and credentials."""
    if provider != "jev":
        _fail("smoke only supports --provider jev")
    cfg = load_config(config)
    if model:
        cfg.raw["jev"]["model"] = model
    evaluator = _build_jev(cfg, cfg.threshold_for("jev"))

    jev_cfg = cfg.section("jev")
    _echo("This will make ONE live, billable request:")
    _echo(f"  POST {jev_cfg['base_url']}{jev_cfg['path']}")
    _echo(f"  model={jev_cfg['model']}  questions=1 noul  state=~10 tokens")
    _echo("  No dataset trace is sent and no result is scored.")
    if not yes and not typer.confirm("Proceed?", default=False):
        _echo("Aborted. Nothing was sent.")
        raise typer.Exit(code=0)

    result = evaluator.smoke()
    evaluator.close()
    _echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def run(
    config: str = typer.Option(DEFAULT_CONFIG),
    provider: str = typer.Option(..., help="rules | tfidf | jev"),
    split: str = typer.Option("test", help="dev | validation | test"),
    model: str | None = typer.Option(None, help="Model ID override (jev only)."),
    repeats: int = typer.Option(1, min=1, help="Repeats per trace."),
    concurrency: str = typer.Option("1", help="Comma-separated sweep, e.g. 1,5,10,20."),
    records: int | None = typer.Option(None, help="Which stage's dataset to run against."),
    runs_dir: str | None = typer.Option(
        None, help="Where run directories go (defaults to the stage's runs root)."
    ),
    confirm_paid: bool = typer.Option(
        False, "--confirm-paid", help="Required for any provider that spends money."
    ),
) -> None:
    """Run a provider over a split. Offline providers need no key and no network."""
    cfg = load_config(config)
    paths = resolve_dataset_paths(cfg, records)
    dataset_path, splits_path = paths.dataset, paths.splits
    runs_root = Path(runs_dir) if runs_dir else paths.runs_root
    if not dataset_path.exists():
        _fail(f"{dataset_path} not found. Run `jev-eval generate` first.")

    all_records = load_dataset(dataset_path)
    splits = load_splits(splits_path)
    selected = select_split(all_records, splits, split)
    if not selected:
        _fail(f"split {split!r} selected no records")

    threshold = cfg.threshold_for(provider)
    arms = [int(c) for c in concurrency.split(",") if c.strip()]
    if provider in OFFLINE_PROVIDERS and arms != [1]:
        _echo(f"note: {provider} runs locally; the concurrency sweep is ignored.")
        arms = [1]

    questions: dict[str, Any] | None = None
    if provider == "jev":
        n_requests = len(selected) * repeats * len(arms)
        jev_cfg = cfg.section("jev")
        _echo("PAID RUN PLAN")
        _echo(f"  endpoint     POST {jev_cfg['base_url']}{jev_cfg['path']}")
        _echo(f"  model        {model or jev_cfg['model']}")
        _echo(f"  split        {split} ({len(selected)} traces)")
        _echo(f"  repeats      {repeats}")
        _echo(f"  concurrency  {arms}")
        _echo(f"  requests     {n_requests} (one per trace per repeat per arm)")
        _echo(f"  pricing      from {cfg.raw['prices']} (input tokens only)")
        _echo("  cost         unknown until tokens are returned; charged per input token.")
        if not confirm_paid:
            _fail("Refusing to spend money without --confirm-paid. Nothing was sent.")
        if model:
            cfg.raw["jev"]["model"] = model
        questions = json.loads(Path(jev_cfg["questions"]).read_text(encoding="utf-8"))

    prices = load_prices(cfg.raw["prices"])

    for arm in arms:
        if provider in OFFLINE_PROVIDERS:
            evaluator = build_offline_evaluator(provider, cfg, all_records, splits)
        elif provider == "jev":
            evaluator = _build_jev(cfg, threshold)
        else:
            _fail(f"unknown provider {provider!r}")

        predictions = finalise_costs(run_predictions(evaluator, selected, repeats, arm), prices)
        run_id = new_run_id(provider, split)
        manifest = write_run(
            run_dir=runs_root / run_id,
            run_id=run_id,
            provider=provider,
            model_id=getattr(evaluator, "model_id", provider),
            split=split,
            repeats=repeats,
            concurrency=arm,
            threshold=threshold,
            records=selected,
            predictions=predictions,
            config=cfg,
            splits_path=splits_path,
            dataset_path=dataset_path,
            questions=questions,
            prices=prices,
            repo_root=REPO_ROOT,
        )
        if hasattr(evaluator, "close"):
            evaluator.close()
        _echo(
            f"{manifest.run_id}: {manifest.n_predictions} predictions, "
            f"{manifest.error_count} errors, {manifest.retry_count} retries, "
            f"cost {manifest.total_cost_usd}"
        )


@app.command()
def report(
    config: str = typer.Option(DEFAULT_CONFIG),
    records: int | None = typer.Option(None, help="Which stage to report on."),
    runs: str | None = typer.Option(None, help="Runs root (defaults to the stage's)."),
    out: str = typer.Option("reports/latest", help="Output directory."),
    split: str = typer.Option("test", help="Which split to report on."),
) -> None:
    """Build tables, plots, results.json and report.md from run artifacts."""
    from .report import build_report

    cfg = load_config(config)
    paths = resolve_dataset_paths(cfg, records)
    runs_root = Path(runs) if runs else paths.runs_root
    if not paths.dataset.exists():
        _fail(f"{paths.dataset} not found. Run `jev-eval generate` first.")
    loaded = load_dataset(paths.dataset)

    results = build_report(
        runs_root=runs_root,
        out_dir=Path(out),
        records=loaded,
        splits_path=paths.splits,
        dataset_path=paths.dataset,
        dataset_sha256=sha256_file(paths.dataset),
        config=cfg,
        split=split,
        audit_key=paths.audit,
        audit_blind=paths.audit_blind,
        n_records_total=len(loaded),
    )
    _echo(f"Wrote {out}/report.md, {out}/results.json and 3 plots.")
    _echo(f"Providers reported: {', '.join(sorted(results['runs']))}")
    _echo(f"Audit status: {results['audit']['status']}")


@app.command("verify-run")
def verify_run_command(
    run: str = typer.Option(..., help="A run directory containing manifest.json."),
) -> None:
    """Recompute checksums and cost arithmetic from a run manifest."""
    run_dir = Path(run)
    if not (run_dir / "manifest.json").exists():
        _fail(f"{run_dir}/manifest.json not found")
    problems = verify_run(run_dir)
    if problems:
        for problem in problems:
            typer.secho(f"  FAIL  {problem}", fg=typer.colors.RED, err=True)
        _fail(f"verify-run failed with {len(problems)} problem(s) for {run_dir}")
    _echo(f"OK  {run_dir}: checksums and cost arithmetic verified against the price manifest.")


@app.command("verify-runs")
def verify_runs_command(
    config: str = typer.Option(DEFAULT_CONFIG),
    records: int | None = typer.Option(None, help="Which stage's runs to verify."),
    runs: str | None = typer.Option(None, help="Runs root (defaults to the stage's)."),
) -> None:
    """Verify every run directory under a root."""
    from .runner import discover_runs

    root = Path(runs) if runs else resolve_dataset_paths(load_config(config), records).runs_root
    found = discover_runs(root)
    if not found:
        _fail(f"no runs found under {root}")
    failures = 0
    for run_dir in found:
        problems = verify_run(run_dir)
        if problems:
            failures += 1
            typer.secho(f"FAIL  {run_dir}", fg=typer.colors.RED, err=True)
            for problem in problems:
                typer.secho(f"      {problem}", fg=typer.colors.RED, err=True)
        else:
            _echo(f"OK    {run_dir}")
    if failures:
        _fail(f"{failures} of {len(found)} runs failed verification")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
