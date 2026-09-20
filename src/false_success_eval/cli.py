"""`jev-eval` command line.

Dry-run discipline: any command that spends money or touches the network prints
exactly what it will do and requires explicit confirmation before it runs.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .budget import load_budget
from .costs import CostProjection, load_prices, project_cost
from .generate import (
    generate_records,
    load_dataset,
    make_splits,
    write_audit_files,
    write_dataset,
)
from .hashing import sha256_file
from .learning_curve import CURVE_PROVIDERS, DEFAULT_SEEDS, TRAIN_SIZES, size_label
from .runner import (
    OFFLINE_PROVIDERS,
    PAID_PROVIDERS,
    RunWriter,
    available_corpora,
    build_offline_evaluator,
    load_config,
    load_splits,
    new_run_id,
    resolve_dataset_paths,
    run_predictions,
    select_split,
    verify_run,
)
from .schemas import LABEL_ORDER, REAL_FAULTS, SYNTHETIC_FAULTS, TraceRecord

app = typer.Typer(
    add_completion=False,
    help="Offline-first evaluation harness for false-success detection in agent traces.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "config/eval.yaml"
CORPUS_HELP = (
    "An ingested real corpus id instead of the synthetic stage (e.g. 'appworld'). "
    "Mutually exclusive with --records."
)


def _paths(cfg: Any, records: int | None, corpus: str | None) -> Any:
    """Resolve dataset paths, refusing an ambiguous selection rather than guessing."""
    if corpus and records is not None:
        _fail(
            "--corpus and --records select different datasets; pass one, not both. "
            "A real corpus has its own artifact namespace and its own record count."
        )
    try:
        return resolve_dataset_paths(cfg, records, corpus)
    except KeyError as exc:
        _fail(str(exc).strip("'"))
        raise


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
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    dataset: str | None = typer.Option(
        None, "--input", help="Dataset path override (defaults to the stage's dataset)."
    ),
) -> None:
    """Check schema, label balance, fault coverage and split leakage.

    Two of these checks mean different things on the two kinds of dataset, so
    they are applied differently rather than applied uniformly and quietly
    failing. Label balance is a *generator contract*: the synthetic set is built
    four ways equal, and a real corpus has whatever prevalence it has. A
    label-homogeneous split unit is a *generation bug* in synthetic data and
    merely a fact about a real corpus. Both are still reported either way; only
    whether they fail the command changes.
    """
    cfg = load_config(config)
    paths = _paths(cfg, records, corpus)
    dataset_path = Path(dataset) if dataset else paths.dataset
    if not dataset_path.exists():
        hint = (
            f"Run `jev-eval ingest --corpus {corpus}` first."
            if corpus
            else "Run `jev-eval generate` first."
        )
        _fail(f"{dataset_path} not found. {hint}")

    built = load_dataset(dataset_path)
    splits = load_splits(paths.splits)
    problems: list[str] = []
    notes: list[str] = []

    actual_sha = sha256_file(dataset_path)
    if actual_sha != splits.dataset_sha256:
        problems.append(
            f"{paths.splits} records dataset sha256 {splits.dataset_sha256} but the file is "
            f"{actual_sha}; regenerate."
        )

    counts = Counter(r.label.value for r in built)
    if paths.is_real:
        notes.append(
            "label prevalence "
            + ", ".join(f"{label.value}={counts.get(label.value, 0)}" for label in LABEL_ORDER)
            + " (a real corpus is not balanced; this is the data, not a defect)"
        )
    else:
        target = len(built) // len(LABEL_ORDER)
        for label in LABEL_ORDER:
            if counts.get(label.value, 0) != target:
                problems.append(
                    f"label {label.value}: {counts.get(label.value, 0)} records, expected {target}"
                )

    expected_faults = REAL_FAULTS if paths.is_real else SYNTHETIC_FAULTS
    seen_faults = {r.fault_type.value for r in built}
    for fault in sorted(expected_faults):
        if fault not in seen_faults:
            problems.append(f"fault type never present: {fault}")
    foreign = sorted(seen_faults - expected_faults)
    if foreign:
        problems.append(
            f"fault types from the other dataset kind are present: {foreign}. A "
            "synthetic fault in a real corpus (or the reverse) means the two have "
            "been mixed."
        )

    dev, validation, test = set(splits.dev), set(splits.validation), set(splits.test)
    for a_name, a, b_name, b in (
        ("dev", dev, "validation", validation),
        ("dev", dev, "test", test),
        ("validation", validation, "test", test),
    ):
        overlap = a & b
        if overlap:
            problems.append(f"split leakage between {a_name} and {b_name}: {sorted(overlap)}")

    families = {r.template_family for r in built}
    uncovered = families - (dev | validation | test)
    if uncovered:
        problems.append(f"split units in no split: {sorted(uncovered)}")

    per_family: dict[str, set[str]] = {}
    for record in built:
        per_family.setdefault(record.template_family, set()).add(record.label.value)
    homogeneous = sorted(f for f, labels in per_family.items() if len(labels) < 2)
    if homogeneous:
        message = (
            f"label-homogeneous split units (a unit split would become a label split): "
            f"{len(homogeneous)} of {len(per_family)}"
        )
        if paths.is_real:
            notes.append(
                message + " -- expected on real data, where a scenario can happen to "
                "draw one label; it is a property of the corpus, not a defect"
            )
        else:
            problems.append(f"{message}: {homogeneous}")

    ids = [r.trace_id for r in built]
    if len(set(ids)) != len(ids):
        problems.append("duplicate trace_id values")

    if problems:
        for problem in problems:
            typer.secho(f"  FAIL  {problem}", fg=typer.colors.RED, err=True)
        _fail(f"{len(problems)} validation problem(s).")

    _echo(f"OK  {paths.dataset_id}: {len(built)} records, {len(families)} split units.")
    _echo(f"OK  splits disjoint: dev={len(dev)} validation={len(validation)} test={len(test)}")
    _echo(f"OK  all {len(expected_faults)} {paths.kind} fault types present, none foreign.")
    if not paths.is_real:
        _echo("OK  every family spans >1 label, so a family split cannot become a label split.")
    _echo(f"OK  dataset sha256 {actual_sha}")
    for note in notes:
        _echo(f"    note: {note}")


@app.command()
def ingest(
    config: str = typer.Option(DEFAULT_CONFIG),
    corpus: str = typer.Option("appworld", help="Which corpus to ingest."),
    source: str | None = typer.Option(
        None,
        "--source",
        help="An already-unpacked release directory. Skips the download entirely.",
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Fetch the published release first. Requires the source project's own package.",
    ),
    cache: str = typer.Option(
        ".cache/ingest", help="Where a downloaded release is cached and unpacked."
    ),
    seed: int | None = typer.Option(None, help="Override the config seed for the split."),
) -> None:
    """Ingest a real, labelled corpus. The licence is verified before anything is read.

    Nothing about a corpus is assumed. If it has no verified licence on record,
    or one that does not permit this use, the command refuses and explains --
    it does not download, does not read and does not write.
    """
    from .ingest.licences import LICENCES, LicenceError, require_licence
    from .ingest.pipeline import ingest_appworld

    cfg = load_config(config)
    corpora = available_corpora(cfg)
    if corpus not in corpora:
        _fail(
            f"unknown corpus {corpus!r}. Declared in config: {sorted(corpora) or 'none'}. "
            f"Licences on record: {sorted(LICENCES)}."
        )
    paths = _paths(cfg, None, corpus)

    _echo(f"Corpus {corpus}")
    try:
        licence = require_licence(corpus)
    except LicenceError as exc:
        _fail(str(exc))
        return
    _echo(f"  licence      {licence.name} ({licence.spdx})  -- permits this use")
    _echo(f"  verified     {licence.verified_utc} from {licence.verified_from}")
    for condition in licence.conditions:
        _echo(f"  condition    {condition}")
    _echo("")

    if corpus != "appworld":
        _fail(
            f"no ingest adapter is implemented for {corpus!r} yet. Its licence is on "
            "record, but a licence is permission, not a reader."
        )

    if source:
        root = Path(source)
    elif download:
        from .ingest.appworld import download_and_unpack

        _echo(f"  downloading  the published release into {cache} ...")
        root = download_and_unpack(Path(cache))
    else:
        _fail(
            "pass --source <unpacked release directory> or --download.\n"
            "  This command does not reach the network unless you ask it to."
        )
        return

    _echo(f"  source       {root}")
    result = ingest_appworld(
        source=root,
        dataset_path=paths.dataset,
        splits_path=paths.splits,
        provenance_path=paths.provenance,
        seed=seed if seed is not None else cfg.seed,
    )

    hashes = {
        paths.dataset.name: result.splits.dataset_sha256,
        paths.splits.name: sha256_file(paths.splits),
        paths.provenance.name: sha256_file(paths.provenance),
        "split_sha256": result.provenance.split_sha256,
        "source_sha256": result.provenance.source_sha256,
        "records": result.provenance.n_records,
    }
    paths.hashes.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rule = result.provenance.label_rule
    _echo("")
    _echo("LABEL RULE  (derived from ground truth, never from the assistant's text)")
    if rule is not None:
        _echo(f"  claim        {rule.claim_field}")
        _echo(f"  truth        {rule.truth_field}")
        for condition, label in rule.mapping:
            _echo(f"    {condition}  ->  {label}")
    _echo("")
    _echo(f"  dataset      {paths.dataset}  sha256 {result.splits.dataset_sha256}")
    _echo(f"  records      {result.provenance.n_records}")
    for label in LABEL_ORDER:
        count = result.provenance.label_counts.get(label.value, 0)
        share = count / max(1, result.provenance.n_records)
        _echo(f"    {label.value:<34} {count:>6}  ({share:.1%})")
    _echo(f"  split unit   {result.provenance.split_unit}")
    _echo(
        f"  splits       dev={len(result.splits.dev)} "
        f"validation={len(result.splits.validation)} test={len(result.splits.test)} units, "
        f"task-disjoint"
    )
    _echo(f"  split hash   {result.provenance.split_sha256}  (frozen)")
    _echo(f"  dropped      {result.stats.as_dict()}")
    _echo(f"  provenance   {paths.provenance}")
    _echo("")
    _echo("Nothing was published. The ingested corpus is git-ignored.")


def _project_run_cost(
    cfg: Any,
    *,
    model_id: str,
    records: Sequence[TraceRecord],
    questions: dict[str, Any],
    n_requests: int,
    prices: Any,
    provider: str = "jev",
) -> CostProjection:
    """Bracket what a paid run will cost, from the bodies it will actually send.

    The request bodies are built here exactly as the adapter builds them, so the
    projection is measured against the real payloads rather than an assumed
    size. Only the tokenizer is unknown, and that is what the band covers.

    ``provider`` selects which adapter's body shape is measured. A general judge
    wraps the same ``state`` in a chat envelope and carries the questions as
    prose plus a JSON schema, so its bodies are a different size from Jev's even
    though the trace inside them is identical -- projecting one from the other
    would misstate the bill for whichever lane was not measured.
    """
    from .budget import fit_events, longest_question_chars
    from .evaluators.jev_http import state_envelope_chars

    jev_cfg = cfg.section("jev")
    evaluation_rule = str(jev_cfg["evaluation_rule"])
    budget = load_budget(cfg.section("context_budget"))
    question_chars = longest_question_chars(questions)

    build_body = _body_builder(cfg, provider, model_id, questions, evaluation_rule)
    per_pass_chars = 0
    truncated = 0
    for record in records:
        view = record.inference_view()
        # Price what will actually be sent. A trace that exceeds the context
        # budget is reduced before it goes out, so projecting from the full
        # trace would over-state the bill -- and a projection that is wrong in
        # the safe direction is still wrong.
        events, truncation = fit_events(
            record.trace_id,
            view.events,
            budget=budget,
            envelope_chars=state_envelope_chars(view, evaluation_rule),
            longest_question_chars=question_chars,
        )
        if truncation is not None:
            truncated += 1
            view = view.model_copy(update={"events": events})
        per_pass_chars += len(json.dumps(build_body(view), separators=(",", ":"), sort_keys=True))

    passes = n_requests / len(records) if records else 0.0
    projection_cfg = cfg.section("cost_projection")
    return project_cost(
        prices,
        model_id,
        request_chars=round(per_pass_chars * passes),
        n_requests=n_requests,
        chars_per_token_low=float(projection_cfg["chars_per_token_low"]),
        chars_per_token_high=float(projection_cfg["chars_per_token_high"]),
        fixed_overhead_tokens=int(projection_cfg["fixed_overhead_tokens"]),
        # Jev is billed on input alone, so its output ceiling is zero and this
        # term vanishes. A general judge is billed for what it writes too.
        output_tokens_per_request=(
            0 if provider == "jev" else int(projection_cfg.get("output_tokens_per_request", 0))
        ),
    )


def _body_builder(
    cfg: Any, provider: str, model_id: str, questions: dict[str, Any], evaluation_rule: str
) -> Any:
    """The request-body function for a provider, as its adapter builds it."""
    from .evaluators.jev_http import build_state

    if provider == "jev":

        def jev_body(view: Any) -> dict[str, Any]:
            return {
                "state": build_state(view, evaluation_rule),
                "model": model_id,
                "questions": questions,
            }

        return jev_body

    from .evaluators.general_model import build_chat_body, build_instructions, response_schema

    vendor = _vendor_config(cfg, provider)
    instructions = build_instructions(questions, evaluation_rule)
    schema = response_schema(questions)

    def judge_body(view: Any) -> dict[str, Any]:
        # The adapter's own builder, so the projection cannot drift from what
        # actually goes out -- and no key and no connection are needed to call it.
        return build_chat_body(
            view,
            model_id=model_id,
            instructions=instructions,
            schema=schema,
            evaluation_rule=evaluation_rule,
            temperature=float(vendor.temperature),
            max_output_tokens=int(vendor.max_output_tokens),
        )

    return judge_body


def _vendor_config(cfg: Any, provider: str) -> Any:
    """The ``general_model`` block for a vendor, refusing an unknown one."""
    from .evaluators.general_model import GeneralModelConfig

    block = cfg.section("general_model").get(provider)
    if not isinstance(block, dict):
        _fail(
            f"no general_model block named {provider!r} in {cfg.path}. Declared vendors: "
            f"{sorted(cfg.section('general_model'))}."
        )
        raise AssertionError("unreachable")
    return GeneralModelConfig.from_mapping(provider, block)


def _traces_over_budget(cfg: Any, records: Sequence[TraceRecord], questions: dict[str, Any]) -> int:
    """How many traces the context budget will reduce, counted before spending."""
    from .budget import fit_events, longest_question_chars
    from .evaluators.jev_http import state_envelope_chars

    budget = load_budget(cfg.section("context_budget"))
    evaluation_rule = str(cfg.section("jev")["evaluation_rule"])
    question_chars = longest_question_chars(questions)
    count = 0
    for record in records:
        view = record.inference_view()
        _, truncation = fit_events(
            record.trace_id,
            view.events,
            budget=budget,
            envelope_chars=state_envelope_chars(view, evaluation_rule),
            longest_question_chars=question_chars,
        )
        count += truncation is not None
    return count


def _format_projection(projection: CostProjection, cfg: Any) -> list[str]:
    """Render the pre-spend projection. A band, labelled as an estimate."""
    fx = float(cfg.raw["fx_rate_gbp_usd"])
    fx_date = str(cfg.raw["fx_rate_date"])
    if projection.usd_low is None or projection.usd_high is None:
        return [
            "PROJECTED COST",
            f"  price manifest   {projection.prices_path} (dated {projection.manifest_date})",
            f"  NO VERIFIED RATE for {projection.model_id}. Cost cannot be projected, and "
            "every cost this run records would be null.",
            "  The run will still record input and output tokens, so the bill can be",
            "  reconstructed later from a rate added with a citation. Nothing is guessed.",
        ]
    if projection.output_tokens_total:
        rate_line = (
            f"  rate             ${projection.input_usd_per_mtok} per Mtok input, "
            f"${projection.output_usd_per_mtok} per Mtok output"
        )
        output_line = (
            f"  output ceiling   {projection.output_tokens_total:,} tokens "
            f"({projection.output_tokens_per_request:,} per request, the schema-bounded "
            "maximum; priced in full)"
        )
    else:
        rate_line = (
            f"  rate             ${projection.input_usd_per_mtok} per Mtok input; "
            "output tokens free under this manifest"
        )
        output_line = ""
    lines = [
        "PROJECTED COST  (estimate, not a quote -- nothing has been sent yet)",
        f"  price manifest   {projection.prices_path} (dated {projection.manifest_date})",
        rate_line,
        f"  request bodies   {projection.request_chars:,} chars across "
        f"{projection.n_requests:,} requests",
        f"  token band       {projection.input_tokens_low:,} - "
        f"{projection.input_tokens_high:,} input tokens",
        f"                   ({projection.chars_per_token_high} to "
        f"{projection.chars_per_token_low} chars/token, plus "
        f"{projection.fixed_overhead_tokens} overhead tokens per request)",
    ]
    if output_line:
        lines.append(output_line)
    lines += [
        f"  PROJECTED USD    ${projection.usd_low:.4f} - ${projection.usd_high:.4f}",
        f"  PROJECTED GBP    £{projection.usd_low * fx:.4f} - £{projection.usd_high * fx:.4f}  "
        f"(fixed rate {fx} GBP/USD, {fx_date}; not fetched)",
        "  No tokenizer is assumed, so the exact input-token count cannot be known",
        "  before the request is sent. The recorded cost after the run is computed",
        "  from the tokens the API actually returned, never from this projection.",
    ]
    return lines


def _spend_gate(projection: CostProjection, cfg: Any, accepted: bool) -> None:
    """Refuse a projection above the configured limit unless it was accepted.

    Checked against the TOP of the band. The point of a threshold is to be
    asked before a surprise, and the optimistic end of an estimate is not the
    number worth being asked about.
    """
    limit = float(cfg.section("cost_projection").get("confirm_above_gbp", 0.0) or 0.0)
    if limit <= 0 or projection.usd_high is None:
        return
    fx = float(cfg.raw["fx_rate_gbp_usd"])
    top_gbp = projection.usd_high * fx
    if top_gbp <= limit:
        return
    if accepted:
        _echo(
            f"  NOTE: projected upper bound £{top_gbp:.2f} is above the £{limit:.2f} "
            "threshold, and --accept-cost-over-threshold was passed."
        )
        return
    _fail(
        f"STOPPED. The projected upper bound is £{top_gbp:.2f}, above the "
        f"£{limit:.2f} threshold in cost_projection.confirm_above_gbp.\n"
        "  Nothing was sent. Reduce the scope (fewer repeats, one concurrency arm, a "
        "smaller split), or re-run with --accept-cost-over-threshold if that figure "
        "is acceptable."
    )


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
        budget=load_budget(cfg.section("context_budget")),
    )


def _build_openai(cfg: Any, threshold: float, model: str | None = None) -> Any:
    """Construct the general-judge arm, refusing rather than guessing anything.

    The model ID comes from config (or ``--model``) and is never defaulted: an
    empty one is a refusal with the listing command in the message, because the
    only acceptable source for it is what the account itself reports.
    """
    from .evaluators.general_model import OpenAIChatJudge, not_run_reason

    vendor = _vendor_config(cfg, "openai")
    if model:
        vendor = replace(vendor, model_id=model)
    blocked = not_run_reason(vendor)
    if blocked is not None:
        hint = ""
        if not vendor.model_id:
            hint = (
                "\n  Run `jev-eval list-models --provider openai` and put the exact id it "
                "prints into general_model.openai.model_id, or pass --model.\n"
                "  No name is assumed here: a judge that silently called a different model "
                "than the one recorded would make the comparison meaningless."
            )
        elif vendor.api_key_env and not os.environ.get(vendor.api_key_env):
            hint = (
                f"\n  Export it in your shell:  export {vendor.api_key_env}=...\n"
                f"  or put {vendor.api_key_env}=... in .env at the repo root and source it.\n"
                "  This harness never accepts a key as a CLI argument and never reads one "
                "from a committed file."
            )
        _fail(f"{blocked}{hint}")

    jev_cfg = cfg.section("jev")
    questions = json.loads(Path(jev_cfg["questions"]).read_text(encoding="utf-8"))
    return OpenAIChatJudge(
        api_key=os.environ.get(vendor.api_key_env, ""),
        model_id=vendor.model_id,
        base_url=vendor.base_url,
        path=vendor.path,
        questions=questions,
        # The same rule Jev is given, from the same config key. If the two lanes
        # were told different things about what counts as evidence, the
        # comparison would be measuring the instruction, not the model.
        evaluation_rule=str(jev_cfg["evaluation_rule"]),
        prices=load_prices(cfg.raw["prices"]),
        policy=cfg.retry_policy(),
        threshold=threshold,
        temperature=vendor.temperature,
        max_output_tokens=vendor.max_output_tokens,
        timeout_s=vendor.request_timeout_s,
        guard=cfg.guard(),
        # The same budget object Jev gets, so both lanes see the same trace.
        budget=load_budget(cfg.section("context_budget")),
        api_key_env=vendor.api_key_env,
    )


@app.command("list-models")
def list_models_command(
    config: str = typer.Option(DEFAULT_CONFIG),
    provider: str = typer.Option("openai", help="Which general_model vendor to ask."),
    contains: str | None = typer.Option(None, help="Only show ids containing this substring."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """List the models this account actually has, from the vendor's own endpoint.

    This exists so no model name in this repository is ever invented. The id
    chosen for a scored run is copied from this output into
    ``general_model.<vendor>.model_id`` and recorded in the run manifest.

    The call is unmetered, but it does reach the network, so it says what it
    will do first like everything else here.
    """
    from .evaluators.general_model import MissingCredentialError, list_models

    cfg = load_config(config)
    vendor = _vendor_config(cfg, provider)
    if not vendor.base_url or not vendor.models_path:
        _fail(f"{provider} has no verified base_url/models_path in {config}.")
    key = os.environ.get(vendor.api_key_env, "")
    if not key:
        _fail(
            f"{vendor.api_key_env} is not set.\n"
            f"  Export it in your shell:  export {vendor.api_key_env}=...\n"
            f"  or put {vendor.api_key_env}=... in .env at the repo root and source it.\n"
            "  This harness never accepts a key as a CLI argument and never reads one "
            "from a committed file."
        )

    _echo("This will make ONE unmetered request to the vendor's model listing:")
    _echo(f"  GET {vendor.base_url}{vendor.models_path}")
    _echo(f"  auth from {vendor.api_key_env}; no trace and no dataset content is sent.")
    if not yes and not typer.confirm("Proceed?", default=False):
        _echo("Aborted. Nothing was sent.")
        raise typer.Exit(code=0)

    try:
        entries = list_models(api_key=key, base_url=vendor.base_url, models_path=vendor.models_path)
    except (MissingCredentialError, RuntimeError) as exc:
        _fail(str(exc))
        return

    rows = sorted(entries, key=lambda e: str(e.get("id", "")))
    if contains:
        rows = [e for e in rows if contains.lower() in str(e.get("id", "")).lower()]
    _echo("")
    suffix = f" matching {contains!r}" if contains else ""
    _echo(f"{len(rows)} model(s) available to this key{suffix}")
    for entry in rows:
        owner = str(entry.get("owned_by", ""))
        _echo(f"  {entry['id']:<44} owned_by={owner}")
    _echo("")
    _echo(
        f"Put the exact id into general_model.{provider}.model_id in {config}, or pass it "
        "to `run --model`. It is written into the run manifest either way."
    )


@app.command()
def smoke(
    config: str = typer.Option(DEFAULT_CONFIG),
    provider: str = typer.Option("jev", help="jev | openai"),
    model: str | None = typer.Option(None, help="Model ID override."),
    records: int | None = typer.Option(None, help="Which stage's dataset to draw from."),
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    split: str = typer.Option("test", help="Which split the smoke trace is drawn from."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Send ONE live request before committing to a run.

    For ``jev`` this is a tiny non-scored connectivity check, as it always was.

    For ``openai`` a connectivity check would prove nothing useful: what has to
    be known before a 702-request run is whether the strict schema comes back
    parseable and what one real trace actually costs. So the general-judge smoke
    sends exactly **one real trace** from the split, prints the tokens it was
    billed for, and then prints the full-split projection twice: the usual
    chars-per-token band, and a point estimate calibrated on what that one
    request measured. Neither is recorded as a cost. The result is not scored
    and no run directory is written.
    """
    if provider not in PAID_PROVIDERS:
        _fail(f"smoke supports {sorted(PAID_PROVIDERS)}, not {provider!r}")
    cfg = load_config(config)

    if provider == "jev":
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
        return

    _smoke_general_judge(
        cfg,
        config=config,
        provider=provider,
        model=model,
        records=records,
        corpus=corpus,
        split=split,
        yes=yes,
    )


def _smoke_general_judge(
    cfg: Any,
    *,
    config: str,
    provider: str,
    model: str | None,
    records: int | None,
    corpus: str | None,
    split: str,
    yes: bool,
) -> None:
    """One real trace through the general judge, then the calibrated projection."""
    paths = _paths(cfg, records, corpus)
    if not paths.dataset.exists():
        _fail(f"{paths.dataset} not found. Ingest or generate it first.")
    all_records = load_dataset(paths.dataset)
    selected = select_split(all_records, load_splits(paths.splits), split)
    if not selected:
        _fail(f"split {split!r} selected no records")

    # The first trace by id: deterministic, so the smoke is repeatable and the
    # figure it calibrates from is not a lucky draw of a short trace.
    one = min(selected, key=lambda r: r.trace_id)
    vendor = _vendor_config(cfg, provider)
    evaluator = _build_openai(cfg, cfg.threshold_for(provider), model)

    _echo("This will make ONE live, billable request:")
    _echo(f"  POST {vendor.base_url}{vendor.path}")
    _echo(f"  model        {evaluator.model_id}")
    _echo(f"  trace        {one.trace_id} (first by id in {paths.dataset_id}/{split})")
    _echo(f"  temperature  {vendor.temperature}, strict JSON schema, 4 questions")
    _echo("  The result is printed, not scored, and no run directory is written.")
    if not yes and not typer.confirm("Proceed?", default=False):
        _echo("Aborted. Nothing was sent.")
        evaluator.close()
        raise typer.Exit(code=0)

    prediction = evaluator.predict(one.inference_view(), one.trace_id)
    truncated = len(evaluator.truncations)
    deviations = list(evaluator.deviations)
    evaluator.close()

    _echo("")
    _echo("SMOKE RESULT")
    _echo(f"  model id reported   {prediction.model_id}")
    _echo(f"  decision            {prediction.decision.value}")
    for deviation in deviations:
        # Better learned here, for one request, than 702 requests into a run.
        _echo(
            f"  DEVIATION           {evaluator.model_id} refused "
            f"{deviation.parameter}={deviation.requested!r} "
            f"(HTTP {deviation.http_status} {deviation.error_code}); "
            f"it was {deviation.applied}."
        )
        _echo(f'                      provider said: "{deviation.error_message}"')
        _echo(
            "                      A run will record this in deviations.jsonl and the "
            "report will state it."
        )
    if prediction.error:
        _echo(f"  error               {prediction.error}")
        _fail(
            "The smoke request did not produce a parseable verdict. Nothing further "
            "was sent, and no projection is printed from a failed measurement."
        )
    predicted = prediction.predicted_label.value if prediction.predicted_label else "n/a"
    _echo(f"  predicted label     {predicted}")
    _echo(f"  P(unsupported)      {prediction.primary_score}")
    _echo(f"  latency             {prediction.end_to_end_latency_ms:.0f} ms")
    _echo(f"  attempts            {len(prediction.attempts)}")
    _echo(f"  truncated           {'yes' if truncated else 'no'}")
    input_tokens = prediction.usage.input_tokens if prediction.usage else 0
    output_tokens = prediction.usage.output_tokens if prediction.usage else 0
    _echo(f"  input tokens        {input_tokens:,}")
    _echo(f"  output tokens       {output_tokens:,}")
    _echo(
        "  cost                "
        + (
            f"${prediction.cost_usd:.6f}"
            if prediction.cost_usd is not None
            else "unavailable -- no verified rate for this model in the price manifest"
        )
    )

    questions = json.loads(Path(cfg.section("jev")["questions"]).read_text(encoding="utf-8"))
    n_requests = len(selected)
    projection = _project_run_cost(
        cfg,
        model_id=evaluator.model_id,
        records=selected,
        questions=questions,
        n_requests=n_requests,
        prices=load_prices(cfg.raw["prices"]),
        provider=provider,
    )
    _echo("")
    _echo(f"FULL-SPLIT PROJECTION  ({n_requests:,} traces, one pass, no repeats)")
    for line in _format_projection(projection, cfg):
        _echo(line)

    _echo("")
    for line in _calibrated_projection(
        cfg,
        projection=projection,
        measured_input_tokens=input_tokens,
        measured_output_tokens=output_tokens,
        smoke_body_chars=_body_chars(cfg, provider, evaluator.model_id, questions, one),
    ):
        _echo(line)
    _echo("")
    _echo(
        "Nothing above is recorded as a cost. Every recorded cost comes from the "
        "tokens the API returned on a real scored run."
    )


def _body_chars(
    cfg: Any, provider: str, model_id: str, questions: dict[str, Any], record: TraceRecord
) -> int:
    """Compact-JSON length of the body the smoke actually sent, for calibration."""
    from .budget import fit_events, longest_question_chars
    from .evaluators.jev_http import state_envelope_chars

    evaluation_rule = str(cfg.section("jev")["evaluation_rule"])
    budget = load_budget(cfg.section("context_budget"))
    view = record.inference_view()
    events, truncation = fit_events(
        record.trace_id,
        view.events,
        budget=budget,
        envelope_chars=state_envelope_chars(view, evaluation_rule),
        longest_question_chars=longest_question_chars(questions),
    )
    if truncation is not None:
        view = view.model_copy(update={"events": events})
    build = _body_builder(cfg, provider, model_id, questions, evaluation_rule)
    return len(json.dumps(build(view), separators=(",", ":"), sort_keys=True))


def _calibrated_projection(
    cfg: Any,
    *,
    projection: CostProjection,
    measured_input_tokens: int,
    measured_output_tokens: int,
    smoke_body_chars: int,
) -> list[str]:
    """Project the split from the tokens one real request was actually billed.

    The band above assumes a chars-per-token range because no tokenizer is
    assumed. This does something narrower and more useful: it measures the ratio
    on one real body and scales it. It is still an estimate -- one trace is one
    trace, and traces vary -- so it is printed as a point estimate beside the
    band rather than instead of it, and it is never recorded.
    """
    if smoke_body_chars <= 0 or measured_input_tokens <= 0:
        return ["CALIBRATED PROJECTION", "  not available: the smoke measured no tokens."]
    chars_per_token = smoke_body_chars / measured_input_tokens
    scaled_input = round(projection.request_chars / chars_per_token)
    scaled_output = measured_output_tokens * projection.n_requests
    lines = [
        "CALIBRATED PROJECTION  (from the one request just sent; a point estimate, not a quote)",
        f"  measured         {measured_input_tokens:,} input + "
        f"{measured_output_tokens:,} output tokens for {smoke_body_chars:,} body chars",
        f"  implied ratio    {chars_per_token:.2f} chars/token "
        f"(the band above assumed {projection.chars_per_token_high} to "
        f"{projection.chars_per_token_low})",
        f"  scaled to split  {scaled_input:,} input + {scaled_output:,} output tokens",
    ]
    rate_in = projection.input_usd_per_mtok
    rate_out = projection.output_usd_per_mtok
    if rate_in is None or rate_out is None:
        lines.append(
            "  COST             unavailable -- this model has no verified rate in the "
            "price manifest, so tokens are projected and money is not."
        )
        return lines
    usd = (scaled_input * rate_in + scaled_output * rate_out) / 1_000_000
    fx = float(cfg.raw["fx_rate_gbp_usd"])
    lines += [
        f"  CALIBRATED USD   ${usd:.4f}",
        f"  CALIBRATED GBP   £{usd * fx:.4f}  (fixed rate {fx} GBP/USD, "
        f"{cfg.raw['fx_rate_date']}; not fetched)",
        "  One trace is one trace. Where this disagrees with the band above, the band "
        "is the safer number to plan against.",
    ]
    return lines


@app.command()
def run(
    config: str = typer.Option(DEFAULT_CONFIG),
    provider: str = typer.Option(..., help="rules | tfidf | tfidf_gbm | jev | openai"),
    split: str = typer.Option("test", help="dev | validation | test"),
    model: str | None = typer.Option(None, help="Model ID override (jev only)."),
    repeats: int = typer.Option(1, min=1, help="Repeats per trace."),
    concurrency: str = typer.Option("1", help="Comma-separated sweep, e.g. 1,5,10,20."),
    records: int | None = typer.Option(None, help="Which stage's dataset to run against."),
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    runs_dir: str | None = typer.Option(
        None, help="Where run directories go (defaults to the stage's runs root)."
    ),
    confirm_paid: bool = typer.Option(
        False, "--confirm-paid", help="Required for any provider that spends money."
    ),
    accept_cost_over_threshold: bool = typer.Option(
        False,
        "--accept-cost-over-threshold",
        help=(
            "Proceed even when the projected upper bound exceeds "
            "cost_projection.confirm_above_gbp. Without it, such a run stops and says so."
        ),
    ),
) -> None:
    """Run a provider over a split. Offline providers need no key and no network."""
    cfg = load_config(config)
    paths = _paths(cfg, records, corpus)
    dataset_path, splits_path = paths.dataset, paths.splits
    runs_root = Path(runs_dir) if runs_dir else paths.runs_root
    if not dataset_path.exists():
        hint = (
            f"Run `jev-eval ingest --corpus {corpus}` first."
            if corpus
            else "Run `jev-eval generate` first."
        )
        _fail(f"{dataset_path} not found. {hint}")

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

    prices = load_prices(cfg.raw["prices"])

    questions: dict[str, Any] | None = None
    if provider in PAID_PROVIDERS:
        n_requests = len(selected) * repeats * len(arms)
        jev_cfg = cfg.section("jev")
        questions = json.loads(Path(jev_cfg["questions"]).read_text(encoding="utf-8"))
        if provider == "jev":
            model_id = model or str(jev_cfg["model"])
            endpoint = f"{jev_cfg['base_url']}{jev_cfg['path']}"
            extra: list[str] = []
        else:
            vendor = _vendor_config(cfg, provider)
            model_id = model or vendor.model_id
            if not model_id:
                _fail(
                    f"{provider} has no model_id in {config} and none was passed.\n"
                    f"  Run `jev-eval list-models --provider {provider}` and use an id it "
                    "prints. No model name is assumed by this harness."
                )
            endpoint = f"{vendor.base_url}{vendor.path}"
            extra = [
                f"  temperature  {vendor.temperature} (matched to the Jev arm)",
                f"  max output   {vendor.max_output_tokens} tokens, strict JSON schema",
                "  questions    the same four, from " + str(jev_cfg["questions"]),
            ]
        projection = _project_run_cost(
            cfg,
            model_id=model_id,
            records=selected,
            questions=questions,
            n_requests=n_requests,
            prices=prices,
            provider=provider,
        )

        _echo("PAID RUN PLAN")
        _echo(f"  provider     {provider}")
        _echo(f"  endpoint     POST {endpoint}")
        _echo(f"  model        {model_id}")
        _echo(f"  split        {split} ({len(selected)} traces)")
        _echo(f"  repeats      {repeats}")
        _echo(f"  concurrency  {arms}")
        _echo(f"  requests     {n_requests} (one per trace per repeat per arm)")
        for line in extra:
            _echo(line)
        over_budget = _traces_over_budget(cfg, selected, questions)
        if over_budget:
            _echo(
                f"  truncation   {over_budget} of {len(selected)} traces exceed the "
                f"context budget and will be reduced by the documented rule; every "
                f"one is recorded in the run's truncation.jsonl"
            )
        _echo("")
        if prices.find(model_id) is None:
            _echo(
                f"  NO PRICE ENTRY for {model_id!r} in {prices.path}.\n"
                "  The run will proceed and will record real input and output token\n"
                "  counts, but every cost will be null and the report and dashboard will\n"
                "  say 'unavailable' rather than print a figure. To price it, add an\n"
                "  entry to a NEW dated manifest with the published rate and a citation;\n"
                "  never edit a dated manifest in place, because runs record its SHA-256."
            )
            _echo("")
        for line in _format_projection(projection, cfg):
            _echo(line)
        _echo("")
        if not confirm_paid:
            _fail("Refusing to spend money without --confirm-paid. Nothing was sent.")
        _spend_gate(projection, cfg, accept_cost_over_threshold)
        if model and provider == "jev":
            cfg.raw["jev"]["model"] = model

    for arm in arms:
        if provider in OFFLINE_PROVIDERS:
            evaluator = build_offline_evaluator(provider, cfg, all_records, splits)
        elif provider == "jev":
            evaluator = _build_jev(cfg, threshold)
        elif provider == "openai":
            evaluator = _build_openai(cfg, threshold, model)
        else:
            _fail(f"unknown provider {provider!r}")

        run_id = new_run_id(provider, split)
        writer = RunWriter(
            run_dir=runs_root / run_id,
            run_id=run_id,
            provider=provider,
            model_id=getattr(evaluator, "model_id", provider),
            split=split,
            repeats=repeats,
            concurrency=arm,
            threshold=threshold,
            n_traces=len(selected),
            n_expected=len(selected) * repeats,
            prices=prices,
            truncation_source=evaluator,
            deviation_source=evaluator,
        )
        _echo(f"{run_id}: writing to {writer.run_dir} as predictions land ...")
        try:
            run_predictions(evaluator, selected, repeats, arm, on_result=writer.append)
        except Exception as exc:  # the partial run must be marked, then re-raised
            writer.abort(f"{type(exc).__name__}: {exc}")
            if hasattr(evaluator, "close"):
                evaluator.close()
            raise
        manifest = writer.finalise(
            config=cfg,
            splits_path=splits_path,
            dataset_path=dataset_path,
            questions=questions,
            repo_root=REPO_ROOT,
        )
        if hasattr(evaluator, "close"):
            evaluator.close()
        _echo(
            f"{manifest.run_id}: {manifest.n_predictions} predictions, "
            f"{manifest.error_count} errors, {manifest.retry_count} retries, "
            f"cost {manifest.total_cost_usd}"
        )
        if manifest.truncated_traces:
            _echo(
                f"  {manifest.truncated_traces} of {manifest.n_traces} traces exceeded "
                f"the context budget and were reduced; every one is listed in "
                f"{manifest.truncation_path}"
            )
        for deviation in getattr(evaluator, "deviations", []) or []:
            # Loud, not a footnote. This run did not send what the frozen
            # config describes, and the report will say so too.
            _echo(
                f"  DEVIATION: {manifest.model_id} refused {deviation.parameter}="
                f"{deviation.requested!r} (HTTP {deviation.http_status} "
                f"{deviation.error_code}); it was {deviation.applied}. "
                f"Recorded in {manifest.deviations_path}."
            )
            _echo(f'    provider said: "{deviation.error_message}"')


def _parse_sizes(raw: str) -> tuple[int | None, ...]:
    """Parse a size sweep. ``full`` means the whole train split."""
    out: list[int | None] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "full":
            out.append(None)
            continue
        try:
            value = int(token)
        except ValueError:
            _fail(f"bad size {token!r}; sizes are integers or the word 'full'")
            raise
        if value <= 0:
            _fail(f"bad size {value}; sizes must be positive")
        out.append(value)
    if not out:
        _fail("no sizes given")
    return tuple(out)


@app.command("learning-curve")
def learning_curve_command(
    config: str = typer.Option(DEFAULT_CONFIG),
    records: int | None = typer.Option(None, help="Which stage's dataset to sweep."),
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    split: str = typer.Option("test", help="Which split the curve is scored on."),
    providers: str = typer.Option(
        ",".join(CURVE_PROVIDERS), help="Comma-separated free classifiers to sweep."
    ),
    sizes: str = typer.Option(
        ",".join(size_label(s) for s in TRAIN_SIZES),
        help="Comma-separated label budgets; 'full' is the whole train split.",
    ),
    seeds: str = typer.Option(
        ",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated subsample seeds."
    ),
    runs: str | None = typer.Option(None, help="Runs root (defaults to the stage's)."),
) -> None:
    """Sweep a free classifier's training-label budget and score on a frozen split.

    Free by construction: a paid provider is refused before anything is fitted,
    and the zero-shot arms are overlaid from their already-recorded predictions
    rather than re-run. This command opens no socket and spends nothing.
    """
    from . import learning_curve as lc

    cfg = load_config(config)
    paths = _paths(cfg, records, corpus)
    runs_root = Path(runs) if runs else paths.runs_root
    if not paths.dataset.exists():
        _fail(f"{paths.dataset} not found. Generate or ingest it first.")

    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    try:
        lc.assert_offline(provider_list)
    except lc.PaidProviderError as exc:
        _fail(str(exc))
    size_list = _parse_sizes(sizes)
    seed_list = tuple(int(s.strip()) for s in seeds.split(",") if s.strip())
    if not seed_list:
        _fail("no seeds given")

    loaded = load_dataset(paths.dataset)
    splits = load_splits(paths.splits)
    dataset_sha = sha256_file(paths.dataset)

    n_fits = sum(len(provider_list) * (1 if s is None else len(seed_list)) for s in size_list)
    _echo("LEARNING CURVE PLAN (free; no key, no network, no spend)")
    _echo(f"  dataset      {paths.dataset}")
    _echo(f"  providers    {provider_list}")
    _echo(f"  sizes        {[lc.size_label(s) for s in size_list]}")
    _echo(f"  seeds        {list(seed_list)}  (one fit at 'full': every seed draws it whole)")
    _echo(f"  fits         {n_fits}")
    _echo(f"  scored on    split {split}")
    _echo("")

    done = {"n": 0}

    def progress(point: lc.CurvePoint) -> None:
        done["n"] += 1
        _echo(
            f"  [{done['n']}/{n_fits}] {point.provider} n={point.n_train} "
            f"(tasks={point.n_train_tasks}) seed={point.seed} "
            f"auprc={point.auprc:.4f} recall={point.recall:.4f}"
        )

    points = lc.run_curve(
        config=cfg,
        records=loaded,
        splits=splits,
        split=split,
        providers=provider_list,
        sizes=size_list,
        seeds=seed_list,
        on_point=progress,
    )

    zero_shot = lc.zero_shot_from_runs(
        runs_root=runs_root,
        records=loaded,
        dataset_sha256=dataset_sha,
        split=split,
        config=cfg,
    )
    missing = [p for p in lc.ZERO_SHOT_PROVIDERS if p not in zero_shot]
    if missing:
        _echo("")
        _echo(
            f"  note: no complete {split} run on this dataset for {missing}; "
            "those flat lines are omitted rather than guessed. Nothing was called."
        )

    curve_id = lc.new_curve_id(split)
    document = lc.curve_document(
        points=points,
        zero_shot=zero_shot,
        split=split,
        dataset_path=paths.dataset,
        dataset_sha256=dataset_sha,
        splits_path=paths.splits,
        config=cfg,
        sizes=size_list,
        seeds=seed_list,
        curve_id=curve_id,
    )
    path = lc.write_curve(document, lc.curve_root(runs_root) / curve_id)

    _echo("")
    _echo(f"Wrote {path}")
    for provider in sorted(document["crossovers"]):
        for reference, result in sorted(document["crossovers"][provider].items()):
            _echo(f"  {result['statement']}")
            lower = result["lower_bound"]
            _echo(
                "    lower 95% bound first clears "
                + (
                    f"{reference} at {lower['n_train']} labelled examples"
                    if lower
                    else f"{reference} at no size tested"
                )
            )
            _echo(
                f"    {reference} leads somewhere on the curve: "
                f"{'yes' if result['reference_leads_anywhere'] else 'no'}"
            )
    _echo("")
    _echo("Paid calls in this step: 0. Run `jev-eval report` to fold the curve in.")


@app.command()
def report(
    config: str = typer.Option(DEFAULT_CONFIG),
    records: int | None = typer.Option(None, help="Which stage to report on."),
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    runs: str | None = typer.Option(None, help="Runs root (defaults to the stage's)."),
    out: str | None = typer.Option(
        None, help="Output directory. Defaults to reports/<dataset>, one per corpus."
    ),
    split: str = typer.Option("test", help="Which split to report on."),
) -> None:
    """Build tables, plots, results.json and report.md from run artifacts."""
    from .report import build_report

    cfg = load_config(config)
    paths = _paths(cfg, records, corpus)
    runs_root = Path(runs) if runs else paths.runs_root
    # One report directory per corpus. Sharing a default would let a real-data
    # report quietly overwrite the synthetic one, leaving a directory whose name
    # says nothing about which dataset produced the numbers inside it.
    out = out or (f"reports/{paths.dataset_id}" if paths.is_real else "reports/latest")
    if not paths.dataset.exists():
        _fail(f"{paths.dataset} not found. Generate or ingest it first.")
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
        provenance_path=paths.provenance,
    )
    plots = results.get("plots") or []
    _echo(f"Wrote {out}/report.md, {out}/results.json and {len(plots)} plots.")
    if "learning_curve.png" not in plots:
        _echo(
            "  No learning curve was folded in: none has been swept for this dataset "
            "and split. `jev-eval learning-curve` builds one; it is free and offline."
        )
    _echo(f"Providers reported: {', '.join(sorted(results['runs']))}")
    _echo(f"Audit status: {results['audit']['status']}")


@app.command()
def serve(
    config: str = typer.Option(DEFAULT_CONFIG),
    run: str | None = typer.Option(
        None, "--run", help="A run directory or run id. Defaults to the newest Jev run."
    ),
    live: bool = typer.Option(
        False, "--live", help="Tail an in-progress run instead of replaying a finished one."
    ),
    replay_speed: float = typer.Option(
        20.0, "--replay-speed", min=0.1, max=1000.0, help="Replay multiplier."
    ),
    port: int = typer.Option(8080, "--port", help="Local port."),
    host: str = typer.Option("127.0.0.1", help="Bind address. Loopback by default."),
    records: int | None = typer.Option(None, help="Which stage's dataset to score against."),
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    runs_dir: str | None = typer.Option(None, help="Runs root (defaults to the stage's)."),
    reports_dir: str = typer.Option("reports", help="Where Export writes PNG and CSV."),
) -> None:
    """Start the local dashboard: with-Jev against without-Jev, live or replayed.

    The server binds loopback by default and makes no outbound request of its
    own. No key is read and no secret reaches the browser: the page is served
    aggregate scoring figures only.
    """
    import uvicorn

    from .dashboard.app import create_app

    try:
        application = create_app(
            config_path=config,
            records=records,
            corpus=corpus,
            runs_dir=runs_dir,
            default_run=run,
            live=live,
            replay_speed=replay_speed,
            reports_dir=reports_dir,
        )
    except FileNotFoundError as exc:
        _fail(str(exc))
        return

    from .dashboard.aggregate import FOOTER_NOTICE

    url = f"http://{host}:{port}/"
    mode = "live (tailing the run as it is written)" if live else "replay"
    _echo(f"Dashboard  {url}")
    _echo(f"  mode      {mode}")
    runs_shown = runs_dir or _paths(load_config(config), records, corpus).runs_root
    _echo(f"  runs      {runs_shown}")
    _echo(f"  run       {run or 'newest Jev run'}")
    _echo("  no network at runtime; no key is read; no secret reaches the browser.")
    _echo(f"  {FOOTER_NOTICE}")
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command("verify-run")
def verify_run_command(
    run: str = typer.Option(..., help="A run directory containing manifest.json."),
    archived: bool = typer.Option(
        False,
        "--archived",
        help=(
            "Verify against the run's own manifest alone; skip the comparison against the "
            "dataset currently on disk. For runs under runs/_archive, whose dataset has "
            "deliberately been replaced."
        ),
    ),
) -> None:
    """Recompute checksums and cost arithmetic from a run manifest."""
    run_dir = Path(run)
    if not (run_dir / "manifest.json").exists():
        _fail(f"{run_dir}/manifest.json not found")
    problems = verify_run(run_dir, archived=archived)
    if problems:
        for problem in problems:
            typer.secho(f"  FAIL  {problem}", fg=typer.colors.RED, err=True)
        _fail(f"verify-run failed with {len(problems)} problem(s) for {run_dir}")
    suffix = " (--archived: the dataset currently on disk was not compared)" if archived else ""
    _echo(
        f"OK  {run_dir}: checksums and cost arithmetic verified against the price manifest.{suffix}"
    )


@app.command("verify-runs")
def verify_runs_command(
    config: str = typer.Option(DEFAULT_CONFIG),
    records: int | None = typer.Option(None, help="Which stage's runs to verify."),
    corpus: str | None = typer.Option(None, help=CORPUS_HELP),
    runs: str | None = typer.Option(None, help="Runs root (defaults to the stage's)."),
) -> None:
    """Verify every run directory under a root."""
    from .runner import discover_runs

    root = Path(runs) if runs else _paths(load_config(config), records, corpus).runs_root
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
