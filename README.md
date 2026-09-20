# jev-false-success-eval

A private, reproducible harness for one question:

> Can TypeSafe's Jev detect **false success** in tool-using agent traces — cases where
> the final assistant message claims an external action succeeded even though the
> ordered tool evidence does not support that exact claim?

Jev is compared against a deterministic rule baseline and a TF-IDF classifier, with an
optional general-model baseline. Everything except the live Jev call runs offline, with
no key and no network.

---

## ⚠️ Publication restriction

**Do not publish, push, or upload any benchmark or performance result produced by this
repository.** TypeSafe's Master Customer Agreement restricts publication of benchmark
results. This harness produces **local artifacts only**. It creates no remote, pushes
nothing, and every report carries the restriction notice in its header.

This constraint was supplied by the operator of this repository. It has not been
independently verified against the MCA text from this machine — see
`preregistration.md`. Treat it as binding regardless.

---

## Requirements

- Python 3.11 (pinned: `>=3.11,<3.12`)
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync --all-extras
```

## Secrets

Keys are read **from the environment only**. This harness never accepts a key as a CLI
argument and never reads one from a committed file.

```bash
cp .env.example .env     # .env is git-ignored
export TYPESAFE_API_KEY=...   # required only for the live path
```

Every artifact written to disk passes through `redact.py`: auth header names are
replaced wholesale, live environment secret values are replaced wherever they appear,
and key-shaped strings are masked. SHA-256 digests are deliberately exempt from the
generic mask so the audit trail survives — but a secret that *looks* like a digest is
still caught, because the environment-value pass runs first.

---

## The two dataset stages

The record count is a config and CLI parameter (`dataset.records` / `--records`).
Two stages are run, and they are **not interchangeable**:

| Stage | Records | Test split | Positives in test | Purpose |
|---|---|---|---|---|
| 1 — pipeline proof | 400 | 8 families, 80 traces | ~20 | Prove the harness end to end |
| 2 — scored comparison | 1000 | 8 families, 200 traces | ~50 | The numbers actually reported |

At 400 records the test split holds only ~20 positives, which is too few for a
10,000-resample paired bootstrap CI to say anything useful. **The useful-signal decision
gate is therefore not reported from a stage-1 run.** The harness enforces this: any
report built from a dataset smaller than 1000 records prints `NOT REPORTED` for that
gate rather than a pass or fail. Only stage 2 produces reportable gate results.

Record count must be divisible by 40 (the template families) and by 4 (the labels), and
must leave at least 4 traces per family.

---

## Commands

Artifact paths carry the record count, so the two stages coexist and each stays
independently verifiable: `data/dataset-400.jsonl` and `runs/400/` alongside
`data/dataset-1000.jsonl` and `runs/1000/`. Pass `--records` to select a stage; it
defaults to `dataset.records` in `config/eval.yaml`.

```bash
# Stage 1 -- prove the pipeline
uv run jev-eval generate --seed 20260919 --records 400
uv run jev-eval validate-data --records 400
uv run jev-eval run --provider rules --split test --records 400
uv run jev-eval run --provider tfidf --split test --records 400
uv run jev-eval verify-runs --records 400
uv run jev-eval report --records 400 --out reports/stage1

# Stage 2 -- the scored comparison
uv run jev-eval generate --seed 20260919 --records 1000
uv run jev-eval validate-data --records 1000
uv run jev-eval run --provider rules --split test --records 1000
uv run jev-eval run --provider tfidf --split test --records 1000
uv run jev-eval verify-runs --records 1000
uv run jev-eval report --records 1000 --out reports/stage2
```

| Command | What it does |
|---|---|
| `generate` | Deterministic dataset, splits, hashes, audit key and blinded audit sheet |
| `validate-data` | Schema, label balance, fault coverage, split leakage, family label-diversity |
| `smoke` | **One** live non-scored Jev request. Fails clearly with no key; asks before sending |
| `run` | Runs a provider over a split. `rules`/`tfidf` are offline; `jev` needs a key **and** `--confirm-paid` |
| `report` | Tables, PR curve, reliability diagram, per-fault heatmap, `results.json`, `report.md` |
| `verify-run` | Recomputes checksums and cost arithmetic for one run from the dated price manifest |
| `verify-runs` | Same, for every run under a root |

### Dry-run discipline

Anything that spends money or touches the network prints exactly what it will do first.
`run --provider jev` prints a full paid-run plan and **refuses** without `--confirm-paid`.
`smoke` prints the single request it will send and asks for confirmation (`--yes` skips).

---

## The live path (built, not run)

No paid run has been made from this repository. These are the commands:

```bash
export TYPESAFE_API_KEY=...

uv run jev-eval smoke --provider jev --model jev-1.13.0

uv run jev-eval run --provider jev --model jev-1.13.0 --split test \
    --records 1000 --repeats 5 --concurrency 1,5,10,20 --confirm-paid
```

`jev-1.13.0` is pinned for any scored run. The alias `jev-latest` currently resolves to
the same model, but an alias moves; the pinned ID does not.

---

## The blinded human audit (required, not optional)

`generate` writes two files:

- `data/audit-<n>.csv` — the **key**: `trace_id, domain, template_family, label, fault_type`
- `data/audit_blind-<n>.csv` — the **sheet a human fills in**: `trace_id`, the rendered
  trace, and empty `auditor_id`, `auditor_label`, `auditor_fault_type`, `auditor_note`
  columns

Fill in `auditor_label` on the blinded sheet without looking at the key, then rebuild the
report. The report's audit section records coverage, agreement with the construction
labels, and every individual correction. With a single auditor it says so and **does not
report inter-rater reliability**, because with one rater there is none to report.

Until the sheet is filled in, the report states plainly that every label rests on the
generator's construction rules and has not been independently checked.

---

## Design notes

**Splits are by template family, never by row.** 8 domains × 5 action families = 40
families, split 24 dev / 8 validation / 8 test. A paraphrase of a tuned template cannot
reach the test split. Every family spans all four labels, so a family split can never
degenerate into a label split — generation *fails* if any family is label-homogeneous.

**The `InferenceView` boundary.** `label`, `fault_type` and `oracle` live on
`TraceRecord` and are unreachable from `InferenceView`, which is the only object any
evaluator receives. The run loop asserts this on every payload before it leaves.

**Honest-failure hard negatives.** Traces labelled `reported_failure_or_uncertainty`
carry *real* faults — errors, timeouts, unconfirmed writes — but the assistant reports
them honestly. Any rule that fires on "an error appears in the trace" is punished by
these. They are a quarter of the dataset.

**The rules baseline is deliberately strong**, and emits a graded score rather than a
boolean so the PR curve has real resolution. Note the ceiling this implies: the
generator constructs faults that code *can* compute exactly, so a deterministic baseline
is close to optimal on this data by construction. That bounds what any comparison here
can claim — see `preregistration.md`.

**`needs_review` is kept separate from P(unsupported_success)** and is never assumed
equal to it. TypeSafe's own documentation states that no arithmetic identity holds
between a Noul answer and a Choice option.

**Noul answers carry no `confidence`.** Only Choice and Score answers do. Confidence is
read from the verdict Choice alone and is never synthesised for a Noul.

**Cost has exactly one source.** Every rate lives in `config/prices-2026-09-19.json`.
No price literal appears anywhere in `src/` — there is a test that fails if one does.
`verify-run` recomputes every cost from that dated manifest and checks the manifest's own
SHA-256, so a changed price file is detected rather than silently used.

---

## Quality gates

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

Tests are offline by default and the whole pytest session blocks sockets, so a test that
tries to reach a real endpoint fails loudly instead of quietly making a live call.

---

## Layout

```
config/       questions.json (frozen), prices-2026-09-19.json, eval.yaml
data/         generated, per stage: dataset-<n>.jsonl, splits-<n>.json, hashes-<n>.json,
              audit-<n>.csv, audit_blind-<n>.csv
runs/<n>/     append-only run directories: manifest.json, predictions.jsonl, attempts.jsonl
reports/      results.json, report.md, plots
src/false_success_eval/
  schemas.py      frozen Pydantic v2 models; the InferenceView boundary
  generate.py     deterministic generator; enforces balance and family label-diversity
  templates.py    8 domains x 5 action families
  redact.py       secret redaction
  retry.py        capped exponential backoff with jitter; error-rate arm guard
  costs.py        the only code that reads a price
  metrics.py      AUPRC, bootstrap CIs, ECE, selective accuracy, prior-shift projection
  report.py       tables, plots, results.json, report.md, audit section
  runner.py       run loop, append-only artifacts, manifest, verify-run
  cli.py          Typer app
  evaluators/     base.py, rules.py, tfidf.py, jev_http.py, general_model.py
```

See `preregistration.md` for the hypothesis, the frozen thresholds and decision gates,
every decision taken where the specification left a choice, and everything that could
not be verified at build time.
