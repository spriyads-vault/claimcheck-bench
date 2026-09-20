# jev-false-success-eval

A private, reproducible harness for one question:

> Can TypeSafe's Jev detect **false success** in tool-using agent traces — cases where
> the final assistant message claims an external action succeeded even though the
> ordered tool evidence does not support that exact claim?

Jev is compared against three free local baselines — a deterministic rule checker, a
TF-IDF classifier, and TF-IDF with a gradient-boosted classifier, which is the published
recipe that beats every LLM judge in the literature and is therefore the bar — and
against **a general LLM judge**, a second paid arm shown the same trace and asked the
same four questions. Everything except the two live calls runs offline, with no key and
no network.

The two comparisons answer different questions and are kept apart:

- **Jev against the free baselines** — is the paid lane worth its bill against free code?
- **Jev against the general judge** — is a *typed* false-success model better than simply
  asking a general LLM the same questions, and at what cost?

The evaluation runs on **real, labelled agent traces** (AppWorld's released experiment
outputs, Apache-2.0, labels from the benchmark's own programmatic evaluation). The
synthetic generator is kept as a controlled diagnostic, not deleted.

---

## ⚠️ Publication restriction

**Do not publish, push, or upload any benchmark or performance result produced by this
repository.** TypeSafe's Master Customer Agreement restricts publication of benchmark
results, and OpenAI's terms restrict publication of benchmark results about its models.
Adding the general-judge arm makes this restriction **stricter**, not looser: a result
now touches two vendors' publication clauses rather than one. This harness produces
**local artifacts only**. It creates no remote, pushes nothing, and every report carries
the restriction notice in its header.

This constraint was supplied by the operator of this repository. It has not been
independently verified against the MCA text from this machine — see
`preregistration.md`. Treat it as binding regardless.

It is the one caveat that is true of every run whatever the data, so it is the
one caveat that is hard-coded. Everything else on the dashboard is generated
from the loaded run — see "The caveat, generated from the run" below.

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
cp .env.example .env          # .env is git-ignored
export TYPESAFE_API_KEY=...   # required only for the live Jev path
export OPENAI_API_KEY=...     # required only for the general-judge path
```

`OPENAI_API_KEY` is read **from the environment only**, by exactly three commands —
`list-models`, `smoke --provider openai` and `run --provider openai`. No other code path
reads it, the dashboard has no code path that reads any key at all, and it is never
logged, never committed and never accepted as a CLI argument. `.env` at the repo root is
git-ignored; export it or source it before running.

Every artifact written to disk passes through `redact.py`: auth header names are
replaced wholesale, live environment secret values are replaced wherever they appear,
and key-shaped strings are masked. SHA-256 digests are deliberately exempt from the
generic mask so the audit trail survives — but a secret that *looks* like a digest is
still caught, because the environment-value pass runs first.

---

## Two kinds of dataset

The evaluation now rests on **real, labelled agent traces**. The synthetic set
is kept, unchanged, as a controlled diagnostic — it isolates fault subclasses no
real corpus separates — but it is no longer where the headline comes from.

| Dataset | Kind | Records | Labels from | Split unit |
|---|---|---|---|---|
| `appworld` | real | 3,507 | AppWorld's programmatic post-episode evaluation | scenario id (195) |
| `synthetic-400` | synthetic | 400 | the generator's construction rules | template family (40) |
| `synthetic-1000` | synthetic | 1000 | the generator's construction rules | template family (40) |

Every dataset owns a disjoint set of artifact paths and its own runs root, so a
real run and a synthetic run cannot be mixed into one report by accident.

---

## The real corpus

```bash
# You already have the release unpacked:
uv run jev-eval ingest --corpus appworld --source path/to/experiment_outputs

# Or let the harness fetch it (needs the `appworld` package, see below):
uv run jev-eval ingest --corpus appworld --download
```

### What is actually being ingested

The brief for this work named "the AppWorld false-success corpus from the
paper". **That corpus does not exist as a release.** The paper —
[*From Confident Closing to Silent Failure: Characterizing False Success in LLM
Agents*](https://arxiv.org/abs/2606.09863), arXiv:2606.09863v1, CC BY 4.0 —
publishes no data of its own and says so: it uses "the publicly released
experiment outputs" of AppWorld.

So what is ingested is **AppWorld's own release**, and what the paper
contributes is the *rule*, which it documents well enough to reimplement. Every
figure quoted from the paper is labelled as the paper's and was not reproduced
here. The sample is not identical to the paper's either — see
"Where this differs from the paper" below.

### Licence, verified before anything was read

`ingest` refuses to touch a corpus that has no verified licence on record, and
refuses one whose licence does not permit this use. The check runs before a
single byte is read. Licences live in `src/false_success_eval/ingest/licences.py`
with the date and URL each was read from.

| Source | Licence | Verdict |
|---|---|---|
| AppWorld experiment outputs | Apache-2.0 | **Permits this use** |
| tau2-bench | MIT | Permits this use (no adapter built — see below) |
| `cx-cmu/agent_trajectories` (HF) | **none declared**, access-gated | **Refused** |

AppWorld attaches one extra condition to the protected bundle, in its own words:
released under Apache 2.0 *"with the additional requirement that any public
redistribution of it (or of its derivatives) must also be done in an encrypted
format."* This harness publishes nothing, and `data/appworld/` is git-ignored,
so the condition is met by not redistributing. Do not commit the ingested
corpus.

The HuggingFace fallback was **not** used. Its API returns no licence tag, its
card declares none, and it is gated. Unclear is not permission, so the harness
records it as a refusal and stops. tau2-bench is cleanly MIT and could be
ingested, but AppWorld was obtainable, so no tau2 adapter was written: a licence
is permission, not a reader.

The bundle is fetched from AppWorld's published S3 URL and its SHA-256 is
checked against the one this harness verified the licence of. It is encrypted —
deliberately, to keep the benchmark out of scrapers and training sets — and this
harness does **not** reimplement that decryption. `--download` calls AppWorld's
own published `unpack_bundle`, so the only code that opens the bundle is the
code its authors shipped for the purpose. Without that package installed the
command says so and stops.

### The label rule

Labels are derived from the environment, never from the assistant's wording.
The claim is a *structured field* the agent wrote as a tool-call argument; the
truth is what AppWorld's evaluator computed by asserting against the app
databases after the episode ran.

| Condition | Label |
|---|---|
| `status=success` **and** ground truth `success=false` | `unsupported_success` |
| `status=success` **and** ground truth `success=true` | `supported_success` |
| `status` in `{fail, failure}` | `reported_failure_or_uncertainty` |
| no `status` argument on the terminal call | `no_success_claim` |

The rule is exported as data (`ingest/labelling.py`), written into
`data/appworld/provenance.json`, printed by `ingest`, and shown in the dashboard
caveat — one source, so the wording cannot drift from the code. An unrecognised
status is an **error**, not a guess.

Ingestion is restricted to the two *self-assessing* architectures
(`legacy_full_code_agent`, `legacy_function_calling_agent`). Only they write a
structured status, so only they can express an honest failure and be
distinguished from a false success. That restriction is the paper's too.

Resulting prevalence, on 3,507 trajectories over 4 model families:

| Label | n | share |
|---|---|---|
| `unsupported_success` | 1,750 | 49.9% |
| `no_success_claim` | 1,007 | 28.7% |
| `reported_failure_or_uncertainty` | 415 | 11.8% |
| `supported_success` | 335 | 9.6% |

### Task-disjoint split

An AppWorld task id is `<scenario>_<variation>`, and the three variations of a
scenario are paraphrases of one task. The split unit is therefore the
**scenario**, not the task: 195 scenarios split 117 dev / 39 validation / 39
test. A paraphrase of a training task cannot reach test. The split hash is
frozen at ingest and recorded in the provenance file.

### Where this differs from the paper

Stated plainly rather than papered over. The paper reports 1,879 AppWorld
trajectories with explicit status claims and a 75.8% false-success share. This
ingest finds 2,500 with an explicit status and a higher share. The paper
released no code, so its exact filtering cannot be reproduced; restricting to
AppWorld's `test_normal` split alone gives 75.1%, which is close enough to
suggest that is the difference, but that is an inference, not a reproduction.
**This harness reports its own numbers on its own sample.**

---

## The two synthetic stages

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
# The real corpus -- the numbers that are reported
uv run jev-eval ingest --corpus appworld --source path/to/experiment_outputs
uv run jev-eval validate-data --corpus appworld
for split in validation test; do
  for provider in rules tfidf tfidf_gbm; do
    uv run jev-eval run --provider $provider --split $split --corpus appworld
  done
done
uv run jev-eval run --provider jev --split test --corpus appworld --concurrency 10 --confirm-paid

# The general LLM judge: one pass, modest concurrency, a smoke first
uv run jev-eval list-models --provider openai
uv run jev-eval smoke --provider openai --corpus appworld --split test
uv run jev-eval run --provider openai --split test --corpus appworld \
    --model <the id you picked> --concurrency 4 --confirm-paid

# The low-label learning curve. Free: no key, no network, nothing billed.
uv run jev-eval learning-curve --corpus appworld --split test

uv run jev-eval verify-runs --corpus appworld
uv run jev-eval report --corpus appworld
uv run jev-eval serve --corpus appworld
```

The baselines need a **validation** run as well as a test run: `tfidf` and
`tfidf_gbm` are calibrated on validation, and the dashboard reads the newest
completed run per provider per split.

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
| `ingest` | Verify a corpus's licence, map a real release into the schema, split it task-disjoint, record provenance |
| `generate` | Deterministic dataset, splits, hashes, audit key and blinded audit sheet |
| `validate-data` | Schema, label balance, fault coverage, split leakage, family label-diversity |
| `list-models` | Ask a vendor which models this key can reach. Unmetered, but asks before touching the network. The only source of a model ID |
| `smoke` | One live request before committing. For `jev`, a non-scored connectivity check; for `openai`, one **real** trace plus the calibrated full-split projection |
| `run` | Runs a provider over a split. `rules`/`tfidf`/`tfidf_gbm` are offline; `jev` and `openai` each need their own key **and** `--confirm-paid` |
| `learning-curve` | Sweeps a free classifier's label budget on task-disjoint, stratified subsets of the train split and scores every fit on the frozen test split. Refuses a paid provider before fitting anything; overlays the zero-shot arms from their **recorded** predictions |
| `report` | Tables, PR curve, reliability diagram, per-fault heatmap, learning curve, `results.json`, `report.md` |
| `verify-run` | Recomputes checksums and cost arithmetic for one run from the dated price manifest |
| `verify-runs` | Same, for every run under a root |
| `serve` | Local dashboard: with-Jev against without-Jev, live or replayed |

### The dashboard

```bash
uv run jev-eval serve --corpus appworld                   # the real corpus
uv run jev-eval serve --records 1000                      # the synthetic diagnostic
uv run jev-eval serve --records 1000 --live               # tail a run that is in flight
uv run jev-eval serve --run runs/1000/jev-test-... --replay-speed 40 --port 8080
```

`serve` starts a FastAPI app on loopback and opens a single build-free page, laid
out as a short evaluation report rather than a console: one reading order from the
finding, through detection quality with its intervals, the label-budget curve, the
per-fault small multiples, cost and speed, and a method-and-limits section given
the same weight as the results. It compares the paid Jev run against the free rule
and TF-IDF baselines on the same split, and it is deliberately as loud about where
Jev buys nothing as it is about where Jev wins. Light and dark are both first-class
and the palette is checked rather than eyeballed -- `tests/test_dashboard_design.py`
recomputes WCAG contrast for every token and the pairwise separation of the series
colours under simulated protanopia, deuteranopia and tritanopia, and fails the build
on a colour edited by eye. Every chart carries the table it was drawn from.

- **Live** tails a run's append-only `predictions.jsonl` and pushes each row over
  SSE as it lands. Run directories are written incrementally now, so an
  in-progress run is a complete, readable artifact at every instant. The page is a
  report rather than a console, so it follows a run that is still in flight and
  otherwise shows the finished figures; the `--replay-speed` stream is still served
  for a client that asks for it, but the page itself has no playback controls.
- **Export** writes a PNG of the current view and a CSV of its rows to
  `reports/dashboard/`. Both carry the MCA notice. Nothing is published.

The server makes **no outbound request of its own**, loads no CDN asset, and never
reads `TYPESAFE_API_KEY` -- there is no code path in `dashboard/` that touches the
environment at all. Predictions reach the browser through a field-by-field public
row that excludes `raw_request` and `raw_response` entirely; there is a test for
each of these.

Currency: TypeSafe bills in USD. Every USD figure is shown beside a GBP figure
converted with `fx_rate_gbp_usd` in `config/eval.yaml`, a fixed dated constant
labelled with its value and date in the UI. No rate is ever fetched.

### The dataset selector

The page switches between the synthetic diagnostic and each ingested real
corpus. One process serves them all; each is loaded lazily and keyed by id, so
switching swaps the dataset, the runs root **and** the caveats together. They
cannot come apart, and a run id from one dataset cannot be scored against
another's labels — the server resolves dataset and run as a pair and 404s
loudly otherwise.

### The caveat, generated from the run

The old three-line banner is gone. It said labels were construction labels and
had not been independently checked — both true of the synthetic set, both false
of a real corpus with programmatic ground truth. Leaving it up would have been
its own dishonesty: it understates the evidence rather than overstating it, but
it is still the page saying something untrue, and a warning that is no longer
true trains a reader to skip the ones that are.

In its place: a **one-line strip** plus an info icon that expands to the method
note. Every line is **generated from the loaded run** (`caveats.py`), never
written into the page, so a synthetic run still shows its synthetic caveats and
a real run shows the real-data ones. On a real corpus the panel carries only
what is still true:

- the data source, its version and its licence, with the date and URL the
  licence was verified from, and its conditions;
- the label rule, including which structured field carried the claim and which
  field carried the truth;
- the task-disjoint split and its frozen hash;
- any truncation, with counts, or a positive statement that nothing was
  shortened;
- the width of the confidence interval on the headline, and whether it crosses
  zero;
- the MCA restriction.

### A negative result renders as loudly as a positive one

The verdict panel is the same size whichever way it goes, and it is driven by
the **interval**, not the point estimate. An interval straddling zero reads "No
difference established", not a win. `tfidf_gbm` beating Jev reads "Jev does not
beat the strong free baseline", in the same space a win would get. Claiming a
win off a point estimate inside a straddling interval is the exact failure mode
this harness exists to measure.

The `semantic_target_mismatch` sample size is still attached to every figure
about that subclass on the synthetic set. On a real corpus that subclass does
not exist — it is a construct of the generator — and the card says so rather
than drawing two empty bars that look like a measured zero.

### Dry-run discipline

Anything that spends money or touches the network prints exactly what it will do first.
`run --provider jev` prints a full paid-run plan and **refuses** without `--confirm-paid`.
`smoke` prints the single request it will send and asks for confirmation (`--yes` skips).

The paid-run plan includes a **projected cost**, computed from the request bodies the
run will actually send and the rate in the dated price manifest. It is printed as a
band, not a single figure: Jev's tokenizer is not published, so the input-token count
cannot be known before the request goes out. The bracketing assumptions live in
`cost_projection` in `config/eval.yaml`, with the smoke-measured per-request overhead
that calibrates them. A projection is never recorded as a cost -- every recorded cost
comes from the tokens the API returned.

---

## The arms

Three free lanes, all local, none metered — and two paid ones.

| Provider | Paid? | What it is | Why it is here |
|---|---|---|---|
| `rules` | no | deterministic rule checker | Exact on the faults code *can* compute. Punished by the honest-failure hard negatives. |
| `tfidf` | no | TF-IDF + logistic regression | The cheap learned baseline. |
| `tfidf_gbm` | no | **TF-IDF + gradient-boosted trees** | **The bar.** |
| `jev` | **yes** | TypeSafe System One, a typed false-success model | The thing under test. |
| `openai` | **yes** | **a general LLM judge** | **The other question:** does the typed model beat simply asking a general LLM the same four questions? |

`tfidf_gbm` is built to the published recipe: bigram word TF-IDF over the
serialised trajectory, gradient-boosted trees on top, fit on dev alone and
calibrated on validation. The paper reports this family of detector beating
*every* LLM judge it tested — AUROC 0.953 on AppWorld and 0.825 on tau2-bench,
against no judge configuration above 0.65. That is why the headline in the
report and the dashboard is stated **against `tfidf_gbm`**, not against the rule
checker: a win over the weak lane would prove nothing.

XGBoost is what the paper ran, and it is what this uses when it loads. On a
machine without an OpenMP runtime it falls back to scikit-learn's histogram
booster — the same family of model, a different implementation — and says so in
the model id (`tfidf-gbm-sklearn-hgb-v1` rather than `tfidf-gbm-xgboost-v1`)
rather than passing the fallback off as the published baseline. On macOS,
`brew install libomp` gets you the real thing.

**The paper's own judge numbers are external context only.** They appear in the
report and on the dashboard's Method tab clearly labelled as the paper's, from
`external_reference` in `config/eval.yaml`. This harness did not reproduce them
and they must not be read as confirmed here.

---

## The general LLM judge

The third paid arm. It exists because "Jev loses to free TF-IDF" and "a typed
model is no better than a general one" are different findings, and only the
first was measured before.

### Fairness is structural, not promised

The comparison is worth nothing unless the two paid arms differ in the model and
in nothing else, so each parity term is enforced by construction and pinned by a
test rather than asserted in prose:

| Term | How |
|---|---|
| Same inference view | The judge is handed `jev_http.build_state` — the *same function*, not a reimplementation. Goal, tool schema, ordered events and evaluation rule are byte-identical. |
| Same four questions | The prompt **and** the response JSON schema are generated from the same frozen `config/questions.json` Jev is sent. Nothing is retyped, so the two cannot drift. |
| Same decoding | `temperature: 0`, strict JSON schema, zero-shot, no examples, no training. **Attempted, not guaranteed** — see "When the model refuses the request" below. |
| Same context budget | The same `ContextBudget` object. A trace shortened for one lane is shortened identically for the other, and both record it. |
| Same primary score | `P(unsupported_success)` from the verdict, so AUPRC is directly comparable. |
| Same split | The task-disjoint test split, unchanged. |

**The one asymmetry is stated, not hidden.** Jev's `probabilities` come from the
API. A general model's are **self-reported** inside its own JSON answer — it is
asked for a distribution and writes one. Both are used identically as the
primary score, so the *ranking* metrics (AUPRC, AUROC, recall) are comparable.
Reading the two lanes' **calibration** as comparable would be over-reading, and
the report says so.

### The model ID is chosen from the account, never invented

No model name is written into this repository. `general_model.openai.model_id`
ships empty, which records `not_run`:

```bash
uv run jev-eval list-models --provider openai
```

That is `GET /v1/models` — unmetered, but it still prints what it will do and
asks first. Copy an exact ID from its output into `config/eval.yaml`, or pass
`--model` for one run. The ID the **API reports back** is what lands in the run
manifest, not the one that was asked for, so an alias that resolved elsewhere is
visible in the artifact.

The `gpt_terra` block remains a permanently refusing stub, but its *reason*
changed on 2026-09-20: `GET /v1/models` on the account does list `gpt-5.6-terra`
(`owned_by=system`), so "no verified model ID" stopped being true and the block
now says it is superseded instead. It stays disabled because the verified route
to the model is the `openai` block, and there should be exactly one.

### When the model refuses the request

`temperature: 0` is what this harness *sends*. Some current model families
reject it outright — `gpt-5.6-terra` answers:

```
HTTP 400 {"code": "unsupported_value", "param": "temperature",
          "message": "Unsupported value: 'temperature' does not support 0.0 with
                      this model. Only the default (1) value is supported."}
```

The first live run of this arm was 702 predictions and 702 errors for exactly
that reason. The adapter now drops the refused parameter, re-sends **the same
trace, the same four questions and the same strict schema**, and records the
refusal — it does not pretend the setting was applied.

- **Only `temperature` and `top_p` may be dropped.** A refusal of
  `response_format`, `messages` or `max_completion_tokens` is recorded as an
  error and the trace is dropped. Relaxing the strict schema or the shared
  inference view to make a call succeed would silently turn this arm into a
  different experiment.
- **Every deviation is an artifact.** `deviations.jsonl` sits beside each run,
  checksummed in the manifest and checked by `verify-run`, carrying the
  provider's **verbatim** message. It is empty for a run that sent exactly what
  `config/eval.yaml` specifies — a positive statement, not a silence.
- **The report stops claiming what is no longer true.** When an arm's
  temperature was refused, the comparison note says so instead of "at
  temperature 0", a deviation table renders under it, and the caveat strip
  carries it.

The consequence has to be read into the numbers: a general judge whose
temperature was refused ran at the provider's own default, so **its figures are
one sample from a non-deterministic decoder**, and its repeat-stability is no
longer a property of the model alone.

The request and response shapes are verified against OpenAI's own published API
description — `openai/openai-openapi`, MIT, spec version 2.3.0, fetched
2026-09-20, SHA-256 `d4e8423…` — recorded in `OPENAI_SHAPE_SOURCE` and in
`preregistration.md` §A8. The test fixture is labelled
`CONSTRUCTED FROM THE DOCUMENTED SCHEMA`, exactly as the Jev one is.

### Cost, which is no longer zero on this side of the page

Jev is billed on input alone. A general model is billed for what it writes too,
so the pre-spend projection prices **both** legs. Jev's band is unchanged,
because its output rate is zero, and there is a test that fails if it moves.

**No OpenAI price is recorded here unless it was read from the vendor's own
published pricing.** Until then the dated manifest carries
`OPENAI_GENERAL_JUDGE_UNPRICED` with `verified: false`, the run records real
token counts with `cost_usd: null`, and the report and dashboard print
**"unavailable"** — never `$0.00`. A lane that looks free because nobody priced
it is the most expensive kind of wrong answer this harness could give. The
paid-run plan says so loudly, before spending, when the chosen model has no
entry at all.

Deliberately cheaper than the Jev run:

- **one pass** over the 702-trace test split, not the 5-repeat concurrency
  sweep. Repeats measure stability, not accuracy, and no point metric reads
  them, so the sweep would buy variance data at several times the price of the
  answer being sought;
- **a 1-trace smoke first.** For Jev, `smoke` is a connectivity check. For a
  general judge that proves nothing useful, so this smoke sends one *real*
  trace, reports the tokens it was billed, and prints the full-split projection
  twice — the usual chars-per-token band, and a point estimate calibrated on
  that measurement. Neither is ever recorded as a cost;
- **a stop-and-ask threshold.** `cost_projection.confirm_above_gbp` is £5. A run
  whose projected *upper* bound exceeds it stops and says so, even with
  `--confirm-paid`. The top of the band is the number worth being asked about.
  `--accept-cost-over-threshold` is the only way past, and it prints that it was
  used.

### Running it

```bash
export OPENAI_API_KEY=...

uv run jev-eval list-models --provider openai            # pick an exact ID
uv run jev-eval smoke --provider openai --corpus appworld --split test
uv run jev-eval run --provider openai --split test --corpus appworld \
    --model <the id you picked> --concurrency 4 --confirm-paid

uv run jev-eval verify-runs --corpus appworld
uv run jev-eval report --corpus appworld
```

### On the dashboard

The judge is a lane beside `rules`, `tfidf`, `tfidf_gbm` and `jev` across
detection, per-fault, and cost and speed — in its own colour, not a fourth shade
of the free-baseline indigo, because it is a different kind of lane.

It is **never folded into the "any free lane" union**: it costs money, and
crediting the free tier with a metered result would be the same class of error
this harness exists to catch. There is a test for that.

The headline it was added for gets its own panel, the same size as the
strong-baseline verdict and built by the same code path: **Jev against the
general judge** on AUPRC and recall, with a paired bootstrap CI on the
Jev-minus-judge difference, and each arm's cost per 1000 traces in USD and GBP
beside it. The verdict is read from the **interval**, never the point estimate.
"Jev loses to a general model" renders exactly as loudly as "Jev wins" — a page
that is only honest when the news is good is not honest.

---

## Long traces and the context budget

Synthetic traces were built short. Real ones are not. Jev's documented budget is
64k tokens in total, of which 32k is the ceiling for `state` plus the longest
question, so a trace that exceeds it has to be reduced before it is sent.

The rule is documented, deterministic and applied in a fixed order
(`budget.py`, `middle-elision-v1`):

1. shorten any tool-result payload over `max_result_chars`, leaving a visible
   `__elided__` marker inside it;
2. then drop whole events **from the middle**, keeping the first 4 and last 12,
   and insert one explicit `__elided__` event saying how many were removed;
3. if it still does not fit, give ground from the tail first, then the head,
   never below one of each.

The middle goes, never the ends. The goal states the task, the opening calls
establish what the agent set out to do, and the closing calls carry the
completion claim and the evidence for it — which is the entire question being
asked. Losing the middle costs intermediate steps; losing either end would
change the answer.

**Nothing is dropped silently.** Every reduction is written to the run's
`truncation.jsonl`, counted in `manifest.truncated_traces`, checksummed in the
manifest, re-checked by `verify-run`, and surfaced in the dashboard caveat. A
run whose manifest says `truncated_traces: 0` provably sent every trace whole.
The evaluator is told too: the marker event means a shortened trace never looks
like a complete one.

The pre-spend cost projection prices the bodies **after** reduction, and the
paid-run plan says how many traces will be reduced before you approve it.

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

- `data/audit-<n>.csv` — the **key**: `trace_id, domain, template_family, label,
  fault_type, audit_stratum, in_sample`
- `data/audit_blind-<n>.csv` — the **sheet a human fills in**: `trace_id`, `in_sample`,
  the rendered trace, and empty `auditor_id`, `auditor_label`, `auditor_fault_type`,
  `auditor_note` columns

`in_sample` marks a deterministic **stratified sample**: up to 8 traces per (label, fault)
cell, chosen by sorted `trace_id`. Auditing those rows is the minimum. Sampling by cell
rather than at random is what guarantees the rarer subclasses are actually put in front of
an auditor — `semantic_target_mismatch` above all, which carries the whole finding and
would otherwise be easy to miss by chance. Because *every* cell is sampled at the same
rate, membership says nothing about a trace's label and the sheet stays blind.

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
boolean so the PR curve has real resolution. 65% of the positives are faults that code
*can* compute exactly — a status field, an entity mismatch, a parameter mismatch, a
zero-row update — and the baseline is close to optimal on those by construction.

**The semantic subclass is the other 35%, and it exists because of that ceiling.**
`semantic_target_mismatch` (amendment A6) names its target only by description: a
read-only probe lists candidates, exactly one satisfies the description, the agent acts on
a different one, and the write *succeeds* and confirms the target the agent chose. There
is no error status, no missing result, and nothing that matches verbatim between the goal
and the evidence — so no string or field comparison resolves it. Generation *fails* if the
rules baseline can. Before A6 the baseline scored AUPRC 1.0000 on the 1000-record test
split; after it, 0.8200.

**`semantic_target_match` is its matched negative control** — same goal shape, same
candidate listing, right target, drawn at the same count. Without it the probe-and-act
*shape* would itself be a label. It earned its keep immediately: TF-IDF flags 0.78 of the
subclass, which looks like partial detection until you see it also flags 0.78 of the
control. Separation is +0.00. The report prints the two rows side by side.

Together these bound what any comparison here can claim — see `preregistration.md` §8.

**`needs_review` is kept separate from P(unsupported_success)** and is never assumed
equal to it. TypeSafe's own documentation states that no arithmetic identity holds
between a Noul answer and a Choice option.

**Noul answers carry no `confidence`.** Only Choice and Score answers do. Confidence is
read from the verdict Choice alone and is never synthesised for a Noul.

**Cost has exactly one source.** Every rate lives in a dated price manifest —
currently `config/prices-2026-09-20-2.json`. No price literal appears anywhere in `src/`;
there is a test that fails if one does. `verify-run` recomputes every cost from the
manifest each run recorded and checks that manifest's own SHA-256, so a changed price
file is detected rather than silently used.

**A dated manifest is superseded, never edited.** Appending two zero-cost entries to
`prices-2026-09-19.json` in place immediately broke verification for all eight runs
recorded against it, which is the check working. The 2026-09-19 file is restored and
kept so those runs still verify; `prices-2026-09-20.json` supersedes it, and
`prices-2026-09-20-2.json` supersedes *that* to add the general-judge entry. A same-day
supersede cannot reuse the filename and must not edit the earlier file — seven runs
record its SHA-256 — so it carries a `revision` to make the ordering explicit rather
than inferred from a suffix. The supersede tests walk the **whole chain**, not the
newest hop: a price that moved two manifests ago and moved back is still a repricing.

**An unknown model records a null cost; it does not abort the run.** `cost_usd` refuses
a model the manifest has never heard of, which is right for a pinned model. It is wrong
for one chosen at run time from an account listing — it would raise on the first
response and destroy the measurement to report a missing rate. The recording path is
`recorded_cost_usd`, which writes `null` instead; `verify-run` recomputes through the
same function, so the two cannot drift, and a run that recorded a *number* for an
unpriced model is still reported as a problem. Neither path ever guesses a rate.

---

## What the real-data run found

Full tables in `reports/appworld/report.md`; the dashboard shows the same
figures live. Summarised here because a negative result belongs in the README,
not only in an artifact nobody opens.

On the 702-trace task-disjoint test split of the AppWorld corpus, at the frozen
threshold of 0.5:

| Provider | AUPRC | AUROC | Precision | Recall | ECE |
|---|---|---|---|---|---|
| `tfidf_gbm` (strong free baseline) | **0.960** | **0.960** | 0.919 | 0.850 | 0.094 |
| `tfidf` (free) | 0.951 | 0.953 | 0.869 | 0.904 | 0.019 |
| `jev` (paid) | 0.848 | 0.823 | 0.820 | 0.511 | 0.146 |
| `rules` (free) | 0.560 | 0.531 | 0.960 | 0.064 | 0.491 |

**Jev did not beat the free baseline.** The AUPRC difference against
`tfidf_gbm` is **−0.113, 95% CI [−0.143, −0.083]** over 10,000 paired bootstrap
resamples — entirely below zero. It loses to the plain logistic-regression
baseline too. It beats only the deterministic
rule checker, which on real traces is barely above chance (AUROC 0.531): those
rules were written against synthetic faults constructed to be exactly
computable, and real traces are not that.

Where Jev actually loses is recall, not precision. At 0.820 its precision is
not far off the boosted baseline's 0.919, but it finds 0.511 of the false
successes against 0.850 — it is cautious, and on this corpus caution costs more
than it saves. Its calibration is also the second-worst of the four (ECE 0.146
against 0.019 for plain TF-IDF).

The strong free baseline is the only provider that clears the deployment gate
outright (precision 0.919, recall 0.850, ECE 0.094). Jev fails it on all three.

**Read the two gates together.** `useful_signal` is preregistered against the
`rules` baseline and frozen there, and Jev passes it (ΔAUPRC lower bound
+0.254). That is now a weak statement, so `beats_strong_baseline` is reported
beside it and Jev fails that one. The frozen gate was not moved to make the
result look worse, and the new one was not omitted to make it look better.

Cost: **$0.1237 for 702 traces**, so **$0.176 per 1000 traces (£0.131)** at the
0.7450 rate in the dated price manifest. 0 errors, 0 retries, 0 parse failures,
and no trace needed truncating — the largest was 62,696 characters of state
against a 79,170-character budget.

One thing worth noting in the other direction: the strong baseline's AUROC of
0.960 here sits close to the 0.953 the paper reports for its own TF-IDF
detector on AppWorld. Different sample, different code, so it is corroboration
rather than reproduction — but it is the kind of agreement that makes the
baseline credible as a bar.

### How many labels before the free classifier wins

The result above is lopsided, which raises the obvious objection: a classifier
needs labels and a zero-shot judge does not. So the label budget was swept —
task-disjoint, stratified, five seeds per size, every fit scored on the same
frozen test split, `jev-eval learning-curve`. No paid call was made to build it;
the two flat lines are the recorded runs' own AUPRC, quoted.

| Labels fit on | `tfidf` AUPRC (95% band) | `tfidf_gbm` AUPRC (95% band) |
|---|---|---|
| 10 | 0.822 [0.784, 0.860] | 0.761 [0.712, 0.810] |
| 25 | **0.892** [0.852, 0.933] | **0.887** [0.811, 0.962] |
| 50 | 0.866 [0.809, 0.923] | 0.903 [0.868, 0.939] |
| 100 | 0.874 [0.812, 0.935] | 0.908 [0.879, 0.936] |
| 200 | 0.885 [0.863, 0.906] | 0.918 [0.884, 0.952] |
| 400 | 0.913 [0.888, 0.938] | 0.932 [0.919, 0.945] |
| 2104 (full) | 0.951 | 0.960 |
| — | `jev` 0.848, zero-shot | `openai` 0.732, zero-shot |

**About 25 labelled examples.** Both classifiers pass Jev's 0.848 on mean AUPRC
at 25 labels and stay past it at every larger budget. On the conservative
reading — the lower edge of the seed band — `tfidf_gbm` clears Jev at **50** and
`tfidf` not until **200**, because plain TF-IDF's spread stays wide until a few
hundred labels.

**Jev leads in exactly one place: the 10-label budget**, where it beats both
classifiers (0.848 against 0.822 and 0.761). That is the whole low-label niche —
one point at the very bottom of the curve, gone by 25 labels. On this corpus 25
labels is roughly **two AppWorld tasks**' worth of trajectories.

The general judge never leads. Both classifiers beat its 0.732 from the smallest
budget tested.

The full-size point is a check, not a measurement: every seed draws the same
subset there, and it reproduces the recorded runs exactly (0.9510 / 0.9602).

Two things this does not say. `N` counts the labels the classifier is *fit* on —
calibration still runs on the whole validation split, as it does for the
full-size runs, so the strict total is `N` + 701. And the band is a Student-t
interval over seeds: it shows how much the draw moves the result, not the
test-split sampling error, and the flat lines are point estimates with their own
unshown uncertainty. See amendment A10 in `preregistration.md`.

### Jev against the general LLM judge

**Run.** `gpt-5.6-terra`, chosen from this account's own `GET /v1/models`
listing, one pass over the same 702-trace test split, the same inference view
and the same four questions.

| Arm | AUPRC | AUROC | Precision | Recall | ECE | Scored |
|---|---|---|---|---|---|---|
| `jev` (typed) | 0.848 | 0.823 | 0.820 | 0.511 | 0.146 | 702 |
| `openai` (general judge) | 0.732 | 0.704 | 0.832 | 0.265 | 0.371 | 701 |

**The typed model beats the general judge**: ΔAUPRC **+0.116, 95% CI [+0.076,
+0.158]**, entirely above zero, on the 701 traces both arms scored. That is the
one comparison in this repository Jev wins — and it does not rescue the main
result, because both arms still lose to the free classifiers by a wide margin.

Two things that happened on this arm and are recorded rather than smoothed:
`gpt-5.6-terra` **refused `temperature: 0.0`** (HTTP 400, "Only the default (1)
value is supported"), so the parameter was dropped, the provider default
applied, and the refusal was written verbatim to the run's `deviations.jsonl`
and carried into the report and the caveat bar. And **one trace of 702 came back
truncated** at the 1200-token output ceiling; the harness refuses to read a
verdict out of a cut-off answer, so it is counted as a parse failure rather than
parsed, which fails `deployment_max_parse_failures: 0` for this arm. No cost is
reported for it: `gpt-5.6-terra` has no entry in the dated price manifest, so
every cost is null and the report says "unavailable" rather than printing a
figure. See amendments A9 and its addenda in `preregistration.md`.

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
config/       questions.json (frozen -- drives BOTH paid arms' prompts and schemas),
              eval.yaml, prices-<date>[-<rev>].json -- dated, superseded rather
              than edited; a same-day supersede carries a `revision`
data/         generated, per stage: dataset-<n>.jsonl, splits-<n>.json, hashes-<n>.json,
              audit-<n>.csv, audit_blind-<n>.csv, provenance-<n>.json
data/<corpus>/ ingested, per real corpus: dataset.jsonl, splits.json, hashes.json,
              provenance.json  -- git-ignored, never redistributed
runs/<n>/     append-only run directories, written as the run happens:
runs/<corpus>/  predictions.jsonl, attempts.jsonl, progress.json (mutable status),
              truncation.jsonl (what was shortened, always written),
              manifest.json (written on completion)
reports/      results.json, report.md, plots
src/false_success_eval/
  schemas.py      frozen Pydantic v2 models; the InferenceView boundary;
                  dataset provenance, licence and label-rule records
  budget.py       the context budget and the documented truncation rule
  caveats.py      caveats generated from the loaded run, never hard-coded
  ingest/         licences.py (verified licences; a refusal is the default),
                  labelling.py (the ground-truth label rule, exported as data),
                  appworld.py (the AppWorld adapter), pipeline.py (orchestration)
  generate.py     deterministic generator; enforces balance, family label-diversity
                  and the semantic subclass's cleanliness guarantees
  templates.py    8 domains x 5 action families
  semantic.py     the semantic-clean subclass: indirection templates per family
  redact.py       secret redaction
  retry.py        capped exponential backoff with jitter; error-rate arm guard
  costs.py        the only code that reads a price
  metrics.py      AUPRC, bootstrap CIs, ECE, selective accuracy, prior-shift projection
  report.py       tables, plots, results.json, report.md, audit section
  runner.py       run loop, append-only artifacts, manifest, verify-run
  cli.py          Typer app
  evaluators/     base.py, rules.py, tfidf.py, tfidf_gbm.py (the strong baseline),
                  _transport.py (the one instrumented HTTP path both paid arms share),
                  jev_http.py, general_model.py (the general LLM judge, and the
                  refusing stubs for vendors whose shapes are unverified)
  dashboard/      aggregate.py (the with/without comparison), app.py (FastAPI + SSE),
                  export.py (PNG and CSV), static/ (one page, vanilla JS, no CDN)
```

See `preregistration.md` for the hypothesis, the frozen thresholds and decision gates,
every decision taken where the specification left a choice, and everything that could
not be verified at build time.
