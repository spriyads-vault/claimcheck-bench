# jev-false-success-eval

A reproducible evaluation of whether a **typed decision model** (TypeSafe Jev) detects
**false success** in tool-using agent traces — a final message claiming an action
succeeded when the ordered tool evidence does not support that claim — measured against
a deterministic rule checker, a TF-IDF classifier, a TF-IDF + gradient-boosted
classifier, and a general OpenAI judge asked the same questions on the same evidence.

> ⚠️ **Publication restriction.** Benchmark and performance figures produced here are
> covered by TypeSafe's Master Customer Agreement and by OpenAI's terms, both of which
> restrict publishing results about their models. This repository is **private** and
> must stay private unless written permission is obtained. Every report carries the same
> notice in its header.

---

## Headline result

On **real AppWorld traces** (702-trace test split, 374 positives, task-disjoint):
**Jev beats the general OpenAI judge**, and **both lose to a trained TF-IDF
classifier.** Jev's case is the **no-labelled-data** case, and only up to about 25
labels.

| arm | model | AUPRC [95% CI] | AUROC | precision | recall | ECE | cost / 1k traces | p50 |
|---|---|---|---|---|---|---|---|---|
| `tfidf_gbm` | tfidf-gbm-xgboost-v1 | **0.9602** [0.943, 0.974] | 0.9604 | 0.9191 | 0.8503 | 0.0943 | $0 | 2 ms |
| `tfidf` | tfidf-logreg-v1 | 0.9510 [0.929, 0.969] | 0.9530 | 0.8689 | 0.9037 | 0.0187 | $0 | 8 ms |
| `jev` | jev-1.13.0 | 0.8476 [0.815, 0.878] | 0.8232 | 0.8197 | 0.5107 | 0.1460 | $0.1762 / £0.1312 | 298 ms |
| `openai` | gpt-5.6-terra | 0.7324 [0.686, 0.776] | 0.7043 | 0.8319 | 0.2647 | 0.3709 | unavailable | 3,276 ms |
| `rules` | deterministic | 0.5602 [0.523, 0.597] | 0.5306 | 0.9600 | 0.0642 | 0.4908 | $0 | 1 ms |

Paired bootstrap on ΔAUPRC, 10,000 resamples, 95% percentile intervals:

- **Jev − OpenAI judge: +0.1157** [+0.0761, +0.1577] — entirely above zero.
- **Jev − tfidf_gbm: −0.1125** [−0.1432, −0.0834] — entirely below zero.
- **Jev − tfidf: −0.1033** [−0.1368, −0.0709] — entirely below zero.

**On cost, precisely.** Jev costs **$0.1762 per 1,000 traces** (£0.1312 at a fixed dated
rate), computed from the tokens the API returned and a cited entry in a dated price
manifest. The general judge's **money cost is unavailable**: no verified published rate
for `gpt-5.6-terra` is on record, so tokens were recorded and no price was guessed. The
dollar comparison between the two paid arms is therefore *not established*. What is
measured is latency: Jev is ~11× faster at p50 (298 ms vs 3,276 ms) and returns
calibrated probabilities from the API rather than self-reported ones inside a JSON body.

**Where Jev's advantage ends.** A zero-shot judge needs no labels. A free classifier does.
Sweeping the classifiers over labelled subsets of the task-disjoint train split (5 seeds
per size, 0 paid calls):

| classifier | labels for its mean to pass Jev | labels for its lower bound to pass Jev | vs the OpenAI judge |
|---|---|---|---|
| `tfidf` | 25 | 200 | already ahead at 10 labels |
| `tfidf_gbm` | 25 | 50 | already ahead at 10 labels |

Jev leads while you have fewer than ~25 labelled examples. The general judge never leads.

Full tables, plots, caveats and decision gates: `reports/appworld/report.md`.

---

## Quickstart

Requires Python 3.11 (pinned `>=3.11,<3.12`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

**Generate the synthetic diagnostic dataset** (deterministic, no network, no key):

```bash
uv run jev-eval generate --seed 20260919 --records 1000
uv run jev-eval validate-data --records 1000
```

**Run the offline baselines and build a report** (free, offline, no key):

```bash
uv run jev-eval run --provider rules      --split test --records 1000
uv run jev-eval run --provider tfidf      --split test --records 1000
uv run jev-eval run --provider tfidf_gbm  --split test --records 1000
uv run jev-eval learning-curve --records 1000 --split test
uv run jev-eval verify-runs --records 1000
uv run jev-eval report --records 1000 --out reports/stage2
```

**Serve the dashboard** (loopback only, makes no outbound request, reads no key):

```bash
uv run jev-eval serve --records 1000
```

**The real corpus** is not in this repository and must be supplied locally. Ingest
refuses any corpus whose licence has not been verified:

```bash
uv run jev-eval ingest --corpus appworld --source path/to/experiment_outputs
uv run jev-eval run --provider rules --split test --corpus appworld
uv run jev-eval report --corpus appworld
```

Paid arms (`--provider jev`, `--provider openai`) read their key from the environment
only, require `--confirm-paid`, and print a projected cost band first. See `.env.example`.

---

## What this is, and is not

**It is** a controlled, preregistered offline comparison on a frozen test split, with
every threshold fixed on validation before test was opened.

**It is not:**

- **Not a production monitor.** Every arm fails the deployment gate except `tfidf_gbm`.
  Jev fails on precision, recall and calibration at the frozen threshold.
- **Audit pending.** The blinded human audit of the real corpus has not been filled in
  (`audit status: no_audit_key`). Until it is, labels rest on AppWorld's own programmatic
  post-episode evaluation and have not been independently checked by a human.
- **Temperature caveat on the general judge.** `gpt-5.6-terra` refused `temperature=0.0`
  with HTTP 400 `unsupported_value`; the parameter was dropped and the provider default
  (1) applied. The arms saw identical evidence and questions but did **not** decode under
  identical settings. The refusal is recorded verbatim in the run's `deviations.jsonl`.
- **Prior-shift numbers are projections,** not observations. At a realistic 1–10%
  production prevalence every arm's projected precision collapses.
- **Paper figures are not reproduced.** Numbers quoted from arXiv:2606.09863 are labelled
  as that paper's and nothing here confirms them.

The split is **task-disjoint**: an AppWorld task id is `<scenario>_<variation>` and the
three variations of a scenario are paraphrases of one task, so splits are cut by
scenario (195 → 117 dev / 39 validation / 39 test, split hash `cac71cf681e3dd63`). A
paraphrase of a training task cannot appear in test.

---

## Reproducibility

- **Seeds are frozen in `config/eval.yaml`:** dataset/split seed `20260919`, bootstrap
  seed `20260919`, 10,000 resamples. The SHA-256 of that config is written into every
  run manifest, so changing it invalidates prior runs rather than silently altering them.
- **Runs are append-only and self-describing.** Each carries `predictions.jsonl`,
  `attempts.jsonl`, `truncation.jsonl` and a `manifest.json` recording the dataset hash,
  the price-manifest hash and the config hash.
- **`verify-run` recomputes, it does not trust.** It re-checks artifact checksums, the
  dataset hash and the cost arithmetic from the dated rates:

  ```bash
  uv run jev-eval verify-run  --run runs/1000/<run_id>
  uv run jev-eval verify-runs --records 1000
  ```

- **Prices are dated and superseded, never edited in place.** No price is hard-coded; a
  model with `verified: false` records `null` cost rather than a guess.
- **Tests block sockets,** so a test that tries to reach a real endpoint fails loudly
  instead of quietly making a live call.

**The five quality gates** — all green as committed:

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

`preregistration.md` records the hypothesis, the frozen thresholds and decision gates,
every amendment, and everything that could not be verified at build time.

---

## Licence and attribution

The evaluation corpus is **AppWorld's publicly released experiment outputs**
(`experiment-outputs-0.1.3`), licensed **Apache-2.0**, with the added condition that any
public redistribution of the protected bundle or its derivatives be encrypted. That
condition is met here by **not redistributing it**: `data/appworld/` is git-ignored and
the corpus must be supplied locally.

> Trivedi et al. *AppWorld: A Controllable World of Apps and People for Benchmarking
> Interactive Coding Agents.* ACL 2024 (Best Resource Paper). <https://appworld.dev/>

The failure mode is characterised by Advani, *From Confident Closing to Silent Failure*,
arXiv:2606.09863v1, CC BY 4.0 — cited as external context, not reproduced here.

**Nothing sensitive is in this repository.** No API key and no corpus data are committed,
in the working tree or anywhere in git history. `.env` is git-ignored and untracked; keys
are read from the process environment only, are never accepted as a CLI argument, and
every artifact written to disk passes through `redact.py`. The generated datasets,
`runs/`, `reports/`, the virtualenv and all caches are git-ignored.
