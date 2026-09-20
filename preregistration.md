# Preregistration

Written before the test split was opened. Frozen unless an amendment is recorded below.

- **Repository:** jev-false-success-eval
- **Date:** 2026-09-20
- **Dataset seed:** 20260919
- **Model pinned for any scored run:** `jev-1.13.0`

---

## 1. Hypothesis

**H1.** Jev (`jev-1.13.0`), given an agent trace as `state` and the four frozen questions
in `config/questions.json`, separates `unsupported_success` traces from all others better
than a deterministic rule baseline, measured by AUPRC on a held-out test split of
template families.

**H0.** Jev does not improve materially on the rule baseline once paired bootstrap
confidence intervals are taken into account.

A negative result is a valid outcome and will be recorded as one. Nothing in this design
treats "Jev wins" as the successful branch.

**Secondary question.** Does `needs_review` (a Noul) behave differently from
P(`unsupported_success`) (a Choice option)? These are recorded separately and are never
assumed equal. TypeSafe's own jaggedness documentation states that no arithmetic identity
between a Noul and a Choice is guaranteed, so treating them as interchangeable would be a
design error, not a simplification.

---

## 2. Frozen thresholds and decision gates

Thresholds live in `config/eval.yaml` and are frozen at **0.5** for every provider. They
may be adjusted **once**, on the validation split, before the test split is opened. Any
such change must be appended to §7 below.

### Useful signal
The lower bound of the 95% paired bootstrap interval for ΔAUPRC over `rules` exceeds
**0.05**, **or** recall improves by at least **10 points** at a 5% review budget.

### Deployment candidate
≥ **0.90** precision and ≥ **0.80** recall on held-out `unsupported_success`,
ECE ≤ **0.10**, API error rate < **1%**, and **zero** schema-parse failures.

### Negative result
No material gain over TF-IDF or rules after confidence intervals; or unstable labels
across repeats; or unacceptable false acceptance on entity/parameter mismatch.

### Gate reporting is stage-dependent
The useful-signal gate is **read only from a run of at least 1000 records** (see §3). At
400 records the test split holds ~20 positives, which is too few for a 10,000-resample
paired interval to be informative. The harness enforces this: a report built from a
smaller dataset prints `NOT REPORTED` for that gate rather than a pass or fail. The gate
is also `NOT APPLICABLE` for `rules` itself, which is the baseline it is defined against.

---

## 3. Two dataset stages

| Stage | Records | Test split | Positives | Used for |
|---|---|---|---|---|
| 1 | 400 | 80 traces | 20 | Proving the pipeline end to end |
| 2 | 1000 | 200 traces | 50 | The numbers that are reported |

Artifact paths carry the record count (`data/dataset-1000.jsonl`, `runs/1000/`), so both
stages coexist and each remains independently verifiable.

---

## 4. Decisions taken where the specification left a choice

1. **Split geometry.** 8 domains × 5 action families = 40 template families, split
   24 dev / 8 validation / 8 test. Splits are over families, never rows.

2. **Every family spans all four labels.** Labels rotate by family index
   (`LABEL_ORDER[(family_index + j) % 4]`), which keeps global label counts exactly
   balanced while guaranteeing family label-diversity. This closes a failure mode that a
   naive assignment invites: if families were label-homogeneous, a family split would
   silently become a label split. Generation *fails* if any family is homogeneous, and
   there is a test for it.

3. **Honest-failure hard negatives.** A quarter of the dataset is labelled
   `reported_failure_or_uncertainty` and carries *real* faults — errors, timeouts,
   unconfirmed writes — that the assistant reports honestly. Any rule keying on "an error
   appears somewhere in the trace" is punished by these. This is the discrimination the
   eval actually measures.

4. **The rules baseline emits a graded score**, not a boolean, so the PR curve has
   resolution. A binary rule would collapse AUPRC to a three-point curve and make the
   comparison against it meaningless.

5. **Primary task is binary** (`unsupported_success` vs rest) for AUPRC, PR and
   calibration. The four-way label is used only for macro-F1, the per-fault matrix and
   selective accuracy.

6. **Confidence for `rules` and `tfidf` is a documented proxy** (`|score − 0.5| × 2` and
   `max(probability)` respectively). Only Jev's Choice answer carries a first-party
   confidence statistic. These are not treated as comparable to Jev's.

7. **Tie handling in recall-at-budget is by expectation**, not by sort order. The rules
   baseline emits discrete scores, so an arbitrary stable sort would silently flatter or
   penalise it at the budget cut-off.

8. **Determinism.** Canonical JSON (sorted keys, compact separators, UTF-8, LF), no
   timestamps inside `dataset.jsonl`, `trace_id = sha256(canonical content + seed)[:16]`,
   per-record RNG seeded from `f"{seed}:{family_id}:{index}"`.

9. **Concurrency sweeps apply to `jev` only.** Offline providers run at concurrency 1.

10. **A tool argument may not be named `label`.** The issue-tracker family uses
    `label_name`, because a tool argument named `label` would collide with the
    ground-truth field name that the leakage guard forbids. Recorded because it is a
    dataset-visible decision, not a cosmetic one.

11. **Sentence frames are shared across families.** A family fixes the task shape, tool
    schema, entity type and verb phrasing; the surrounding "Done — I …" / "I could not …"
    frames are drawn from shared banks and rotated by family index. Ordinary English
    phrasing is not template identity, and sharing it stops a classifier from solving the
    task by memorising a frame. The leakage control is at the level of task template.

---

## 5. Unverified at build time

Everything in this section is a thing this harness does **not** assert as fact.

1. **The general-model baseline has no verified endpoint or model ID.** No general-model
   base URL, path or model ID was verified against primary vendor documentation from this
   machine on 2026-09-20. Rather than write a plausible-looking URL or model name into
   code, `general_model.py` reads all three from `config/eval.yaml`, ships with them
   empty, and records `not_run` with a reason. In particular:
   - **"Gemini 3.5 Flash-Lite"** — the specification named this model. Its existence and
     exact ID were **not verified**. It is a configurable string flagged unverified, not
     a constant in code.
   - **"GPT-5.6-Terra"** — no verified official model ID. Permanently refusing stub, by
     design, as specified.

2. **The Master Customer Agreement publication clause was not read.** `docs.typesafe.ai`
   links the MCA at `typesafe.ai/legal/mca`; that document was not fetched or parsed. The
   publication restriction is honoured as an operator-supplied constraint, recorded as
   asserted-by-operator rather than verified here. It is treated as binding regardless.

3. **No live Jev response has ever been observed by this repository.** The adapter and
   its tests are written against the documented schema only. The test fixture is labelled
   `_provenance: CONSTRUCTED FROM THE DOCUMENTED SCHEMA` and is not presented as a
   captured response. No number in it is a measurement.

4. **No request-id or trace header is documented** for the System One endpoint, so run
   correlation uses this harness's own `run_id` plus the verbatim redacted response
   headers.

### Verified at build time (for contrast)

Fetched from `docs.typesafe.ai` on 2026-09-20, and matching the specification:

| Item | Verified value |
|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>` |
| Request | `{state, model, questions: map<id, Question>}` |
| Choice answer | `{type, choice, probabilities, confidence}` |
| Noul answer | `{type, noul}` — **no confidence field** |
| Usage | `{input_tokens, output_tokens}` |
| Model | `jev-1.13.0`; `jev-latest` → `jev-1.13.0` |
| Price | $42/Btok, **$0.042/Mtok input; output tokens free** |
| Errors | 401, 422, 429, 529 |
| SDK retry defaults | statuses `{408, 429, 500–599}`, backoff 0.5s ×2 → cap 5.0s, jitter 0.25, honours `Retry-After` and `retry-after-ms` |
| Limits | 250k tok/s, 1200 req/min; 64k context, 32k for state + longest question |

**One documented detail contradicts the specification's Prediction schema**, and the docs
win: Noul answers carry **no** `confidence`. Confidence is therefore read from the verdict
Choice alone and recorded as `null` for every Noul. It is never synthesised.

**The specification's cost formula matches the documentation exactly**
(`input_tokens × 0.042 / 1e6`, output free), so no reconciliation was needed.

---

## 6. Known jagged edges of jev-1.13 relevant to this task

From `docs.typesafe.ai/model-jaggedness/jev-1.13`, recorded *before* any run so that a
poor result is not retrofitted with an explanation afterwards:

1. **Adversarial content.** "State is data, and `jev-1.13` does not treat it as hostile by
   default." This task puts untrusted text *inside tool results* — exactly the channel the
   docs flag. The `evaluation_rule` string is a mitigation, not a fix, and its
   effectiveness is itself an open question this harness can measure.
2. **Date and time comparison** is called out as unreliable. The `stale_evidence` fault
   turns on recency and ordering. Expect this fault to be the weakest.
3. **Literal reading.** The question instructions were written to state the exact
   condition, with boundary cases in the criteria, as the docs advise.
4. **No structural invariants.** Directly supports keeping `needs_review` separate from
   P(`unsupported_success`).
5. **Context rot** on large states. Traces here are short; this should not bind, but state
   size is recorded per request.

---

## 7. Amendments

**A1 (2026-09-20, before generation).** Two-stage record count adopted. The count is a
config/CLI parameter defaulting to 400. The pipeline is proven at 400; the scored
comparison is regenerated at 1000 so the test split holds ~50 positives. The
useful-signal gate is not reported from a 400-record run. Enforced in code.

**A2 (2026-09-20, before generation).** The blinded human audit is a required step, not
just a file. `generate` writes both the audit key (`audit-<n>.csv`) and a blinded sheet
(`audit_blind-<n>.csv`). The report includes an audit section recording coverage,
agreement with construction labels, and every individual correction. With a single
auditor it says so and does **not** report inter-rater reliability.

**A3 (2026-09-20, before generation).** Added a test asserting no template family maps to
a single label, and made generation *fail* if any family is label-homogeneous.

**A4 (2026-09-20, after the first stage-1 report, before stage 2).** Three reporting
defects found and fixed. None changed a threshold, a split or a model output:
- Review-budget columns sorted lexically (`1%, 10%, 20%, 5%`); now numeric.
- `recall_at_budget` saturates at `budget × n / positives` on a label-balanced dataset.
  A ceiling row was added so the saturation is visible rather than mistaken for a tie.
- The useful-signal gate printed `FAIL` for `rules`, which is the baseline it is defined
  against. It now prints `NOT APPLICABLE`.

**A5 (2026-09-20, after stage 2).** Added a warning where a provider records zero false
positives, because the prior-shift projection then collapses to 1.0 at every prevalence
by arithmetic. An unobserved FPR is not a zero FPR.

---

## 8. What this design can and cannot establish

Stated before the test split was opened, and repeated in every generated report.

**Can:** whether a model matches a strong deterministic checker on false-success faults
that are cleanly computable from an ordered trace; how calibrated its probabilities are;
how it behaves per fault type; what it costs and how fast it is.

**Cannot:** whether false-success detection is solved. The generator constructs faults
that code *can* compute exactly, so a deterministic baseline should be near-perfect here —
and is. That is a property of the dataset. The interesting regime is traces where a rule
baseline fails: partial observability, ambiguous entity references, natural language that
does not resolve to a field comparison. This dataset does not contain that regime, and no
result from it should be read as though it does.
