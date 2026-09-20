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

**A7 (2026-09-20, before the paid run).** Three changes ahead of the live Jev run and
the dashboard built on it. None touches a threshold, a split, a label or a metric.

- **Run artifacts are now written as the run happens.** `predictions.jsonl` and
  `attempts.jsonl` were previously written once, at the end. They are now appended and
  flushed per prediction, still under exclusive create, so a run in flight is a complete
  readable artifact at every instant. A mutable `progress.json` sits beside them to say
  whether a run is running, complete or failed; the append-only files cannot say that
  themselves. It is not part of the verified artifact set and no metric is read from it.
- **`config/eval.yaml` gained three keys**, which changes its SHA-256 and therefore the
  `eval_config_sha256` recorded in runs made after this point. Runs made before it keep
  the old digest, which is correct: they were made under the old config. `verify-run`
  does not compare the config digest against the file on disk, so no prior run's
  verification is affected. The keys are `fx_rate_gbp_usd` / `fx_rate_date` /
  `fx_rate_source` (a fixed, dated, operator-supplied GBP-per-USD constant, never
  fetched, used for display only and never for a recorded cost) and `cost_projection`
  (bracketing assumptions for the pre-spend projection).
- **The paid-run plan now prints a projected cost.** It is computed from the request
  bodies the run will actually send and the rate in the dated price manifest, and is
  printed as a band because Jev's tokenizer is not published. The per-request overhead
  that calibrates the band was measured from the smoke request on 2026-09-20: a
  177-character compact body was billed 290 input tokens. **A projection is never
  recorded as a cost.** Every recorded cost still comes from `costs.py` and the tokens
  the API returned.

**A8 (2026-09-20, with the dashboard).** The dashboard states the with/without
comparison against **two** free baselines, not one: `rules` alone (the deterministic
free guard, and the default) and `rules OR tfidf` (the strongest thing the free tier
can do). Both headline numbers are shown together, so the headline cannot be read as
having been measured against whichever baseline flattered it. The dashboard also
carries a panel, built from the same rows as the win panel, listing the fault groups
where the free guard already catches everything and Jev therefore adds only cost and
latency -- including the rows where Jev catches *fewer* than the free guard does.

**A5 (2026-09-20, after stage 2).** Added a warning where a provider records zero false
positives, because the prior-shift projection then collapses to 1.0 at every prevalence
by arithmetic. An unobserved FPR is not a zero FPR.

**A6 (2026-09-20, after the first stage-2 report, before regeneration).** *The operator
proposed this as "A5"; A5 was already taken, so it is recorded here as A6. Nothing else
about it is changed.*

**This amendment changes dataset composition.** It is logged before the dataset was
regenerated and before the test split was reopened.

*Motivation.* The first stage-2 report put `rules` at AUPRC 1.0000, precision 1.0000 and
recall 1.0000. That is not a strong baseline; it is a dataset with no headroom. Every
`unsupported_success` fault as built was exactly rule-computable — a status field, an
entity that differs from one named in the goal, a parameter that differs from one named
in the goal, a zero-row update. On such a corpus no model can demonstrate a contribution
over a deterministic checker, because there is nothing left for it to contribute. §8
already said the interesting regime was absent. A6 puts some of it in.

*The new fault.* `semantic_target_mismatch`, under the `unsupported_success` label.
Defining property: the goal refers to its **target** by description or indirection, never
by an ID that appears verbatim in the tool arguments or the tool result. A read-only probe
enumerates candidates; exactly one satisfies the description; the agent acts on a
different, plausible one. The write **succeeds** and the tool result confirms success for
the target the agent chose, so there is no error status, no missing or stale result, and
no verbatim entity or parameter mismatch for a string or field check to catch. Only
reading the goal against the evidence resolves it.

*One reading had to be fixed, and it is recorded rather than left implicit.* The action
**parameter** is still stated verbatim in the goal. It has to be: the parameter is not the
target, and withholding it would hand the rules baseline a `parameter_mismatch` on every
record of the subclass — reopening exactly the exact-match shortcut this amendment exists
to close. "Never by an ID or parameter value that appears verbatim" is therefore read as
constraining how the goal identifies its *target*, not as forbidding the goal to state the
value being written. A test asserts the parameter is present in the goal, with that reason
attached.

*Templates.* 38 of the 40 template families carry the subclass, spanning all 8 domains,
including the five named in the proposal: ticketing/close ("the ticket the customer
replied to this morning"), refund_sandbox/refund ("the duplicate charge"), retrieval/tag
("the passage that supports the refund claim"), calendar/reschedule ("the meeting the
client asked to push") and database_update/status ("the record for the account flagged
overdue"). Covering nearly every family, rather than only five, is what guarantees the
subclass reaches dev, validation **and** test under a random family split instead of
depending on luck.

*Two families are excluded, and the exclusion is derived rather than chosen.*
`ticketing__merge` and `issue_tracker__link_pr` take an entity-shaped identifier
(`TKT-1002`, `PR-884`) as their action parameter. That identifier enters the goal's entity
set, and the rules baseline then fires `entity_mismatch` on the acted-on ID — for the
wrong reason, and on the matched control as well. Those families would not be clean, so
they do not carry the subclass. A test asserts the exclusion set is exactly the set of
families with that property.

*Oracle fields.* `described_target` (the goal's referent, as a string), `acted_target`
(the ID the agent operated on), `resolved_target_id` (the ID the description resolves to)
and a one-line `human_resolvable_note` stating how a careful reader tells them apart from
the goal and the evidence alone. `described_target` is the descriptor string, not the
resolved ID — the resolved ID necessarily appears in the candidate listing, so the
verbatim guarantee below would be unstatable against it.

*Volume.* Reallocated **within** the existing positives. Label balance is untouched:
exactly `records / 4` per label, as before. 35% of `unsupported_success` carries the new
subclass and 65% is spread over the seven rule-computable faults — exactly 35 / 65 at 400
records and 88 / 162 at 1000. The 60/20/20 family split and per-split label balance are
unchanged.

*Addition beyond the proposal, and why it was necessary.* A matched negative control,
`semantic_target_match`, under `supported_success`: identical goal shape, identical
candidate listing, agent acts on the **right** target. Drawn at the same count as the
mismatch fault. Without it the probe-and-act *shape* is itself a label — every trace with
that shape is a positive — and a bag of n-grams can flag the whole subclass without
reading anything, making a meaningless 0.78 flag rate look like detection. This changes
the fault mix inside `supported_success` (the 60/40 none/retry split now applies to the
records that are not controls) but not the label balance. **It immediately earned its
keep:** see the result below.

*Guarantees, structural rather than metric, in `tests/test_semantic_subclass.py` and
enforced at generation time so a violation is a loud failure rather than a quiet number:*

1. For every `semantic_target_mismatch` record, `described_target` does not appear
   verbatim (case-insensitively) in any tool argument or tool result in the inference
   view. That is what leaves an exact-match check nothing to match on.
2. The rules baseline fires neither `entity_mismatch` nor `parameter_mismatch` on any of
   them — and in fact fires **no** signal at all, so it does not flag them.
3. The rules baseline does not flag the matched control either.
4. The subclass spans at least 4 domains and reaches dev, validation and test.
5. No family carrying it is label-homogeneous.
6. The enumeration probe is advertised in the tool schema of **every** record of a family,
   not only the semantic ones. A tool that appeared alongside one fault type only would be
   a label sitting in the tool schema.

Generation raises `GenerationError` on any violation of 1–4.

*Blinded audit.* The audit sheet now marks a deterministic **stratified sample**
(`in_sample`), up to 8 traces per (label, fault) cell. Every cell is sampled at the same
rate, so membership leaks nothing about a trace's label and the sheet stays blind. The
point is that `semantic_target_mismatch` and its control are guaranteed in front of an
auditor rather than sampled in by luck, because that subclass now carries the whole
finding. The report records sample coverage and names any cell with no audited trace.

*Threshold re-freeze (on validation, before test was reopened).* Swept
0.30 / 0.40 / 0.45 / 0.50 / 0.55 / 0.60 / 0.70 for both offline providers on the
regenerated validation split at both stages. `rules` sits on a flat plateau — identical
precision, recall and F1 from 0.30 to 0.60 at both 400 and 1000 — so 0.50 is not a
knife-edge. `tfidf` peaks slightly lower (F1 0.7407 at 0.40 against 0.7174 at 0.50 on the
1000-record validation split), but spending the single permitted adjustment to flatter a
baseline would weaken the comparison it exists to support. **Thresholds remain 0.50 for
every provider.** The permitted one-time adjustment is therefore still unused.

*Two defects found while implementing this, both fixed, neither a threshold or split
change:*

- `_confirmed()` wrote a literal `status: "confirmed"` key **after** the applied
  parameter. For the one family whose `param_key` is itself `status`
  (`database_update__status`), that silently replaced the applied parameter with the
  string `"confirmed"`, so the rules baseline saw a `parameter_mismatch` on every record
  of that family including its supported successes. It went unnoticed because that family
  fell in the dev split. The parameter is now written last; the event already carries
  `status="ok"`, so the marker was redundant and the parameter was not.
- `build_report` picked the most recent run per provider without checking the run had been
  scored against the dataset being reported on. Regenerating the corpus leaves earlier runs
  behind, and they would have been mixed in silently. Runs whose `dataset_sha256` does not
  match are now skipped and named in the report.

*Result, against the prediction.* Predicted: `rules` and `tfidf` near 0.00 flag rate on
the subclass, and `rules` total AUPRC no longer 1.0. On the regenerated 1000-record test
split:

| | before A6 | after A6 |
|---|---|---|
| `rules` AUPRC | 1.0000 | **0.8200** |
| `rules` recall | 1.0000 | 0.6400 |
| `rules` flag rate on `semantic_target_mismatch` | n/a | **0.00** |
| `tfidf` flag rate on `semantic_target_mismatch` | n/a | 0.78 |
| `tfidf` flag rate on the matched control | n/a | 0.78 |

`rules` behaved exactly as predicted: 0.00, and total AUPRC off the ceiling. `tfidf` did
**not** — it flags 0.78 of the subclass. But it flags 0.78 of the matched control too.
Separation, the difference of the two, is **+0.00 for both baselines**. TF-IDF has learned
the probe-and-act shape and gets no discrimination out of it; its 0.78 is not detection.
Reported without the control, that 0.78 would have read as a baseline partially solving
the subclass. This is the result the control was added to make visible, and the report now
carries the two rows side by side with the separation computed.

---

## 8. What this design can and cannot establish

Stated before the test split was opened, and repeated in every generated report.

**Can:** whether a model matches a strong deterministic checker on false-success faults
that are cleanly computable from an ordered trace; how calibrated its probabilities are;
how it behaves per fault type; what it costs and how fast it is.

**Cannot:** whether false-success detection is solved.

*Superseded in part by A6, and the original wording is kept above so the change is
visible.* Before A6 the generator constructed only faults that code *can* compute exactly,
a deterministic baseline was near-perfect, and the section said flatly that the interesting
regime — traces where a rule baseline fails, because the reference is by description rather
than by a field that can be compared — was absent. A6 puts 35% of the positives into that
regime, and the deterministic baseline's AUPRC fell from 1.0000 to 0.8200 accordingly.

What still holds, restated for the corpus as it now stands:

- The other 65% of positives remain exactly rule-computable, and a checker remains
  near-perfect on them. A headline AUPRC is a blend of the two regimes and should not be
  read as a single difficulty.
- `semantic_target_mismatch` is one *shape* of semantic reference failure — a candidate
  listing present in the trace, exactly one candidate satisfying the description, the
  wrong one acted on. Real traces also fail through partial observability (the evidence
  needed is simply absent), through genuine ambiguity (no candidate is clearly right), and
  through references that resolve only against context outside the trace. None of those
  are in this corpus.
- Every trace is still synthetic and every label still comes from construction. The
  blinded audit is what turns a construction rule into a checked label, and until it is
  filled in the `semantic_target_mismatch` labels rest on the generator alone — which
  matters more now than before, because that subclass carries the finding.

---

## Amendment A7 — the evaluation moves onto real, labelled traces

Recorded 2026-09-20, before any run on the new corpus was scored.

### What changed and why

Up to A6 every trace was synthetic and every label came from construction. That
bounded what any result here could claim, and §8 said so. A7 moves the headline
onto a real corpus with labels the environment computed, and keeps the synthetic
set — unchanged — as a controlled diagnostic. Both are now first-class: each owns
disjoint artifact paths and its own runs root, so they cannot be mixed into one
report by accident.

### The corpus, and a correction to the brief

The work was specified as ingesting "the AppWorld false-success corpus from the
paper" (arXiv:2606.09863). **That corpus is not released.** The paper publishes
no data and states that it uses AppWorld's publicly released experiment outputs.

So the ingest reads AppWorld's own release — `experiment-outputs-0.1.3.bundle`,
SHA-256 `e5ec6367…`, 8,190 trajectories over 4 architectures and 4 model
families — and reimplements the rule the paper documents. The paper's reported
figures are recorded in `external_reference` in `config/eval.yaml`, shown
labelled as the paper's, and were **not** reproduced here.

### Licences, checked before anything was read

| Source | Licence | Verdict |
|---|---|---|
| AppWorld experiment outputs | Apache-2.0, plus an encrypted-redistribution condition on the protected bundle | ingested |
| tau2-bench | MIT | permitted, no adapter written |
| `cx-cmu/agent_trajectories` | none declared, access-gated | **refused** |

The HuggingFace fallback was refused rather than used. It declares no licence
and is gated; unclear is not permission. That refusal is encoded, not just
documented: `require_licence` raises and the ingest stops, and there is a test
for it. AppWorld's extra condition constrains *public redistribution*; this
harness publishes nothing and `data/appworld/` is git-ignored, so it is met by
not redistributing.

### The label rule

Derived from the environment, never from the assistant's wording. The claim is
the structured `status` the agent passed to `apis.supervisor.complete_task`; the
truth is the pass/fail AppWorld's evaluator computed by asserting against the
app databases after the episode.

- `status=success` and ground truth false → `unsupported_success`
- `status=success` and ground truth true → `supported_success`
- `status` in {fail, failure} → `reported_failure_or_uncertainty`
- no `status` → `no_success_claim`

An unrecognised status is an error, not a guess. A success claim with no ground
truth is dropped and counted, not assumed. Restricted to the two self-assessing
architectures, because only they can express an honest failure and therefore be
distinguished from a false success.

Result: 3,507 trajectories, 49.9% positive. That prevalence is roughly balanced
by accident, not by design; it is the data.

### Split

Task-disjoint by AppWorld **scenario**, not by task id. A task id is
`<scenario>_<variation>` and the variations are paraphrases of one task, so
splitting on the task would let a paraphrase of a training task reach test. 195
scenarios, 117/39/39. The split hash is frozen at ingest and recorded in
`data/appworld/provenance.json`.

### Baseline strengthened

`tfidf_gbm` (TF-IDF + gradient-boosted trees) is added and is now the baseline
the headline is stated against. The paper reports this family of detector
beating every LLM judge it tested; measuring Jev against the weak rule checker
instead would be a straw man. The rule baseline and the logistic-regression
baseline are kept and still reported.

### Thresholds

`tfidf_gbm` is frozen at 0.5, the same default as every other provider, set
before the test split was opened on the real corpus. No threshold was tuned on
real test data.

### Context budget

Real traces are long enough to overflow Jev's 64k/32k budget. The reduction rule
(`middle-elision-v1`) is documented, deterministic, applied in a fixed order, and
fully accounted: every reduction is written to the run's `truncation.jsonl`,
counted in the manifest, checksummed, re-verified by `verify-run`, and shown in
the caveat. The evaluator is told a trace was shortened by an explicit marker
event, so a reduced trace never presents as a whole one.

### Caveat honesty

The construction-label and not-independently-checked lines are **removed on real
data**, because on a corpus with programmatic ground truth they are false.
Leaving a false warning up is not a conservative choice: it trains a reader to
skip the warnings that are true. Caveats are now generated from the loaded run's
provenance rather than written into the page, so a synthetic run keeps its
warnings and a real run cannot inherit them. Tested both ways.

### What this still cannot establish

- The corpus is one benchmark, one environment family, structured API calls with
  programmatic rewards. Whether false success behaves the same in unstructured
  or open-ended settings is untested here.
- Restricted to self-assessing architectures, which is a selected slice of agent
  behaviour, not a random one.
- The sample is not the paper's sample, and the difference has not been fully
  explained — see the README. No number here should be read as reproducing a
  published one.
- Prevalence on this corpus is ~50%. Production prevalence is far lower, and the
  prior-shift projection remains a projection.

### A7 addendum — a redaction bug that silently dropped 117 traces

Found while checking why the report said 585 traces scored when the test split
holds 702, and recorded here because it changed reported numbers.

Layer 3 of `redact.py` masks "long opaque tokens", `[A-Za-z0-9_-]{32,}`. Real
AppWorld trace ids embed the model family, and
`deepseek-coder-33b-instruct_together` is 36 characters that match it exactly.
Every affected `trace_id` was rewritten to `[REDACTED]` on the way to disk, so
117 of 702 predictions no longer joined to the record they were made about and
were dropped from every metric — silently, and not at random: the loss was
exactly the DeepSeek `test_challenge` slice.

Two changes, both tested:

1. Structural identifiers (`trace_id`, `run_id`, `model_id`, `git_commit`, and
   the recorded artifact paths) are exempt from the *heuristic* layers. They
   are join keys, not values. The exact live-secret scrub still applies to
   them, and `contains_secret` still refuses to write a run if a live secret
   survives anywhere, exempt fields included. A URL is deliberately **not**
   exempt: it can carry a key as a query parameter.
2. `evaluate_run` now **refuses** to report when any prediction's `trace_id` is
   absent from the dataset, rather than scoring the remainder. A broken join is
   how a biased subset gets reported as a whole split.

Every run on the real corpus was discarded and redone after the fix, including
the paid one. The numbers in `reports/appworld/` are post-fix and cover all 702
test traces.

### A7 addendum — a second gate, because the frozen one stopped being the question

§2 freezes the useful-signal gate against the `rules` baseline. That was the
right bar when `rules` was the only free comparator. It is not the right bar
now: on the real corpus Jev clears it comfortably (ΔAUPRC lower bound +0.253)
while losing to TF-IDF with gradient boosting (−0.114, CI [−0.145, −0.085]).
Reporting only the frozen gate would have let a PASS stand for a result that
is, on the question actually asked, a loss.

The frozen gate is **not** moved or retuned — retuning a preregistered gate
after seeing the data is the thing preregistration exists to prevent. Instead
a second gate, `beats_strong_baseline`, is reported beside it and labelled in
the artifact itself as not preregistered. Its rule: a provider beats the
strong free baseline only when the *whole* 95% interval on ΔAUPRC is above
zero. An interval straddling zero is `no_difference`, not a win. The
useful-signal gate's own note now states what clearing it does and does not
mean, so the weaker gate cannot be quoted on its own.

Also fixed while adding it, and material to any gate reading: the deployment
criteria were computed from numpy scalars and serialised as `numpy.bool_`,
which lands in JSON as the **string** `"False"`. A non-empty string is truthy,
so `all()` would have passed a run that failed only on calibration. Every
criterion is now coerced with `bool()`, with a regression test that fails on a
calibration-only failure. No previously reported gate verdict changes: the
runs that failed on ECE also failed on precision and recall.

### A7 addendum — a price manifest edited in place, and the runs it invalidated

Found by `verify-runs`, which is the outcome the check exists for, and recorded
here because it briefly made eight runs unverifiable.

`config/prices-2026-09-19.json` gained two entries mid-session — the local
gradient-boosted evaluators added by A7, both at zero cost. The edit was made
in place. Every run already recorded against that file had stored its SHA-256,
so all eight synthetic runs began failing verification with a price-manifest
mismatch. No cost figure was wrong: the two added models are free, and no
metered rate changed. The integrity check could not know that, and was right
not to assume it.

Resolution, in the order it matters:

1. `prices-2026-09-19.json` is restored byte-for-byte to the content the
   synthetic runs recorded. They verify again. A superseded manifest is kept,
   not deleted, precisely so old runs stay checkable.
2. `prices-2026-09-20.json` carries the additions under its own date and names
   what it supersedes. `config/eval.yaml` points at it.
3. The AppWorld runs were re-run against the new manifest rather than having
   their manifests rewritten. Editing a recorded artifact to match a changed
   input is the failure this harness measures; it is not available as a fix.
4. Three tests now pin the discipline: the active manifest is the one
   `eval.yaml` names, it declares its date and what it supersedes, and a
   supersede may add a model but never silently reprices one. The last is the
   invariant that made this incident harmless, so it is now enforced rather
   than relied on.

---

## Amendment A8 — a general LLM judge as a third paid arm

Added 2026-09-20, after the A7 real-data run. The question this amendment adds
is **not** the one the harness was built for, and it is kept separate from it.

A7 established that on the AppWorld test split Jev does not beat the free
TF-IDF baselines. That answers "is the paid lane worth its bill against free
code". It does not answer "is a *typed* false-success model better than simply
asking a general LLM the same questions", which is the question a reader who
has already accepted the cost of an LLM call actually has. A8 adds that arm.

### What was fixed in advance, and why each choice

The comparison is only worth making if the two paid arms differ in the *model*
and in nothing else. Each parity term below is enforced by construction rather
than by convention, with the test that pins it:

| Term | How it is enforced | Test |
|---|---|---|
| Same inference view | The judge is handed `jev_http.build_state` — the same function, not a reimplementation | `test_the_judge_is_sent_byte_identical_state_to_jev` |
| Same four questions | Prompt *and* response JSON schema are generated from the same frozen `config/questions.json` Jev is sent | `test_the_four_questions_come_from_the_same_frozen_file_jev_is_sent` |
| Same decoding | `temperature: 0`, strict JSON schema, no examples | `test_the_request_is_deterministic_and_strictly_schema_bound` |
| Same context budget | The same `ContextBudget` object; a trace shortened for one is shortened identically for the other | `test_both_arms_reduce_an_oversized_trace_identically` |
| Same primary score | `P(unsupported_success)` read from the verdict, so AUPRC is comparable | `test_the_primary_score_is_p_unsupported_success_exactly_as_it_is_for_jev` |
| Same split, zero-shot | The A7 task-disjoint test split; no training, no fitting, no calibration | shared `select_split` path |

**One asymmetry cannot be engineered away, and is stated rather than hidden.**
Jev returns `probabilities` from the API. A general model's probabilities are
**self-reported** inside its own JSON answer — the model is asked for a
distribution and writes one. Both are used identically as the primary score, so
the ranking metrics are comparable, but they are not the same kind of number.
Reading a general judge's calibration (ECE) as comparable to Jev's would be
over-reading; the ranking comparison (AUPRC, AUROC, recall) is the one this
amendment is making.

### The model ID is chosen, never invented

Section 5 of this document refuses to write an unverified model name into code,
and that refusal stands. The `openai` block in `config/eval.yaml` ships with
`model_id: ""`, which records `not_run`. The ID is obtained by calling the
account's own listing — `jev-eval list-models --provider openai`, which is
`GET /v1/models` — and copied from what that prints. It is a config value,
overridable per run with `--model`, and the ID the **API reports back** is what
goes into the run manifest, not the one that was asked for, so an alias that
resolved elsewhere is visible in the artifact.

`GPT-5.6-Terra` remains a permanently refusing stub. It was never a verified
model ID and adding a working OpenAI arm does not make it one.

### Verified request shape

Unlike the general-model slot in section 5, this one **is** verified, against
OpenAI's own published API description rather than a blog post or recall:
`https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml`
(MIT, spec version 2.3.0, fetched 2026-09-20), SHA-256
`d4e842399c2e9e63aca79aa9cb04b766f1cb9e1e98d48300a22e3dddc2add6fb`.

| Item | Verified value |
|---|---|
| Listing | `GET /v1/models` → `{object, data[{id, object, created, owned_by}]}` |
| Endpoint | `POST /v1/chat/completions`, server `https://api.openai.com/v1` |
| Request | `{model, messages[{role, content}], temperature, max_completion_tokens, response_format}` |
| Structured output | `response_format: {type: "json_schema", json_schema: {name, schema, strict}}` |
| Response | `{model, choices[{index, finish_reason, message{content, refusal}}], usage}` |
| Usage | `{prompt_tokens, completion_tokens}` |
| Roles | `developer` replaces `system` on current models |

The test fixture is labelled `_provenance: CONSTRUCTED FROM THE DOCUMENTED
SCHEMA` exactly as the Jev one is. No number in it is a measurement.

### Cost, and what is *not* claimed about it

OpenAI bills for output as well as input; Jev does not. The pre-spend projection
therefore prices both legs, and the Jev band is unchanged because its output
rate is zero (`test_jevs_projection_is_unchanged_by_the_output_leg`).

**No OpenAI price is recorded in this repository unless it has been read from
the vendor's own published pricing.** Until then the manifest carries
`OPENAI_GENERAL_JUDGE_UNPRICED` with `verified: false`, and the run records real
token counts with `cost_usd: null`. The report and the dashboard print
**"unavailable"**, never `$0.00`: a lane that looks free because nobody priced
it is the most expensive kind of wrong answer this harness could give.

### A change to the cost path that this amendment forced

`cost_usd` refuses a model the manifest has never heard of. That is right for a
pinned model whose price should already be on record, and it is **wrong** for a
model chosen at run time from an account listing: it would raise on the first
response and destroy a 702-request measurement to report a missing rate.

So the recording path is now `recorded_cost_usd`, which returns `None` for an
unknown model. `cost_usd` is unchanged and still refuses. Both share one
arithmetic helper, and `verify-run` recomputes through `recorded_cost_usd`, so a
recorded cost and a recomputed one cannot drift — while a run that recorded a
*number* for a model with no rate is still reported as a problem
(`test_a_cost_invented_for_an_unpriced_model_is_caught_by_verify_run`). Nothing
is guessed on either path; the difference is only whether a missing entry stops
the run or is written down.

The paid-run plan now says loudly, before spending, when the chosen model has no
entry at all.

### Cost control, and the stop-and-ask threshold

Deliberately narrower than the Jev run:

- **A single pass over the test split**, not the 5-repeat concurrency sweep.
  Repeats measure stability, not accuracy, and no point metric in this harness
  reads them, so the sweep would have bought variance data at several times the
  price of the answer being sought.
- **A 1-trace smoke first.** For Jev, `smoke` is a connectivity check. For a
  general judge that proves nothing worth knowing, so the general-judge smoke
  sends one real trace, reports the tokens it was billed, and prints the
  full-split projection twice: the usual chars-per-token band, and a point
  estimate calibrated on that measurement. Neither is ever recorded as a cost.
- **`cost_projection.confirm_above_gbp: 5.0`.** A run whose projected **upper**
  bound exceeds it stops and says so, even with `--confirm-paid`. The top of the
  band is the number worth being asked about; the optimistic end is not.
  `--accept-cost-over-threshold` is the only way past, and it prints that it was
  used.

### A new price manifest, on the same date

The OpenAI entry could not be appended to `prices-2026-09-20.json`: seven runs
record that file's SHA-256, and the A7 addendum above is the record of what an
in-place edit costs. It could not reuse the filename either. So
`prices-2026-09-20-2.json` supersedes it, carries `revision: 2` to make the
ordering explicit rather than inferred from a filename, and the supersede tests
now walk the **whole chain** rather than the newest hop — a price that moved two
manifests ago and moved back is still a repricing.

### What a result either way will mean

The paper this work sits beside reports TF-IDF beating every LLM judge it
tested. A7 found Jev below TF-IDF. The expectation is therefore that **both**
LLM arms sit below the free baselines, and that expectation is not a prediction
this amendment is allowed to protect.

- **Jev above the judge, interval clear of zero** — the typed model is worth
  something over a general one *at whatever it costs*, which is reported beside
  it. It still says nothing about the free baselines.
- **Interval spanning zero** — no difference established. This is the most
  likely outcome at this sample size and is reported as such, not as a win.
- **Judge above Jev** — reported in the same space and at the same size as a win
  would get. The dashboard panel and the report section are built by one code
  path for all three outcomes.

The verdict is read from the **interval**, never the point estimate, exactly as
the strong-baseline verdict is.

### What A8 does not establish

- Nothing about any general model other than the exact ID recorded in the run
  manifest, at temperature 0, with this prompt, on this split.
- Nothing about a general model given a *better* prompt. Parity with Jev's four
  questions is the point, and it is also a ceiling: a prompt tuned for the
  general model would be a different experiment, and would break the comparison
  it was tuned to win.
- Nothing about calibration parity, for the self-reported-probability reason
  above.
- Nothing publishable. OpenAI's terms restrict publication of benchmark results
  and so do TypeSafe's. This arm makes the MCA restriction *stricter*, not
  looser, and the footer stays.

---

## Amendment A9 — the general judge's first live run failed 702/702, and what was done about it

**Logged 2026-09-20, after the failed run `openai-test-20260920T042135Z-6e941d` and
before the re-run. It changes no threshold, no split, no label, no metric and no
question. It changes what the adapter does when a provider refuses a request
parameter, and it adds a run artifact.**

### The real error

The run recorded 702 predictions and 702 errors. The manifest said `HTTP 400` and
nothing more, and `attempts.jsonl` carried the status without a message. The message
was in each prediction's `raw_response`, and it was the same one 661 times:

```json
{"error": {"code": "unsupported_value",
           "param": "temperature",
           "type": "invalid_request_error",
           "message": "Unsupported value: 'temperature' does not support 0.0 with this
                       model. Only the default (1) value is supported."}}
```

The other 41 were `429 rate_limit_exceeded` on the organisation's 500k tokens-per-minute
limit — a consequence of 702 doomed requests being retried at concurrency 4, not a
second cause.

**None of the other candidate diagnoses were the problem, and the artifacts say so
rather than this paragraph asserting it.** The failed run's own `raw_request.body`
shows `max_completion_tokens` (not `max_tokens`), the `developer` role, and the strict
`json_schema` response format; `POST /v1/chat/completions` returned `400` on
`temperature` alone. Replaying that exact recorded body against the live API with
`temperature` removed and nothing else changed returned `200`, `finish_reason: stop`,
`model: gpt-5.6-terra`, and a schema-valid answer. Chat Completions is the right
endpoint for this model, the token parameter was already correct, and the strict schema
is accepted. The single blocker was the decoding temperature.

`GET /v1/models` on this account lists `gpt-5.6-terra` (`owned_by=system`), so the
model ID is verified by the account's own listing, as §A8 requires. The `gpt_terra`
block in `config/eval.yaml` refused on the grounds that no official ID had been
verified; that ground no longer holds, so the reason is corrected. The block stays
disabled, because the verified path to the model is the `openai` block and there
should only be one.

### What changed

**The adapter may now drop a decoding parameter the model refuses — and only a
decoding parameter.** On a `400` whose `error.code` is `unsupported_value` or
`unsupported_parameter` and whose `error.param` is in `DROPPABLE_PARAMS`
(`temperature`, `top_p`), the adapter removes that parameter, re-sends **the same
trace, the same four questions and the same strict schema**, and records the refusal.
The refusal is learned once per run, so later traces do not spend a request — or a
slice of the TPM limit — rediscovering it.

**A `400` naming anything else is still an error and the trace is still dropped.**
`response_format`, `messages` and `max_completion_tokens` are explicitly *not*
droppable, and `build_chat_body` raises if asked to omit them. Relaxing the strict
output schema or the shared inference view to make a call succeed would quietly turn
this arm into a different experiment, which is the failure mode this whole repository
is built against.

**Every deviation is an artifact, not a sentence in a report.** A new
`deviations.jsonl` sits beside every run — written on finalise, checksummed in the
manifest, verified by `verify-run`, and **empty for a run that sent exactly what
`config/eval.yaml` specifies**, which is a positive statement rather than a silence.
It carries the parameter, the value requested, what was applied, the HTTP status, the
provider's error code and the provider's **verbatim** message, so a reader checks the
deviation against the API rather than against this repository's summary of it. The
manifest gains `deviation_count`, `deviations_path`, `deviations_sha256`.

**The report and the dashboard stop claiming what is no longer true.** The
general-judge note read "asked the same four questions … at temperature 0" as a
constant string. It is now built from the runs: when an arm's temperature was refused
it reads "at the decoding settings each provider accepted", the deviation is stated in
the same sentence, a deviation table renders under the comparison, and the caveat strip
carries "a provider refused a request parameter" so it cannot be missed by a reader who
never opens the panel.

### The fairness that was kept, and the one that was not

Kept, and kept structurally rather than by promise: the same `build_state`, the same
frozen `questions.json`, the same strict four-answer schema, the same `ContextBudget`
with the same truncation accounting, the same task-disjoint test split, zero-shot, and
`P(unsupported_success)` as the primary score.

**Not kept: decoding temperature.** Jev is called at temperature 0. `gpt-5.6-terra`
will not accept it and runs at its own default, which its error message states is 1.
This is a real asymmetry and it is not repairable from this side — the alternative is
no general-judge arm at all. It is recorded in the run, printed by `run`, stated in the
report, and carried in the caveat bar. **Any figure from this arm must be read as a
single sample from a non-deterministic decoder, not as a deterministic evaluation.**
In particular, this arm's `repeat_stability` is no longer a property of the model alone.

### Cost

Unchanged and still unavailable. `config/prices-2026-09-20-2.json` has no verified rate
for `gpt-5.6-terra`, so every prediction records `cost_usd: null` and the manifest total
is `null`. **No rate was guessed and none was added.** Tokens are recorded, so the bill
can be reconstructed exactly if a published rate with a citation is later added to a new
dated manifest. The report prints "unavailable", never zero.

### Tests

New tests assert: the recorded 400 body reproduces the original failure without the
fallback; a refused `temperature` is dropped and the trace is still scored; the second
request differs from the first in that one key and nothing else; the deviation carries
the provider's verbatim message; both rounds' attempts stay in the record; the refusal
is learned once across traces; a refusal of `response_format`, `messages` or
`max_completion_tokens` is an error with no second request; `build_chat_body` refuses to
omit a non-droppable parameter; a clean run records no deviation; every run writes a
deviation account; and `verify-run` detects a tampered one.

### A9 addendum — one trace lost to reasoning tokens, and the ceiling that caused it

The re-run (`openai-test-20260920T044110Z-cdf451`) produced **702 predictions, 1 error,
4 retries**. The single error is not a transport failure and not a refusal. It is:

```
ParseFailureError: finish_reason is 'length': the answer was cut off by
max_completion_tokens
```

on `appworld/legacy_function_calling_agent/gpt-4o-2024-05-13/test_normal/09b0ee6_1`, a
21,465-token prompt. The response body says why: `completion_tokens: 1200`, of which
`reasoning_tokens: 1200`, and `content: ""`. The model spent the entire output
allowance thinking and emitted no answer at all.

**The comment in `config/eval.yaml` was written against a non-reasoning model and is
wrong for this family.** It says 1200 output tokens is "enough for the four answers as
strict JSON", and for the *answer* it is — the measured distribution over the 701
parsed traces is median 207, p95 567, p99 875, max 1,106, and **nothing else hit the
ceiling.** But for a reasoning model `max_completion_tokens` bounds reasoning *plus*
answer, so the four answers are competing with the model's own deliberation for the
same budget, and on the longest prompt in the split the deliberation won outright.

**This run is reported as it stands: 701 of 702 traces scored, 1 parse failure, and the
failure counted rather than parsed out of what arrived.** The harness behaved correctly
— it refuses to read a verdict out of a truncated answer — and the deployment gate
`deployment_max_parse_failures: 0` fails for this arm on that one trace, which is the
gate doing its job.

**The ceiling was deliberately not raised for this run, and that is a choice worth
recording.** Raising it is defensible: it is a ceiling, not an instruction, so a larger
one does not change what the model is asked, and it would likely produce a
zero-parse-failure run. But it would be a second live pass over the full split, and the
value of the frozen config is that it is not adjusted after seeing the result it
produced. The finding is logged here instead, with the measured distribution, so the
decision to raise it — if taken — is taken in the open and before the next run rather
than as a silent repair of this one.

### A9 addendum — a paired comparison that assumed both arms scored everything

Building the report on this run crashed:

```
ValueError: Found input variables with inconsistent numbers of samples: [702, 701]
```

`compare()` took two score vectors and zipped them positionally, which is correct only
while every provider scores every trace. Until now every provider did. The judge's one
parse failure made the two vectors different lengths for the first time, and the
crash is the *lucky* outcome: had the lengths happened to match while the trace sets
differed, every row after the first gap would have been compared against the wrong
trace, silently.

`compare()` now takes the two summaries, intersects them on trace id via
`pair_on_shared_traces()`, and computes the paired bootstrap on the traces **both**
arms scored. The count of paired traces, the count that could not be paired and the
first few unpaired ids go into the comparison entry, and the report prints a line
saying the interval is on the smaller sample whenever anything was dropped. Each arm's
own row in the headline table is unchanged — it is still that arm's own `n_scored`.
Three tests pin it, including one asserting that a gap shifts nothing: with the judge
missing trace `b`, its `c` must be matched against the baseline's `c` and not its `b`.

---

## Amendment A10 — a low-label learning curve, because "needs no labels" is the last open question

**Not preregistered.** Added after the real-data result, and reported beside the frozen
gates rather than in place of them. Recorded here in full because it was designed with
the full-size result already known, which is exactly the situation in which a
measurement can be quietly shaped to produce a wanted answer.

### Why it exists

A7 and A9 left the comparison lopsided in one direction: on the AppWorld test split both
free TF-IDF baselines beat both paid arms on AUPRC, comfortably and with intervals
entirely clear of zero. That settles the question the harness was built to ask, but it
does not settle the question a reader will ask next. A classifier needs labels. A
zero-shot judge does not. If the crossover were at a few thousand labels, "the free
baseline wins" would be true and useless — nobody starting from nothing has a few
thousand labels.

So the honest form of the remaining question is: **at what label budget does the free
classifier overtake the zero-shot arms, and is there any budget at which they lead?**

### Method, frozen before the sweep was run

- **Budgets.** 10, 25, 50, 100, 200, 400 and the whole train split, each with five
  subsample seeds.
- **Sampling is task-first, never row-first.** Distinct AppWorld task ids are shuffled
  with the seed; tasks are admitted in that order, each contributing **all** of its rows
  to per-label buckets; the quota is then taken off the front of each bucket. A row can
  only be reached by admitting its whole task, so sibling trajectories of one task are
  never split across a fitted set and an unseen one. Sampling draws from the **train**
  split only, so the validation and test tasks are unreachable; the evaluators' own
  `FitLeakageError` is still armed underneath.
- **Stratified.** Each budget keeps the train split's own label prevalence, by largest
  remainder, with a floor of one example of every label present. The floor is not
  cosmetic: a subset missing a class produces a pipeline whose `predict_proba` has fewer
  columns than the calibration set has classes, which fails deep inside scikit-learn
  with an error that names nothing relevant.
- **Scored on the frozen test split**, with the same records, threshold and metric
  functions the report uses everywhere else.
- **The zero-shot arms are quoted, not re-run.** Jev and the general judge have no
  training size to sweep. Their flat lines come from their already-recorded
  `predictions.jsonl` on the same split, scored through the same `evaluate_run`. The
  sweep constructs no paid evaluator, reads no key and opens no socket; a paid provider
  passed to `learning-curve` is refused before anything is fitted, and any prediction
  carrying a cost or a metered token stops the curve.
- **The band** is a Student-t 95% interval on the mean over seeds. It states how much
  the subsample draw moves the result. It is **not** a confidence interval for the
  test-split estimate, and the flat zero-shot lines are point estimates whose own
  uncertainty is not drawn. Both facts are carried into the report and the dashboard as
  caveats rather than left to the reader.

### Two decisions that could have been taken the flattering way

**What `N` counts.** `N` is the number of labelled examples the classifier is *fit* on.
Probability calibration still runs on the whole validation split at every budget,
exactly as it does for the reported full-size runs. The alternative — carving the
calibration set out of the same `N` — would have made the headline stricter, but it
would also have changed the fitting recipe between the curve and the run the curve's
last point is supposed to reproduce, and at `N`=10 it cannot produce a four-class
calibration set at all. The recipe was held fixed and the accounting stated instead:
every artifact records `n_calibration` beside `n_train`, and the caveat travels with the
number. **A reader who wants the strict figure should add the validation count to `N`.**

**What counts as a crossing.** A budget is only reported as the crossover if it *and
every larger budget* stay above the line. A curve that pops above the reference at one
size and falls back below it at the next has not crossed over, and reporting the first
touch would be the most flattering possible reading of seed noise.

### The check that would have caught a broken sweep

At the full budget every seed draws the identical subset, so the band collapses to a
point there by construction — and that point must reproduce the already-recorded
full-size run exactly. It does: `tfidf` 0.9510 AUPRC / 0.9037 recall and `tfidf_gbm`
0.9602 / 0.8503, the same figures `reports/appworld/report.md` prints for those runs.
A subsampler that had quietly changed the fit, the ordering or the feature rendering
would have shown up as a disagreement at that point.

### What this cannot establish

The curve says how many labels this classifier needs **on this corpus, with this task
structure, at this prevalence**. AppWorld tasks come in families of 16–18 sibling
trajectories, so a task-disjoint budget of 50 labels is drawn from three or four tasks,
not fifty independent situations. A corpus with one trajectory per task would need more
labels to reach the same coverage. The crossover is a property of this sample, not a
constant.
