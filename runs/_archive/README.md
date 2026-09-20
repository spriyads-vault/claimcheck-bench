# Archived runs

Runs kept as history, not as results. Nothing here is scored by the current report.

---

## `pre-a6/` — the runs that motivated amendments A5 and A6

### What they are

Four offline runs on the **test** split, scored against the dataset as it stood
**before amendment A6** added the `semantic_target_mismatch` subclass and its matched
control. Stage directories are preserved, so each run sits at
`pre-a6/<records>/<run_id>/` exactly as it did under `runs/<records>/`.

| stage | run | provider | n | dataset | `dataset_sha256` |
|---|---|---|---|---|---|
| 400 | `rules-test-20260920T002338Z-d9a45f` | rules | 80 | `data/dataset-400.jsonl` | `0b3a77cee2a193d6e401f57af25d4a112b2957a1c74f17dff3e9d22d6cc3c343` |
| 400 | `tfidf-test-20260920T002340Z-9d6f73` | tfidf | 80 | `data/dataset-400.jsonl` | `0b3a77cee2a193d6e401f57af25d4a112b2957a1c74f17dff3e9d22d6cc3c343` |
| 1000 | `rules-test-20260920T002401Z-b7e460` | rules | 200 | `data/dataset-1000.jsonl` | `0ab631cbe6fd324a7788eee35b702e4caa404ef62c4d1868065981cb20330314` |
| 1000 | `tfidf-test-20260920T002403Z-a8aad6` | tfidf | 200 | `data/dataset-1000.jsonl` | `0ab631cbe6fd324a7788eee35b702e4caa404ef62c4d1868065981cb20330314` |

Those two dataset hashes no longer correspond to any file in `data/`. The paths in the
manifests are the paths those datasets occupied at the time; regenerating at A6 replaced
both files in place. The manifests are what make these runs self-describing.

### Why they are retained

They are the record of the result that motivated the amendments: on the pre-A6
1000-record test split the deterministic `rules` baseline reached **AUPRC 1.0000,
precision 1.0000, recall 1.0000**. Every `unsupported_success` fault in that corpus was
exactly rule-computable — a status field, an entity mismatch, a parameter mismatch, a
zero-row update — so a checker had nothing left to miss and a model had no headroom in
which to demonstrate a contribution over one. That is a property of the dataset, not of
the baseline, and it is why A6 exists.

`rules-test-...-b7e460/predictions.jsonl` still carries the signature directly: 50 flags
and 150 passes on a 200-trace split whose prevalence is exactly 50 positives. The
400-record run is the same shape at 20 / 80.

The headline figures above are quoted from the stage-2 report those runs produced. That
report has since been rebuilt over the A6 dataset at `reports/stage2/`, and the pre-A6
dataset is not retained, so the figures **cannot be recomputed from this directory**.
Treat them as a quotation, not as a measurement this repository can reproduce. The
before/after comparison is recorded in `preregistration.md` §7, amendment A6.

A5 (zero false positives collapsing the prior-shift projection to 1.0 at every prevalence)
came out of the same report.

### How to verify them

Use `--archived`:

```bash
jev-eval verify-run --run runs/_archive/pre-a6/1000/rules-test-20260920T002401Z-b7e460 --archived
```

That verifies each run against **its own manifest** — predictions and attempts checksums,
the price manifest, and the cost arithmetic recomputed from the dated rates — and skips
the comparison against whatever dataset currently sits at the manifest's path. All four
runs pass. The manifest's own `dataset_sha256` is left exactly as recorded; the flag
changes what is compared, never what is stored.

**Without `--archived` these runs fail, and that is correct.** The default comparison is
against the dataset on disk, which A6 deliberately replaced:

```
FAIL  dataset sha256 mismatch: manifest 0b3a77cee2a193d6..., file 6025a98749aec63d...
```

That mismatch means the archive is correctly labelled, not that a run is corrupt. It stays
the default because for a *live* run a dataset that has moved underneath it is a real
problem. Do not "fix" it by re-pointing a manifest at the current dataset — that would
destroy the only thing these runs are kept for.

`jev-eval verify-runs` walks a *stage* root (`runs/400`, `runs/1000`) and does not reach
this directory.

### They are not reportable

`build_report` discovers runs with `runs_root.glob("*/manifest.json")` against
`runs/<records>/`, one level deep. `runs/_archive/pre-a6/<records>/<run_id>/` is outside
every stage root and is never discovered. Independently of that, `build_report` skips any
run whose `dataset_sha256` does not match the dataset being reported on and names it in
the report — so even if one of these were moved back into a stage root, it would be
excluded rather than silently blended with current runs.
