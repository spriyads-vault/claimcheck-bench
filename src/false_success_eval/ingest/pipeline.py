"""Orchestration for ``jev-eval ingest``: licence, map, split, record.

The order here is the contract. The licence is checked before the source is
opened; the split is computed before anything is written; the provenance file is
written last and describes exactly what the other files contain. A caveat shown
in the dashboard is generated from that provenance, so the page cannot claim
something about the data that the ingest did not record.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..generate import make_splits
from ..hashing import canonical_json, sha256_file, sha256_obj
from ..schemas import (
    LABEL_ORDER,
    DatasetKind,
    DatasetProvenance,
    Splits,
    TraceRecord,
)
from .appworld import (
    BUNDLE_SHA256,
    BUNDLE_URL,
    BUNDLE_VERSION,
    SELF_ASSESSING_ARCHITECTURES,
    IngestStats,
    build_records,
)
from .labelling import APPWORLD_LABEL_RULE
from .licences import require_licence

#: Proportions of split units, matching the synthetic 24/8/8 family split.
DEV_FRACTION = 0.60
VALIDATION_FRACTION = 0.20


@dataclass(frozen=True)
class IngestResult:
    provenance: DatasetProvenance
    splits: Splits
    records: tuple[TraceRecord, ...]
    stats: IngestStats
    dataset_path: Path
    splits_path: Path
    provenance_path: Path


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_sizes(n_families: int) -> tuple[int, int, int]:
    """Family counts for dev/validation/test, mirroring the synthetic 60/20/20.

    Every split gets at least one family, and the three always sum to the total,
    so no family is silently left out of every split.
    """
    if n_families < 3:
        raise ValueError(f"{n_families} split units is too few to make three disjoint splits")
    dev = max(1, round(n_families * DEV_FRACTION))
    validation = max(1, round(n_families * VALIDATION_FRACTION))
    if dev + validation >= n_families:
        dev = n_families - 2
        validation = 1
    test = n_families - dev - validation
    return dev, validation, test


def write_dataset_jsonl(records: Sequence[TraceRecord], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record.model_dump(mode="json")))
            handle.write("\n")
    return sha256_file(path)


def ingest_appworld(
    *,
    source: Path,
    dataset_path: Path,
    splits_path: Path,
    provenance_path: Path,
    seed: int,
) -> IngestResult:
    """Ingest AppWorld's released experiment outputs. Licence is checked first."""
    licence = require_licence("appworld")

    records, stats = build_records(source)
    if not records:
        raise RuntimeError(
            f"no usable trajectories found under {source}. Expected the "
            "self-assessing architectures "
            f"{list(SELF_ASSESSING_ARCHITECTURES)} with evaluations/*.json beside "
            "each split's tasks/ directory."
        )

    dataset_sha = write_dataset_jsonl(records, dataset_path)

    families = sorted({r.template_family for r in records})
    dev, validation, test = split_sizes(len(families))
    splits = make_splits(tuple(records), seed, dev, validation, test, dataset_sha)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(splits.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    label_counts = {
        label.value: sum(1 for r in records if r.label is label) for label in LABEL_ORDER
    }
    fault_counts: dict[str, int] = {}
    for record in records:
        fault_counts[record.fault_type.value] = fault_counts.get(record.fault_type.value, 0) + 1

    provenance = DatasetProvenance(
        dataset_id="appworld",
        kind=DatasetKind.real,
        source_name="AppWorld released experiment outputs",
        source_url=BUNDLE_URL,
        source_version=f"experiment-outputs-{BUNDLE_VERSION}",
        source_sha256=BUNDLE_SHA256,
        retrieved_utc=_now(),
        licence=licence,
        label_rule=APPWORLD_LABEL_RULE,
        split_unit="appworld_scenario_id",
        split_unit_description=(
            "An AppWorld task id is <scenario>_<variation>, and the three variations "
            "of a scenario are paraphrases of one task. Splits are disjoint by "
            "scenario, so a paraphrase of a training task cannot appear in test. "
            f"{len(families)} scenarios split {dev} dev / {validation} validation / "
            f"{test} test."
        ),
        split_sha256=sha256_obj(splits.model_dump(mode="json")),
        n_records=len(records),
        label_counts=label_counts,
        fault_counts=dict(sorted(fault_counts.items())),
        notes=(
            "The paper that characterises this failure mode (arXiv:2606.09863, "
            "CC BY 4.0) releases no corpus of its own; it states that it uses "
            "AppWorld's publicly released experiment outputs. This ingest reads the "
            "same release and reimplements the documented rule. Figures quoted from "
            "the paper are external context and were not reproduced here.",
            "Restricted to the two self-assessing architectures "
            f"{list(SELF_ASSESSING_ARCHITECTURES)}: only they write a structured "
            "completion status, so only they can express an honest failure and be "
            "distinguished from a false success.",
            "tool_schema entries carry no required-argument list. The release "
            "records the calls an agent made, not the signatures it could have "
            "called, and inventing one would state something the source does not.",
            "An AppWorld agent writes code, not a closing paragraph. The final "
            "assistant message is the agent's own last code cell verbatim; no prose "
            "was synthesised for it.",
            f"Dropped during ingest: {stats.dropped_no_goal} with no task "
            f"instruction, {stats.dropped_no_ground_truth} with a success claim but "
            f"no ground-truth record, out of {stats.seen} trajectories seen.",
        ),
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return IngestResult(
        provenance=provenance,
        splits=splits,
        records=records,
        stats=stats,
        dataset_path=dataset_path,
        splits_path=splits_path,
        provenance_path=provenance_path,
    )


def synthetic_provenance(
    *,
    dataset_id: str,
    records: Sequence[TraceRecord],
    splits: Splits,
    seed: int,
) -> DatasetProvenance:
    """Provenance for a generated dataset, so both kinds describe themselves alike."""
    label_counts = {
        label.value: sum(1 for r in records if r.label is label) for label in LABEL_ORDER
    }
    fault_counts: dict[str, int] = {}
    for record in records:
        fault_counts[record.fault_type.value] = fault_counts.get(record.fault_type.value, 0) + 1
    return DatasetProvenance(
        dataset_id=dataset_id,
        kind=DatasetKind.synthetic,
        source_name="Deterministic generator in this repository",
        source_url="src/false_success_eval/generate.py",
        source_version=f"seed={seed}",
        retrieved_utc=_now(),
        licence=None,
        label_rule=None,
        split_unit="template_family",
        split_unit_description=(
            "8 domains x 5 action families = 40 template families, split by family "
            "so a paraphrase of a tuned template cannot reach the test split."
        ),
        split_sha256=sha256_obj(splits.model_dump(mode="json")),
        n_records=len(records),
        label_counts=label_counts,
        fault_counts=dict(sorted(fault_counts.items())),
        notes=(
            "Labels are construction labels: the generator knows what it built. "
            "They have not been independently checked until the blinded audit sheet "
            "is filled in.",
            "This dataset is a controlled diagnostic. It isolates fault subclasses "
            "that real corpora do not separate, and it is not evidence about "
            "production traffic.",
        ),
    )


def load_provenance(path: Path) -> DatasetProvenance | None:
    """Read a provenance sidecar, or None for a dataset written before they existed."""
    if not path.exists():
        return None
    try:
        return DatasetProvenance.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
