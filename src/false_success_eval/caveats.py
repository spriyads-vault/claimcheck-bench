"""Caveats generated from the loaded run, never written into the page.

The rule this module exists to enforce: **a caveat is only shown if it is true
of the data currently loaded.** On the synthetic diagnostic the labels really
are construction labels and really have not been independently checked, so that
warning belongs on screen. On a real corpus with programmatic ground truth both
of those statements are false, and leaving them up would be its own dishonesty
-- it would understate the evidence rather than overstate it, but it would still
be the page saying something untrue.

So nothing here is a constant string keyed to a screen. Every line is derived
from the dataset's provenance record and the run's own manifest, which means a
synthetic run and a real run served by the same binary show different caveats,
and neither can show the other's.

The output is a small structured object: one ``strip`` line for the compact bar,
and an ordered list of ``items`` for the panel the info icon expands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .schemas import DatasetKind, DatasetProvenance

#: The one line that is true of every run this harness produces, whatever the
#: data. It is a contractual restriction, not a property of the evidence.
MCA_NOTICE = (
    "Internal only. Performance figures are covered by TypeSafe's MCA and must not be "
    "published without written permission."
)


@dataclass(frozen=True)
class CaveatItem:
    """One expandable caveat: a short title and the sentence behind it."""

    key: str
    title: str
    body: str
    severity: str = "info"

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "body": self.body, "severity": self.severity}


@dataclass(frozen=True)
class Caveats:
    strip: str
    items: list[CaveatItem] = field(default_factory=list)
    kind: str = "synthetic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strip": self.strip,
            "kind": self.kind,
            "items": [item.as_dict() for item in self.items],
            "mca": MCA_NOTICE,
        }


def _fmt_int(value: float | int | None) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _ci_item(ci: dict[str, Any] | None, n_positives: int, n_traces: int) -> CaveatItem:
    """State the interval width, because a point estimate alone invites over-reading."""
    if not ci or ci.get("lower") is None or ci.get("upper") is None:
        return CaveatItem(
            key="interval",
            title="No interval computed",
            body=(
                f"This view has {_fmt_int(n_positives)} positives in "
                f"{_fmt_int(n_traces)} scored traces. No paired bootstrap interval is "
                "attached to the headline yet, so read the difference as a point "
                "estimate with unknown precision."
            ),
            severity="warn",
        )
    lower = float(ci["lower"])
    upper = float(ci["upper"])
    width = upper - lower
    crosses = lower <= 0.0 <= upper
    return CaveatItem(
        key="interval",
        title=f"95% interval spans {width:.3f}",
        body=(
            f"The paired bootstrap interval on the headline difference is "
            f"[{lower:+.3f}, {upper:+.3f}], a width of {width:.3f} over "
            f"{_fmt_int(n_positives)} positives in {_fmt_int(n_traces)} scored traces."
            + (
                " It crosses zero, so this data does not establish a difference in "
                "either direction."
                if crosses
                else " It does not cross zero."
            )
        ),
        severity="warn" if crosses else "info",
    )


def _truncation_item(truncation: dict[str, Any] | None) -> CaveatItem | None:
    if not truncation:
        return None
    count = int(truncation.get("truncated_traces", 0) or 0)
    total = int(truncation.get("n_traces", 0) or 0)
    if count == 0:
        return CaveatItem(
            key="truncation",
            title="No trace was shortened",
            body=(
                f"Every one of the {_fmt_int(total)} traces fitted the evaluator's "
                "context budget whole. Nothing was dropped."
            ),
        )
    rule = str(truncation.get("rule", "")) or "the documented reduction rule"
    dropped = int(truncation.get("dropped_events", 0) or 0)
    share = (count / total * 100.0) if total else 0.0
    return CaveatItem(
        key="truncation",
        title=f"{_fmt_int(count)} traces shortened to fit the context budget",
        body=(
            f"{_fmt_int(count)} of {_fmt_int(total)} traces ({share:.1f}%) exceeded the "
            f"budget and were reduced by {rule}, removing {_fmt_int(dropped)} events in "
            "total from the middles of those traces. The goal, the opening calls and "
            "the closing claim were kept in every case, and an explicit marker event "
            "tells the evaluator the trace was shortened. Every reduction is listed in "
            "the run's truncation file and counted in its manifest."
        ),
        severity="warn",
    )


def _deviation_item(deviation: dict[str, Any] | None) -> CaveatItem | None:
    """What the run did not send as configured, in the provider's own words.

    ``None`` when no run wrote a deviation account: the caveat then says nothing
    about deviations, which is not the same claim as "there were none".
    """
    if deviation is None:
        return None
    rows = list(deviation.get("deviations") or [])
    if not rows:
        return CaveatItem(
            key="deviation",
            title="No provider refused a request parameter",
            body=(
                "Every paid arm sent exactly the request config/eval.yaml specifies, "
                "including temperature 0. Nothing was dropped to make a call succeed."
            ),
        )
    sentences = []
    for row in rows:
        sentences.append(
            f"{row.get('run_model_id') or row.get('model_id')} "
            f"(arm `{row.get('provider', '')}`) refused {row['parameter']}="
            f"{row['requested']!r} with HTTP {row['http_status']} {row['error_code']}: "
            f'"{row["error_message"]}" It was {row["applied"]}, and the '
            f"{int(row.get('occurrences', 1)):,} refusals are recorded in the run's "
            "deviation file."
        )
    return CaveatItem(
        key="deviation",
        title=f"{len(rows)} request parameter(s) the provider refused",
        body=(
            "This run did not send exactly what the frozen config describes. "
            + " ".join(sentences)
            + " Nothing that changes the trace, the four questions or the output "
            "schema can be dropped this way, so the arms still see the same "
            "evidence and answer the same thing -- but they are no longer decoding "
            "under identical settings, and no figure here should be read as if "
            "they were."
        ),
        severity="warn",
    )


def _real_items(provenance: DatasetProvenance) -> list[CaveatItem]:
    items: list[CaveatItem] = []

    licence = provenance.licence
    licence_line = (
        f"{provenance.source_name} ({provenance.source_version}), "
        f"licensed {licence.name} ({licence.spdx})."
        if licence
        else f"{provenance.source_name} ({provenance.source_version})."
    )
    body = licence_line
    if licence:
        body += f" Verified {licence.verified_utc} from {licence.verified_from}"
        if licence.conditions:
            body += " Conditions: " + " ".join(licence.conditions)
        if licence.attribution:
            body += f" Attribution: {licence.attribution}"
    items.append(
        CaveatItem(
            key="source",
            title="Data source and licence",
            body=body,
        )
    )

    rule = provenance.label_rule
    if rule is not None:
        mapping = "; ".join(f"{cond} -> {label}" for cond, label in rule.mapping)
        items.append(
            CaveatItem(
                key="label_rule",
                title="Labels come from the environment, not the wording",
                body=(
                    f"{rule.description} Claim field: {rule.claim_field}. Ground truth: "
                    f"{rule.truth_field}. Rule: {mapping}."
                ),
            )
        )

    items.append(
        CaveatItem(
            key="split",
            title="Task-disjoint split",
            body=provenance.split_unit_description
            + f" Split hash {provenance.split_sha256[:16]}, frozen at ingest.",
        )
    )

    if provenance.notes:
        items.append(
            CaveatItem(
                key="scope",
                title="What this corpus does and does not cover",
                body=" ".join(provenance.notes),
            )
        )
    return items


def _synthetic_items(
    provenance: DatasetProvenance | None, audit: dict[str, Any] | None
) -> list[CaveatItem]:
    items = [
        CaveatItem(
            key="source",
            title="Synthetic diagnostic, not production traffic",
            body=(
                "These traces were built by the generator in this repository to "
                "isolate fault subclasses a real corpus does not separate. They are a "
                "controlled diagnostic and say nothing about how often any of this "
                "happens in production."
            ),
            severity="warn",
        )
    ]
    audit_text = (audit or {}).get("text")
    items.append(
        CaveatItem(
            key="label_rule",
            title="Construction labels",
            body=str(
                audit_text
                or "Labels are construction labels from the generator and have not "
                "been independently checked."
            ),
            severity="warn",
        )
    )
    if provenance is not None:
        items.append(
            CaveatItem(
                key="split",
                title="Family-disjoint split",
                body=provenance.split_unit_description
                + f" Split hash {provenance.split_sha256[:16]}.",
            )
        )
    return items


def build_caveats(
    *,
    provenance: DatasetProvenance | None,
    audit: dict[str, Any] | None = None,
    headline_ci: dict[str, Any] | None = None,
    n_positives: int = 0,
    n_traces: int = 0,
    truncation: dict[str, Any] | None = None,
    deviation: dict[str, Any] | None = None,
    extra: Sequence[CaveatItem] = (),
) -> Caveats:
    """Build the caveat block for one loaded run.

    ``provenance`` decides the whole shape: a real corpus never gets the
    construction-label line and a synthetic one never gets the licence line.
    """
    kind = provenance.kind.value if provenance is not None else DatasetKind.synthetic.value

    if provenance is not None and provenance.kind is DatasetKind.real:
        items = _real_items(provenance)
        licence = provenance.licence
        strip_source = provenance.source_name
        if licence:
            strip_source += f", {licence.spdx}"
        strip = f"{strip_source} · labels from programmatic ground truth · task-disjoint split"
    else:
        items = _synthetic_items(provenance, audit)
        strip = (
            "Synthetic diagnostic · construction labels, not independently checked · "
            "family-disjoint split"
        )

    truncation_item = _truncation_item(truncation)
    if truncation_item is not None:
        items.append(truncation_item)
        if truncation_item.severity == "warn":
            strip += " · some traces shortened to fit context"

    deviation_item = _deviation_item(deviation)
    if deviation_item is not None:
        items.append(deviation_item)
        if deviation_item.severity == "warn":
            strip += " · a provider refused a request parameter"

    items.append(_ci_item(headline_ci, n_positives, n_traces))
    items.extend(extra)
    items.append(
        CaveatItem(
            key="mca",
            title="Publication restriction",
            body=MCA_NOTICE,
            severity="warn",
        )
    )
    return Caveats(strip=strip, items=items, kind=kind)
