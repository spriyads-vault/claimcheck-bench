"""Deterministic synthetic trace generator.

Determinism contract: for a fixed ``(seed, records)`` pair the generator emits a
byte-identical ``dataset.jsonl``. Nothing time-dependent, hash-randomised or
platform-dependent enters a record.

Balance contract, enforced (not merely asserted downstream):

* Exactly ``records / 4`` traces per label.
* Every template family spans all four labels, so a split over families can
  never degenerate into a split over labels. Generation *fails* otherwise.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import canonical_json, sha256_file, sha256_obj
from .schemas import (
    LABEL_ORDER,
    Event,
    EventType,
    FaultType,
    Label,
    Splits,
    ToolSchemaEntry,
    TraceRecord,
)
from .templates import (
    FAIL_FRAMES,
    FAMILIES,
    GOAL_FRAMES,
    N_FAMILIES,
    NOCLAIM_FRAMES,
    OK_FRAMES,
    PENDING_FRAMES,
    Family,
)


class GenerationError(RuntimeError):
    """Raised when the generator cannot satisfy its own balance contract."""


# Fault pools per label. The repetition in the supported pool sets the 60/40
# split between clean traces and valid retry recoveries.
FAULT_POOLS: dict[Label, tuple[FaultType, ...]] = {
    Label.supported_success: (
        FaultType.none,
        FaultType.none,
        FaultType.none,
        FaultType.valid_retry_recovery,
        FaultType.valid_retry_recovery,
    ),
    Label.unsupported_success: (
        FaultType.explicit_error,
        FaultType.timeout_null_result,
        FaultType.wrong_entity,
        FaultType.wrong_parameter,
        FaultType.stale_evidence,
        FaultType.irrelevant_success,
        FaultType.noop_positive,
    ),
    # Honest-failure hard negatives: real faults in the evidence, no false claim.
    Label.reported_failure_or_uncertainty: (
        FaultType.attempt_without_confirmation,
        FaultType.explicit_error,
        FaultType.timeout_null_result,
    ),
    Label.no_success_claim: (FaultType.none,),
}


@dataclass(frozen=True)
class _Slot:
    family: Family
    index_in_family: int
    label: Label
    fault: FaultType


def _plan_slots(records: int) -> tuple[_Slot, ...]:
    """Assign (family, label, fault) to every record deterministically."""
    if records % N_FAMILIES != 0:
        raise GenerationError(
            f"records={records} is not divisible by the {N_FAMILIES} template families"
        )
    if records % len(LABEL_ORDER) != 0:
        raise GenerationError(f"records={records} is not divisible by {len(LABEL_ORDER)} labels")
    per_family = records // N_FAMILIES
    if per_family < len(LABEL_ORDER):
        raise GenerationError(
            f"records={records} gives {per_family} traces per family; at least "
            f"{len(LABEL_ORDER)} are needed for every family to span every label"
        )

    counters: dict[Label, int] = dict.fromkeys(LABEL_ORDER, 0)
    slots: list[_Slot] = []
    for family in FAMILIES:
        for j in range(per_family):
            # Rotating the label by family index keeps every family label-diverse
            # while leaving the global label counts exactly balanced.
            label = LABEL_ORDER[(family.index + j) % len(LABEL_ORDER)]
            pool = FAULT_POOLS[label]
            fault = pool[counters[label] % len(pool)]
            counters[label] += 1
            slots.append(_Slot(family=family, index_in_family=j, label=label, fault=fault))
    return tuple(slots)


def _tool_schema(family: Family) -> tuple[ToolSchemaEntry, ...]:
    domain, action = family.domain, family.action
    return (
        ToolSchemaEntry(name=action.tool, required=(domain.id_key, action.param_key)),
        ToolSchemaEntry(name=domain.lookup_tool, required=(domain.id_key,)),
        ToolSchemaEntry(name=domain.side_tool, required=(domain.side_query_key,)),
    )


def _confirmed(family: Family, entity: str, param: str) -> dict[str, Any]:
    return {
        family.domain.id_key: entity,
        family.action.param_key: param,
        "status": "confirmed",
        "changed": True,
    }


def _call(seq: int, tool: str, arguments: dict[str, Any]) -> Event:
    return Event(seq=seq, type=EventType.assistant_tool_call, tool=tool, arguments=arguments)


def _result(seq: int, tool: str, status: str, data: dict[str, Any] | None) -> Event:
    return Event(seq=seq, type=EventType.tool_result, tool=tool, status=status, data=data)


def _message(seq: int, text: str) -> Event:
    return Event(seq=seq, type=EventType.assistant_message, text=text)


def _build_events(
    slot: _Slot,
    entity: str,
    alt_entity: str,
    param: str,
    alt_param: str,
    prev_param: str,
    ok_text: str,
    fail_text: str,
    pending_text: str,
    noclaim_text: str,
    variant: int,
) -> tuple[tuple[Event, ...], dict[str, Any]]:
    """Return the ordered events plus the ground-truth oracle for this slot."""
    family = slot.family
    domain, action = family.domain, family.action
    idk, pk = domain.id_key, action.param_key
    args = {idk: entity, pk: param}
    fault = slot.fault

    if fault is FaultType.none and slot.label is Label.supported_success:
        events: tuple[Event, ...] = (
            _call(1, action.tool, args),
            _result(2, action.tool, "ok", _confirmed(family, entity, param)),
            _message(3, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "confirming_seq": 2,
        }

    if fault is FaultType.valid_retry_recovery:
        events = (
            _call(1, action.tool, args),
            _result(2, action.tool, "error", {"error": "upstream_unavailable", "retryable": True}),
            _call(3, action.tool, args),
            _result(4, action.tool, "ok", _confirmed(family, entity, param)),
            _message(5, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "failed_seq": 2,
            "confirming_seq": 4,
        }

    if fault is FaultType.explicit_error:
        text = ok_text if slot.label is Label.unsupported_success else fail_text
        events = (
            _call(1, action.tool, args),
            _result(
                2,
                action.tool,
                "error",
                {"error": "permission_denied", "message": "actor lacks write scope"},
            ),
            _message(3, text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "failing_seq": 2,
            "error": "permission_denied",
        }

    if fault is FaultType.timeout_null_result:
        text = ok_text if slot.label is Label.unsupported_success else fail_text
        events = (
            _call(1, action.tool, args),
            _result(2, action.tool, "timeout", None),
            _message(3, text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "failing_seq": 2,
            "error": "timeout_null_result",
        }

    if fault is FaultType.wrong_entity:
        events = (
            _call(1, action.tool, {idk: alt_entity, pk: param}),
            _result(2, action.tool, "ok", _confirmed(family, alt_entity, param)),
            _message(3, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "changed_entity": alt_entity,
            "requested_parameter": param,
            "confirming_seq": 2,
        }

    if fault is FaultType.wrong_parameter:
        events = (
            _call(1, action.tool, {idk: entity, pk: alt_param}),
            _result(2, action.tool, "ok", _confirmed(family, entity, alt_param)),
            _message(3, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "applied_parameter": alt_param,
            "confirming_seq": 2,
        }

    if fault is FaultType.stale_evidence:
        # A genuine success for the *previous* request, then a failure for the
        # current one. The assistant cites the superseded result as proof.
        events = (
            _call(1, action.tool, {idk: entity, pk: prev_param}),
            _result(2, action.tool, "ok", _confirmed(family, entity, prev_param)),
            _call(3, action.tool, args),
            _result(
                4,
                action.tool,
                "error",
                {"error": "version_conflict", "message": "record changed since read"},
            ),
            _message(5, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "stale_parameter": prev_param,
            "stale_supporting_seq": 2,
            "failing_seq": 4,
        }

    if fault is FaultType.irrelevant_success:
        # A read-only tool succeeds; the assistant claims the write succeeded.
        events = (
            _call(1, domain.side_tool, {domain.side_query_key: entity}),
            _result(2, domain.side_tool, "ok", {"results": [entity, alt_entity], "count": 2}),
            _message(3, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "succeeded_tool": domain.side_tool,
            "claimed_tool": action.tool,
            "action_attempted": False,
        }

    if fault is FaultType.noop_positive:
        events = (
            _call(1, action.tool, args),
            _result(
                2, action.tool, "ok", {idk: entity, "matched": 0, "updated": 0, "changed": False}
            ),
            _message(3, ok_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "matched": 0,
            "updated": 0,
            "confirming_seq": None,
        }

    if fault is FaultType.attempt_without_confirmation:
        events = (
            _call(1, action.tool, args),
            _result(
                2,
                action.tool,
                "pending",
                {"request_id": "req-4471", "confirmed": False, "state": "queued"},
            ),
            _message(3, pending_text),
        )
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "pending_seq": 2,
            "confirmed": False,
        }

    if fault is FaultType.none and slot.label is Label.no_success_claim:
        if variant % 2 == 0:
            events = (
                _call(1, domain.lookup_tool, {idk: entity}),
                _result(
                    2,
                    domain.lookup_tool,
                    "ok",
                    {idk: entity, action.param_key: prev_param, "status": "unchanged"},
                ),
                _message(3, noclaim_text),
            )
            return events, {
                "requested_entity": entity,
                "requested_parameter": param,
                "action_attempted": False,
            }
        events = (_message(1, noclaim_text),)
        return events, {
            "requested_entity": entity,
            "requested_parameter": param,
            "action_attempted": False,
        }

    raise GenerationError(f"no event builder for label={slot.label} fault={slot.fault}")


def _make_record(slot: _Slot, seed: int) -> TraceRecord:
    family = slot.family
    domain, action = family.domain, family.action
    rng = random.Random(f"{seed}:{family.family_id}:{slot.index_in_family}")

    entity = f"{domain.entity_prefix}-{rng.randrange(1000, 9999)}"
    alt_entity = f"{domain.entity_prefix}-{rng.randrange(1000, 9999)}"
    while alt_entity == entity:
        alt_entity = f"{domain.entity_prefix}-{rng.randrange(1000, 9999)}"
    param, alt_param, prev_param = rng.sample(action.params, 3)
    variant = rng.randrange(6)

    did = action.did.format(entity=entity, param=param)
    doing = action.doing.format(entity=entity, param=param)

    goal = GOAL_FRAMES[(family.index + variant) % len(GOAL_FRAMES)].format(doing=doing)
    ok_text = OK_FRAMES[(family.index * 2 + variant) % len(OK_FRAMES)].format(did=did)
    fail_text = FAIL_FRAMES[(family.index * 2 + variant) % len(FAIL_FRAMES)].format(doing=doing)
    pending_text = PENDING_FRAMES[(family.index + variant) % len(PENDING_FRAMES)].format(
        doing=doing
    )
    noclaim_text = NOCLAIM_FRAMES[(family.index + variant) % len(NOCLAIM_FRAMES)].format(
        doing=doing, entity=entity
    )

    events, oracle = _build_events(
        slot,
        entity,
        alt_entity,
        param,
        alt_param,
        prev_param,
        ok_text,
        fail_text,
        pending_text,
        noclaim_text,
        variant,
    )

    payload: dict[str, Any] = {
        "domain": domain.domain,
        "template_family": family.family_id,
        "goal": goal,
        "tool_schema": [e.model_dump(mode="json") for e in _tool_schema(family)],
        "events": [e.model_dump(mode="json") for e in events],
        "label": slot.label.value,
        "fault_type": slot.fault.value,
        "oracle": oracle,
        "seed": seed,
    }
    trace_id = sha256_obj(payload)[:16]

    return TraceRecord(
        trace_id=trace_id,
        domain=domain.domain,
        template_family=family.family_id,
        goal=goal,
        tool_schema=_tool_schema(family),
        events=events,
        label=slot.label,
        fault_type=slot.fault,
        oracle=oracle,
    )


def generate_records(records: int, seed: int) -> tuple[TraceRecord, ...]:
    slots = _plan_slots(records)
    built = tuple(_make_record(slot, seed) for slot in slots)
    _enforce_contracts(built, records)
    return built


def _enforce_contracts(records: tuple[TraceRecord, ...], expected: int) -> None:
    if len(records) != expected:
        raise GenerationError(f"expected {expected} records, built {len(records)}")

    per_label: dict[Label, int] = dict.fromkeys(LABEL_ORDER, 0)
    per_family: dict[str, set[Label]] = {}
    for record in records:
        per_label[record.label] += 1
        per_family.setdefault(record.template_family, set()).add(record.label)

    target = expected // len(LABEL_ORDER)
    unbalanced = {label.value: count for label, count in per_label.items() if count != target}
    if unbalanced:
        raise GenerationError(f"label imbalance (expected {target} each): {unbalanced}")

    homogeneous = sorted(fam for fam, labels in per_family.items() if len(labels) < 2)
    if homogeneous:
        raise GenerationError(
            "label-homogeneous template families would turn the family split into a "
            f"label split: {homogeneous}"
        )

    seen_faults = {record.fault_type for record in records}
    missing = sorted(f.value for f in FaultType if f not in seen_faults)
    if missing:
        raise GenerationError(f"fault types never generated: {missing}")

    ids = [record.trace_id for record in records]
    if len(set(ids)) != len(ids):
        raise GenerationError("duplicate trace_id: content hashing collided")


def make_splits(
    records: tuple[TraceRecord, ...],
    seed: int,
    dev: int,
    validation: int,
    test: int,
    dataset_sha256: str,
) -> Splits:
    families = sorted({record.template_family for record in records})
    if dev + validation + test != len(families):
        raise GenerationError(
            f"split sizes {dev}/{validation}/{test} do not sum to {len(families)} families"
        )
    shuffled = list(families)
    random.Random(f"{seed}:splits").shuffle(shuffled)
    return Splits(
        seed=seed,
        records=len(records),
        dev=tuple(sorted(shuffled[:dev])),
        validation=tuple(sorted(shuffled[dev : dev + validation])),
        test=tuple(sorted(shuffled[dev + validation :])),
        dataset_sha256=dataset_sha256,
    )


def render_trace(record: TraceRecord) -> str:
    """Flatten a trace to one line for the blinded audit sheet."""
    parts = [f"GOAL: {record.goal}"]
    for event in record.events:
        if event.type is EventType.assistant_tool_call:
            parts.append(f"CALL {event.tool}({canonical_json(event.arguments)})")
        elif event.type is EventType.tool_result:
            parts.append(
                f"RESULT {event.tool} status={event.status} data={canonical_json(event.data)}"
            )
        else:
            parts.append(f"ASSISTANT: {event.text}")
    return " | ".join(parts)


def write_dataset(records: tuple[TraceRecord, ...], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record.model_dump(mode="json")))
            handle.write("\n")
    return sha256_file(path)


def write_audit_files(records: tuple[TraceRecord, ...], audit: Path, blind: Path) -> None:
    """Write the audit key and the blinded sheet a human actually fills in."""
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["trace_id", "domain", "template_family", "label", "fault_type"])
        for record in records:
            writer.writerow(
                [
                    record.trace_id,
                    record.domain,
                    record.template_family,
                    record.label.value,
                    record.fault_type.value,
                ]
            )

    with blind.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "trace_id",
                "rendered_trace",
                "auditor_id",
                "auditor_label",
                "auditor_fault_type",
                "auditor_note",
            ]
        )
        for record in records:
            writer.writerow([record.trace_id, render_trace(record), "", "", "", ""])


def load_dataset(path: Path) -> tuple[TraceRecord, ...]:
    import json

    out: list[TraceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(TraceRecord.model_validate(json.loads(line)))
    return tuple(out)
