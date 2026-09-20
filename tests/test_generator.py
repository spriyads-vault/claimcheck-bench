from __future__ import annotations

from collections import Counter

import pytest

from false_success_eval.generate import (
    GenerationError,
    generate_records,
    render_trace,
    write_dataset,
)
from false_success_eval.hashing import canonical_json, sha256_file
from false_success_eval.schemas import (
    LABEL_ORDER,
    REAL_FAULTS,
    SYNTHETIC_FAULTS,
    FaultType,
    Label,
)
from false_success_eval.templates import N_FAMILIES

RECORDS = 160
SEED = 20260919


def test_generation_is_byte_identical_for_a_fixed_seed(tmp_path):
    first = write_dataset(generate_records(RECORDS, SEED), tmp_path / "a.jsonl")
    second = write_dataset(generate_records(RECORDS, SEED), tmp_path / "b.jsonl")
    assert first == second
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert sha256_file(tmp_path / "a.jsonl") == first


def test_a_different_seed_produces_a_different_dataset(tmp_path):
    a = write_dataset(generate_records(RECORDS, SEED), tmp_path / "a.jsonl")
    b = write_dataset(generate_records(RECORDS, SEED + 1), tmp_path / "b.jsonl")
    assert a != b


def test_labels_are_exactly_balanced():
    records = generate_records(RECORDS, SEED)
    counts = Counter(r.label for r in records)
    assert set(counts) == set(LABEL_ORDER)
    assert set(counts.values()) == {RECORDS // len(LABEL_ORDER)}


def test_every_fault_type_appears():
    """Every *synthetic* fault, and none belonging to an ingested corpus.

    The enum also carries the real-corpus fault names so one schema serves both
    kinds of dataset. The generator must produce all of the former and none of
    the latter: a real fault name in a generated dataset would mean the two had
    been mixed.
    """
    records = generate_records(RECORDS, SEED)
    seen = {r.fault_type.value for r in records}
    assert seen == set(SYNTHETIC_FAULTS)
    assert not seen & set(REAL_FAULTS)


def test_all_eight_domains_appear():
    records = generate_records(RECORDS, SEED)
    assert len({r.domain for r in records}) == 8


def test_no_template_family_maps_to_a_single_label():
    """A family split must never become a label split."""
    records = generate_records(RECORDS, SEED)
    per_family: dict[str, set[Label]] = {}
    for record in records:
        per_family.setdefault(record.template_family, set()).add(record.label)
    assert len(per_family) == N_FAMILIES
    homogeneous = {f for f, labels in per_family.items() if len(labels) < 2}
    assert not homogeneous, f"label-homogeneous families: {sorted(homogeneous)}"
    # Stronger than the contract requires, and true by construction:
    assert all(len(labels) == len(LABEL_ORDER) for labels in per_family.values())


def test_generation_refuses_a_label_homogeneous_corpus():
    """The contract is enforced, not merely asserted downstream.

    Labels here stay globally balanced -- 10 families per label, 4 records each --
    so only the family-homogeneity check can catch it.
    """
    from false_success_eval.generate import _enforce_contracts
    from false_success_eval.templates import FAMILIES

    family_label = {f.family_id: LABEL_ORDER[f.index % len(LABEL_ORDER)] for f in FAMILIES}
    doctored = tuple(
        record.model_copy(update={"label": family_label[record.template_family]})
        for record in generate_records(RECORDS, SEED)
    )
    assert len(Counter(r.label for r in doctored).values()) == len(LABEL_ORDER)
    assert set(Counter(r.label for r in doctored).values()) == {RECORDS // len(LABEL_ORDER)}

    with pytest.raises(GenerationError, match="label-homogeneous"):
        _enforce_contracts(doctored, RECORDS)


def test_record_count_must_divide_into_families():
    with pytest.raises(GenerationError, match="not divisible"):
        generate_records(RECORDS + 1, SEED)


def test_record_count_must_allow_every_family_to_span_every_label():
    with pytest.raises(GenerationError, match="at least"):
        generate_records(N_FAMILIES * 2, SEED)


def test_trace_ids_are_unique_and_content_derived():
    records = generate_records(RECORDS, SEED)
    ids = [r.trace_id for r in records]
    assert len(set(ids)) == len(ids)
    assert all(len(i) == 16 for i in ids)


def test_events_are_sequentially_ordered_and_end_with_an_assistant_message():
    for record in generate_records(RECORDS, SEED):
        seqs = [e.seq for e in record.events]
        assert seqs == sorted(seqs)
        assert record.events[-1].type.value == "assistant_message"
        assert record.events[-1].text


def test_oracle_carries_ground_truth_detail():
    records = generate_records(RECORDS, SEED)
    wrong_entity = [r for r in records if r.fault_type is FaultType.wrong_entity]
    assert wrong_entity
    for record in wrong_entity:
        assert record.oracle["requested_entity"] != record.oracle["changed_entity"]

    wrong_param = [r for r in records if r.fault_type is FaultType.wrong_parameter]
    assert wrong_param
    for record in wrong_param:
        assert record.oracle["requested_parameter"] != record.oracle["applied_parameter"]


def test_render_trace_is_single_line():
    record = generate_records(RECORDS, SEED)[0]
    rendered = render_trace(record)
    assert "\n" not in rendered
    assert record.goal in rendered


def test_canonical_json_is_stable():
    record = generate_records(RECORDS, SEED)[0]
    payload = record.model_dump(mode="json")
    assert canonical_json(payload) == canonical_json(payload)
