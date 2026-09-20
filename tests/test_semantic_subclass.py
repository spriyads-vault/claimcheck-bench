"""Guarantees for the semantic-clean fault subclass (amendment A6).

These are **structural**, not metric. They assert properties of the constructed
records, so they hold whatever any evaluator happens to score. A metric test
would only tell us what one baseline did on one split; these tell us the
subclass has no exact-match handle for *any* string or field comparison to grab.
"""

from __future__ import annotations

from collections import Counter

import pytest

from false_success_eval.evaluators.rules import RulesEvaluator
from false_success_eval.generate import (
    SEMANTIC_POSITIVE_SHARE,
    GenerationError,
    audit_sample,
    audit_stratum,
    evidence_strings,
    generate_records,
    make_splits,
)
from false_success_eval.schemas import FaultType, Label
from false_success_eval.semantic import ENTITY_SHAPED, EXCLUDED_FAMILIES, SPECS
from false_success_eval.templates import FAMILIES

RECORDS = 400
SEED = 20260919


@pytest.fixture(scope="module")
def corpus():
    return generate_records(RECORDS, SEED)


@pytest.fixture(scope="module")
def semantic(corpus):
    records = [r for r in corpus if r.fault_type is FaultType.semantic_target_mismatch]
    assert records
    return records


# -- the guarantee the subclass exists for ------------------------------
def test_described_target_never_appears_verbatim_in_the_evidence(semantic):
    """The property that makes an exact-match check unable to fire.

    The goal names its target only by description. If that description appeared
    anywhere in a tool argument or tool result, a string check could resolve the
    reference without reading anything, and the subclass would be no better than
    the seven faults it was added to complement.
    """
    for record in semantic:
        descriptor = record.oracle["described_target"].lower()
        assert descriptor
        for blob in evidence_strings(record):
            assert descriptor not in blob.lower(), (
                f"{record.trace_id} ({record.template_family}): described_target "
                f"{descriptor!r} is present in the evidence as {blob!r}"
            )


def test_the_goal_names_no_entity_id(semantic):
    """Indirection, not an ID. This is what empties the rules baseline's goal-entity set."""
    for record in semantic:
        assert not ENTITY_SHAPED.findall(record.goal), record.goal


def test_the_rules_baseline_fires_no_signal_at_all(semantic):
    """No exact-entity hit, no exact-parameter hit -- and in fact no hit of any kind."""
    rules = RulesEvaluator(threshold=0.5)
    for record in semantic:
        prediction = rules.predict(record.inference_view(), record.trace_id)
        signals = prediction.raw_response["signals"]
        assert signals["entity_mismatch"] is False, record.trace_id
        assert signals["parameter_mismatch"] is False, record.trace_id
        assert not any(signals.values()), (record.trace_id, signals)
        assert prediction.decision.value == "pass"


def test_the_action_parameter_is_stated_verbatim_in_the_goal(semantic):
    """Why the parameter is not withheld along with the target.

    The parameter is not the target. Withholding it would hand the rules
    baseline a `parameter_mismatch` on every record of the subclass, which is
    exactly the exact-match shortcut A6 exists to close.
    """
    for record in semantic:
        assert record.oracle["requested_parameter"].lower() in record.goal.lower()


def test_the_write_succeeds_and_confirms_the_agents_own_target(semantic):
    """No error status, no null result, no missing confirmation to key on."""
    for record in semantic:
        write, confirm, claim = record.events[2], record.events[3], record.events[4]
        assert write.type.value == "assistant_tool_call"
        assert confirm.type.value == "tool_result"
        assert confirm.status == "ok"
        assert confirm.data["changed"] is True
        acted = record.oracle["acted_target"]
        assert acted in write.arguments.values()
        assert acted in confirm.data.values()
        assert acted in claim.text


def test_the_described_target_resolves_to_exactly_one_candidate(semantic):
    """'Genuinely wrong, not ambiguous': the acted target is never the resolved one."""
    for record in semantic:
        listing = record.events[1].data["results"]
        ids = [row[k] for row in listing for k in row if k.endswith("_id")]
        assert len(ids) == len(set(ids)) == 3
        assert record.oracle["resolved_target_id"] in ids
        assert record.oracle["acted_target"] in ids
        assert record.oracle["acted_target"] != record.oracle["resolved_target_id"]


def test_every_record_carries_a_human_resolvable_note(semantic):
    for record in semantic:
        note = record.oracle["human_resolvable_note"]
        assert isinstance(note, str) and len(note) > 40
        assert "{param}" not in note


# -- composition --------------------------------------------------------
def test_the_subclass_is_35_percent_of_the_positives(corpus):
    positives = [r for r in corpus if r.label is Label.unsupported_success]
    semantic = [r for r in positives if r.fault_type is FaultType.semantic_target_mismatch]
    assert len(semantic) == round(SEMANTIC_POSITIVE_SHARE * len(positives))
    assert len(positives) - len(semantic) == 65


def test_label_balance_is_untouched(corpus):
    counts = Counter(r.label for r in corpus)
    assert set(counts.values()) == {RECORDS // 4}


def test_the_subclass_spans_at_least_four_domains(semantic):
    assert len({r.domain for r in semantic}) >= 4


def test_the_subclass_reaches_dev_validation_and_test(corpus, semantic):
    splits = make_splits(corpus, SEED, 24, 8, 8, "0" * 64)
    for name in ("dev", "validation", "test"):
        families = set(getattr(splits, name))
        present = [r for r in semantic if r.template_family in families]
        assert present, f"no semantic_target_mismatch record in the {name} split"


def test_no_family_carrying_the_subclass_is_label_homogeneous(corpus, semantic):
    """The existing family guard, restated against the new subclass specifically."""
    carriers = {r.template_family for r in semantic}
    per_family: dict[str, set[Label]] = {}
    for record in corpus:
        per_family.setdefault(record.template_family, set()).add(record.label)
    for family in carriers:
        assert len(per_family[family]) > 1, family


# -- the exclusion set is derived, not hand-picked ----------------------
def test_excluded_families_are_exactly_those_with_entity_shaped_parameters():
    """Why two families cannot carry the subclass.

    Their action parameter is itself an entity-shaped identifier, so it enters
    the goal's entity set and the rules baseline fires `entity_mismatch` on the
    acted-on ID. A subclass that a checker resolves is not a clean one.
    """
    derived = {
        family.family_id
        for family in FAMILIES
        if any(ENTITY_SHAPED.fullmatch(p) for p in family.action.params)
    }
    assert derived == EXCLUDED_FAMILIES
    assert set(SPECS) == {f.family_id for f in FAMILIES} - EXCLUDED_FAMILIES


# -- the matched negative control ---------------------------------------
def test_the_control_shares_the_shape_but_acts_on_the_right_target(corpus):
    """Without this, the probe-and-act shape would itself be a label."""
    control = [r for r in corpus if r.fault_type is FaultType.semantic_target_match]
    semantic = [r for r in corpus if r.fault_type is FaultType.semantic_target_mismatch]
    assert len(control) == len(semantic)
    rules = RulesEvaluator(threshold=0.5)
    for record in control:
        assert record.label is Label.supported_success
        assert record.oracle["acted_target"] == record.oracle["resolved_target_id"]
        # Same five-event shape as the mismatch fault: probe, listing, write,
        # confirmation, claim. Only the chosen candidate differs.
        assert [e.type.value for e in record.events] == [e.type.value for e in semantic[0].events]
        assert len(record.events) == 5
        assert record.events[1].data["count"] == 3
        assert rules.predict(record.inference_view(), record.trace_id).decision.value == "pass"


def test_the_probe_tool_is_advertised_on_every_record_not_just_semantic_ones(corpus):
    """A tool that appeared only alongside one fault would be a label in the schema."""
    by_family: dict[str, set[tuple[str, ...]]] = {}
    for record in corpus:
        by_family.setdefault(record.template_family, set()).add(
            tuple(e.name for e in record.tool_schema)
        )
    for family, schemas in by_family.items():
        assert len(schemas) == 1, f"{family} advertises more than one tool schema: {schemas}"
        assert len(next(iter(schemas))) == 4


# -- generation fails loudly if the subclass stops being clean ----------
def test_generation_fails_if_the_rules_baseline_can_resolve_the_subclass(corpus):
    from false_success_eval.generate import _enforce_semantic_subclass

    doctored = []
    done = False
    for record in corpus:
        if record.fault_type is FaultType.semantic_target_mismatch and not done:
            done = True
            # Put the goal's referent verbatim into the evidence.
            events = list(record.events)
            probe = events[0]
            events[0] = probe.model_copy(
                update={"arguments": {"query": record.oracle["described_target"]}}
            )
            doctored.append(record.model_copy(update={"events": tuple(events)}))
        else:
            doctored.append(record)

    with pytest.raises(GenerationError, match="appears verbatim"):
        _enforce_semantic_subclass(tuple(doctored))


def test_generation_fails_if_the_subclass_is_absent():
    from false_success_eval.generate import _enforce_semantic_subclass

    with pytest.raises(GenerationError, match="no semantic_target_mismatch"):
        _enforce_semantic_subclass(())


# -- the blinded audit reaches the subclass -----------------------------
def test_the_audit_sample_covers_every_label_fault_cell_including_the_subclass(corpus):
    sample = audit_sample(corpus)
    sampled_strata = {audit_stratum(r) for r in corpus if r.trace_id in sample}
    all_strata = {audit_stratum(r) for r in corpus}
    assert sampled_strata == all_strata
    assert "unsupported_success/semantic_target_mismatch" in sampled_strata
    assert "supported_success/semantic_target_match" in sampled_strata


def test_the_audit_sample_is_deterministic(corpus):
    assert audit_sample(corpus) == audit_sample(generate_records(RECORDS, SEED))


def test_sample_membership_leaks_no_label(corpus):
    """Every cell is sampled at the same rate, so `in_sample` says nothing about the label."""
    sample = audit_sample(corpus)
    by_stratum = Counter(audit_stratum(r) for r in corpus)
    sampled_by_stratum = Counter(audit_stratum(r) for r in corpus if r.trace_id in sample)
    for stratum, total in by_stratum.items():
        assert sampled_by_stratum[stratum] == min(8, total)
