"""Caveats: generated from the run, so a run cannot show another run's warnings.

The central property: the construction-label warning and the
not-independently-checked warning are true of the synthetic diagnostic and false
of a real corpus with programmatic ground truth. A caveat block built for a real
run must not contain them, and a caveat block built for a synthetic run must.
"""

from __future__ import annotations

import pytest

from false_success_eval.caveats import MCA_NOTICE, build_caveats
from false_success_eval.ingest.labelling import APPWORLD_LABEL_RULE
from false_success_eval.ingest.licences import require_licence
from false_success_eval.schemas import DatasetKind, DatasetProvenance


def _real_provenance(**overrides) -> DatasetProvenance:
    defaults: dict = {
        "dataset_id": "appworld",
        "kind": DatasetKind.real,
        "source_name": "AppWorld released experiment outputs",
        "source_url": "https://example.invalid/bundle",
        "source_version": "experiment-outputs-0.1.3",
        "source_sha256": "a" * 64,
        "licence": require_licence("appworld"),
        "label_rule": APPWORLD_LABEL_RULE,
        "split_unit": "appworld_scenario_id",
        "split_unit_description": "195 scenarios split 117 dev / 39 validation / 39 test.",
        "split_sha256": "b" * 64,
        "n_records": 3507,
    }
    defaults.update(overrides)
    return DatasetProvenance(**defaults)


def _synthetic_provenance() -> DatasetProvenance:
    return DatasetProvenance(
        dataset_id="synthetic-1000",
        kind=DatasetKind.synthetic,
        source_name="Deterministic generator in this repository",
        source_url="src/false_success_eval/generate.py",
        split_unit="template_family",
        split_unit_description="40 template families split 24/8/8.",
        split_sha256="c" * 64,
        n_records=1000,
    )


def _bodies(caveats) -> str:
    return " ".join(item["title"] + " " + item["body"] for item in caveats["items"]).lower()


# ---------------------------------------------------------------------------
# the two lines that had to go
# ---------------------------------------------------------------------------


def test_real_data_drops_the_construction_label_line():
    caveats = build_caveats(provenance=_real_provenance()).as_dict()
    text = _bodies(caveats) + " " + caveats["strip"].lower()
    assert "construction label" not in text
    assert "blinded" not in text
    assert "not been independently checked" not in text


def test_real_data_ignores_a_stale_audit_status():
    """An audit sheet left over from the synthetic set must not leak through."""
    audit = {
        "status": "not_performed",
        "text": "Labels are construction labels from the generator. The blinded human audit "
        "is pending.",
    }
    caveats = build_caveats(provenance=_real_provenance(), audit=audit).as_dict()
    assert "construction label" not in _bodies(caveats)


def test_synthetic_data_keeps_the_construction_label_line():
    audit = {
        "status": "not_performed",
        "text": "Labels are construction labels from the generator.",
    }
    caveats = build_caveats(provenance=_synthetic_provenance(), audit=audit).as_dict()
    text = _bodies(caveats)
    assert "construction label" in text
    assert "synthetic" in caveats["strip"].lower()


def test_a_dataset_with_no_provenance_gets_the_cautious_wording():
    """Absent provenance must over-warn, not under-warn."""
    caveats = build_caveats(provenance=None).as_dict()
    assert caveats["kind"] == "synthetic"
    assert "construction label" in _bodies(caveats)


# ---------------------------------------------------------------------------
# what a real caveat must say instead
# ---------------------------------------------------------------------------


def test_real_caveats_state_source_licence_rule_and_split():
    caveats = build_caveats(provenance=_real_provenance()).as_dict()
    keys = {item["key"] for item in caveats["items"]}
    assert {"source", "label_rule", "split", "interval", "mca"} <= keys
    text = _bodies(caveats)
    assert "apache-2.0" in text
    assert "appworld" in text
    assert "task-disjoint" in text or "disjoint" in text
    assert "complete_task" in text


def test_the_strip_is_one_line():
    caveats = build_caveats(provenance=_real_provenance()).as_dict()
    assert "\n" not in caveats["strip"]
    assert len(caveats["strip"]) < 200


def test_mca_notice_survives_on_both_kinds():
    for provenance in (_real_provenance(), _synthetic_provenance()):
        caveats = build_caveats(provenance=provenance).as_dict()
        assert caveats["mca"] == MCA_NOTICE
        assert any(item["body"] == MCA_NOTICE for item in caveats["items"])


# ---------------------------------------------------------------------------
# interval width and truncation
# ---------------------------------------------------------------------------


def test_an_interval_that_crosses_zero_is_called_out():
    caveats = build_caveats(
        provenance=_real_provenance(),
        headline_ci={"point": 0.02, "lower": -0.04, "upper": 0.08},
        n_positives=350,
        n_traces=700,
    ).as_dict()
    interval = next(item for item in caveats["items"] if item["key"] == "interval")
    assert "crosses zero" in interval["body"]
    assert interval["severity"] == "warn"
    assert "0.120" in interval["title"]


def test_an_interval_clear_of_zero_is_not_flagged_as_a_warning():
    caveats = build_caveats(
        provenance=_real_provenance(),
        headline_ci={"point": 0.2, "lower": 0.12, "upper": 0.28},
        n_positives=350,
        n_traces=700,
    ).as_dict()
    interval = next(item for item in caveats["items"] if item["key"] == "interval")
    assert "does not cross zero" in interval["body"]
    assert interval["severity"] == "info"


def test_a_missing_interval_is_stated_rather_than_omitted():
    caveats = build_caveats(provenance=_real_provenance(), headline_ci=None).as_dict()
    interval = next(item for item in caveats["items"] if item["key"] == "interval")
    assert "unknown precision" in interval["body"]


def test_truncation_is_reported_when_traces_were_shortened():
    caveats = build_caveats(
        provenance=_real_provenance(),
        truncation={
            "truncated_traces": 120,
            "n_traces": 700,
            "dropped_events": 4300,
            "rule": "middle-elision-v1",
        },
    ).as_dict()
    item = next(i for i in caveats["items"] if i["key"] == "truncation")
    assert "120" in item["title"]
    assert "middle-elision-v1" in item["body"]
    assert "4,300" in item["body"]
    assert item["severity"] == "warn"
    assert "shortened" in caveats["strip"]


def test_no_truncation_is_stated_positively():
    caveats = build_caveats(
        provenance=_real_provenance(),
        truncation={"truncated_traces": 0, "n_traces": 700},
    ).as_dict()
    item = next(i for i in caveats["items"] if i["key"] == "truncation")
    assert "No trace was shortened" in item["title"]
    assert "shortened" not in caveats["strip"]


def test_a_run_with_no_truncation_accounting_says_nothing_about_it():
    """Silence is different from 'everything fitted'."""
    caveats = build_caveats(provenance=_real_provenance(), truncation=None).as_dict()
    assert not any(item["key"] == "truncation" for item in caveats["items"])


@pytest.mark.parametrize("kind", ["real", "synthetic"])
def test_kind_is_carried_so_the_page_can_badge_it(kind):
    provenance = _real_provenance() if kind == "real" else _synthetic_provenance()
    assert build_caveats(provenance=provenance).as_dict()["kind"] == kind


# ---------------------------------------------------------------------------
# the caveat is only honest if the run actually fills it in
# ---------------------------------------------------------------------------
#
# build_caveats() takes the counts and the truncation account as arguments, so
# a caller that forgets them still produces a plausible-looking caveat block
# that quietly reads "0 positives in 0 scored traces" and says nothing about
# truncation. These pin the wiring, not the formatting.


def test_the_interval_line_reports_the_counts_it_was_given():
    caveats = build_caveats(
        provenance=_real_provenance(),
        headline_ci={"point": -0.114, "lower": -0.145, "upper": -0.085},
        n_positives=289,
        n_traces=702,
    ).as_dict()
    body = next(i for i in caveats["items"] if i["key"] == "interval")["body"]
    assert "289" in body and "702" in body
    assert "0 positives in 0 scored traces" not in body


def test_the_real_report_fills_in_counts_and_truncation():
    """The wiring the report actually uses, not a hand-built call."""
    import json
    from pathlib import Path

    results = Path("reports/appworld/results.json")
    if not results.exists():
        pytest.skip("no real-corpus report built in this checkout")
    caveats = json.loads(results.read_text(encoding="utf-8"))["caveats"]
    interval = next(i for i in caveats["items"] if i["key"] == "interval")
    assert "0 positives in 0 scored traces" not in interval["body"]
    assert any(i["key"] == "truncation" for i in caveats["items"]), (
        "the Jev run writes a truncation account, so the caveat must speak to it"
    )


# ---------------------------------------------------------------------------
# Deviations from the frozen request
# ---------------------------------------------------------------------------

DEVIATION_ROW = {
    "parameter": "temperature",
    "requested": 0.0,
    "applied": "omitted from the request; the provider default applies",
    "model_id": "gpt-5.6-terra",
    "run_model_id": "gpt-5.6-terra",
    "provider": "openai",
    "http_status": 400,
    "error_code": "unsupported_value",
    "error_message": (
        "Unsupported value: 'temperature' does not support 0.0 with this model. "
        "Only the default (1) value is supported."
    ),
    "first_seen_trace_id": "appworld/x/y/test_challenge/1_1",
    "occurrences": 702,
}


def _deviation_caveat(deviation):
    caveats = build_caveats(provenance=_real_provenance(), deviation=deviation).as_dict()
    return next((i for i in caveats["items"] if i["key"] == "deviation"), None), caveats


def test_no_deviation_account_means_the_caveat_says_nothing_about_deviations():
    """Silence, not a claim that nothing deviated."""
    item, _ = _deviation_caveat(None)
    assert item is None


def test_an_empty_deviation_account_is_a_positive_statement():
    item, caveats = _deviation_caveat({"deviations": []})
    assert item is not None
    assert item["severity"] == "info"
    assert "exactly the request" in item["body"]
    assert "refused a request parameter" not in caveats["strip"]


def test_a_refused_parameter_is_named_in_the_providers_own_words():
    item, caveats = _deviation_caveat({"deviations": [DEVIATION_ROW]})
    assert item is not None
    assert item["severity"] == "warn"
    assert "gpt-5.6-terra" in item["body"]
    assert "temperature" in item["body"]
    assert DEVIATION_ROW["error_message"] in item["body"]
    assert "702" in item["body"]
    # And the compact strip carries it too, so it cannot be missed by someone
    # who never opens the panel.
    assert "refused a request parameter" in caveats["strip"]


def test_the_deviation_caveat_says_what_could_not_have_been_dropped():
    item, _ = _deviation_caveat({"deviations": [DEVIATION_ROW]})
    assert "four questions" in item["body"]
    assert "schema" in item["body"]
