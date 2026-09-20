"""End-to-end offline acceptance: generate -> validate -> run -> report -> verify.

Nothing here has a key or a network. The session-wide socket block in conftest
means any accidental live call fails the test rather than making a request.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from false_success_eval.cli import app
from false_success_eval.runner import discover_runs, load_run, verify_run

RECORDS = 160
DATA = f"data/dataset-{RECORDS}.jsonl"
RUNS = f"runs/{RECORDS}"
REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A throwaway copy of the repo config, with a cheap bootstrap budget.

    The fixture also lays down a dataset and two offline runs, so every test in
    this module can be selected and run on its own rather than depending on the
    ones numbered before it.
    """
    root = tmp_path_factory.mktemp("harness")
    shutil.copytree(REPO / "config", root / "config")
    config_path = root / "config" / "eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["records"] = RECORDS
    config["metrics"]["bootstrap_resamples"] = 200
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "data").mkdir()
    (root / "runs").mkdir()

    for args in (
        ("generate",),
        ("run", "--provider", "rules", "--split", "test"),
        ("run", "--provider", "tfidf", "--split", "test"),
    ):
        result = invoke(root, *args)
        assert result.exit_code == 0, f"fixture setup failed for {args}: {result.output}"
    return root


def invoke(workspace: Path, *args: str):
    import os

    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        return runner.invoke(app, list(args))
    finally:
        os.chdir(cwd)


def test_01_generate(workspace):
    result = invoke(workspace, "generate")
    assert result.exit_code == 0, result.output
    for name in (
        f"dataset-{RECORDS}.jsonl",
        f"splits-{RECORDS}.json",
        f"hashes-{RECORDS}.json",
        f"audit-{RECORDS}.csv",
        f"audit_blind-{RECORDS}.csv",
    ):
        assert (workspace / "data" / name).exists(), name
    lines = (workspace / DATA).read_text().strip().splitlines()
    assert len(lines) == RECORDS


def test_02_generation_is_reproducible(workspace):
    before = (workspace / DATA).read_bytes()
    assert invoke(workspace, "generate").exit_code == 0
    assert (workspace / DATA).read_bytes() == before


def test_03_validate_data_passes(workspace):
    result = invoke(workspace, "validate-data", "--input", DATA)
    assert result.exit_code == 0, result.output
    assert "splits disjoint" in result.output
    assert "every family spans >1 label" in result.output


def test_04_validate_data_catches_a_tampered_dataset(workspace, tmp_path):
    corrupt = workspace / "data" / "corrupt.jsonl"
    lines = (workspace / DATA).read_text().splitlines()
    corrupt.write_text("\n".join(lines[:-4]) + "\n")
    result = invoke(workspace, "validate-data", "--input", "data/corrupt.jsonl")
    assert result.exit_code == 1
    corrupt.unlink()


def test_05_run_rules_offline(workspace):
    result = invoke(workspace, "run", "--provider", "rules", "--split", "test")
    assert result.exit_code == 0, result.output
    assert "0 errors" in result.output


def test_06_run_tfidf_offline(workspace):
    result = invoke(workspace, "run", "--provider", "tfidf", "--split", "test")
    assert result.exit_code == 0, result.output


def test_07_runs_are_append_only(workspace):
    """A second run never overwrites the first."""
    before = set(discover_runs(workspace / RUNS))
    assert invoke(workspace, "run", "--provider", "rules", "--split", "test").exit_code == 0
    after = set(discover_runs(workspace / RUNS))
    assert before < after
    for run_dir in before:
        assert (run_dir / "manifest.json").exists()


def test_08_predictions_carry_the_raw_artifacts(workspace):
    run_dir = discover_runs(workspace / RUNS)[0]
    manifest, predictions = load_run(run_dir)
    assert len(predictions) == manifest.n_predictions
    assert all(p.primary_score is not None for p in predictions)
    assert all(p.threshold == manifest.threshold for p in predictions)


def test_09_verify_run_passes_on_every_run(workspace):
    for run_dir in discover_runs(workspace / RUNS):
        assert verify_run(run_dir) == [], run_dir


def test_10_verify_run_cli_passes(workspace):
    run_dir = discover_runs(workspace / RUNS)[0]
    result = invoke(workspace, "verify-run", "--run", str(run_dir.relative_to(workspace)))
    assert result.exit_code == 0, result.output
    assert "cost arithmetic verified" in result.output


def test_11_verify_run_detects_a_tampered_prediction_file(workspace):
    run_dir = discover_runs(workspace / RUNS)[-1]
    path = run_dir / "predictions.jsonl"
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + "\n")
    try:
        problems = verify_run(run_dir)
        assert any("sha256 mismatch" in p for p in problems)
    finally:
        path.write_text(original)
    assert verify_run(run_dir) == []


def test_12_verify_run_detects_bad_cost_arithmetic(workspace):
    run_dir = discover_runs(workspace / RUNS)[0]
    path = run_dir / "predictions.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["cost_usd"] = 12.34
    rows[0]["usage"] = {"input_tokens": 0, "output_tokens": 0}
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    try:
        problems = verify_run(run_dir)
        assert any("cost" in p for p in problems)
    finally:
        path.write_text(original)


def test_13_report_builds(workspace):
    result = invoke(workspace, "report", "--out", "reports/test")
    assert result.exit_code == 0, result.output
    out = workspace / "reports" / "test"
    for name in (
        "report.md",
        "results.json",
        "pr_curve.png",
        "reliability.png",
        "fault_heatmap.png",
    ):
        assert (out / name).exists(), name
    report = (out / "report.md").read_text()
    assert "rules" in report and "tfidf" in report
    assert "Not for publication" in report


def test_14_report_withholds_the_useful_signal_gate_at_stage_one(workspace):
    results = json.loads((workspace / "reports" / "test" / "results.json").read_text())
    assert results["n_records"] == RECORDS
    for provider, gates in results["gates"].items():
        useful = gates["useful_signal"]
        assert useful["reported"] is False
        assert "passed" not in useful
        # 'rules' is the baseline the gate is defined against, so it is not merely
        # withheld at stage 1 -- it never applies.
        expected = "NOT APPLICABLE" if provider == "rules" else "NOT REPORTED"
        assert expected in useful["note"], (provider, useful["note"])


def test_15_report_records_that_the_audit_has_not_been_performed(workspace):
    results = json.loads((workspace / "reports" / "test" / "results.json").read_text())
    assert results["audit"]["status"] == "not_performed"
    assert "Blinded human audit" in (workspace / "reports" / "test" / "report.md").read_text()


def test_16_report_reads_a_filled_in_audit_sheet(workspace):
    """A filled sheet produces an agreement figure and lists every correction."""
    import csv

    blind = workspace / "data" / f"audit_blind-{RECORDS}.csv"
    original = blind.read_text()
    key_rows = list(csv.DictReader((workspace / "data" / f"audit-{RECORDS}.csv").open()))
    rows = list(csv.DictReader(blind.open()))
    truth = {r["trace_id"]: r for r in key_rows}
    for i, row in enumerate(rows[:20]):
        correct = truth[row["trace_id"]]["label"]
        row["auditor_id"] = "auditor-1"
        disagreement = "no_success_claim" if correct != "no_success_claim" else "supported_success"
        row["auditor_label"] = disagreement if i == 3 else correct
        row["auditor_note"] = "disagrees: reads as a question" if i == 3 else ""
    with blind.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    try:
        assert invoke(workspace, "report", "--out", "reports/audited").exit_code == 0
        results = json.loads((workspace / "reports" / "audited" / "results.json").read_text())
        audit = results["audit"]
        assert audit["status"] == "performed"
        assert audit["n_audited"] == 20
        assert audit["n_auditors"] == 1
        assert audit["inter_rater_reliability"] is None
        assert "not defined and is not reported" in audit["irr_note"]
        assert audit["agreement_with_construction_labels"] == pytest.approx(0.95)
        assert audit["n_corrections"] == 1
        report = (workspace / "reports" / "audited" / "report.md").read_text()
        assert "Agreement with construction labels" in report
        assert "95.0%" in report
    finally:
        blind.write_text(original)


def test_17_a_paid_run_is_refused_without_the_confirm_flag(workspace, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key-for-a-refusal-test")
    result = invoke(workspace, "run", "--provider", "jev", "--split", "test")
    assert result.exit_code == 1
    assert "PAID RUN PLAN" in result.output
    assert "Refusing to spend money without --confirm-paid" in result.output
    assert not any(d.name.startswith("jev-") for d in (workspace / RUNS).iterdir())


def test_18_jev_without_a_key_fails_with_a_clear_message(workspace):
    result = invoke(workspace, "run", "--provider", "jev", "--split", "test", "--confirm-paid")
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY is not set" in result.output


def test_19_smoke_without_a_key_fails_before_prompting(workspace):
    result = invoke(workspace, "smoke", "--provider", "jev")
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY is not set" in result.output
    assert "Proceed?" not in result.output


def test_20_general_model_records_not_run_without_configuration(workspace):
    from false_success_eval.evaluators.general_model import (
        GeneralModelConfig,
        GeneralModelEvaluator,
    )
    from false_success_eval.generate import generate_records

    config = yaml.safe_load((workspace / "config" / "eval.yaml").read_text())
    view = generate_records(RECORDS, 20260919)[0].inference_view()

    for name in ("gemini", "claude", "gpt_terra"):
        evaluator = GeneralModelEvaluator(
            GeneralModelConfig.from_mapping(name, config["general_model"][name])
        )
        prediction = evaluator.predict(view, "t1")
        assert prediction.decision.value == "not_run"
        assert prediction.error.startswith("not_run:")
        assert prediction.cost_usd is None

    terra = GeneralModelEvaluator(
        GeneralModelConfig.from_mapping("gpt_terra", config["general_model"]["gpt_terra"])
    )
    assert "No verified official model ID" in terra.blocked


def test_21_no_artifact_contains_a_secret(workspace):
    for path in list(workspace.rglob("*.json")) + list(workspace.rglob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        assert "not-a-real-key-for-a-refusal-test" not in text
        assert "Bearer " not in text or "[REDACTED]" in text
