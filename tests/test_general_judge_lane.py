"""The general-judge lane, end to end: metrics, report, dashboard, export.

These run against a throwaway workspace with real generated data, real offline
baseline runs, and hand-built Jev and general-judge runs written through the
real ``RunWriter``. No network: the session-wide socket block in conftest means
anything that tried to reach one would fail loudly rather than quietly succeed.

The load-bearing assertion in this file is the parity one. The dashboard is
allowed to present figures however it likes, but it is not allowed to *compute*
a different recall or AUPRC than ``jev-eval report`` does for the same run, so
every lane is checked against the metrics module rather than against a
hand-written expectation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from false_success_eval import metrics
from false_success_eval.cli import app as cli_app
from false_success_eval.costs import load_prices
from false_success_eval.dashboard.app import create_app
from false_success_eval.runner import (
    RunWriter,
    load_config,
    load_run,
    load_splits,
    new_run_id,
    resolve_dataset_paths,
    select_split,
)
from false_success_eval.schemas import Attempt, Decision, Prediction, Usage

RECORDS = 160
REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()

JEV_MODEL = "jev-1.13.0"
#: Deliberately a name no price manifest carries. That is the state this arm
#: ships in until a published rate is on record, and it must not crash a run.
JUDGE_MODEL = "a-general-model-id-no-manifest-prices"


def _invoke(workspace: Path, *args: str):
    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        return runner.invoke(cli_app, list(args))
    finally:
        os.chdir(cwd)


def _write_fake_run(workspace: Path, provider: str, model_id: str, scores) -> str:
    """A run written through the real RunWriter, with no network.

    ``scores`` maps a record to its P(unsupported_success). The two paid lanes
    are given deliberately different, deterministic score functions so the
    comparison between them has something to find; nothing here is a claim
    about either model.
    """
    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        from false_success_eval.generate import load_dataset

        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, RECORDS, None)
        records = load_dataset(paths.dataset)
        selected = select_split(records, load_splits(paths.splits), "test")
        prices = load_prices(config.raw["prices"])

        run_id = new_run_id(provider, "test")
        writer = RunWriter(
            run_dir=paths.runs_root / run_id,
            run_id=run_id,
            provider=provider,
            model_id=model_id,
            split="test",
            repeats=1,
            concurrency=1,
            threshold=0.5,
            n_traces=len(selected),
            n_expected=len(selected),
            prices=prices,
        )
        for index, record in enumerate(selected):
            score = scores(record, index)
            writer.append(
                Prediction(
                    trace_id=record.trace_id,
                    provider=provider,
                    model_id=model_id,
                    repeat=0,
                    predicted_label=record.label,
                    primary_score=score,
                    probabilities={record.label.value: score},
                    threshold=0.5,
                    decision=Decision.flag if score >= 0.5 else Decision.pass_,
                    usage=Usage(input_tokens=4000, output_tokens=90),
                    attempts=(Attempt(attempt_number=1, http_status=200, latency_ms=900.0),),
                    end_to_end_latency_ms=900.0 + index,
                )
            )
        writer.finalise(
            config=config,
            splits_path=paths.splits,
            dataset_path=paths.dataset,
            questions={"verdict": {"type": "choice"}},
            repo_root=workspace,
        )
        return run_id
    finally:
        os.chdir(cwd)


def _jev_score(record, index):
    from false_success_eval.schemas import Label

    return 0.92 if record.label is Label.unsupported_success else 0.11


def _judge_score(record, index):
    """A weaker, noisier ranker: right more often than not, but not by much."""
    from false_success_eval.schemas import Label

    positive = record.label is Label.unsupported_success
    wobble = ((index * 37) % 11) / 20.0
    return (0.45 + wobble) if positive else (0.30 + wobble)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("judge-lane")
    shutil.copytree(REPO / "config", root / "config")
    config_path = root / "config" / "eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["records"] = RECORDS
    config["metrics"]["bootstrap_resamples"] = 200
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    for name in ("data", "runs", "reports"):
        (root / name).mkdir()

    for args in (
        ("generate",),
        ("run", "--provider", "rules", "--split", "test"),
        ("run", "--provider", "tfidf", "--split", "test"),
        ("run", "--provider", "tfidf_gbm", "--split", "validation"),
        ("run", "--provider", "tfidf_gbm", "--split", "test"),
    ):
        result = _invoke(root, *args)
        assert result.exit_code == 0, result.output

    jev_run = _write_fake_run(root, "jev", JEV_MODEL, _jev_score)
    _write_fake_run(root, "openai", JUDGE_MODEL, _judge_score)
    return root, jev_run


@pytest.fixture
def client(workspace):
    root, jev_run = workspace
    cwd = os.getcwd()
    os.chdir(root)
    try:
        application = create_app(config_path="config/eval.yaml", records=RECORDS)
        with TestClient(application) as test_client:
            yield test_client, root, jev_run
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# A model with no rate must not abort the run
# ---------------------------------------------------------------------------


def test_an_unpriced_model_records_tokens_and_a_null_cost_without_crashing(workspace):
    """The regression this arm introduced: a run-time model id no manifest knows.

    ``cost_usd`` refuses an unknown model outright, which is right for a pinned
    model and catastrophic for one chosen at run time -- it would raise on the
    first response and lose the whole measurement. The recording path records
    null instead, beside the real token counts.
    """
    root, _ = workspace
    run_dir = next((root / "runs" / str(RECORDS)).glob("openai-test-*"))
    manifest, predictions = load_run(run_dir)
    assert manifest.total_cost_usd is None
    assert manifest.total_input_tokens > 0
    assert manifest.total_output_tokens > 0
    assert all(p.cost_usd is None for p in predictions)


def test_such_a_run_still_verifies(workspace):
    """A null cost is a verifiable record, not an unverifiable gap."""
    root, _ = workspace
    result = _invoke(root, "verify-runs", "--records", str(RECORDS))
    assert result.exit_code == 0, result.output


def test_a_cost_invented_for_an_unpriced_model_is_caught_by_verify_run(workspace, tmp_path):
    """The other direction: verification must still refuse a made-up number."""
    from false_success_eval.runner import verify_run

    root, _ = workspace
    source = next((root / "runs" / str(RECORDS)).glob("openai-test-*"))
    tampered = tmp_path / "tampered"
    shutil.copytree(source, tampered)
    lines = (tampered / "predictions.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["cost_usd"] = 0.0001
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    (tampered / "predictions.jsonl").write_text("\n".join(lines) + "\n")

    cwd = os.getcwd()
    os.chdir(root)
    try:
        problems = verify_run(tampered)
    finally:
        os.chdir(cwd)
    assert any("no verified rate" in p for p in problems), problems


# ---------------------------------------------------------------------------
# Parity with the metrics module
# ---------------------------------------------------------------------------


def test_the_judge_lane_matches_the_metrics_module_exactly(client):
    """The dashboard reads metrics; it does not compute a second opinion."""
    from false_success_eval.generate import load_dataset
    from false_success_eval.report import evaluate_run

    test_client, root, jev_run = client
    snapshot = test_client.get(f"/api/snapshot?run={jev_run}").json()
    reference = snapshot["reference"]
    assert "openai" in reference, "the general judge must appear as a scored provider"

    config = load_config("config/eval.yaml")
    paths = resolve_dataset_paths(config, RECORDS, None)
    records = load_dataset(paths.dataset)
    selected = select_split(records, load_splits(paths.splits), "test")

    run_dir = next((root / "runs" / str(RECORDS)).glob("openai-test-*"))
    _, predictions = load_run(run_dir)
    expected = evaluate_run(selected, predictions, 0.5, config)

    for key in ("auprc", "auroc", "precision", "recall", "ece", "n_scored"):
        assert reference["openai"][key] == pytest.approx(expected[key]), key


def test_the_judge_lane_flag_count_matches_a_direct_count(client):
    """The streamed lane and the metrics module agree on the finished run."""
    from false_success_eval.generate import load_dataset
    from false_success_eval.schemas import Label

    test_client, root, jev_run = client
    snapshot = test_client.get(f"/api/snapshot?run={jev_run}").json()

    config = load_config("config/eval.yaml")
    paths = resolve_dataset_paths(config, RECORDS, None)
    records = load_dataset(paths.dataset)
    selected = select_split(records, load_splits(paths.splits), "test")
    positives = {r.trace_id for r in selected if r.label is Label.unsupported_success}

    run_dir = next((root / "runs" / str(RECORDS)).glob("openai-test-*"))
    _, predictions = load_run(run_dir)
    flagged = sum(
        1
        for p in predictions
        if p.trace_id in positives and p.primary_score is not None and p.primary_score >= 0.5
    )
    assert snapshot["lanes"]["openai"]["flagged"] == flagged
    assert snapshot["kpi"]["caught_by_general_judge"] == flagged


# ---------------------------------------------------------------------------
# The headline: Jev against the general judge
# ---------------------------------------------------------------------------


def test_the_snapshot_carries_a_paired_bootstrap_ci_on_the_difference(client):
    test_client, _, jev_run = client
    judge = test_client.get(f"/api/snapshot?run={jev_run}").json()["general_judge"]
    assert judge["computable"] is True
    assert judge["rival"] == "openai"
    assert judge["ci"]["lower"] <= judge["ci"]["point"] <= judge["ci"]["upper"]
    assert judge["recall_ci"]["lower"] <= judge["recall_ci"]["upper"]
    assert judge["n_paired"] > 0
    assert judge["verdict"] in {
        "jev_wins",
        "judge_wins",
        "no_difference",
        "not_computable",
    }


def test_the_verdict_is_read_from_the_interval_not_the_point_estimate(client):
    """An interval spanning zero is 'no difference', whatever the point says."""
    test_client, _, jev_run = client
    judge = test_client.get(f"/api/snapshot?run={jev_run}").json()["general_judge"]
    lower, upper = judge["ci"]["lower"], judge["ci"]["upper"]
    if lower <= 0.0 <= upper:
        assert judge["verdict"] == "no_difference"
        assert judge["jev_beats_judge"] is False
    elif lower > 0:
        assert judge["verdict"] == "jev_wins"
    else:
        assert judge["verdict"] == "judge_wins"


def test_the_judge_panel_reports_cost_as_unavailable_never_as_zero(client):
    """A lane that looks free because nobody priced it is the worst answer here."""
    test_client, _, jev_run = client
    judge = test_client.get(f"/api/snapshot?run={jev_run}").json()["general_judge"]
    arm = judge["cost"]["arms"]["openai"]
    assert arm["available"] is True
    assert arm["cost_available"] is False
    assert arm["cost_per_1000_traces_usd"] is None
    assert arm["cost_per_1000_traces_gbp"] is None
    assert arm["input_tokens"] > 0
    assert arm["output_tokens"] > 0


def test_the_paid_arms_are_never_folded_into_the_free_union(client):
    """The general judge costs money. Crediting the free tier with it would lie."""
    from false_success_eval.dashboard.aggregate import FREE_BASELINES, GENERAL_JUDGE

    assert GENERAL_JUDGE not in FREE_BASELINES
    test_client, _, jev_run = client
    snapshot = test_client.get(f"/api/snapshot?run={jev_run}").json()
    union = snapshot["lanes"]["free_union"]["flagged"]
    best_free = max(snapshot["lanes"][name]["flagged"] for name in FREE_BASELINES)
    assert union >= best_free
    # The union is the free lanes only, so it cannot exceed what they can reach
    # together even when the judge flags more.
    assert union <= snapshot["kpi"]["positives_total"]


def test_selecting_the_judge_run_does_not_compare_it_against_itself(client):
    test_client, root, _ = client
    run_id = next((root / "runs" / str(RECORDS)).glob("openai-test-*")).name
    judge = test_client.get(f"/api/snapshot?run={run_id}").json()["general_judge"]
    assert judge["computable"] is False
    assert judge["verdict"] == "not_applicable"


# ---------------------------------------------------------------------------
# The report says the same things
# ---------------------------------------------------------------------------


def test_the_report_carries_the_general_judge_headline(workspace):
    root, _ = workspace
    result = _invoke(root, "report", "--records", str(RECORDS), "--out", "reports/judge")
    assert result.exit_code == 0, result.output

    results = json.loads((root / "reports" / "judge" / "results.json").read_text())
    block = results["general_judge"]
    assert block["reported"] is True
    assert block["primary"] == "jev"
    assert block["general_judge"] == "openai"
    assert set(block["arms"]) == {"jev", "openai"}
    assert block["auprc_delta"]["lower"] <= block["auprc_delta"]["upper"]

    # The comparison table gains the judge as a target for every other provider.
    assert "openai" in results["comparisons"]["jev"]

    markdown = (root / "reports" / "judge" / "report.md").read_text()
    assert "## Typed model against general judge" in markdown
    assert "unavailable" in markdown, (
        "an arm with no verified rate must say so in the report, not print a zero"
    )


def test_the_report_and_the_dashboard_agree_on_the_difference(client, workspace):
    """One number, two surfaces. They are computed from the same metrics module."""
    root, jev_run = workspace
    results = json.loads((root / "reports" / "judge" / "results.json").read_text())
    test_client, _, _ = client
    snapshot = test_client.get(f"/api/snapshot?run={jev_run}").json()

    report_delta = results["general_judge"]["auprc_delta"]["point"]
    dashboard_delta = snapshot["general_judge"]["delta_auprc"]
    assert report_delta == pytest.approx(dashboard_delta, abs=1e-12)

    for provider in ("jev", "openai"):
        assert results["runs"][provider]["auprc"] == pytest.approx(
            snapshot["reference"][provider]["auprc"]
        )
        assert results["runs"][provider]["recall"] == pytest.approx(
            snapshot["reference"][provider]["recall"]
        )


def test_the_difference_is_exactly_the_two_arms_auprc_gap(workspace):
    """No smoothing, no shrinkage: the point estimate is a subtraction."""
    root, _ = workspace
    results = json.loads((root / "reports" / "judge" / "results.json").read_text())
    jev = results["runs"]["jev"]["auprc"]
    judge = results["runs"]["openai"]["auprc"]
    assert results["general_judge"]["auprc_delta"]["point"] == pytest.approx(jev - judge)


# ---------------------------------------------------------------------------
# Export and the row contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("view", ["overview", "detection", "cost", "per-fault"])
def test_export_includes_the_judge_lane_and_still_writes_both_files(client, view):
    test_client, root, jev_run = client
    response = test_client.post("/api/export", json={"run": jev_run, "view": view})
    assert response.status_code == 200
    written = [Path(p) for p in response.json()["written"]]
    assert len(written) == 2
    for path in written:
        assert (root / path).exists() or path.exists()

    if view in {"detection", "per-fault", "cost"}:
        csv_path = next(p for p in written if str(p).endswith(".csv"))
        text = (root / csv_path).read_text() if (root / csv_path).exists() else csv_path.read_text()
        assert "openai" in text or "general_judge" in text


def test_no_request_or_response_body_reaches_the_browser_for_the_judge_lane(client):
    test_client, _, _ = client
    rows = test_client.get("/api/rows").json()["rows"]
    blob = json.dumps(rows)
    assert "raw_request" not in blob
    assert "raw_response" not in blob


def test_the_detection_rows_carry_a_gap_against_the_judge(client):
    test_client, _, jev_run = client
    snapshot = test_client.get(f"/api/snapshot?run={jev_run}").json()
    for group in snapshot["detection"]:
        assert "openai" in group
        assert "gap_vs_general_judge" in group
        if group["jev"]["rate"] is not None and group["openai"]["rate"] is not None:
            assert group["gap_vs_general_judge"] == pytest.approx(
                group["jev"]["rate"] - group["openai"]["rate"]
            )


def test_an_unscored_lane_reads_as_none_rather_than_a_zero(client):
    """A rate over nothing is None. Zero is a finding; empty is not."""
    test_client, _, jev_run = client
    snapshot = test_client.get(f"/api/snapshot?run={jev_run}").json()
    for row in snapshot["per_fault"]:
        if row["general_judge_scored"] == 0:
            assert row["general_judge_rate"] is None
            assert row["gap_vs_general_judge"] is None


def test_the_bootstrap_is_seeded_so_the_interval_is_reproducible(client):
    test_client, _, jev_run = client
    first = test_client.get(f"/api/snapshot?run={jev_run}").json()["general_judge"]
    second = test_client.get(f"/api/snapshot?run={jev_run}").json()["general_judge"]
    assert first["ci"] == second["ci"]
    assert first["delta_auprc"] == second["delta_auprc"]


def test_the_paired_bootstrap_uses_only_traces_both_arms_scored(workspace):
    """A lane that skipped a trace must not have it counted against the other."""
    from false_success_eval.generate import load_dataset

    root, _ = workspace
    cwd = os.getcwd()
    os.chdir(root)
    try:
        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, RECORDS, None)
        selected = select_split(load_dataset(paths.dataset), load_splits(paths.splits), "test")
    finally:
        os.chdir(cwd)

    jev_dir = next((root / "runs" / str(RECORDS)).glob("jev-test-*"))
    judge_dir = next((root / "runs" / str(RECORDS)).glob("openai-test-*"))
    _, jev_predictions = load_run(jev_dir)
    _, judge_predictions = load_run(judge_dir)
    jev_points = metrics.point_predictions(jev_predictions)
    judge_points = metrics.point_predictions(judge_predictions)

    both = [
        r
        for r in selected
        if r.trace_id in jev_points
        and r.trace_id in judge_points
        and jev_points[r.trace_id].primary_score is not None
        and judge_points[r.trace_id].primary_score is not None
    ]
    results = json.loads((root / "reports" / "judge" / "results.json").read_text())
    y = metrics.binary_targets(both)
    a = np.array([jev_points[r.trace_id].primary_score for r in both], dtype=float)
    b = np.array([judge_points[r.trace_id].primary_score for r in both], dtype=float)
    expected = metrics.auprc(y, a) - metrics.auprc(y, b)
    assert results["general_judge"]["auprc_delta"]["point"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Dry-run discipline on the paid arm
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def priced_workspace(workspace, tmp_path_factory):
    """A copy of the workspace where the judge model has a rate on record.

    The spend gate can only fire for a model that can be priced, so this exists
    to exercise it. The rate is a test fixture, not a published price, and it
    lives in a manifest of its own rather than in ``config/``.
    """
    root, _ = workspace
    priced = tmp_path_factory.mktemp("priced")
    shutil.copytree(root / "config", priced / "config")
    for name in ("data", "runs", "reports"):
        shutil.copytree(root / name, priced / name)

    manifest_path = priced / "config" / "prices-test-fixture.json"
    manifest = json.loads((priced / "config" / "prices-2026-09-20-2.json").read_text())
    # Deliberately absurd, and far above any real rate: the split here is 32
    # small synthetic traces, so a realistic price would project well under the
    # threshold and the gate would never fire. The number's job is to cross the
    # line, not to resemble anything.
    manifest["models"]["expensive-judge"] = {
        "input_usd_per_mtok": 3000.0,
        "output_usd_per_mtok": 12000.0,
        "verified": True,
        "source": "TEST FIXTURE. Not a published rate and not used by any real run.",
    }
    manifest["supersedes"] = "config/prices-2026-09-20-2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    config_path = priced / "config" / "eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["prices"] = "config/prices-test-fixture.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return priced


def test_a_projection_over_the_threshold_stops_and_says_so(priced_workspace):
    """The £5 gate. Checked against the top of the band, not the optimistic end."""
    result = _invoke(
        priced_workspace,
        "run",
        "--provider",
        "openai",
        "--model",
        "expensive-judge",
        "--split",
        "test",
        "--records",
        str(RECORDS),
        "--confirm-paid",
    )
    assert result.exit_code == 1, result.output
    assert "STOPPED" in result.output
    assert "above the £5.00 threshold" in result.output
    assert "Nothing was sent" in result.output
    assert "--accept-cost-over-threshold" in result.output


def test_the_projection_prices_the_output_leg_for_a_general_judge(priced_workspace):
    result = _invoke(
        priced_workspace,
        "run",
        "--provider",
        "openai",
        "--model",
        "expensive-judge",
        "--split",
        "test",
        "--records",
        str(RECORDS),
        "--confirm-paid",
    )
    assert "per Mtok input, $12000.0 per Mtok output" in result.output
    assert "output ceiling" in result.output
    assert "PROJECTED GBP" in result.output


def test_the_gate_lets_an_accepted_overrun_through_to_the_key_check(priced_workspace):
    """Past the gate, the next refusal is the missing key -- not a silent call."""
    result = _invoke(
        priced_workspace,
        "run",
        "--provider",
        "openai",
        "--model",
        "expensive-judge",
        "--split",
        "test",
        "--records",
        str(RECORDS),
        "--confirm-paid",
        "--accept-cost-over-threshold",
    )
    assert "--accept-cost-over-threshold was passed" in result.output
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_a_paid_run_refuses_without_confirm_paid(workspace):
    root, _ = workspace
    result = _invoke(
        root,
        "run",
        "--provider",
        "openai",
        "--model",
        "some-model-from-the-listing",
        "--split",
        "test",
        "--records",
        str(RECORDS),
    )
    assert result.exit_code == 1
    assert "Refusing to spend money without --confirm-paid" in result.output
    assert "Nothing was sent" in result.output


def test_the_plan_states_the_parity_terms_before_spending(workspace):
    root, _ = workspace
    result = _invoke(
        root,
        "run",
        "--provider",
        "openai",
        "--model",
        "some-model-from-the-listing",
        "--split",
        "test",
        "--records",
        str(RECORDS),
    )
    assert "temperature  0.0 (matched to the Jev arm)" in result.output
    assert "strict JSON schema" in result.output
    assert "config/questions.json" in result.output
    assert "NO PRICE ENTRY" in result.output


def test_an_unknown_general_model_vendor_is_refused_by_name(workspace):
    root, _ = workspace
    result = _invoke(root, "list-models", "--provider", "nobody", "--yes")
    assert result.exit_code == 1
    assert "no general_model block named 'nobody'" in result.output
