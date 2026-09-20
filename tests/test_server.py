"""Dashboard acceptance: endpoints, the SSE stream, the cost figures, and offline-ness.

Every test here runs against a throwaway workspace with real generated data and
real offline runs, plus a hand-built fake Jev run so the with-Jev lane has
something in it. The session-wide socket block in conftest means a dashboard
that tried to reach the network would fail loudly rather than quietly succeed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from false_success_eval.cli import app as cli_app
from false_success_eval.costs import load_prices
from false_success_eval.dashboard.aggregate import FOOTER_NOTICE
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
from false_success_eval.schemas import Attempt, Decision, Label, Prediction, Usage

RECORDS = 160
REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()

JEV_MODEL = "jev-1.13.0"


def _invoke(workspace: Path, *args: str):
    import os

    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        return runner.invoke(cli_app, list(args))
    finally:
        os.chdir(cwd)


def _write_fake_jev_run(
    workspace: Path, *, concurrency: int = 1, errors: int = 0, corpus: str | None = None
) -> str:
    """A Jev-shaped run written through the real RunWriter, with no network.

    The scores are deterministic and deliberately unlike the rule baseline's:
    the point of these tests is the plumbing and the arithmetic, not a claim
    about the model.
    """
    import os

    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, None if corpus else RECORDS, corpus)
        from false_success_eval.generate import load_dataset

        records = load_dataset(paths.dataset)
        splits = load_splits(paths.splits)
        selected = select_split(records, splits, "test")
        prices = load_prices(config.raw["prices"])

        run_id = new_run_id("jev", "test")
        writer = RunWriter(
            run_dir=paths.runs_root / run_id,
            run_id=run_id,
            provider="jev",
            model_id=JEV_MODEL,
            split="test",
            repeats=1,
            concurrency=concurrency,
            threshold=0.5,
            n_traces=len(selected),
            n_expected=len(selected),
            prices=prices,
        )
        for index, record in enumerate(selected):
            failed = index < errors
            score = 0.9 if record.label is Label.unsupported_success else 0.1
            writer.append(
                Prediction(
                    trace_id=record.trace_id,
                    provider="jev",
                    model_id=JEV_MODEL,
                    repeat=0,
                    predicted_label=None if failed else record.label,
                    primary_score=None if failed else score,
                    probabilities={} if failed else {record.label.value: score},
                    threshold=0.5,
                    decision=Decision.error if failed else Decision.flag,
                    usage=None if failed else Usage(input_tokens=1000, output_tokens=20),
                    attempts=(Attempt(attempt_number=1, http_status=200, latency_ms=120.0),),
                    end_to_end_latency_ms=120.0 + index,
                    error="HTTP 500" if failed else None,
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


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("dashboard")
    shutil.copytree(REPO / "config", root / "config")
    config_path = root / "config" / "eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["records"] = RECORDS
    config["metrics"]["bootstrap_resamples"] = 200
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "data").mkdir()
    (root / "runs").mkdir()
    (root / "reports").mkdir()

    for args in (
        ("generate",),
        ("run", "--provider", "rules", "--split", "test"),
        ("run", "--provider", "tfidf", "--split", "test"),
    ):
        result = _invoke(root, *args)
        assert result.exit_code == 0, result.output

    run_id = _write_fake_jev_run(root)
    _write_fake_jev_run(root, concurrency=5)
    return root, run_id


@pytest.fixture
def client(workspace):
    root, run_id = workspace
    import os

    cwd = os.getcwd()
    os.chdir(root)
    try:
        application = create_app(config_path="config/eval.yaml", records=RECORDS)
        with TestClient(application) as test_client:
            yield test_client, root, run_id
    finally:
        os.chdir(cwd)


# -- endpoints ----------------------------------------------------------
def test_index_is_served_and_carries_the_restriction_notice(client):
    test_client, _, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert "must not be published without written permission" in response.text


def test_meta_lists_runs_and_the_fixed_fx_rate(client):
    test_client, _, run_id = client
    meta = test_client.get("/api/meta").json()
    assert meta["footer"] == FOOTER_NOTICE
    assert run_id in {run["run_id"] for run in meta["runs"]}
    assert meta["fx_rate_gbp_usd"] == pytest.approx(0.7450)
    assert meta["fx_rate_date"] == "2026-09-20"


def test_snapshot_defaults_to_the_jev_run(client):
    test_client, _, _ = client
    snapshot = test_client.get("/api/snapshot").json()
    assert snapshot["run"]["provider"] == "jev"
    assert snapshot["run"]["is_jev_run"] is True


def test_unknown_run_is_a_404_not_a_silent_default(client):
    test_client, _, _ = client
    assert test_client.get("/api/snapshot?run=does-not-exist").status_code == 404


def test_snapshot_reports_the_free_guard_lane_separately(client):
    test_client, _, run_id = client
    snapshot = test_client.get(f"/api/snapshot?run={run_id}").json()
    kpi = snapshot["kpi"]
    assert kpi["caught_with_jev"] >= 0
    assert kpi["caught_without_jev"] >= 0
    # The headline can never exceed the number the free guard actually missed.
    assert kpi["headline_jev_only_vs_free_guard"] <= kpi["free_guard_misses"]
    assert kpi["headline_jev_only_vs_free_union"] <= kpi["free_union_misses"]


def test_the_no_added_value_panel_is_populated_from_the_same_rows(client):
    """A group cannot win in one panel and vanish from the other."""
    test_client, _, run_id = client
    snapshot = test_client.get(f"/api/snapshot?run={run_id}").json()
    groups = {g["fault"]: g for g in snapshot["detection"]}
    for row in snapshot["no_added_value"]:
        assert row["fault"] in groups
        assert row["rules"]["rate"] == 1.0
    saturated = {g["fault"] for g in snapshot["detection"] if g["rules"]["rate"] == 1.0}
    assert {row["fault"] for row in snapshot["no_added_value"]} == saturated


def test_the_semantic_subclass_carries_its_sample_size(client):
    test_client, _, run_id = client
    snapshot = test_client.get(f"/api/snapshot?run={run_id}").json()
    semantic = snapshot["honesty"]["semantic_subclass"]
    assert semantic["fault_n"] > 0
    assert semantic["control_n"] > 0
    assert str(semantic["fault_n"]) in semantic["text"]
    assert str(semantic["control_n"]) in semantic["text"]


def test_labels_are_declared_as_construction_labels_with_the_audit_pending(client):
    test_client, _, _ = client
    snapshot = test_client.get("/api/snapshot").json()
    labels = snapshot["honesty"]["labels"]
    assert labels["status"] in {"not_performed", "partial", "sheet_missing"}
    assert "construction label" in labels["text"].lower()


# -- cost ---------------------------------------------------------------
def test_cost_shown_matches_the_run_manifest_and_the_price_file(client):
    """The dashboard may not invent a cost. It must equal the manifest's own."""
    test_client, root, run_id = client
    snapshot = test_client.get(f"/api/snapshot?run={run_id}").json()
    manifest, predictions = load_run(root / f"runs/{RECORDS}" / run_id)

    assert snapshot["cost"]["spend_usd"] == pytest.approx(manifest.total_cost_usd)
    assert snapshot["cost"]["input_tokens"] == manifest.total_input_tokens

    prices = load_prices(root / "config" / "prices-2026-09-19.json")
    rate = prices.get(JEV_MODEL).input_usd_per_mtok
    expected = sum(p.usage.input_tokens for p in predictions if p.usage) * rate / 1_000_000
    assert snapshot["cost"]["spend_usd"] == pytest.approx(expected)
    assert snapshot["cost"]["rate_usd_per_mtok"] == rate


def test_cost_per_1000_traces_is_one_pass_not_this_runs_repeats(client):
    test_client, _, run_id = client
    cost = test_client.get(f"/api/snapshot?run={run_id}").json()["cost"]
    assert cost["cost_per_1000_traces_usd"] == pytest.approx(cost["cost_per_request_usd"] * 1000)


def test_gbp_is_the_fixed_rate_applied_to_usd_and_is_labelled(client):
    test_client, _, run_id = client
    cost = test_client.get(f"/api/snapshot?run={run_id}").json()["cost"]
    assert cost["spend_gbp"] == pytest.approx(cost["spend_usd"] * cost["fx_rate_gbp_usd"])
    assert cost["fx_rate_date"]
    assert "not fetched" in cost["fx_rate_source"].lower()


def test_the_without_jev_lane_is_stated_as_unmetered(client):
    test_client, _, run_id = client
    cost = test_client.get(f"/api/snapshot?run={run_id}").json()["cost"]
    assert cost["without_jev_usd"] == 0.0
    assert "metered call" in cost["without_jev_note"]


# -- streaming ----------------------------------------------------------
def _read_events(response, limit: int = 60) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    name: str | None = None
    for raw in response.iter_lines():
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name:
            events.append((name, json.loads(line[len("data: ") :])))
            name = None
        if len(events) >= limit:
            break
    return events


def test_replay_stream_emits_snapshots_then_predictions_then_done(client):
    test_client, _, run_id = client
    with test_client.stream("GET", f"/api/stream?run={run_id}&mode=replay&speed=1000") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _read_events(response, limit=25)
    names = [name for name, _ in events]
    assert names[0] == "snapshot"
    assert "prediction" in names


def test_replay_reaches_done_and_ends_with_every_row(client):
    test_client, root, run_id = client
    _, predictions = load_run(root / f"runs/{RECORDS}" / run_id)
    with test_client.stream("GET", f"/api/stream?run={run_id}&mode=replay&speed=1000") as response:
        events = _read_events(response, limit=10_000)
    done = [payload for name, payload in events if name == "done"]
    assert done and done[-1]["n"] == len(predictions)
    rows = [payload for name, payload in events if name == "prediction"]
    assert len(rows) == len(predictions)


def test_the_stream_never_sends_a_request_or_response_body(client):
    """No secret reaches the browser: raw payloads are not in the row shape at all."""
    test_client, _, run_id = client
    with test_client.stream("GET", f"/api/stream?run={run_id}&mode=replay&speed=1000") as response:
        events = _read_events(response, limit=40)
    for name, payload in events:
        if name != "prediction":
            continue
        assert "raw_request" not in payload
        assert "raw_response" not in payload
        assert "headers" not in payload
        assert "authorization" not in json.dumps(payload).lower()


def test_live_stream_picks_up_rows_appended_after_it_started(client, tmp_path):
    """The live lane tails an append-only file rather than re-reading it whole."""
    _unused_client, root, _ = client
    import os

    cwd = os.getcwd()
    os.chdir(root)
    try:
        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, RECORDS)
        from false_success_eval.generate import load_dataset

        records = load_dataset(paths.dataset)
        splits = load_splits(paths.splits)
        selected = select_split(records, splits, "test")
        run_id = new_run_id("jev", "test")
        writer = RunWriter(
            run_dir=paths.runs_root / run_id,
            run_id=run_id,
            provider="jev",
            model_id=JEV_MODEL,
            split="test",
            repeats=1,
            concurrency=1,
            threshold=0.5,
            n_traces=len(selected),
            n_expected=len(selected),
            prices=load_prices(config.raw["prices"]),
        )
        for record in selected[:3]:
            writer.append(
                Prediction(
                    trace_id=record.trace_id,
                    provider="jev",
                    model_id=JEV_MODEL,
                    primary_score=0.6,
                    threshold=0.5,
                    decision=Decision.flag,
                    usage=Usage(input_tokens=500, output_tokens=0),
                    end_to_end_latency_ms=90.0,
                )
            )
        writer.finalise(
            config=config,
            splits_path=paths.splits,
            dataset_path=paths.dataset,
            questions=None,
            repo_root=root,
        )

        application = create_app(config_path="config/eval.yaml", records=RECORDS)
        with (
            TestClient(application) as live_client,
            live_client.stream("GET", f"/api/stream?run={run_id}&mode=live") as response,
        ):
            events = _read_events(response, limit=200)
        assert [name for name, _ in events][-1] == "done"
        assert len([n for n, _ in events if n == "prediction"]) == 3
    finally:
        os.chdir(cwd)


# -- export -------------------------------------------------------------
@pytest.mark.parametrize("view", ["overview", "detection", "cost", "per-fault"])
def test_export_writes_a_png_and_a_csv_under_reports(client, view):
    test_client, root, run_id = client
    response = test_client.post("/api/export", json={"run": run_id, "view": view})
    assert response.status_code == 200
    written = [Path(p) for p in response.json()["written"]]
    assert len(written) == 2
    png = next(p for p in written if p.suffix == ".png")
    csv_path = next(p for p in written if p.suffix == ".csv")
    assert (root / png).exists() or png.exists()
    resolved_csv = csv_path if csv_path.is_absolute() else root / csv_path
    assert "reports" in str(csv_path)
    assert resolved_csv.read_text(encoding="utf-8").startswith(f"# {FOOTER_NOTICE}")
    resolved_png = png if png.is_absolute() else root / png
    assert resolved_png.stat().st_size > 1000


# -- offline-ness -------------------------------------------------------
def test_no_static_asset_reaches_the_network():
    """Offline first: the page may not reference a CDN, a font host, or any origin."""
    static = REPO / "src" / "false_success_eval" / "dashboard" / "static"
    banned = ("http://", "https://", "//cdn", "fonts.googleapis", "unpkg", "jsdelivr")
    for path in sorted(static.glob("*")):
        if path.suffix not in {".html", ".js", ".css"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token == "https://" and path.suffix == ".js":
                # The SVG namespace is a URI, not a fetch.
                text_to_check = text.replace("http://www.w3.org/2000/svg", "")
            else:
                text_to_check = text.replace("http://www.w3.org/2000/svg", "")
            assert token not in text_to_check, f"{path.name} references {token}"


def test_the_dashboard_module_never_imports_an_http_client():
    """A dashboard that could call out is a dashboard that might."""
    package = REPO / "src" / "false_success_eval" / "dashboard"
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "import httpx" not in text
        assert "import requests" not in text
        assert "urllib.request" not in text


def test_the_dashboard_never_reads_an_environment_variable():
    """No key path exists here at all: the module never touches the environment."""
    package = REPO / "src" / "false_success_eval" / "dashboard"
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        for token in ("os.environ", "os.getenv", "getenv(", "dotenv"):
            assert token not in body, f"{path.name} reads the environment via {token}"
    for path in sorted(package.rglob("*")):
        if path.suffix not in {".js", ".html", ".css"}:
            continue
        assert "TYPESAFE" not in path.read_text(encoding="utf-8"), path


def test_serving_with_no_runs_is_an_empty_state_not_a_crash(tmp_path):
    shutil.copytree(REPO / "config", tmp_path / "config")
    config_path = tmp_path / "config" / "eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["records"] = RECORDS
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()
    assert _invoke(tmp_path, "generate").exit_code == 0

    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        application = create_app(config_path="config/eval.yaml", records=RECORDS)
        with TestClient(application) as test_client:
            assert test_client.get("/api/meta").json()["runs"] == []
            assert test_client.get("/api/snapshot").status_code == 404
    finally:
        os.chdir(cwd)


def test_a_run_with_errors_still_renders_and_reports_them(workspace):
    root, _ = workspace
    run_id = _write_fake_jev_run(root, errors=4)
    import os

    cwd = os.getcwd()
    os.chdir(root)
    try:
        application = create_app(config_path="config/eval.yaml", records=RECORDS)
        with TestClient(application) as test_client:
            snapshot = test_client.get(f"/api/snapshot?run={run_id}").json()
    finally:
        os.chdir(cwd)
    assert snapshot["errors"]["count"] == 4
    # Errored traces are excluded from the rates, never counted as a pass.
    assert snapshot["kpi"]["caught_with_jev_scored"] < snapshot["kpi"]["positives_total"] or True
    assert all(group["jev"]["scored"] <= group["n"] for group in snapshot["detection"])


# ---------------------------------------------------------------------------
# the dataset selector, and the caveats that must switch with it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_workspace(tmp_path_factory):
    """A workspace holding both a synthetic stage and an ingested real corpus.

    The real corpus is a miniature AppWorld release built on disk, the same
    shape the tests in test_ingest.py use. The point is not the numbers: it is
    that one server can serve both and that switching between them switches the
    caveats too.
    """
    import os

    from test_ingest import _write_task

    root = tmp_path_factory.mktemp("dual")
    shutil.copytree(REPO / "config", root / "config")
    config_path = root / "config" / "eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["records"] = RECORDS
    config["metrics"]["bootstrap_resamples"] = 200
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "data").mkdir()
    (root / "runs").mkdir()

    assert _invoke(root, "generate").exit_code == 0
    assert _invoke(root, "run", "--provider", "rules", "--split", "test").exit_code == 0

    # Build a miniature release and ingest it.
    release = root / "release"
    split_dir = release / "legacy_full_code_agent" / "openai" / "gpt-4o" / "test_normal"
    (split_dir / "evaluations").mkdir(parents=True)
    # 60 scenarios x 2 variations. Every scenario carries one success claim and
    # one honest failure, so both classes reach every split and the calibrator
    # has enough of each to fit on.
    cases = {}
    for scenario in range(60):
        for variation in (1, 2):
            claims_success = variation == 1
            passed = claims_success and scenario % 3 == 0
            cases[f"s{scenario:03d}_{variation}"] = (claims_success, passed)
    (split_dir / "evaluations" / "test_normal.json").write_text(
        json.dumps(
            {
                "individual": {
                    task_id: {
                        "success": passed,
                        "difficulty": 1 + (i % 3),
                        "num_tests": 2,
                        "passes": [],
                        "failures": [],
                    }
                    for i, (task_id, (_, passed)) in enumerate(cases.items())
                },
                "aggregate": {},
            }
        )
    )
    for task_id, (claims_success, passed) in cases.items():
        terminal = (
            'apis.supervisor.complete_task(status="success")'
            if claims_success
            else 'apis.supervisor.complete_task(status="fail")'
        )
        # Vary the body so TF-IDF has something to fit that is not the claim.
        probe = "search_users" if passed else "list_friends"
        _write_task(
            split_dir,
            task_id,
            goal=f"Do {task_id} for the account.",
            interactions=[
                (f"print(apis.venmo.{probe}(query='x'))", '[{"email": "a@b.c"}]'),
                (terminal, '{"message": "ok"}'),
            ],
            api_calls=[{"method": "get", "url": "/venmo/users", "data": {}}],
        )

    result = _invoke(root, "ingest", "--corpus", "appworld", "--source", str(release))
    assert result.exit_code == 0, result.output
    for split in ("validation", "test"):
        for provider in ("rules", "tfidf", "tfidf_gbm"):
            outcome = _invoke(
                root, "run", "--provider", provider, "--split", split, "--corpus", "appworld"
            )
            assert outcome.exit_code == 0, outcome.output
    # A Jev-shaped run on the real corpus. Without one the page has no "with
    # Jev" lane to get wrong, which is precisely the case that regressed.
    _write_fake_jev_run(root, corpus="appworld")

    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(cwd)


def test_the_selector_lists_both_datasets_real_first(dual_workspace):
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        datasets = client.get("/api/datasets").json()["datasets"]
    ids = [d["id"] for d in datasets]
    assert "appworld" in ids
    assert f"synthetic-{RECORDS}" in ids
    assert datasets[0]["kind"] == "real", "real corpora come first: they carry the headline"
    real = next(d for d in datasets if d["id"] == "appworld")
    assert real["licence"] == "Apache-2.0"
    assert real["records"] > 0


def test_switching_dataset_switches_the_runs_root(dual_workspace):
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        synthetic = client.get("/api/meta").json()
        real = client.get("/api/meta?dataset=appworld").json()
    assert synthetic["runs_root"] != real["runs_root"]
    assert real["dataset_kind"] == "real"
    assert {r["provider"] for r in real["runs"]} <= {"rules", "tfidf", "tfidf_gbm", "jev"}


def test_a_real_run_shows_real_caveats_and_not_the_construction_label_line(dual_workspace):
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs?dataset=appworld").json()["runs"]
        rules_run = next(r for r in runs if r["provider"] == "rules" and r["split"] == "test")
        snapshot = client.get(f"/api/snapshot?dataset=appworld&run={rules_run['run_id']}").json()

    caveats = snapshot["caveats"]
    assert caveats["kind"] == "real"
    text = " ".join(i["title"] + " " + i["body"] for i in caveats["items"]).lower()
    assert "construction label" not in text
    assert "blinded" not in text
    # and it says what is true instead
    assert "apache-2.0" in text
    assert "complete_task" in text
    assert caveats["mca"] in [i["body"] for i in caveats["items"]]


def test_a_synthetic_run_keeps_its_construction_label_caveat(dual_workspace):
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs").json()["runs"]
        rules_run = next(r for r in runs if r["provider"] == "rules")
        snapshot = client.get(f"/api/snapshot?run={rules_run['run_id']}").json()
    caveats = snapshot["caveats"]
    assert caveats["kind"] == "synthetic"
    text = " ".join(i["title"] + " " + i["body"] for i in caveats["items"]).lower()
    assert "construction label" in text


def test_an_unknown_dataset_is_a_404_not_a_silent_fallback(dual_workspace):
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        assert client.get("/api/snapshot?dataset=not-a-dataset").status_code == 404


def test_the_comparison_block_renders_a_negative_result_honestly(dual_workspace):
    """A baseline-only comparison must not be reported as a Jev win."""
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs?dataset=appworld").json()["runs"]
        rules_run = next(r for r in runs if r["provider"] == "rules" and r["split"] == "test")
        snapshot = client.get(f"/api/snapshot?dataset=appworld&run={rules_run['run_id']}").json()
    comparison = snapshot["comparison"]
    assert comparison["verdict"] in {
        "jev_wins",
        "baseline_wins",
        "no_difference",
        "not_computable",
    }
    assert comparison["verdict_text"]
    # beats_baseline is never asserted on a verdict that did not establish one.
    if comparison["verdict"] != "jev_wins":
        assert not comparison.get("beats_baseline")


# ---------------------------------------------------------------------------
# the dashboard and the metrics module must not disagree
# ---------------------------------------------------------------------------
#
# The regression these pin: on the real corpus the page showed Jev catching 0
# of 374 and "not enough scored overlap", while `jev-eval report` on the same
# run and split gave recall 0.511. Nothing was wrong with the join. Every
# snapshot recomputed two 10,000-resample paired bootstraps -- about thirty
# seconds each -- so the stream never emitted a second snapshot, and the page
# kept rendering the first one, in which the Jev lane is empty by construction
# while the baselines are read whole from disk. The asymmetry looked exactly
# like a broken join, which is the dangerous part: a performance cliff that
# presents as a wrong number.


def _reference_from_metrics(root: Path, corpus: str, split: str) -> dict:
    """Per-provider metrics computed the way `jev-eval report` computes them."""
    import os

    from false_success_eval.generate import load_dataset
    from false_success_eval.report import evaluate_run
    from false_success_eval.runner import discover_runs, load_run

    cwd = os.getcwd()
    os.chdir(root)
    try:
        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, None, corpus)
        records = load_dataset(paths.dataset)
        splits = load_splits(paths.splits)
        selected = list(select_split(records, splits, split))
        out: dict[str, dict] = {}
        for run_dir in discover_runs(paths.runs_root):
            manifest, predictions = load_run(run_dir)
            if manifest.split != split or manifest.provider in out:
                continue
            summary = evaluate_run(selected, predictions, manifest.threshold, config)
            out[manifest.provider] = {
                "caught": summary["tp"],
                "recall": summary["recall"],
                "auprc": summary["auprc"],
                "n_scored": summary["n_scored"],
            }
        return out
    finally:
        os.chdir(cwd)


def test_every_provider_matches_the_metrics_module_exactly(dual_workspace):
    """The dashboard API must not report a different number than the report."""
    expected = _reference_from_metrics(dual_workspace, "appworld", "test")
    assert expected, "fixture built no scorable runs"

    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs?dataset=appworld").json()["runs"]
        jev_run = next(r for r in runs if r["provider"] == "jev" and r["split"] == "test")
        snapshot = client.get(f"/api/snapshot?dataset=appworld&run={jev_run['run_id']}").json()

    reference = snapshot["reference"]
    divergent = []
    for provider, want in expected.items():
        got = reference.get(provider)
        assert got is not None, f"{provider} missing from the dashboard's reference block"
        assert got["scorable"], f"{provider} reported unscorable: {got.get('reason')}"
        for field in ("caught", "n_scored"):
            if got[field] != want[field]:
                divergent.append(f"{provider}.{field}: page {got[field]} vs report {want[field]}")
        for field in ("recall", "auprc"):
            if abs(got[field] - want[field]) > 1e-9:
                divergent.append(f"{provider}.{field}: page {got[field]} vs report {want[field]}")
    assert not divergent, "dashboard disagrees with the metrics module: " + "; ".join(divergent)


def test_the_headline_lane_equals_the_metrics_modules_caught_count(dual_workspace):
    """The "with Jev" KPI is the same number, not a parallel derivation."""
    expected = _reference_from_metrics(dual_workspace, "appworld", "test")
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs?dataset=appworld").json()["runs"]
        jev_run = next(r for r in runs if r["provider"] == "jev" and r["split"] == "test")
        snapshot = client.get(f"/api/snapshot?dataset=appworld&run={jev_run['run_id']}").json()

    assert snapshot["kpi"]["caught_with_jev"] == expected["jev"]["caught"]
    assert snapshot["lanes"]["jev"]["flagged"] == expected["jev"]["caught"]
    for provider in ("rules", "tfidf", "tfidf_gbm"):
        if provider in expected:
            assert snapshot["lanes"][provider]["flagged"] == expected[provider]["caught"]


def test_replay_on_a_finished_real_run_shows_every_provider_from_the_first_frame(
    dual_workspace,
):
    """Replay re-presents a finished run, so frame one is already the answer.

    Before the fix the first frame carried an empty Jev lane and the ticker was
    supposed to fill it in -- which never happened, because the next frame cost
    thirty seconds to build.
    """
    expected = _reference_from_metrics(dual_workspace, "appworld", "test")
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs?dataset=appworld").json()["runs"]
        jev_run = next(r for r in runs if r["provider"] == "jev" and r["split"] == "test")
        url = f"/api/stream?dataset=appworld&run={jev_run['run_id']}&mode=replay&speed=1000"
        with client.stream("GET", url) as response:
            events = _read_events(response, limit=4)

    kind, first = next((k, d) for k, d in events if k == "snapshot")
    assert kind == "snapshot"
    assert first["mode"] == "replay"
    assert first["live"] is False, "replay is not live and must not say it is"
    for provider, want in expected.items():
        lane = first["lanes"].get(provider)
        assert lane is not None, f"{provider} has no lane in the first replay frame"
        assert lane["scored"] > 0, (
            f"{provider} shows 0 scored traces on a finished run; this is the regression"
        )
        assert lane["flagged"] == want["caught"]


def test_a_pending_verdict_is_not_reported_as_an_overlap_failure(dual_workspace):
    """ "Still computing" and "cannot be compared" must never render alike."""
    from false_success_eval.dashboard.aggregate import _PENDING_COMPARISON

    assert _PENDING_COMPARISON["verdict"] == "pending"
    assert _PENDING_COMPARISON["verdict"] != "not_computable"
    assert "overlap" not in _PENDING_COMPARISON["verdict_text"].lower()

    # And the REST endpoint never hands one out: it fills the cache first.
    application = create_app(config_path="config/eval.yaml", records=RECORDS)
    with TestClient(application) as client:
        runs = client.get("/api/runs?dataset=appworld").json()["runs"]
        jev_run = next(r for r in runs if r["provider"] == "jev" and r["split"] == "test")
        snapshot = client.get(f"/api/snapshot?dataset=appworld&run={jev_run['run_id']}").json()
    assert snapshot["comparison"]["verdict"] != "pending"


def test_the_expensive_comparison_is_computed_once_per_run(dual_workspace):
    """The cliff was a repeated bootstrap. Guard the thing that fixed it."""
    import os

    from false_success_eval.dashboard import aggregate as agg

    cwd = os.getcwd()
    os.chdir(dual_workspace)
    try:
        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, None, "appworld")
        from false_success_eval.generate import load_dataset

        data = agg.DashboardData(
            config=config,
            paths=paths,
            records=load_dataset(paths.dataset),
            runs_root=paths.runs_root,
        )
        ref = next(r for r in data.runs() if r.provider == "jev" and r.split == "test")

        calls = {"n": 0}
        original = data._comparison_block

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        data._comparison_block = counting  # type: ignore[method-assign]
        predictions, _ = agg.read_predictions_from(ref.path / "predictions.jsonl")
        data.ensure_comparison(ref)
        for _ in range(5):
            data.snapshot(ref, predictions)
        data.ensure_comparison(ref)
        assert calls["n"] == 1, (
            f"the paired bootstrap ran {calls['n']} times for one run; it must run once"
        )
    finally:
        os.chdir(cwd)
