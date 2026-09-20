"""The design rules of the report page, checked rather than eyeballed.

Three of them are the kind that quietly rot: a colour edited by eye, a table
that stops agreeing with the chart above it, a transition added after the
reduced-motion block was written. Each is recomputed here from the files the
browser is actually served.

The colour maths is implemented in this file rather than imported, so the test
is an independent check on the stylesheet and not a restatement of it:

* WCAG 2.x relative luminance and contrast ratio.
* The Machado et al. (2009) severity-1.0 matrices for protanopia, deuteranopia
  and tritanopia, and CIE L*a*b* distance between the simulated colours.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from false_success_eval.cli import app as cli_app
from false_success_eval.costs import load_prices
from false_success_eval.dashboard import export
from false_success_eval.dashboard.aggregate import (
    FIELD_INDEX,
    PROVIDER_META,
    DashboardData,
    build_figure,
    read_predictions_from,
)
from false_success_eval.generate import load_dataset
from false_success_eval.runner import (
    RunWriter,
    load_config,
    load_splits,
    new_run_id,
    resolve_dataset_paths,
    select_split,
)
from false_success_eval.schemas import Attempt, Decision, Label, Prediction, Usage

REPO = Path(__file__).resolve().parents[1]
RECORDS = 160
cli = CliRunner()

STATIC = Path(export.__file__).resolve().parent / "static"
TOKENS = STATIC / "tokens.css"
APP_CSS = STATIC / "app.css"
APP_JS = STATIC / "app.js"
CHARTS_JS = STATIC / "charts.js"
INDEX = STATIC / "index.html"

#: WCAG AA for body text. Applied to every token used as type.
TEXT_CONTRAST = 4.5

#: WCAG 1.4.11 for a graphical object that carries meaning. A series mark is
#: one: if it cannot be seen against the page, the chart is not readable.
MARK_CONTRAST = 3.0

#: CIE76 distance below which two colours read as the same at a glance. 20 is a
#: conservative threshold for marks that appear side by side.
SEPARATION = 20.0

CVD_MATRICES = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}

SERIES_TOKENS = (
    "--series-trained",
    "--series-trained-alt",
    "--series-typed",
    "--series-general",
    "--series-heuristic",
)

#: The two TF-IDF arms are one role in two steps, so they are not required to be
#: separable from each other -- they are always direct-labelled and they never
#: appear as a contrast the reader has to resolve by hue.
SAME_ROLE = {frozenset({"--series-trained", "--series-trained-alt"})}


# -- one real snapshot, built offline ------------------------------------
#
# The parity checks below are only worth something against figures the page
# would actually draw, so this fixture builds a throwaway workspace with real
# generated data, real offline classifier runs, a short learning curve, and
# hand-built Jev and judge runs written through the real RunWriter. No network:
# the socket block in conftest would fail loudly if anything reached for one.


def _invoke(workspace: Path, *args: str):
    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        return cli.invoke(cli_app, list(args))
    finally:
        os.chdir(cwd)


def _write_fake_run(workspace: Path, provider: str, model_id: str, score) -> str:
    cwd = os.getcwd()
    os.chdir(workspace)
    try:
        config = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(config, RECORDS, None)
        selected = select_split(load_dataset(paths.dataset), load_splits(paths.splits), "test")
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
            prices=load_prices(config.raw["prices"]),
        )
        for index, record in enumerate(selected):
            value = score(record, index)
            writer.append(
                Prediction(
                    trace_id=record.trace_id,
                    provider=provider,
                    model_id=model_id,
                    repeat=0,
                    predicted_label=record.label,
                    primary_score=value,
                    probabilities={record.label.value: value},
                    threshold=0.5,
                    decision=Decision.flag if value >= 0.5 else Decision.pass_,
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


def _typed_score(record, index):
    return 0.92 if record.label is Label.unsupported_success else 0.11


def _judge_score(record, index):
    """A weaker, noisier ranker. Nothing here is a claim about any model."""
    wobble = ((index * 37) % 11) / 20.0
    return (0.45 + wobble) if record.label is Label.unsupported_success else (0.30 + wobble)


@pytest.fixture(scope="module")
def dashboard_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("design")
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
        ("run", "--provider", "tfidf_gbm", "--split", "test"),
        ("learning-curve", "--providers", "tfidf", "--sizes", "10,25", "--seeds", "0,1"),
    ):
        result = _invoke(root, *args)
        assert result.exit_code == 0, result.output

    typed_run = _write_fake_run(root, "jev", "jev-1.13.0", _typed_score)
    _write_fake_run(root, "openai", "a-general-model-id-no-manifest-prices", _judge_score)

    cwd = os.getcwd()
    os.chdir(root)
    try:
        loaded = load_config("config/eval.yaml")
        paths = resolve_dataset_paths(loaded, RECORDS, None)
        data = DashboardData(
            config=loaded,
            paths=paths,
            records=load_dataset(paths.dataset),
            runs_root=paths.runs_root,
        )
        ref = next(run for run in data.runs() if run.run_id == typed_run)
        predictions, _ = read_predictions_from(ref.path / "predictions.jsonl")
        data.ensure_comparison(ref)
        return data.snapshot(ref, predictions)
    finally:
        os.chdir(cwd)


# -- colour maths, implemented here on purpose ---------------------------


def _rgb(value: str) -> tuple[float, float, float]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(value: str) -> float:
    r, g, b = (_linear(c) for c in _rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def simulate(value: str, kind: str) -> str:
    matrix = CVD_MATRICES[kind]
    r, g, b = _rgb(value)
    return _hex(
        tuple(max(0.0, min(1.0, row[0] * r + row[1] * g + row[2] * b)) for row in matrix)  # type: ignore[arg-type]
    )


def _lab(value: str) -> tuple[float, float, float]:
    r, g, b = (_linear(c) for c in _rgb(value))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def separation(a: str, b: str) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(_lab(a), _lab(b), strict=True)))


def worst_separation(a: str, b: str) -> float:
    """The closest these two ever get: normal vision or any simulated CVD."""
    return min(
        [separation(a, b)]
        + [separation(simulate(a, kind), simulate(b, kind)) for kind in CVD_MATRICES]
    )


def theme_tokens(selector: str) -> dict[str, str]:
    """The custom properties a selector block declares, parsed from tokens.css."""
    text = TOKENS.read_text(encoding="utf-8")
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\n\}", text, re.S)
    assert match, f"no {selector} block in {TOKENS.name}"
    return {
        name: value.strip()
        for name, value in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", match.group(1))
    }


LIGHT = theme_tokens(":root")
DARK = {**LIGHT, **theme_tokens(':root[data-theme="dark"]')}
THEMES = {"light": LIGHT, "dark": DARK}


# -- 1. contrast tokens --------------------------------------------------


@pytest.mark.parametrize("theme", sorted(THEMES))
@pytest.mark.parametrize("name", ["--text", "--muted"])
def test_text_tokens_meet_wcag_aa_against_the_page(theme, name):
    tokens = THEMES[theme]
    ratio = contrast(tokens[name], tokens["--bg"])
    assert ratio >= TEXT_CONTRAST, f"{theme} {name} is {ratio:.2f}:1 on --bg, needs {TEXT_CONTRAST}"


@pytest.mark.parametrize("theme", sorted(THEMES))
@pytest.mark.parametrize("name", ["--text", "--muted"])
def test_text_tokens_meet_wcag_aa_against_the_raised_surface(theme, name):
    """Cards are gone, but the surface token is still used behind notices."""
    tokens = THEMES[theme]
    ratio = contrast(tokens[name], tokens["--surface"])
    assert ratio >= TEXT_CONTRAST, f"{theme} {name} is {ratio:.2f}:1 on --surface"


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_text_on_the_accent_meets_wcag_aa(theme):
    tokens = THEMES[theme]
    ratio = contrast(tokens["--on-accent"], tokens["--series-trained"])
    assert ratio >= TEXT_CONTRAST, f"{theme} --on-accent is {ratio:.2f}:1 on the accent"


@pytest.mark.parametrize("theme", sorted(THEMES))
@pytest.mark.parametrize(
    "name", [*SERIES_TOKENS, "--semantic-for", "--semantic-against", "--focus"]
)
def test_every_mark_colour_clears_the_non_text_contrast_floor(theme, name):
    """A series mark and a focus ring are graphical objects, not decoration."""
    tokens = THEMES[theme]
    ratio = contrast(tokens[name], tokens["--bg"])
    assert ratio >= MARK_CONTRAST, f"{theme} {name} is {ratio:.2f}:1 on --bg, needs {MARK_CONTRAST}"


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_series_stay_apart_under_every_simulated_colour_vision_deficiency(theme):
    tokens = THEMES[theme]
    for a, b in itertools.combinations(SERIES_TOKENS, 2):
        if frozenset({a, b}) in SAME_ROLE:
            continue
        worst = worst_separation(tokens[a], tokens[b])
        assert worst >= SEPARATION, (
            f"{theme}: {a} and {b} collapse to {worst:.1f} Lab units under simulated "
            "colour-vision deficiency; they are drawn side by side"
        )


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_the_reserved_semantic_pair_stays_apart_under_colour_vision_deficiency(theme):
    tokens = THEMES[theme]
    worst = worst_separation(tokens["--semantic-for"], tokens["--semantic-against"])
    assert worst >= SEPARATION, f"{theme}: for/against collapse to {worst:.1f} Lab units"


def test_the_semantic_pair_is_never_used_as_a_series_colour():
    """Reserved means reserved: win and loss are not a category."""
    charts = CHARTS_JS.read_text(encoding="utf-8")
    assert "--semantic-for" not in charts
    assert "--semantic-against" not in charts


def test_dark_is_re_stepped_rather_than_inverted():
    """Every series token differs between themes, and none is the light value."""
    for name in SERIES_TOKENS:
        assert LIGHT[name] != DARK[name] or name == "--series-trained-alt", (
            f"{name} is identical in both themes; dark must take its own step off the hue"
        )
    assert LIGHT["--bg"] != DARK["--bg"]


def test_the_page_uses_tokens_rather_than_literal_colours():
    """A hex literal in the layout is a colour that escapes every check above."""
    css = APP_CSS.read_text(encoding="utf-8")
    leaked = re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
    assert not leaked, f"app.css hard-codes colours {leaked}; they belong in tokens.css"


def test_the_export_reads_its_palette_from_the_same_tokens():
    """The PNG and the page cannot drift if there is only one palette."""
    for role, colour in export.ROLE_COLOUR.items():
        token = f"--series-{role}"
        assert colour.lower() == LIGHT[token].lower(), f"{role} differs from {token}"


# -- 2. table parity -----------------------------------------------------


def _flatten_series(figure: dict) -> list[tuple[str, ...]]:
    """(series label, key, x, y, lower, upper) for every point the chart draws."""
    digits = figure.get("digits", 3)

    def fmt(value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return f"{float(value):.{digits}f}"  # type: ignore[arg-type]

    out: list[tuple[str, ...]] = []
    for entry in figure["series"]:
        for point in entry["points"]:
            out.append(
                (
                    entry["label"],
                    entry["key"],
                    str(point["x"]),
                    fmt(point.get("y")),
                    fmt(point.get("lower")),
                    fmt(point.get("upper")),
                )
            )
    return out


def _flatten_table(figure: dict) -> list[tuple[str, ...]]:
    index = figure["table"]["field_index"]
    order = ("series", "key", "x", "y", "lower", "upper")
    return [tuple(row[index[field]] for field in order) for row in figure["table"]["rows"]]


def sample_figure() -> dict:
    return build_figure(
        figure_id="fig-test",
        kind="interval-dots",
        title="t",
        caption="c",
        value_label="AUPRC",
        x_label="Detector",
        series=[
            {
                "key": "tfidf_gbm",
                "label": "TF-IDF + boosting",
                "short": "GBM",
                "role": "trained",
                "group": "Trained on labels",
                "trains": True,
                "paid": False,
                "points": [
                    {"x": "GBM", "y": 0.96, "lower": 0.94, "upper": 0.97, "extra": {"n": 702}}
                ],
            },
            {
                "key": "openai",
                "label": "General LLM judge",
                "short": "Judge",
                "role": "general",
                "group": "Zero-shot, no training data",
                "trains": False,
                "paid": True,
                "points": [{"x": "Judge", "y": None, "lower": None, "upper": None, "extra": {}}],
            },
        ],
        extra_columns=["n"],
    )


def test_a_figures_table_is_the_same_points_the_chart_draws():
    figure = sample_figure()
    assert _flatten_table(figure) == _flatten_series(figure)


def test_the_table_carries_the_machine_readable_key_as_well_as_the_label():
    """An exported CSV has to be joinable back to a run directory."""
    figure = sample_figure()
    keys = [row[figure["table"]["field_index"]["key"]] for row in figure["table"]["rows"]]
    assert keys == ["tfidf_gbm", "openai"]
    assert figure["table"]["columns"][FIELD_INDEX["key"]] == "Key"


def test_a_missing_value_is_a_dash_in_the_table_not_a_zero():
    """A reader must be able to tell 'unavailable' from 'zero'."""
    figure = sample_figure()
    judge = figure["table"]["rows"][1]
    assert judge[FIELD_INDEX["y"]] == "—"
    assert "0.000" not in judge


def test_every_figure_in_a_live_snapshot_round_trips_to_its_table(dashboard_snapshot):
    """The invariant on real data, not only on a fixture."""
    figures = dashboard_snapshot["figures"]
    assert set(figures) >= {"headline_auprc", "learning_curve", "per_fault", "cost", "latency"}
    for name, figure in figures.items():
        assert _flatten_table(figure) == _flatten_series(figure), f"{name} table differs"


def test_every_figure_declares_a_table_with_a_row_per_point(dashboard_snapshot):
    for name, figure in dashboard_snapshot["figures"].items():
        points = sum(len(entry["points"]) for entry in figure["series"])
        assert len(figure["table"]["rows"]) == points, f"{name} has {points} points"
        for row in figure["table"]["rows"]:
            assert len(row) == len(figure["table"]["columns"])


def test_the_csv_export_writes_the_figures_own_table(dashboard_snapshot):
    for view, figure_key in export.VIEW_FIGURE.items():
        rows = export.view_table(dashboard_snapshot, view)
        figure = dashboard_snapshot["figures"][figure_key]
        assert len(rows) == len(figure["table"]["rows"]), view
        if rows:
            assert list(rows[0].keys()) == figure["table"]["columns"], view
            assert list(rows[0].values()) == figure["table"]["rows"][0], view


# -- 3. reduced motion ---------------------------------------------------


def test_the_stylesheet_honours_prefers_reduced_motion():
    css = APP_CSS.read_text(encoding="utf-8")
    assert "@media (prefers-reduced-motion: reduce)" in css
    block = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    block = block[: block.index("\n}\n\n")]
    for declaration in (
        "animation-duration",
        "animation-iteration-count",
        "transition-duration",
        "scroll-behavior",
    ):
        assert declaration in block, f"reduced-motion block does not neutralise {declaration}"


def test_reduced_motion_uses_a_wildcard_so_a_new_transition_cannot_escape_it():
    css = APP_CSS.read_text(encoding="utf-8")
    block = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert "*," in block and "*::before" in block and "*::after" in block


def test_reduced_motion_declarations_are_important_enough_to_win():
    css = APP_CSS.read_text(encoding="utf-8")
    block = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    block = block[: block.index("\n}\n\n")]
    for line in block.splitlines():
        if ":" in line and ("animation" in line or "transition" in line):
            assert "!important" in line, f"reduced-motion declaration can be overridden: {line}"


def test_smooth_scrolling_is_switched_off_for_reduced_motion():
    css = APP_CSS.read_text(encoding="utf-8")
    assert "scroll-behavior: smooth" in css
    block = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert "scroll-behavior: auto" in block


# -- house rules the page must keep --------------------------------------


#: Wording that would crown the paid detector. The bare word "winner" is not on
#: this list on purpose: the page uses it to say there is no winner, and a test
#: that banned the word would push the prose into vaguer language rather than
#: honester language. These patterns match the claim, not the vocabulary.
CROWNING = (
    r"best detector",
    r"\bjev\b\s+wins\b",
    r"\bjev\b\s+is\s+the\s+best\b",
    r"\bjev\b[^.]{0,60}\bbeats\s+(?:all|every|both|them)\b",
    r"\bjev\b[^.]{0,60}\boutperforms\s+(?:all|every|both|them)\b",
    r"state[ -]of[ -]the[ -]art",
    r"\bjev\b[^.]{0,60}\bbest\b",
    r"\bbest\b[^.]{0,60}\bjev\b",
)


@pytest.mark.parametrize("pattern", CROWNING)
def test_no_wording_on_the_page_crowns_the_paid_detector(pattern):
    for path in (INDEX, APP_JS, CHARTS_JS):
        text = path.read_text(encoding="utf-8").lower()
        found = re.search(pattern, text)
        assert not found, f"{path.name} reads {found.group(0)!r}"


def test_the_finding_sentence_never_ends_on_the_paid_arm(dashboard_snapshot):
    """The uncomfortable clause is last, so the sentence cannot close on a win."""
    finding = dashboard_snapshot["finding"]
    sentence = finding["sentence"].lower()
    if "trained on labels" in sentence:
        assert sentence.rstrip(" .").endswith("beats them both"), sentence
    for pattern in CROWNING:
        assert not re.search(pattern, sentence), sentence


def test_the_honest_claim_is_carried_as_a_first_class_claim(dashboard_snapshot):
    """Not a footnote: the result against the paid arms is in the claim list."""
    claims = dashboard_snapshot["finding"]["claims"]
    assert claims, "the finding carries no claims"
    stances = {claim["stance"] for claim in claims}
    assert stances & {"against", "caveat"}, (
        "every claim on the page is a point in favour; the counter-evidence is missing"
    )
    for claim in claims:
        assert claim["text"] and claim["stance"] in {"for", "against", "caveat"}


def test_the_page_carries_the_restriction_without_javascript():
    assert "must not be published without written permission" in INDEX.read_text(encoding="utf-8")


def test_the_page_has_no_emoji():
    """Every character on the page is text, punctuation or a typographic mark."""
    for path in (INDEX, APP_JS, CHARTS_JS, APP_CSS, TOKENS):
        for char in path.read_text(encoding="utf-8"):
            assert not (0x1F000 <= ord(char) <= 0x1FAFF), f"{path.name} contains an emoji"
            assert not (0x2600 <= ord(char) <= 0x27BF), f"{path.name} contains an emoji"


def test_a_figure_never_spends_colour_on_a_rank(dashboard_snapshot):
    """p50/p95/p99 is a rank, so it does not get three hues.

    Every latency line measures the same arm. They share that arm's colour and
    are separated by their dash pattern and their own label, so a hue means the
    same thing in this figure as it does in every other one.
    """
    latency = dashboard_snapshot["figures"]["latency"]
    roles = {entry["role"] for entry in latency["series"]}
    assert len(roles) == 1, f"the latency figure colours percentiles by rank: {roles}"
    dashes = [entry.get("dash", "") for entry in latency["series"]]
    assert len(set(dashes)) == len(dashes), "series sharing a hue must differ by dash"


def test_the_legend_swatch_carries_the_dash_pattern():
    """A legend that dropped the dash would identify those series by hue alone."""
    charts = CHARTS_JS.read_text(encoding="utf-8")
    legend = charts.split("function legendFor(", 1)[1].split("\n  }", 1)[0]
    assert "s.dash" in legend


def test_every_provider_shown_has_a_role_and_a_group():
    for provider, meta in PROVIDER_META.items():
        assert meta["role"] in {"trained", "trained-alt", "typed", "general", "heuristic"}
        assert meta["group"], provider
        assert meta["label"] and meta["short"], provider


def test_colour_is_assigned_by_role_and_never_by_provider_name():
    """charts.js may map role to hue; mapping a provider id to a hue is rank."""
    charts = CHARTS_JS.read_text(encoding="utf-8")
    role_map = charts.split("var ROLE_VAR = {", 1)[1].split("};", 1)[0]
    for provider in PROVIDER_META:
        assert provider not in role_map, f"charts.js colours {provider} by name"


def test_the_snapshot_never_ships_a_raw_request_or_response(dashboard_snapshot):
    blob = json.dumps(dashboard_snapshot)
    assert "raw_request" not in blob
    assert "raw_response" not in blob
