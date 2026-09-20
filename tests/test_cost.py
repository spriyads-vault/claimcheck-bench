from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from false_success_eval.costs import PriceError, cost_usd, load_prices
from false_success_eval.schemas import Usage

MANIFEST = "config/prices-2026-09-19.json"


def test_manifest_loads_with_its_date():
    prices = load_prices(MANIFEST)
    assert prices.manifest_date == "2026-09-19"
    assert prices.currency == "USD"


def test_jev_rate_matches_the_documented_price():
    """$0.042 per Mtok input, output free."""
    price = load_prices(MANIFEST).get("jev-1.13.0")
    assert price.input_usd_per_mtok == 0.042
    assert price.output_usd_per_mtok == 0.0
    assert price.verified is True
    assert "docs.typesafe.ai" in price.source


def test_cost_is_input_tokens_times_rate_over_one_million():
    prices = load_prices(MANIFEST)
    cost = cost_usd(prices, "jev-1.13.0", Usage(input_tokens=1_000_000, output_tokens=5_000))
    assert cost == pytest.approx(0.042)


def test_output_tokens_are_free_under_this_manifest():
    prices = load_prices(MANIFEST)
    a = cost_usd(prices, "jev-1.13.0", Usage(input_tokens=2000, output_tokens=0))
    b = cost_usd(prices, "jev-1.13.0", Usage(input_tokens=2000, output_tokens=999_999))
    assert a == b == pytest.approx(2000 * 0.042 / 1_000_000)


def test_offline_providers_cost_nothing_but_still_price_through_the_manifest():
    prices = load_prices(MANIFEST)
    assert cost_usd(prices, "rules", Usage()) == 0.0
    assert cost_usd(prices, "tfidf-logreg-v1", Usage()) == 0.0


def test_an_unverified_rate_yields_none_rather_than_a_guess():
    prices = load_prices(MANIFEST)
    assert cost_usd(prices, "UNVERIFIED_GENERAL_MODEL", Usage(input_tokens=10)) is None


def test_missing_usage_yields_none():
    assert cost_usd(load_prices(MANIFEST), "jev-1.13.0", None) is None


def test_an_unknown_model_refuses_rather_than_defaulting():
    with pytest.raises(PriceError, match="no price entry"):
        cost_usd(load_prices(MANIFEST), "some-model-we-never-priced", Usage(input_tokens=1))


def test_no_price_literal_is_hard_coded_outside_the_manifest():
    """The rate may live in exactly one place."""
    import pathlib

    offenders = []
    for path in pathlib.Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "0.042" in line and "docs.typesafe.ai" not in line:
                offenders.append(f"{path}: {line.strip()}")
    assert not offenders, offenders


def test_manifest_is_valid_json_with_every_entry_shaped_the_same():
    import pathlib

    raw = json.loads(pathlib.Path(MANIFEST).read_text(encoding="utf-8"))
    for name, entry in raw["models"].items():
        assert {"input_usd_per_mtok", "output_usd_per_mtok", "verified", "source"} <= set(entry), (
            name
        )
        if entry["verified"]:
            assert entry["input_usd_per_mtok"] is not None, name


# ---------------------------------------------------------------------------
# dated manifests supersede, they do not get edited
# ---------------------------------------------------------------------------
#
# A price manifest is dated because runs record its SHA-256 and `verify-run`
# recomputes every cost against it. Editing one in place invalidates every run
# already recorded against it -- which is exactly what happened when the two
# zero-cost gradient-boosted evaluators were appended to the 2026-09-19 file:
# all eight synthetic runs stopped verifying. The fix was a new dated file, and
# these pin the discipline so the shortcut cannot be taken again quietly.

ACTIVE_MANIFEST = "config/prices-2026-09-20-2.json"

#: The full supersede chain, oldest first. Every link is walked rather than
#: only the newest hop, because "no rate changed" has to hold end to end: a
#: price that moved two manifests ago and moved back is still a repricing that
#: invalidates comparisons between runs on either side of it.
MANIFEST_CHAIN = (
    "config/prices-2026-09-19.json",
    "config/prices-2026-09-20.json",
    "config/prices-2026-09-20-2.json",
)


def _models(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["models"]


def test_the_manifest_eval_yaml_points_at_is_the_one_under_test():
    import yaml

    raw = yaml.safe_load(Path("config/eval.yaml").read_text(encoding="utf-8"))
    assert raw["prices"] == ACTIVE_MANIFEST, (
        "the active manifest changed; point ACTIVE_MANIFEST at it and re-run the "
        "runs recorded against the previous one"
    )


def test_the_active_manifest_declares_its_own_date_and_what_it_supersedes():
    prices = load_prices(ACTIVE_MANIFEST)
    assert prices.manifest_date == "2026-09-20"
    raw = json.loads(Path(ACTIVE_MANIFEST).read_text(encoding="utf-8"))
    assert raw["supersedes"] == MANIFEST_CHAIN[-2]


def test_two_same_day_manifests_are_told_apart_by_revision_not_by_editing():
    """A second manifest on one date needs a distinguishing mark of its own.

    Adding the OpenAI arm on the same day as the previous manifest could not be
    an in-place edit -- eight runs record that file's SHA-256 -- and it could not
    reuse the filename either. A `revision` makes the ordering explicit rather
    than leaving it to be inferred from a filename suffix.
    """
    previous = json.loads(Path(MANIFEST_CHAIN[-2]).read_text(encoding="utf-8"))
    active = json.loads(Path(ACTIVE_MANIFEST).read_text(encoding="utf-8"))
    assert previous["manifest_date"] == active["manifest_date"]
    assert active.get("revision", 1) > previous.get("revision", 1)


def test_the_chain_is_linked_and_every_link_is_still_loadable():
    """Runs recorded against any manifest must keep verifying, so all stay readable."""
    for older, newer in pairwise(MANIFEST_CHAIN):
        assert load_prices(older).currency == "USD"
        linked = json.loads(Path(newer).read_text(encoding="utf-8"))["supersedes"]
        assert linked == older, f"{newer} says it supersedes {linked}, not {older}"


def test_superseding_may_add_a_model_but_never_reprices_one_silently():
    """The invariant that made the in-place edit survivable, now enforced."""
    for older, newer in pairwise(MANIFEST_CHAIN):
        old = _models(older)
        new = _models(newer)
        assert set(old) <= set(new), (
            f"{newer} drops a model {older} carried; a run may still reference it"
        )
        repriced = {
            name: (old[name], new[name])
            for name in old
            if (old[name]["input_usd_per_mtok"], old[name]["output_usd_per_mtok"])
            != (new[name]["input_usd_per_mtok"], new[name]["output_usd_per_mtok"])
        }
        assert not repriced, (
            f"these models were repriced between {older} and {newer}: {sorted(repriced)}. "
            "That is allowed, but it invalidates cost comparisons between runs on either "
            "side of it, so it has to be a deliberate, recorded decision -- not a test fix."
        )


def test_the_superseded_manifest_is_still_loadable():
    """Runs recorded against it must keep verifying, so it stays readable."""
    assert load_prices(MANIFEST).manifest_date == "2026-09-19"
