from __future__ import annotations

import json

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
