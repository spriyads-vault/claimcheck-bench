"""Cost arithmetic. Every rate is read from the dated price manifest.

No price literal appears anywhere else in this package. A model whose entry is
marked ``verified: false`` has no usable rate, and its cost is recorded as
``None`` rather than guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import Usage

MTOK = 1_000_000


class PriceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    input_usd_per_mtok: float | None
    output_usd_per_mtok: float | None
    verified: bool
    source: str


@dataclass(frozen=True)
class PriceManifest:
    path: str
    manifest_date: str
    currency: str
    models: dict[str, ModelPrice]

    def get(self, model_id: str) -> ModelPrice:
        if model_id not in self.models:
            raise PriceError(
                f"no price entry for {model_id!r} in {self.path}. "
                "Add one with a citation rather than assuming a rate."
            )
        return self.models[model_id]


def load_prices(path: str | Path) -> PriceManifest:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = {
        name: ModelPrice(
            model_id=name,
            input_usd_per_mtok=entry.get("input_usd_per_mtok"),
            output_usd_per_mtok=entry.get("output_usd_per_mtok"),
            verified=bool(entry.get("verified", False)),
            source=str(entry.get("source", "")),
        )
        for name, entry in raw["models"].items()
    }
    return PriceManifest(
        path=str(path),
        manifest_date=str(raw["manifest_date"]),
        currency=str(raw["currency"]),
        models=models,
    )


def cost_usd(manifest: PriceManifest, model_id: str, usage: Usage | None) -> float | None:
    """Cost for one request, or None when usage is absent or the rate is unverified."""
    if usage is None:
        return None
    price = manifest.get(model_id)
    if not price.verified or price.input_usd_per_mtok is None or price.output_usd_per_mtok is None:
        return None
    return (
        usage.input_tokens * price.input_usd_per_mtok
        + usage.output_tokens * price.output_usd_per_mtok
    ) / MTOK
