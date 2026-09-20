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
        """The rate for a model, refusing rather than defaulting.

        Used where a missing entry is a mistake worth stopping for -- a pinned
        model whose price should already be on record.
        """
        if model_id not in self.models:
            raise PriceError(
                f"no price entry for {model_id!r} in {self.path}. "
                "Add one with a citation rather than assuming a rate."
            )
        return self.models[model_id]

    def find(self, model_id: str) -> ModelPrice | None:
        """The rate for a model, or None when the manifest has never seen it.

        Used where a missing entry is a *fact to record* rather than a mistake to
        stop for. A general-model arm chooses its model at run time from the
        account's own listing, so the id can legitimately be one no manifest
        mentions. Aborting a 700-request run on its first response because a
        rate is missing would lose the measurement and teach nobody anything;
        recording ``cost_usd: null`` alongside the real token counts loses
        nothing, because the bill can be reconstructed from a rate added later
        with a citation. Neither path ever invents a number.
        """
        return self.models.get(model_id)


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


def _apply(price: ModelPrice | None, usage: Usage | None) -> float | None:
    if usage is None or price is None:
        return None
    if not price.verified or price.input_usd_per_mtok is None or price.output_usd_per_mtok is None:
        return None
    return (
        usage.input_tokens * price.input_usd_per_mtok
        + usage.output_tokens * price.output_usd_per_mtok
    ) / MTOK


def cost_usd(manifest: PriceManifest, model_id: str, usage: Usage | None) -> float | None:
    """Cost for one request, or None when usage is absent or the rate is unverified.

    Refuses an unknown model outright. This is the strict path.
    """
    if usage is None:
        return None
    return _apply(manifest.get(model_id), usage)


def recorded_cost_usd(manifest: PriceManifest, model_id: str, usage: Usage | None) -> float | None:
    """Cost for one request as a *run artifact* writes it.

    Identical to :func:`cost_usd` except that a model the manifest has never
    heard of records ``None`` rather than raising. See
    :meth:`PriceManifest.find` for why the two paths differ: a general-model arm
    picks its model at run time, so an unknown id is a fact about the manifest
    rather than a bug in the run, and the honest record of it is a null cost
    beside the real token counts.

    ``verify-run`` recomputes through this same function, so a recorded cost and
    a recomputed one are produced by one code path and cannot drift. A run that
    recorded a number for a model with no rate is still reported as a problem.
    """
    return _apply(manifest.find(model_id), usage)


@dataclass(frozen=True)
class CostProjection:
    """A pre-spend estimate. A band, never a quote.

    Jev's tokenizer is not published, so the number of input tokens a request
    will be billed for cannot be known before it is sent. Everything here is
    bracketed by two chars-per-token assumptions read from ``eval.yaml`` and is
    reported as a range. Nothing in this class is ever recorded as a cost: a
    recorded cost always comes from :func:`cost_usd` and the tokens the API
    actually returned.
    """

    model_id: str
    prices_path: str
    manifest_date: str
    n_requests: int
    request_chars: int
    fixed_overhead_tokens: int
    chars_per_token_low: float
    chars_per_token_high: float
    input_tokens_low: int
    input_tokens_high: int
    usd_low: float | None
    usd_high: float | None
    input_usd_per_mtok: float | None
    #: Output is free on Jev's manifest and is not on a general model's, so the
    #: projection prices both and each lane's rate decides what that is worth.
    #: This is the schema-bounded ceiling on the answer, not a measurement.
    output_tokens_per_request: int = 0
    output_tokens_total: int = 0
    output_usd_per_mtok: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "prices_path": self.prices_path,
            "manifest_date": self.manifest_date,
            "n_requests": self.n_requests,
            "request_chars": self.request_chars,
            "fixed_overhead_tokens": self.fixed_overhead_tokens,
            "chars_per_token_low": self.chars_per_token_low,
            "chars_per_token_high": self.chars_per_token_high,
            "input_tokens_low": self.input_tokens_low,
            "input_tokens_high": self.input_tokens_high,
            "usd_low": self.usd_low,
            "usd_high": self.usd_high,
            "input_usd_per_mtok": self.input_usd_per_mtok,
            "output_tokens_per_request": self.output_tokens_per_request,
            "output_tokens_total": self.output_tokens_total,
            "output_usd_per_mtok": self.output_usd_per_mtok,
        }


def project_cost(
    manifest: PriceManifest,
    model_id: str,
    *,
    request_chars: int,
    n_requests: int,
    chars_per_token_low: float,
    chars_per_token_high: float,
    fixed_overhead_tokens: int,
    output_tokens_per_request: int = 0,
) -> CostProjection:
    """Bracket the cost of a run before any of it is sent.

    ``request_chars`` is the total compact-JSON length of every request body the
    run will send. A *low* chars-per-token figure means *more* tokens, so it
    produces the upper end of the band.

    ``output_tokens_per_request`` prices what the provider will *write*. It is
    zero for a lane that is billed on input alone, and the schema-bounded
    ceiling on the answer for one that is not; costing the ceiling means the
    projection errs high on the side the operator is being asked to approve.
    A lane whose output rate is unverified projects no cost at all rather than
    a partial one, because half a bill is not a smaller bill.
    """
    price = manifest.find(model_id)
    overhead = fixed_overhead_tokens * n_requests
    tokens_high = round(request_chars / chars_per_token_low) + overhead
    tokens_low = round(request_chars / chars_per_token_high) + overhead
    output_total = max(0, output_tokens_per_request) * n_requests

    rate = price.input_usd_per_mtok if price is not None and price.verified else None
    output_rate = price.output_usd_per_mtok if price is not None and price.verified else None
    # `is not None` rather than truthiness throughout: a rate of 0.0 is a
    # verified free lane, not a missing price, and the two must not collapse.
    usd_low: float | None = None
    usd_high: float | None = None
    if rate is not None and (output_total == 0 or output_rate is not None):
        output_usd = (output_total * output_rate / MTOK) if output_rate is not None else 0.0
        usd_low = tokens_low * rate / MTOK + output_usd
        usd_high = tokens_high * rate / MTOK + output_usd

    return CostProjection(
        model_id=model_id,
        prices_path=manifest.path,
        manifest_date=manifest.manifest_date,
        n_requests=n_requests,
        request_chars=request_chars,
        fixed_overhead_tokens=fixed_overhead_tokens,
        chars_per_token_low=chars_per_token_low,
        chars_per_token_high=chars_per_token_high,
        input_tokens_low=tokens_low,
        input_tokens_high=tokens_high,
        usd_low=usd_low,
        usd_high=usd_high,
        input_usd_per_mtok=rate,
        output_tokens_per_request=max(0, output_tokens_per_request),
        output_tokens_total=output_total,
        output_usd_per_mtok=output_rate,
    )
