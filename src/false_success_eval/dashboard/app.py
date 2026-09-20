"""The local dashboard server: FastAPI, server-sent events, no outbound calls.

Offline-first, like the rest of the harness. The application never opens a
socket of its own: it reads run directories from disk and pushes what it finds
to a page that loads every asset from this same process. There is no CDN, no
font fetch, no telemetry, and no code path that reads ``TYPESAFE_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..runner import (
    DatasetPaths,
    EvalConfig,
    available_corpora,
    load_config,
    read_progress,
    resolve_dataset_paths,
)
from ..schemas import Prediction
from .aggregate import (
    FOOTER_NOTICE,
    DashboardData,
    RunRef,
    public_row,
    read_predictions_from,
)
from .export import export_view

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: How often a live stream looks for new lines. Fast enough to feel immediate,
#: slow enough that tailing a run costs nothing measurable.
LIVE_POLL_S = 0.25

#: A full snapshot every this many predictions, so a page that missed an event
#: or joined late is corrected rather than left drifting.
SNAPSHOT_EVERY = 10

REPLAY_MIN_INTERVAL_S = 0.002
REPLAY_MAX_INTERVAL_S = 2.0


class ExportRequest(BaseModel):
    run: str | None = None
    dataset: str | None = None
    view: str = Field(default="overview")
    scale: int = Field(default=1_000_000, ge=1)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=float)}\n\n"


def _rows_for(
    data: DashboardData, ref: RunRef, predictions: Sequence[Prediction]
) -> list[dict[str, Any]]:
    by_id = {r.trace_id: r for r in data.split_records(ref.split)}
    return [public_row(p, by_id.get(p.trace_id)) for p in predictions]


def discover_datasets(config: EvalConfig) -> list[DatasetPaths]:
    """Every dataset this dashboard can switch between, real and synthetic.

    A dataset is listed only when its file is actually on disk, so the selector
    never offers something that would 404 when chosen. Real corpora come first:
    they are the ones the evaluation now rests on, and the synthetic set is
    kept as a controlled diagnostic rather than deleted.
    """
    found: list[DatasetPaths] = []
    for corpus in sorted(available_corpora(config)):
        paths = resolve_dataset_paths(config, None, corpus)
        if paths.dataset.exists():
            found.append(paths)
    section = config.section("dataset")
    pattern = str(section.get("output", "data/dataset-{records}.jsonl"))
    seen: set[str] = set()
    for candidate in sorted(
        Path(pattern.format(records="*")).parent.glob(Path(pattern.format(records="*")).name)
    ):
        stem = candidate.stem.rsplit("-", 1)[-1]
        if not stem.isdigit() or stem in seen:
            continue
        seen.add(stem)
        paths = resolve_dataset_paths(config, int(stem), None)
        if paths.dataset.exists():
            found.append(paths)
    return found


def _dataset_row(paths: DatasetPaths, selected: bool) -> dict[str, Any]:
    from ..ingest.pipeline import load_provenance

    provenance = load_provenance(paths.provenance)
    return {
        "id": paths.dataset_id,
        "kind": paths.kind,
        "records": provenance.n_records if provenance else paths.records,
        "runs_root": str(paths.runs_root),
        "selected": selected,
        "source_name": provenance.source_name if provenance else "",
        "licence": (provenance.licence.spdx if provenance and provenance.licence else ""),
    }


def create_app(
    *,
    config_path: str = "config/eval.yaml",
    records: int | None = None,
    corpus: str | None = None,
    runs_dir: str | None = None,
    default_run: str | None = None,
    live: bool = False,
    replay_speed: float = 20.0,
    reports_dir: str = "reports",
) -> FastAPI:
    """Build the dashboard application, able to switch between datasets.

    One process serves every dataset that exists on disk. Each is loaded lazily
    into its own :class:`DashboardData` and keyed by id, so switching between
    the synthetic diagnostic and a real corpus swaps the dataset, the runs root
    *and* the caveats together. They cannot come apart.
    """
    from ..generate import load_dataset

    config: EvalConfig = load_config(config_path)
    paths = resolve_dataset_paths(config, records, corpus)
    if not paths.dataset.exists():
        hint = f"Run `jev-eval ingest --corpus {corpus}` first." if corpus else "Generate it first."
        raise FileNotFoundError(
            f"{paths.dataset} not found. The dashboard scores runs against the dataset "
            f"they were run on. {hint}"
        )
    runs_root = Path(runs_dir) if runs_dir else paths.runs_root
    data = DashboardData(
        config=config,
        paths=paths,
        records=load_dataset(paths.dataset),
        runs_root=runs_root,
    )

    datasets: dict[str, DashboardData] = {paths.dataset_id: data}
    dataset_paths: dict[str, DatasetPaths] = {paths.dataset_id: paths}
    for candidate in discover_datasets(config):
        dataset_paths.setdefault(candidate.dataset_id, candidate)

    def dataset_for(dataset_id: str | None) -> DashboardData:
        """Resolve a dataset by id, loading it on first use."""
        if not dataset_id or dataset_id == paths.dataset_id:
            return data
        if dataset_id in datasets:
            return datasets[dataset_id]
        candidate = dataset_paths.get(dataset_id)
        if candidate is None or not candidate.dataset.exists():
            raise HTTPException(
                status_code=404,
                detail=(f"no dataset {dataset_id!r} on disk. Available: {sorted(dataset_paths)}."),
            )
        loaded = DashboardData(
            config=config,
            paths=candidate,
            records=load_dataset(candidate.dataset),
            runs_root=candidate.runs_root,
        )
        datasets[dataset_id] = loaded
        return loaded

    app = FastAPI(title="False-success evaluation dashboard", docs_url=None, redoc_url=None)
    app.state.data = data
    app.state.dataset_for = dataset_for
    app.state.dataset_paths = dataset_paths
    app.state.default_dataset = paths.dataset_id
    app.state.default_run = default_run
    app.state.default_mode = "live" if live else "replay"
    app.state.replay_speed = replay_speed
    app.state.reports_dir = Path(reports_dir)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def resolve(run: str | None, dataset: str | None = None) -> tuple[DashboardData, RunRef]:
        """Find a run within a dataset. Both are resolved together, never apart."""
        chosen = dataset_for(dataset)
        default = app.state.default_run if chosen is data else None
        ref = chosen.find_run(run or default)
        if ref is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no run found under {chosen.runs_root} for dataset "
                    f"{chosen.paths.dataset_id!r}. Run an evaluation against it first, "
                    "or point --run at a run directory."
                ),
            )
        return chosen, ref

    # -- pages ----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # -- api ------------------------------------------------------------
    def _datasets_payload(selected: str) -> list[dict[str, Any]]:
        rows = [
            _dataset_row(candidate, candidate.dataset_id == selected)
            for candidate in dataset_paths.values()
        ]
        # Real corpora first: they are what the evaluation rests on now.
        rows.sort(key=lambda row: (row["kind"] != "real", row["id"]))
        return rows

    @app.get("/api/meta")
    def meta(dataset: str | None = Query(default=None)) -> dict[str, Any]:
        """Everything the page needs before it picks a dataset and a run."""
        chosen = dataset_for(dataset)
        return {
            "footer": FOOTER_NOTICE,
            "records": chosen.paths.records,
            "runs_root": str(chosen.runs_root),
            "dataset": chosen.paths.dataset_id,
            "dataset_kind": chosen.paths.kind,
            "datasets": _datasets_payload(chosen.paths.dataset_id),
            "default_mode": app.state.default_mode,
            "default_replay_speed": app.state.replay_speed,
            "default_run": app.state.default_run if chosen is data else None,
            "fx_rate_gbp_usd": float(config.raw["fx_rate_gbp_usd"]),
            "fx_rate_date": str(config.raw["fx_rate_date"]),
            "fx_rate_source": str(config.raw.get("fx_rate_source", "")),
            "prices_path": chosen.prices.path,
            "prices_date": chosen.prices.manifest_date,
            "external_reference": config.raw.get("external_reference") or {},
            "runs": [r.as_dict() for r in chosen.runs()],
        }

    @app.get("/api/datasets")
    def datasets_endpoint() -> dict[str, Any]:
        return {"datasets": _datasets_payload(app.state.default_dataset)}

    @app.get("/api/runs")
    def runs(dataset: str | None = Query(default=None)) -> dict[str, Any]:
        chosen = dataset_for(dataset)
        return {"dataset": chosen.paths.dataset_id, "runs": [r.as_dict() for r in chosen.runs()]}

    @app.get("/api/snapshot")
    def snapshot(
        run: str | None = Query(default=None), dataset: str | None = Query(default=None)
    ) -> dict[str, Any]:
        chosen, ref = resolve(run, dataset)
        predictions, _ = read_predictions_from(ref.path / "predictions.jsonl")
        # A sync endpoint runs in the threadpool, so this cannot stall the
        # event loop, and it means /api/snapshot never answers with a pending
        # verdict -- the report and this endpoint agree on every field.
        chosen.ensure_comparison(ref)
        return chosen.snapshot(_refresh(ref), predictions)

    @app.get("/api/rows")
    def rows(
        run: str | None = Query(default=None), dataset: str | None = Query(default=None)
    ) -> dict[str, Any]:
        """The underlying per-prediction rows, as the CSV export writes them."""
        chosen, ref = resolve(run, dataset)
        predictions, _ = read_predictions_from(ref.path / "predictions.jsonl")
        return {"rows": _rows_for(chosen, ref, predictions)}

    @app.post("/api/export")
    def export(request: ExportRequest) -> JSONResponse:
        chosen, ref = resolve(request.run, request.dataset)
        predictions, _ = read_predictions_from(ref.path / "predictions.jsonl")
        chosen.ensure_comparison(ref)
        snap = chosen.snapshot(_refresh(ref), predictions)
        written = export_view(
            snapshot=snap,
            rows=_rows_for(chosen, ref, predictions),
            view=request.view,
            out_dir=Path(app.state.reports_dir) / "dashboard",
            scale=request.scale,
        )
        return JSONResponse({"written": [str(p) for p in written]})

    # -- stream ---------------------------------------------------------
    @app.get("/api/stream")
    async def stream(
        run: str | None = Query(default=None),
        dataset: str | None = Query(default=None),
        mode: str | None = Query(default=None),
        speed: float = Query(default=0.0, ge=0.0, le=1000.0),
    ) -> StreamingResponse:
        chosen, ref = resolve(run, dataset)
        chosen_mode = mode or app.state.default_mode
        chosen_speed = speed or float(app.state.replay_speed)
        generator = (
            _live_stream(chosen, ref, _refresh)
            if chosen_mode == "live"
            else _replay_stream(chosen, ref, chosen_speed)
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    def _refresh(ref: RunRef) -> RunRef:
        """Re-read the run header, so an in-flight run's status stays current."""
        from .aggregate import describe_run

        return describe_run(ref.path) or ref

    return app


async def _live_stream(data: DashboardData, ref: RunRef, refresh: Any) -> AsyncIterator[str]:
    """Tail an append-only predictions file and push each row as it lands."""
    path = ref.path / "predictions.jsonl"
    predictions: list[Prediction] = []
    offset = 0
    since_snapshot = 0
    by_id = {r.trace_id: r for r in data.split_records(ref.split)}

    current = refresh(ref)
    yield _sse("snapshot", {**data.snapshot(current, predictions), "mode": "live", "live": True})

    idle_polls = 0
    while True:
        fresh, offset = read_predictions_from(path, offset)
        current = refresh(ref)
        if fresh:
            idle_polls = 0
            for prediction in fresh:
                predictions.append(prediction)
                yield _sse("prediction", public_row(prediction, by_id.get(prediction.trace_id)))
                since_snapshot += 1
            if since_snapshot >= SNAPSHOT_EVERY:
                since_snapshot = 0
                yield _sse(
                    "snapshot",
                    {**data.snapshot(current, predictions), "mode": "live", "live": True},
                )
        else:
            idle_polls += 1
            if idle_polls % 16 == 0:
                # A heartbeat, so a proxy or a sleeping tab does not drop the
                # connection during a slow serial arm.
                yield ": keep-alive\n\n"

        progress = read_progress(ref.path)
        status = str(progress.get("status")) if progress else current.status
        if status in {"complete", "failed"} and not fresh:
            # The run has stopped, so the page is no longer live even though
            # this was a live stream. Say so on the way out, and give the
            # finished run the verdict it has earned.
            if status == "complete":
                await asyncio.to_thread(data.ensure_comparison, current)
            yield _sse(
                "snapshot",
                {**data.snapshot(current, predictions), "mode": "live", "live": False},
            )
            yield _sse("done", {"status": status, "n": len(predictions), "mode": "live"})
            return
        await asyncio.sleep(LIVE_POLL_S)


async def _replay_stream(data: DashboardData, ref: RunRef, speed: float) -> AsyncIterator[str]:
    """Replay a finished run at an adjustable speed. No API cost, same feel.

    Replay re-presents a run that has already finished, so the figures it
    shows are the finished run's figures from the first frame. Seeding the
    snapshot from the complete prediction file, rather than from an empty list
    that the ticker slowly refills, is what makes the page agree with
    ``jev-eval report`` on the same run -- and it is why replay no longer
    depends on the stream arriving at all. The ticker below is presentation.
    """
    all_predictions, _ = read_predictions_from(ref.path / "predictions.jsonl")
    by_id = {r.trace_id: r for r in data.split_records(ref.split)}

    mean_latency_ms = (
        sum(p.end_to_end_latency_ms for p in all_predictions) / len(all_predictions)
        if all_predictions
        else 0.0
    )
    arm = max(1, ref.concurrency)
    base_interval_s = (mean_latency_ms / 1000.0) / arm
    interval = min(
        REPLAY_MAX_INTERVAL_S,
        max(REPLAY_MIN_INTERVAL_S, base_interval_s / max(speed, 0.01)),
    )

    # Computed once. Every snapshot this stream emits is the same finished-run
    # snapshot, so the ticker costs nothing beyond the rows themselves.
    def _frame() -> dict[str, Any]:
        return {
            **data.snapshot(ref, all_predictions),
            "mode": "replay",
            "live": False,
            "replay": {"speed": speed, "interval_s": interval, "total": len(all_predictions)},
        }

    # Frame one carries the finished run's counts, which need no bootstrap and
    # are correct immediately. The verdict is the slow part, so it is computed
    # off the event loop and pushed in a second frame rather than holding the
    # first one back for half a minute.
    yield _sse("snapshot", _frame())
    await asyncio.to_thread(data.ensure_comparison, ref)
    complete = _frame()
    yield _sse("snapshot", complete)

    for prediction in all_predictions:
        yield _sse("prediction", public_row(prediction, by_id.get(prediction.trace_id)))
        await asyncio.sleep(interval)
    yield _sse("snapshot", complete)
    yield _sse("done", {"status": "replayed", "n": len(all_predictions), "mode": "replay"})
