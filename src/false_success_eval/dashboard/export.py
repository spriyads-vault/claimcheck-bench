"""Export one figure of the report: a PNG and a CSV, written under ``reports/``.

Rendered here rather than in the browser so an export is reproducible and needs
no screenshot library. Two things keep it honest:

* **The figure spec is the input.** The same ``figures`` block the page draws
  from is what these renderers read, and the CSV is the figure's own table.
  A PNG, the page and the table cannot show three different numbers.
* **The palette is read from ``tokens.css``.** Not copied into constants here,
  because a copy drifts. The light theme's values are parsed at import; a colour
  changed in the stylesheet changes the export with it, and the contrast and
  colour-vision tests that police that file police this module too.

Every figure carries the MCA notice in its footer: an exported image travels
further than the page it came from, so the restriction travels with it.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .aggregate import FOOTER_NOTICE

TOKENS_PATH = Path(__file__).resolve().parent / "static" / "tokens.css"


def _light_tokens() -> dict[str, str]:
    """The light theme's custom properties, parsed from the stylesheet."""
    text = TOKENS_PATH.read_text(encoding="utf-8")
    match = re.search(r":root\s*\{(.*?)\}", text, re.S)
    if not match:  # pragma: no cover - the file is ours and is tested
        raise RuntimeError(f"no :root block in {TOKENS_PATH}")
    return {
        name: value.strip()
        for name, value in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", match.group(1))
    }


_TOKENS = _light_tokens()


def token(name: str, fallback: str = "#000000") -> str:
    return _TOKENS.get(name, fallback)


PAGE = token("--bg", "#ffffff")
INK = token("--text", "#17171a")
MUTED = token("--muted", "#5b5b66")
GRID = token("--grid", "#ededea")
LINE = token("--line", "#e3e3e0")

#: Role to colour, exactly as charts.js maps it.
ROLE_COLOUR = {
    "trained": token("--series-trained"),
    "trained-alt": token("--series-trained-alt"),
    "typed": token("--series-typed"),
    "general": token("--series-general"),
    "heuristic": token("--series-heuristic"),
}

#: A view name to the figure it exports. ``detection`` and ``per-fault`` are the
#: same figure now: the report shows one per-fault view, as small multiples, and
#: the exhaustive label-by-fault matrix lives in ``reports/*/report.md`` where an
#: exhaustive table belongs.
VIEW_FIGURE = {
    "overview": "headline_auprc",
    "detection": "per_fault",
    "learning-curve": "learning_curve",
    "per-fault": "per_fault",
    "cost": "cost",
    "latency": "latency",
}

VIEWS = tuple(VIEW_FIGURE)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _finish(fig: Any, path: Path) -> None:
    """Lay out, then reserve a strip at the foot for the restriction notice."""
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    fig.text(0.5, 0.022, FOOTER_NOTICE, ha="center", va="bottom", fontsize=7, color=MUTED)
    fig.savefig(path, dpi=160, facecolor=PAGE)
    plt.close(fig)


def _style(ax: Any) -> None:
    ax.set_facecolor(PAGE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def _colour(series: dict[str, Any]) -> str:
    return ROLE_COLOUR.get(str(series.get("role")), token("--series-heuristic"))


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _title(ax: Any, figure: dict[str, Any]) -> None:
    ax.set_title(figure.get("title", ""), color=INK, fontsize=12, loc="left", pad=12)


def _placeholder(figure: dict[str, Any], message: str) -> Any:
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=PAGE)
    _style(ax)
    ax.axis("off")
    _title(ax, figure)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, wrap=True)
    return fig


def _interval_dots_figure(figure: dict[str, Any]) -> Any:
    """A point and its interval, one row per series, grouped as the page groups."""
    series = [s for s in figure.get("series", []) if s.get("points")]
    if not series:
        return _placeholder(figure, "No scored runs on this split yet")
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.62 * len(series) + 2.0)), facecolor=PAGE)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)

    labels: list[str] = []
    positions: list[float] = []
    y = float(len(series))
    last_group = None
    for entry in series:
        if entry.get("group") != last_group:
            last_group = entry.get("group")
            ax.text(
                0.0,
                y + 0.55,
                str(last_group).upper(),
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                color=MUTED,
                va="center",
            )
        point = entry["points"][0]
        value = _num(point.get("y"))
        lower, upper = _num(point.get("lower")), _num(point.get("upper"))
        if value is None:
            ax.text(
                0.02,
                y,
                entry.get("unavailable_reason") or "not available",
                transform=ax.get_yaxis_transform(),
                fontsize=8.5,
                color=MUTED,
                va="center",
            )
        else:
            if lower is not None and upper is not None and upper > lower:
                ax.plot([lower, upper], [y, y], color=_colour(entry), lw=2, alpha=0.55)
                ax.plot(
                    [lower, upper],
                    [y, y],
                    "|",
                    color=_colour(entry),
                    markersize=9,
                    markeredgewidth=2,
                )
            ax.plot([value], [y], "o", color=_colour(entry), markersize=8)
            ax.annotate(
                f"{value:.{figure.get('digits', 3)}f}",
                xy=(value, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK,
            )
        labels.append(entry["label"])
        positions.append(y)
        y -= 1.0

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(min(positions) - 0.8, max(positions) + 1.2)
    if figure.get("domain"):
        ax.set_xlim(figure["domain"][0], figure["domain"][1])
    ax.set_xlabel(figure.get("value_label", ""), color=MUTED, fontsize=9)
    _title(ax, figure)
    return fig


def _curve_band_figure(figure: dict[str, Any]) -> Any:
    series = [s for s in figure.get("series", []) if s.get("points")]
    if not series:
        return _placeholder(figure, figure.get("reason") or "No label sweep on this split yet")
    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor=PAGE)
    _style(ax)
    ax.grid(True, which="both", color=GRID, linewidth=0.7)

    for entry in series:
        xs = [p["x"] for p in entry["points"]]
        colour = _colour(entry)
        ax.fill_between(
            xs,
            [p["lower"] for p in entry["points"]],
            [p["upper"] for p in entry["points"]],
            color=colour,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            xs,
            [p["y"] for p in entry["points"]],
            marker="o",
            ms=5,
            lw=2,
            color=colour,
            label=entry["label"],
        )

    for reference in figure.get("references", []):
        value = _num(reference.get("value"))
        if value is None:
            continue
        ax.axhline(
            value,
            ls="--",
            lw=1.6,
            color=ROLE_COLOUR.get(str(reference.get("role")), MUTED),
            label=f"{reference['short']}, zero-shot ({value:.3f})",
        )

    for index, marker in enumerate(figure.get("markers", [])):
        ax.axvline(marker["x"], ls=":", lw=1, color=MUTED)
        ax.annotate(
            "\n".join([marker["label"], *marker.get("lines", [])]),
            xy=(marker["x"], 0.03 + 0.15 * index),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7.5,
            va="bottom",
            color=MUTED,
        )

    if figure.get("x_scale") == "log":
        ax.set_xscale("log")
    ax.set_xlabel(figure.get("x_label", ""), color=MUTED, fontsize=9)
    ax.set_ylabel(figure.get("value_label", ""), color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _title(ax, figure)
    return fig


def _small_multiples_figure(figure: dict[str, Any]) -> Any:
    panels = figure.get("panels", [])
    series = [s for s in figure.get("series", []) if s.get("points")]
    if not panels or not series:
        return _placeholder(figure, "No scored fault groups on this split yet")

    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 0.46 * len(series) * rows + 1.6 * rows + 1.0),
        facecolor=PAGE,
        squeeze=False,
    )
    for index, panel in enumerate(panels):
        ax = axes[index // columns][index % columns]
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        names: list[str] = []
        positions: list[float] = []
        y = float(len(series))
        for entry in series:
            point = next((p for p in entry["points"] if p.get("panel") == panel["key"]), None)
            value = _num((point or {}).get("y"))
            if point is not None and value is not None:
                ax.barh([y], [value], height=0.52, color=_colour(entry))
                lower, upper = _num(point.get("lower")), _num(point.get("upper"))
                if lower is not None and upper is not None:
                    ax.plot([lower, upper], [y, y], color=INK, lw=1.3, alpha=0.55)
            elif point is not None:
                ax.text(0.02, y, "not scored", fontsize=7.5, color=MUTED, va="center")
            names.append(entry["short"])
            positions.append(y)
            y -= 1.0
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8, color=INK)
        ax.set_ylim(min(positions) - 0.7, max(positions) + 0.7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title(
            f"{panel['label']}\n{panel['n']} false successes",
            color=INK,
            fontsize=9.5,
            loc="left",
            pad=8,
        )
    for spare in range(len(panels), rows * columns):
        axes[spare // columns][spare % columns].axis("off")
    fig.suptitle(figure.get("title", ""), color=INK, fontsize=12, x=0.01, ha="left")
    return fig


def _lines_figure(figure: dict[str, Any]) -> Any:
    series = [s for s in figure.get("series", []) if s.get("points")]
    if not series:
        return _placeholder(figure, "No latency arms recorded")
    if len(series[0]["points"]) < 2:
        # One arm is not a line. A line through a single point would imply a
        # trend that was never measured.
        return _interval_dots_figure(
            {
                **figure,
                "domain": None,
                "digits": 0,
                "series": [
                    {
                        **s,
                        "group": f"Concurrency {s['points'][0]['x']}",
                        "points": [{**s["points"][0], "lower": None, "upper": None}],
                    }
                    for s in series
                ],
            }
        )
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=PAGE)
    _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    for entry in series:
        # Same arm, same hue: the dash pattern carries the percentile, exactly
        # as it does on the page. Colour is never spent on a rank.
        dash = str(entry.get("dash") or "")
        ax.plot(
            [p["x"] for p in entry["points"]],
            [p["y"] for p in entry["points"]],
            marker="o",
            ms=5,
            lw=2,
            color=_colour(entry),
            linestyle=(0, tuple(float(v) for v in dash.split())) if dash else "-",
            label=entry["label"],
        )
    ax.set_xlabel(figure.get("x_label", ""), color=MUTED, fontsize=9)
    ax.set_ylabel(figure.get("value_label", ""), color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _title(ax, figure)
    return fig


RENDERERS = {
    "interval-dots": _interval_dots_figure,
    "curve-band": _curve_band_figure,
    "small-multiples": _small_multiples_figure,
    "lines": _lines_figure,
}


def figure_for(snapshot: dict[str, Any], view: str) -> dict[str, Any] | None:
    return (snapshot.get("figures") or {}).get(VIEW_FIGURE.get(view, "headline_auprc"))


def view_table(snapshot: dict[str, Any], view: str, scale: int = 0) -> list[dict[str, Any]]:
    """The rows behind the exported figure -- the figure's own table, verbatim.

    Not a second derivation of the same numbers. ``scale`` is accepted and
    ignored: it was a projection knob on the old cost panel, and the cost figure
    now reports a measured spend rather than a projection.
    """
    figure = figure_for(snapshot, view)
    if not figure:
        return []
    columns = figure["table"]["columns"]
    return [dict(zip(columns, row, strict=False)) for row in figure["table"]["rows"]]


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {FOOTER_NOTICE}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def export_view(
    *,
    snapshot: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    view: str,
    out_dir: Path,
    scale: int = 1_000_000,
) -> list[Path]:
    """Write ``<run>-<view>-<stamp>.png`` and ``.csv``. Local files, nothing published."""
    if view not in VIEW_FIGURE:
        view = "overview"
    run_id = str(snapshot.get("run", {}).get("run_id", "run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{run_id}-{view}-{_stamp()}"

    figure = figure_for(snapshot, view)
    if figure is None:
        rendered = _placeholder({"title": view}, "This figure is not available for this run")
    else:
        rendered = RENDERERS.get(str(figure.get("kind")), _interval_dots_figure)(figure)
    png_path = out_dir / f"{base}.png"
    _finish(rendered, png_path)

    table = view_table(snapshot, view, scale)
    csv_path = write_csv(table or [dict(r) for r in rows], out_dir / f"{base}.csv")
    return [png_path, csv_path]
