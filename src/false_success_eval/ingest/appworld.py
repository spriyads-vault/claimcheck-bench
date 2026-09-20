"""Map AppWorld's publicly released experiment outputs into this harness's schema.

What is actually being ingested, stated plainly, because it is not quite what
the brief assumed. The paper *From Confident Closing to Silent Failure:
Characterizing False Success in LLM Agents* (arXiv:2606.09863, CC BY 4.0)
releases **no corpus of its own**: it says so itself -- "we use the publicly
released experiment outputs" of AppWorld. So the corpus ingested here is
AppWorld's own release, and the paper contributes the *rule*, which it documents
well enough to reimplement. The paper's reported figures are therefore external
context, never something this harness reproduces; see ``reports/`` and the
README for how they are labelled.

The source is ``experiment-outputs-0.1.3.bundle``: 8,190 trajectories over 4
agent architectures and 4 model families. Two of those architectures --
``legacy_full_code_agent`` and ``legacy_function_calling_agent`` -- are
*self-assessing*: they pass a structured ``status`` to the terminal
``complete_task`` call and sometimes write a failure. Only those two can express
an honest failure, so only those two can distinguish a false success from one,
and the ingest is restricted to them. That restriction is the paper's too.

Everything an evaluator sees is built from the trajectory log. Everything used
to label it is built from ``evaluations/<split>.json``, which AppWorld computed
by asserting against the app databases after the episode ran. The two never
touch: see :mod:`.labelling`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import (
    Event,
    EventType,
    ToolSchemaEntry,
    TraceRecord,
)
from .labelling import fault_for, label_for, normalise_claim

#: The published bundle. Public, unauthenticated, and checksummed on ingest.
BUNDLE_VERSION = "0.1.3"
BUNDLE_URL = (
    f"https://s3.us-west-2.amazonaws.com/appworld.dev/experiment-outputs-{BUNDLE_VERSION}.bundle"
)
BUNDLE_SHA256 = "e5ec6367d32b1883d28aaa25e4fe5026c89a08d5fb7a5742a83bfb65fe6bb2da"

#: The two self-assessing architectures. The other two never write a status, so
#: every one of their trajectories would be ``no_success_claim`` and none could
#: ever be a false success.
SELF_ASSESSING_ARCHITECTURES = ("legacy_full_code_agent", "legacy_function_calling_agent")

#: Apps that carry no task domain of their own.
_INFRASTRUCTURE_APPS = frozenset({"supervisor", "admin", "api_docs"})

_INTERACTION_SPLIT = re.compile(r"^### Environment Interaction \d+\s*$", re.M)
_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
_COMPLETE_TASK = re.compile(r"complete_task\(([^\n]*)")
_STATUS = re.compile(r"""['"]?status['"]?\s*[=:]\s*['"](\w+)['"]""")
_API_CALL = re.compile(r"apis\.([a-z_]+)\.([a-z_0-9]+)")
_TASK_INSERT = re.compile(r"INSERT INTO tasks \(([^)]*)\)")


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRow:
    """One AppWorld trajectory, before it becomes a :class:`TraceRecord`."""

    architecture: str
    model_family: str
    benchmark_split: str
    task_id: str
    scenario_id: str
    goal: str
    interactions: tuple[tuple[str, str], ...]
    api_calls: tuple[dict[str, Any], ...]
    raw_status: str | None
    ground_truth_success: bool | None
    difficulty: int | None
    num_tests: int
    n_failed_tests: int

    @property
    def trace_id(self) -> str:
        return (
            f"appworld/{self.architecture}/{self.model_family}/"
            f"{self.benchmark_split}/{self.task_id}"
        )


# ---------------------------------------------------------------------------
# reading the release
# ---------------------------------------------------------------------------


def _read_goal(task_dir: Path) -> str:
    """The task instruction, read out of the episode's own supervisor database.

    AppWorld seeds each episode's database with the task, so the goal travels
    with the trajectory and no second download is needed. The row is a raw
    ``INSERT INTO tasks`` statement with its values alongside.
    """
    db = task_dir / "dbs" / "supervisor.jsonl"
    if not db.exists():
        return ""
    with db.open(encoding="utf-8") as handle:
        for line in handle:
            if "INSERT INTO tasks" not in line:
                continue
            try:
                statement, values, *_ = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            match = _TASK_INSERT.search(statement)
            if not match:
                continue
            columns = [c.strip() for c in match.group(1).split(",")]
            if "instruction" not in columns:
                continue
            index = columns.index("instruction")
            if index < len(values) and isinstance(values[index], str):
                return str(values[index])
    return ""


def _read_interactions(task_dir: Path) -> tuple[tuple[str, str], ...]:
    """Parse ``environment_io.md`` into (code, output) pairs, in order."""
    log = task_dir / "logs" / "environment_io.md"
    if not log.exists():
        return ()
    text = log.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[str, str]] = []
    for chunk in _INTERACTION_SPLIT.split(text)[1:]:
        blocks = _FENCE.findall(chunk)
        code = blocks[0].strip() if blocks else ""
        output = blocks[1].strip() if len(blocks) > 1 else ""
        if not code and not output:
            continue
        out.append((code, output))
    return tuple(out)


def _read_api_calls(task_dir: Path) -> tuple[dict[str, Any], ...]:
    log = task_dir / "logs" / "api_calls.jsonl"
    if not log.exists():
        return ()
    rows: list[dict[str, Any]] = []
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                rows.append(loaded)
    return tuple(rows)


def _terminal_status(interactions: Sequence[tuple[str, str]]) -> str | None:
    """The structured status on the last ``complete_task`` call, if any.

    Read from the agent's emitted code, which is where the claim actually is.
    The last such call wins: an agent that retries and then completes has made
    its claim on the final call, not the abandoned one.
    """
    for code, _ in reversed(interactions):
        for call in reversed(_COMPLETE_TASK.findall(code)):
            match = _STATUS.search(call)
            if match:
                return match.group(1)
    return None


def _primary_domain(
    api_calls: Sequence[dict[str, Any]], interactions: Sequence[tuple[str, str]]
) -> str:
    """The app the task is actually about: the most-called non-infrastructure app."""
    counts: dict[str, int] = {}
    for call in api_calls:
        url = str(call.get("url", ""))
        parts = [p for p in url.split("/") if p]
        if not parts:
            continue
        app = parts[0]
        if app in _INFRASTRUCTURE_APPS:
            continue
        counts[app] = counts.get(app, 0) + 1
    if not counts:
        for code, _ in interactions:
            for app, _fn in _API_CALL.findall(code):
                if app not in _INFRASTRUCTURE_APPS:
                    counts[app] = counts.get(app, 0) + 1
    if not counts:
        return "supervisor"
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


def _tool_schema(interactions: Sequence[tuple[str, str]]) -> tuple[ToolSchemaEntry, ...]:
    """The AppWorld APIs this trajectory touched.

    ``required`` is left empty on purpose. The experiment outputs record the
    calls an agent made, not the signatures it could have called, and inventing
    a required-argument list from observed arguments would state something the
    source does not say.
    """
    names: list[str] = []
    seen: set[str] = set()
    for code, _ in interactions:
        for app, fn in _API_CALL.findall(code):
            name = f"{app}.{fn}"
            if name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(ToolSchemaEntry(name=name, required=()) for name in sorted(names))


def _events(row: SourceRow) -> tuple[Event, ...]:
    """The ordered trace an evaluator sees.

    Each AppWorld interaction becomes a call event carrying the agent's code and
    a result event carrying the environment's response verbatim. The final event
    is the agent's own last code cell, repeated as the assistant message,
    because that cell *is* the agent's closing output -- it is where the
    structured completion claim was made. No prose is synthesised for it: an
    AppWorld agent writes code, not a closing paragraph, and putting words in
    its mouth would invent the very thing being measured.
    """
    events: list[Event] = []
    seq = 0
    for code, output in row.interactions:
        events.append(
            Event(
                seq=seq,
                type=EventType.assistant_tool_call,
                tool="execute_python",
                arguments={"code": code},
            )
        )
        seq += 1
        events.append(
            Event(
                seq=seq,
                type=EventType.tool_result,
                tool="execute_python",
                status="returned",
                data={"output": output},
            )
        )
        seq += 1
    final_code = row.interactions[-1][0] if row.interactions else ""
    events.append(Event(seq=seq, type=EventType.assistant_message, text=final_code))
    return tuple(events)


def iter_source_rows(root: Path) -> Iterator[SourceRow]:
    """Walk an unpacked experiment-outputs directory, self-assessing runs only."""
    if not root.exists():
        raise IngestError(
            f"{root} does not exist. Point --source at an unpacked AppWorld "
            "experiment-outputs directory, or pass --download."
        )
    architectures = [root / name for name in SELF_ASSESSING_ARCHITECTURES]
    present = [d for d in architectures if d.is_dir()]
    if not present:
        raise IngestError(
            f"{root} contains none of the self-assessing architectures "
            f"{list(SELF_ASSESSING_ARCHITECTURES)}. Found: "
            f"{sorted(p.name for p in root.iterdir() if p.is_dir())}. This does not "
            "look like an unpacked AppWorld experiment-outputs directory."
        )
    for arch_dir in present:
        for provider_dir in sorted(p for p in arch_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in provider_dir.iterdir() if p.is_dir()):
                for split_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                    evaluations = split_dir / "evaluations" / f"{split_dir.name}.json"
                    tasks_dir = split_dir / "tasks"
                    if not evaluations.exists() or not tasks_dir.is_dir():
                        continue
                    truth = json.loads(evaluations.read_text(encoding="utf-8"))["individual"]
                    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
                        interactions = _read_interactions(task_dir)
                        if not interactions:
                            continue
                        record = truth.get(task_dir.name, {})
                        yield SourceRow(
                            architecture=arch_dir.name,
                            model_family=model_dir.name,
                            benchmark_split=split_dir.name,
                            task_id=task_dir.name,
                            scenario_id=task_dir.name.rsplit("_", 1)[0],
                            goal=_read_goal(task_dir),
                            interactions=interactions,
                            api_calls=_read_api_calls(task_dir),
                            raw_status=_terminal_status(interactions),
                            ground_truth_success=record.get("success"),
                            difficulty=record.get("difficulty"),
                            num_tests=int(record.get("num_tests", 0) or 0),
                            n_failed_tests=len(record.get("failures") or []),
                        )


# ---------------------------------------------------------------------------
# mapping into the schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestStats:
    """What the ingest kept, and what it dropped and why. Printed and recorded."""

    seen: int = 0
    kept: int = 0
    dropped_no_goal: int = 0
    dropped_no_ground_truth: int = 0

    def merged(self, **changes: int) -> IngestStats:
        current = {
            "seen": self.seen,
            "kept": self.kept,
            "dropped_no_goal": self.dropped_no_goal,
            "dropped_no_ground_truth": self.dropped_no_ground_truth,
        }
        for key, value in changes.items():
            current[key] += value
        return IngestStats(**current)

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped_no_goal": self.dropped_no_goal,
            "dropped_no_ground_truth": self.dropped_no_ground_truth,
        }


def to_record(row: SourceRow) -> TraceRecord:
    """Build one :class:`TraceRecord`. The label comes only from ground truth."""
    claim = normalise_claim(row.raw_status)
    label = label_for(claim, row.ground_truth_success)
    fault = fault_for(label, claim, row.difficulty)
    return TraceRecord(
        trace_id=row.trace_id,
        domain=_primary_domain(row.api_calls, row.interactions),
        # The AppWorld task id is <scenario>_<variation>: the three variations
        # of a scenario are paraphrases of one task. Splitting on the scenario
        # is therefore the real-data form of this harness's template-family
        # split, and it is what stops a paraphrase crossing into test.
        template_family=row.scenario_id,
        goal=row.goal,
        tool_schema=_tool_schema(row.interactions),
        events=_events(row),
        label=label,
        fault_type=fault,
        oracle={
            "source": "appworld",
            "architecture": row.architecture,
            "model_family": row.model_family,
            "benchmark_split": row.benchmark_split,
            "task_id": row.task_id,
            "raw_status": row.raw_status,
            "normalised_claim": claim,
            "ground_truth_success": row.ground_truth_success,
            "difficulty": row.difficulty,
            "num_tests": row.num_tests,
            "n_failed_tests": row.n_failed_tests,
            "n_interactions": len(row.interactions),
            "n_api_calls": len(row.api_calls),
        },
    )


def build_records(root: Path) -> tuple[tuple[TraceRecord, ...], IngestStats]:
    """Map every usable trajectory under ``root``. Drops are counted, not hidden."""
    records: list[TraceRecord] = []
    stats = IngestStats()
    for row in iter_source_rows(root):
        stats = stats.merged(seen=1)
        if not row.goal:
            stats = stats.merged(dropped_no_goal=1)
            continue
        claim = normalise_claim(row.raw_status)
        if claim == "success" and row.ground_truth_success is None:
            stats = stats.merged(dropped_no_ground_truth=1)
            continue
        records.append(to_record(row))
        stats = stats.merged(kept=1)
    records.sort(key=lambda r: r.trace_id)
    return tuple(records), stats


# ---------------------------------------------------------------------------
# obtaining the release
# ---------------------------------------------------------------------------


def download_and_unpack(destination: Path) -> Path:
    """Fetch and unpack the published bundle using AppWorld's own tooling.

    The bundle is encrypted, and AppWorld encrypts it deliberately -- to keep
    the benchmark out of scrapers and training sets. This harness therefore does
    not reimplement that decryption: it calls AppWorld's own published
    ``unpack_bundle``, so the only code that opens the bundle is the code its
    authors shipped for the purpose. If the package is not installed, the ingest
    says so and stops rather than working around it.
    """
    try:
        from appworld.common.constants import PASSWORD, SALT  # type: ignore[import-not-found]
        from appworld.common.utils import unpack_bundle  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only with --download
        raise IngestError(
            "--download needs the 'appworld' package, whose own tooling this harness "
            "uses to unpack the release rather than reimplementing its decryption.\n"
            "  pip install appworld\n"
            "Alternatively run `appworld download experiment-outputs` yourself and "
            "pass --source <that directory>."
        ) from exc

    import httpx

    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / f"experiment-outputs-{BUNDLE_VERSION}.bundle"
    if not bundle.exists():
        with httpx.stream("GET", BUNDLE_URL, timeout=900.0, follow_redirects=True) as response:
            response.raise_for_status()
            with bundle.open("wb") as handle:
                for chunk in response.iter_bytes(65536):
                    handle.write(chunk)

    from ..hashing import sha256_file

    digest = sha256_file(bundle)
    if digest != BUNDLE_SHA256:
        raise IngestError(
            f"downloaded bundle sha256 {digest} does not match the recorded "
            f"{BUNDLE_SHA256}. Refusing to ingest a release that is not the one "
            "this harness verified the licence of."
        )
    unpacked = destination / "outputs"
    if not unpacked.exists():
        unpacked.mkdir(parents=True, exist_ok=True)
        unpack_bundle(
            bundle_file_path=str(bundle),
            base_directory=str(unpacked),
            password=PASSWORD,
            salt=SALT,
        )
    return unpacked
