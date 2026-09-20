"""Canonical JSON serialisation and SHA-256 helpers.

Every hash in this repository is taken over *canonical* JSON: sorted keys,
compact separators, UTF-8, no trailing whitespace. That makes the dataset and
every manifest byte-reproducible across machines and Python builds.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

CANONICAL_SEPARATORS = (",", ":")


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to the one JSON string this repo considers canonical."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=CANONICAL_SEPARATORS,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: str | Path) -> str | None:
    """Return the current commit SHA, or None when this is not a git checkout.

    Never raises: a missing git binary or an uninitialised repository is a
    recordable fact, not an error.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None
