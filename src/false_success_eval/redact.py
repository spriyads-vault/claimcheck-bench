"""Secret redaction for every artifact this harness writes to disk.

Three layers, applied in this order:

1. **Header names.** Any mapping key that names an auth header is replaced
   wholesale, regardless of its value.
2. **Live environment values.** The exact value of any secret-bearing
   environment variable is replaced wherever it appears. This runs before the
   digest exemption, so a key that happens to look like a hash is still caught.
3. **Key-shaped strings.** Known vendor prefixes, ``Bearer <token>``, and long
   opaque tokens.

A 64-character lowercase hex string is exempt from layer 3 because this
repository writes SHA-256 digests into the same artifacts, and redacting those
would destroy the audit trail. Layer 2 still applies to it.

**Structural identifiers are exempt from layers 1 and 3 too.** A ``trace_id`` is
a join key, not a value: if it is rewritten, the prediction can no longer be
matched to the record it was made about, and the trace silently vanishes from
every metric. That is exactly what happened when real trace ids arrived --
``deepseek-coder-33b-instruct_together`` is 36 characters of ``[A-Za-z0-9_-]``
and layer 3's long-opaque-token rule swallowed it whole, dropping 117 of 702
traces out of the scored set without a word. Silent data loss is a worse
failure than a conservative redaction, so these keys get layer 2 -- the exact
live-secret scrub -- and nothing heuristic.

The safety net is unchanged: :func:`contains_secret` runs over the whole row
before it is written and refuses the run if a live secret survives *anywhere*,
exempt keys included.
"""

from __future__ import annotations

import os
import re
from typing import Any

REDACTED = "[REDACTED]"

SECRET_ENV_VARS = ("TYPESAFE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")

SENSITIVE_KEY_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "api_key",
        "access_token",
        "secret",
        "password",
        "token",
        "bearer",
    }
)

#: Keys whose values are structural identifiers this harness generates itself.
#: They must round-trip byte-exact or the artifact stops joining to the dataset.
#: Layer 2 still scrubs a live secret out of them; only the heuristics are off.
STRUCTURAL_KEY_NAMES = frozenset(
    {
        "trace_id",
        "run_id",
        "model_id",
        "provider",
        "split",
        "template_family",
        "domain",
        "dataset_path",
        "predictions_path",
        "attempts_path",
        "truncation_path",
        "prices_path",
        "git_commit",
    }
)

_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]+")
_VENDOR_PREFIXED = re.compile(r"\b(?:sk|pk|rk|ts|AIza|ghp|gho|xox[abps])[-_A-Za-z0-9]{12,}\b")
_LONG_OPAQUE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")
_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")


def _live_secret_values() -> tuple[str, ...]:
    values = []
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            values.append(value)
    return tuple(values)


def _scrub_env_values(text: str) -> str:
    for secret in _live_secret_values():
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


def redact_text(text: str) -> str:
    """Redact secrets from a single string."""
    text = _scrub_env_values(text)
    text = _BEARER.sub(f"Bearer {REDACTED}", text)
    text = _VENDOR_PREFIXED.sub(REDACTED, text)
    if _SHA256_HEX.match(text):
        # A bare SHA-256 digest. Already cleared by the env-value pass above.
        return text
    return _LONG_OPAQUE.sub(
        lambda m: m.group(0) if _SHA256_HEX.match(m.group(0)) else REDACTED, text
    )


def redact(obj: Any, _structural: bool = False) -> Any:
    """Recursively redact secrets from any JSON-shaped object.

    ``_structural`` is set for the value of a key in
    :data:`STRUCTURAL_KEY_NAMES`, where only the exact live-secret scrub runs.
    """
    if isinstance(obj, str):
        return _scrub_env_values(obj) if _structural else redact_text(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            name = key.strip().lower() if isinstance(key, str) else ""
            if name in SENSITIVE_KEY_NAMES:
                out[key] = REDACTED
            else:
                out[key] = redact(value, _structural or name in STRUCTURAL_KEY_NAMES)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(item, _structural) for item in obj]
    return obj


def contains_secret(obj: Any) -> bool:
    """True when a live environment secret survives anywhere inside ``obj``."""
    secrets = _live_secret_values()
    if not secrets:
        return False
    from .hashing import canonical_json

    blob = canonical_json(obj) if not isinstance(obj, str) else obj
    return any(secret in blob for secret in secrets)
