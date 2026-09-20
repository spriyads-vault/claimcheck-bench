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


def redact(obj: Any) -> Any:
    """Recursively redact secrets from any JSON-shaped object."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and key.strip().lower() in SENSITIVE_KEY_NAMES:
                out[key] = REDACTED
            else:
                out[key] = redact(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(item) for item in obj]
    return obj


def contains_secret(obj: Any) -> bool:
    """True when a live environment secret survives anywhere inside ``obj``."""
    secrets = _live_secret_values()
    if not secrets:
        return False
    from .hashing import canonical_json

    blob = canonical_json(obj) if not isinstance(obj, str) else obj
    return any(secret in blob for secret in secrets)
