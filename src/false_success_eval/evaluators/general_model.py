"""General-model baseline, env-guarded and refusing by default.

Anti-fabrication note. At build time (2026-09-20) no general-model endpoint,
path or model ID was verified against primary vendor documentation from this
machine. Rather than write a plausible-looking URL or model name into code, this
adapter reads all three from ``config/eval.yaml`` and refuses to run until a
human fills them in. An unconfigured or unkeyed provider records ``not_run``.

That is a deliberate choice: a baseline that silently called the wrong model
would be worse than no baseline at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..schemas import Decision, InferenceView, Prediction


class RefusedError(RuntimeError):
    """Raised when an adapter is asked to run something unverified."""


@dataclass(frozen=True)
class GeneralModelConfig:
    name: str
    enabled: bool
    base_url: str
    path: str
    model_id: str
    api_key_env: str
    refuse: bool = False
    reason: str = ""

    @classmethod
    def from_mapping(cls, name: str, raw: dict[str, Any]) -> GeneralModelConfig:
        return cls(
            name=name,
            enabled=bool(raw.get("enabled", False)),
            base_url=str(raw.get("base_url", "")),
            path=str(raw.get("path", "")),
            model_id=str(raw.get("model_id", "")),
            api_key_env=str(raw.get("api_key_env", "")),
            refuse=bool(raw.get("refuse", False)),
            reason=str(raw.get("reason", "")),
        )


def not_run_reason(config: GeneralModelConfig) -> str | None:
    """Why this adapter will not run, or None when it is ready."""
    if config.refuse:
        return config.reason or f"{config.name} is configured to refuse by design."
    if not config.enabled:
        return (
            f"{config.name} is disabled in config/eval.yaml. No endpoint or model ID was "
            "verified at build time; fill them in and set enabled: true to use it."
        )
    missing = [
        field
        for field, value in (
            ("base_url", config.base_url),
            ("path", config.path),
            ("model_id", config.model_id),
        )
        if not value
    ]
    if missing:
        return f"{config.name} is missing verified config: {', '.join(missing)}."
    if not config.api_key_env:
        return f"{config.name} has no api_key_env configured."
    if not os.environ.get(config.api_key_env):
        return f"{config.name} requires {config.api_key_env}, which is not set."
    return None


class GeneralModelEvaluator:
    """Records ``not_run`` unless explicitly configured and keyed."""

    provider = "general_model"

    def __init__(self, config: GeneralModelConfig, threshold: float = 0.5) -> None:
        self.config = config
        self.model_id = config.model_id or f"{config.name}:UNCONFIGURED"
        self.threshold = threshold
        self.blocked = not_run_reason(config)

    def predict(self, view: InferenceView, trace_id: str, repeat: int = 0) -> Prediction:
        if self.blocked is not None:
            return Prediction(
                trace_id=trace_id,
                provider=f"{self.provider}:{self.config.name}",
                model_id=self.model_id,
                repeat=repeat,
                threshold=self.threshold,
                decision=Decision.not_run,
                cost_usd=None,
                error=f"not_run: {self.blocked}",
            )
        raise RefusedError(
            f"{self.config.name} is marked ready in config, but no verified request shape for it "
            "exists in this repository. Implement it against primary vendor documentation and "
            "record the citation in preregistration.md before running it."
        )
