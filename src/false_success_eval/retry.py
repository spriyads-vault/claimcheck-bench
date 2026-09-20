"""Capped exponential backoff with jitter, plus a concurrency-arm error guard.

Defaults mirror the documented TypeSafe SDK retry policy (backoff 0.5s doubling
to a 5.0s cap, 0.25 jitter fraction, ``Retry-After`` honoured). The delay
function takes an explicit RNG so retry accounting is testable without sleeping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 5.0
    backoff_jitter: float = 0.25
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 429, 500, 502, 503, 504, 529})
    )
    respect_retry_after: bool = True
    total_timeout_s: float = 30.0

    def is_retryable_status(self, status: int | None) -> bool:
        return status is not None and status in self.retry_statuses


def compute_delay(
    policy: RetryPolicy,
    attempt_number: int,
    retry_after_s: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Delay before the retry that follows ``attempt_number`` (1-indexed).

    ``Retry-After`` wins outright when present and honoured: the server's
    instruction is not something to jitter away.
    """
    if attempt_number < 1:
        raise ValueError("attempt_number is 1-indexed")
    if policy.respect_retry_after and retry_after_s is not None:
        return max(0.0, float(retry_after_s))
    base = float(min(policy.backoff_initial_s * (2 ** (attempt_number - 1)), policy.backoff_max_s))
    if policy.backoff_jitter <= 0:
        return base
    draw = float((rng or random).random())
    return base * (1.0 - policy.backoff_jitter * draw)


def parse_retry_after(headers: dict[str, str]) -> float | None:
    """Read ``retry-after-ms`` or ``Retry-After`` (seconds form) from headers."""
    lowered = {k.lower(): v for k, v in headers.items()}
    if "retry-after-ms" in lowered:
        try:
            return float(lowered["retry-after-ms"]) / 1000.0
        except ValueError:
            return None
    if "retry-after" in lowered:
        try:
            return float(lowered["retry-after"])
        except ValueError:
            return None
    return None


class ArmAbortedError(RuntimeError):
    """Raised when a concurrency arm exceeds its sustained error-rate budget."""


@dataclass
class ErrorRateGuard:
    """Abort a concurrency arm whose sustained error rate exceeds the budget."""

    max_error_rate: float = 0.05
    min_samples: int = 20
    total: int = 0
    errors: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    def record(self, *, failed: bool) -> None:
        self.total += 1
        if failed:
            self.errors += 1

    def check(self) -> None:
        if self.total >= self.min_samples and self.error_rate > self.max_error_rate:
            raise ArmAbortedError(
                f"error rate {self.error_rate:.1%} over {self.total} requests exceeds the "
                f"{self.max_error_rate:.1%} budget; aborting this concurrency arm"
            )
