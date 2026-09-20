"""The one instrumented HTTP path every paid adapter uses.

Extracted from ``jev_http`` when a second paid arm was added. Both adapters
have to record the same things -- every attempt, its status, its error class,
the delay before the retry and the wall time it took -- or the operational
columns of a comparison mean different things for each lane, and the comparison
stops being one.

The SDKs are still deliberately not used, for the reason they never were: a
client that retries internally hides exactly the attempts this harness exists to
count.
"""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

from ..retry import RetryPolicy, compute_delay, parse_retry_after
from ..schemas import Attempt


def post_with_retries(
    client: httpx.Client,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str],
    *,
    policy: RetryPolicy,
    rng: random.Random,
    sleep: Any,
) -> tuple[httpx.Response | None, list[Attempt], str | None]:
    """POST ``body``, retrying per ``policy``. Returns (response, attempts, error).

    Every attempt is recorded whether it succeeded, failed or was retried, so a
    run's ``attempts.jsonl`` is a complete account rather than a summary of the
    ones that happened to matter.
    """
    attempts: list[Attempt] = []
    last_error: str | None = None
    for attempt_number in range(1, policy.max_attempts + 1):
        start_ns = time.perf_counter_ns()
        status: int | None = None
        error_class: str | None = None
        response: httpx.Response | None = None
        try:
            response = client.post(path, json=body, headers=headers)
            status = response.status_code
        except httpx.HTTPError as exc:
            error_class = type(exc).__name__
            last_error = f"{error_class}: {exc}"
        end_ns = time.perf_counter_ns()

        retryable = error_class is not None or policy.is_retryable_status(status)
        is_last = attempt_number >= policy.max_attempts
        delay = 0.0
        if retryable and not is_last:
            retry_after = (
                parse_retry_after(dict(response.headers)) if response is not None else None
            )
            delay = compute_delay(policy, attempt_number, retry_after, rng)

        attempts.append(
            Attempt(
                attempt_number=attempt_number,
                http_status=status,
                error_class=error_class,
                retry_delay_seconds=delay,
                start_perf_ns=start_ns,
                end_perf_ns=end_ns,
                latency_ms=(end_ns - start_ns) / 1e6,
            )
        )

        if not retryable:
            if response is not None and status is not None and status >= 400:
                last_error = f"HTTP {status}"
            return response, attempts, last_error
        if is_last:
            if status is not None:
                last_error = f"HTTP {status} after {attempt_number} attempts"
            return response, attempts, last_error
        sleep(delay)
    return None, attempts, last_error
