from __future__ import annotations

import random

import pytest

from false_success_eval.retry import (
    ArmAbortedError,
    ErrorRateGuard,
    RetryPolicy,
    compute_delay,
    parse_retry_after,
)


def test_backoff_doubles_and_caps():
    policy = RetryPolicy(backoff_initial_s=0.5, backoff_max_s=5.0, backoff_jitter=0.0)
    delays = [compute_delay(policy, n) for n in range(1, 7)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0]


def test_jitter_only_subtracts_and_stays_in_bounds():
    policy = RetryPolicy(backoff_initial_s=1.0, backoff_max_s=5.0, backoff_jitter=0.25)
    rng = random.Random(7)
    for _ in range(200):
        delay = compute_delay(policy, 2, rng=rng)
        assert 2.0 * 0.75 <= delay <= 2.0


def test_delay_is_deterministic_for_a_seeded_rng():
    policy = RetryPolicy()
    a = [compute_delay(policy, n, rng=random.Random(42)) for n in range(1, 4)]
    b = [compute_delay(policy, n, rng=random.Random(42)) for n in range(1, 4)]
    assert a == b


def test_retry_after_wins_outright():
    policy = RetryPolicy(backoff_initial_s=0.5, backoff_jitter=0.25)
    assert compute_delay(policy, 1, retry_after_s=12.0, rng=random.Random(1)) == 12.0


def test_retry_after_is_ignored_when_not_honoured():
    policy = RetryPolicy(respect_retry_after=False, backoff_jitter=0.0)
    assert compute_delay(policy, 1, retry_after_s=12.0) == 0.5


def test_attempt_number_is_one_indexed():
    with pytest.raises(ValueError, match="1-indexed"):
        compute_delay(RetryPolicy(), 0)


def test_parse_retry_after_prefers_milliseconds():
    assert parse_retry_after({"retry-after-ms": "1500", "Retry-After": "9"}) == 1.5
    assert parse_retry_after({"Retry-After": "3"}) == 3.0
    assert parse_retry_after({"retry-after": "not-a-number"}) is None
    assert parse_retry_after({}) is None


def test_retryable_statuses():
    policy = RetryPolicy()
    for status in (408, 429, 500, 502, 503, 504, 529):
        assert policy.is_retryable_status(status)
    for status in (200, 400, 401, 404, 422):
        assert not policy.is_retryable_status(status)
    assert not policy.is_retryable_status(None)


def test_error_rate_guard_aborts_only_past_the_budget_and_sample_floor():
    guard = ErrorRateGuard(max_error_rate=0.05, min_samples=20)
    for _ in range(19):
        guard.record(failed=True)
    guard.check()  # below the sample floor, no abort
    guard.record(failed=True)
    with pytest.raises(ArmAbortedError, match="exceeds"):
        guard.check()


def test_error_rate_guard_tolerates_a_rate_inside_the_budget():
    guard = ErrorRateGuard(max_error_rate=0.05, min_samples=20)
    for i in range(100):
        guard.record(failed=(i < 5))
    assert guard.error_rate == 0.05
    guard.check()
