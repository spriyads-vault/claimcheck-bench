from __future__ import annotations

from false_success_eval.redact import REDACTED, contains_secret, redact, redact_text


def test_authorization_header_is_replaced_wholesale():
    payload = {"headers": {"Authorization": "Bearer ts-live-abcdefghijklmnop12345678"}}
    assert redact(payload)["headers"]["Authorization"] == REDACTED


def test_header_name_matching_ignores_case_and_spacing():
    for name in ("authorization", "AUTHORIZATION", " Authorization ", "x-api-key", "API_KEY"):
        assert redact({name: "supersecretvalue123456789"})[name] == REDACTED


def test_bearer_token_inside_free_text_is_redacted():
    assert "abcdefghijk" not in redact_text("sent Bearer abcdefghijklmnopqrstuvwxyz123456")


def test_vendor_prefixed_keys_are_redacted():
    for token in (
        "sk-abcdefghijklmnop123456",
        "AIzaSyABCDEFGHIJKLMNOPQRST",
        "ghp_abcdefghijklmnop12",
    ):
        assert token not in redact_text(f"key is {token} ok")


def test_live_environment_values_are_redacted_anywhere(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "hunter2-hunter2-hunter2")
    payload = {
        "body": {"note": "the key hunter2-hunter2-hunter2 leaked"},
        "list": ["hunter2-hunter2-hunter2"],
    }
    cleaned = redact(payload)
    assert "hunter2" not in str(cleaned)
    assert contains_secret(payload) is True
    assert contains_secret(cleaned) is False


def test_a_secret_that_looks_like_a_digest_is_still_redacted(monkeypatch):
    digest_shaped = "a" * 64
    monkeypatch.setenv("TYPESAFE_API_KEY", digest_shaped)
    assert redact_text(digest_shaped) == REDACTED


def test_sha256_digests_survive_so_the_audit_trail_survives():
    digest = "3b" + "0" * 62
    assert redact_text(digest) == digest
    assert redact({"dataset_sha256": digest})["dataset_sha256"] == digest


def test_ordinary_trace_content_is_untouched():
    payload = {
        "goal": "Please reschedule event CAL-4821 to 2026-10-09 11:00 UTC",
        "status": "ok",
        "matched": 0,
    }
    assert redact(payload) == payload


def test_redaction_recurses_through_lists_and_nesting():
    payload = {"a": [{"b": {"Authorization": "Bearer xyzxyzxyzxyzxyzxyzxyz"}}]}
    assert redact(payload)["a"][0]["b"]["Authorization"] == REDACTED


def test_contains_secret_is_false_without_live_keys():
    assert contains_secret({"anything": "at all"}) is False


# ---------------------------------------------------------------------------
# structural identifiers must survive redaction intact
# ---------------------------------------------------------------------------


def test_a_long_model_name_in_a_trace_id_is_not_redacted():
    """The regression that dropped 117 of 702 real traces from every metric.

    ``deepseek-coder-33b-instruct_together`` is 36 characters of
    ``[A-Za-z0-9_-]``, which the long-opaque-token heuristic matched. Rewriting
    a trace_id breaks the join to the dataset, and the trace then vanishes from
    the scored set with no error anywhere.
    """
    trace_id = (
        "appworld/legacy_full_code_agent/deepseek-coder-33b-instruct_together/"
        "test_challenge/4441ee9_2"
    )
    row = redact({"trace_id": trace_id, "provider": "jev"})
    assert row["trace_id"] == trace_id


def test_a_git_commit_survives_redaction():
    """40 hex characters is long and opaque, and it is also the audit trail."""
    commit = "595ee32b7cb3955077ac70193d960f30b80a244f"
    assert redact({"git_commit": commit})["git_commit"] == commit


def test_structural_exemption_still_scrubs_a_live_secret(monkeypatch):
    """Exempt from the heuristics, never exempt from the exact-value scrub."""
    secret = "ts-live-abcdefghijklmnopqrstuvwxyz0123456789"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    row = redact({"trace_id": f"trace/{secret}/1"})
    assert secret not in row["trace_id"]
    assert "[REDACTED]" in row["trace_id"]


def test_a_non_structural_long_token_is_still_redacted():
    """The heuristic is narrowed, not switched off."""
    row = redact({"note": "abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"})
    assert row["note"] == "[REDACTED]"


def test_a_url_is_not_treated_as_structural():
    """A URL can carry a key as a query parameter, so it keeps the heuristics."""
    row = redact({"url": "https://api.example.com/v1?api_key=abcdefghijklmnopqrstuvwxyz012345"})
    assert "abcdefghijklmnopqrstuvwxyz012345" not in row["url"]
