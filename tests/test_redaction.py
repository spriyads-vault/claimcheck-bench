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
