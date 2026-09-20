"""Test configuration.

The default test run must never touch the network. That is enforced here rather
than trusted: sockets are blocked for the whole session, so a test that tries to
reach a real endpoint fails loudly instead of quietly making a live call.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

BLOCK_MESSAGE = (
    "Network access is blocked in the default test run. A live call belongs behind "
    "an explicit CLI command and an environment variable, never in pytest."
)


class BlockedNetworkError(RuntimeError):
    pass


class _BlockedSocket(socket.socket):
    def connect(self, *args: Any, **kwargs: Any) -> Any:
        raise BlockedNetworkError(BLOCK_MESSAGE)

    def connect_ex(self, *args: Any, **kwargs: Any) -> Any:
        raise BlockedNetworkError(BLOCK_MESSAGE)


@pytest.fixture(autouse=True, scope="session")
def block_network() -> Iterator[None]:
    original_socket = socket.socket
    original_create = socket.create_connection

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise BlockedNetworkError(BLOCK_MESSAGE)

    socket.socket = _BlockedSocket  # type: ignore[misc]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[misc]
        socket.create_connection = original_create  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let an operator's real key leak into a test artifact."""
    for name in ("TYPESAFE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
