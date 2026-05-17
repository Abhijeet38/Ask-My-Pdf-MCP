"""Protocol every LLM provider must implement."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """A minimal interface: take system + user text, return a completion string."""

    name: str

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str: ...
