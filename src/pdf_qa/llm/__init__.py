"""LLM provider factory.

Two providers supported in this build:
  - bedrock  — Amazon Bedrock (Claude). Requires working AWS credentials.
  - ollama   — Fully local via Ollama. Requires `ollama serve` + a pulled model.

Optional auto-fallback: set `LLM_FALLBACK=ollama` and any failure during the
primary provider's `generate()` call will be retried on the fallback once.
This is the recommended way to use Ollama as a safety net for Bedrock when
AWS creds expire mid-session or the gamma cluster throttles.
"""

from __future__ import annotations

import logging

from ..config import settings
from .base import LLMClient

log = logging.getLogger(__name__)


_VALID = {"bedrock", "ollama"}


def _construct(provider: str) -> LLMClient:
    if provider == "bedrock":
        from .bedrock import BedrockClient
        return BedrockClient()
    if provider == "ollama":
        from .ollama import OllamaClient
        return OllamaClient()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Choose one of: {sorted(_VALID)}"
    )


class _FallbackClient:
    """Wraps a primary LLM client; on any exception in `generate`, retries
    once on the fallback. Logs the failure so it isn't silent.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}+fallback({fallback.name})"

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        try:
            return self._primary.generate(system=system, user=user, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Primary LLM (%s) failed: %s — falling back to %s",
                self._primary.name, e, self._fallback.name,
            )
            return self._fallback.generate(system=system, user=user, max_tokens=max_tokens)


def make_client() -> LLMClient:
    primary_name = (settings.llm_provider or "bedrock").lower()
    fallback_name = (settings.llm_fallback or "").lower()

    primary = _construct(primary_name)
    if not fallback_name or fallback_name == primary_name:
        return primary

    fallback = _construct(fallback_name)
    return _FallbackClient(primary, fallback)


__all__ = ["make_client", "LLMClient"]
