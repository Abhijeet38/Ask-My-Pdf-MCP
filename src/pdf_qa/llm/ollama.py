"""Ollama provider (fully local; no API key required).

Requires Ollama to be running locally with the configured model pulled, e.g.
    ollama pull llama3.1:8b && ollama serve
"""

from __future__ import annotations

import httpx

from ..config import settings


class OllamaClient:
    name = "ollama"

    def __init__(self) -> None:
        self._url = settings.ollama_host.rstrip("/") + "/api/chat"
        self._model = settings.ollama_model
        self._client = httpx.Client(timeout=120.0)

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        body = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = self._client.post(self._url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
