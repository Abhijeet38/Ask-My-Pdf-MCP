"""Embedding model wrapper.

Loads a sentence-transformers model once per process. Auto-detects the best
device (mps on Apple Silicon, cuda if available, else cpu).

BGE models perform best when *queries* are prefixed; passages are embedded
plain. This wrapper handles that automatically via separate `embed_passages`
and `embed_query` methods.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np

from ..config import settings

log = logging.getLogger(__name__)


_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # local import keeps cold start fast for tools that don't embed
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        device = _resolve_device(settings.embedding_device)
        log.info("Loading embedding model %s on %s", settings.embedding_model, device)
        self._model = SentenceTransformer(settings.embedding_model, device=device)
        self._device = device
        self._is_bge = "bge" in settings.embedding_model.lower()

    @classmethod
    def instance(cls) -> "Embedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            items,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prompt = (_BGE_QUERY_PREFIX + text) if self._is_bge else text
        vec = self._model.encode(
            [prompt],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vec.astype(np.float32)
