"""Shared state used by every MCP tool.

Lazily initializes the OpenSearch store, the embedder, and the LLM client
so importing the tools module is fast and side-effect free.
"""

from __future__ import annotations

import logging
import threading

from ..config import settings
from ..ingest.embed import Embedder
from ..llm import LLMClient, make_client
from ..manifest import manifest
from ..store.client import OpenSearchStore

log = logging.getLogger(__name__)


class AppState:
    _instance: "AppState | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._store: OpenSearchStore | None = None
        self._embedder: Embedder | None = None
        self._llm: LLMClient | None = None

    @classmethod
    def instance(cls) -> "AppState":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ---- lazy accessors ---------------------------------------------------
    @property
    def store(self) -> OpenSearchStore:
        if self._store is None:
            self._store = OpenSearchStore(embedding_dim=settings.embedding_dim)
            self._store.ensure_index()
        return self._store

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder.instance()
        return self._embedder

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = make_client()
            log.info("LLM provider in use: %s", self._llm.name)
        return self._llm

    # ---- bootstrap --------------------------------------------------------
    def reconcile_manifest(self) -> tuple[int, int]:
        """Walk DATA_DIR + reconcile against OpenSearch.
        Returns (n_discovered, n_already_indexed).
        """
        new = manifest.discover(settings.data_dir)
        already = self.store.indexed_doc_ids()
        for entry in manifest.all():
            if entry.doc_id in already:
                count = self.store.count_for_doc(entry.doc_id)
                # we don't know page count without parsing; cheap fallback
                manifest.mark_indexed(entry.doc_id, pages=entry.pages, chunks=count)
        return len(new), len(already)
