"""Thin wrapper around opensearchpy.OpenSearch.

Owns lifecycle of both the chunks index (vector + BM25, used for retrieval)
and the pages index (raw page text, used for overlap-free literal counting).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from ..config import settings
from ..ingest.chunker import Chunk
from .schema import index_body, pages_index_body

log = logging.getLogger(__name__)


class OpenSearchStore:
    def __init__(self, *, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.index = settings.os_index
        self.pages_index = settings.os_pages_index
        self.client = OpenSearch(**settings.opensearch_kwargs())

    # ---- lifecycle ---------------------------------------------------------
    def ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index):
            log.info("Creating OpenSearch index %s (dim=%d)", self.index, self.embedding_dim)
            self.client.indices.create(index=self.index, body=index_body(self.embedding_dim))
        if not self.client.indices.exists(index=self.pages_index):
            log.info("Creating OpenSearch index %s (pages)", self.pages_index)
            self.client.indices.create(index=self.pages_index, body=pages_index_body())

    def reset_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            log.info("Dropping OpenSearch index %s", self.index)
            self.client.indices.delete(index=self.index)
        if self.client.indices.exists(index=self.pages_index):
            log.info("Dropping OpenSearch index %s", self.pages_index)
            self.client.indices.delete(index=self.pages_index)
        self.ensure_index()

    def health(self) -> dict:
        return self.client.cluster.health(wait_for_status="yellow", timeout=10)

    # ---- writes (chunks) ---------------------------------------------------
    def bulk_index(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        actions = ({
            "_index": self.index,
            "_id": f"{c.doc_id}:{c.chunk_index}",
            "_source": {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "kind": c.kind,
                "text": c.text,
                "embedding": vectors[i].tolist(),
            },
        } for i, c in enumerate(chunks))
        bulk(self.client, actions, request_timeout=120, refresh=True)

    # ---- writes (pages) ----------------------------------------------------
    def bulk_index_pages(
        self, *, doc_id: str, doc_name: str, page_texts: dict[int, str]
    ) -> int:
        """Write one record per page into the pages index.
        Idempotent — uses doc_id:page as the primary key.
        Returns the number of pages written.
        """
        if not page_texts:
            return 0
        actions = ({
            "_index": self.pages_index,
            "_id": f"{doc_id}:{page}",
            "_source": {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page,
                "text": text,
            },
        } for page, text in page_texts.items())
        success, _ = bulk(self.client, actions, request_timeout=120, refresh=True)
        return success

    def delete_doc(self, doc_id: str) -> int:
        """Delete every chunk AND every page record for a doc_id.
        Returns total deleted count across both indices.
        """
        total = 0
        for idx in (self.index, self.pages_index):
            if not self.client.indices.exists(index=idx):
                continue
            resp = self.client.delete_by_query(
                index=idx,
                body={"query": {"term": {"doc_id": doc_id}}},
                refresh=True,
            )
            total += resp.get("deleted", 0)
        return total

    # ---- introspection -----------------------------------------------------
    def count_for_doc(self, doc_id: str) -> int:
        if not self.client.indices.exists(index=self.index):
            return 0
        return self.client.count(
            index=self.index,
            body={"query": {"term": {"doc_id": doc_id}}},
        )["count"]

    def indexed_doc_ids(self) -> set[str]:
        """Return the set of doc_ids that have at least one chunk in the index.
        Used at startup to reconcile the manifest with what's already stored.
        """
        if not self.client.indices.exists(index=self.index):
            return set()
        body = {
            "size": 0,
            "aggs": {"docs": {"terms": {"field": "doc_id", "size": 10_000}}},
        }
        resp = self.client.search(index=self.index, body=body)
        return {b["key"] for b in resp["aggregations"]["docs"]["buckets"]}

    def pages_indexed_doc_ids(self) -> set[str]:
        """Return the set of doc_ids that have records in the pages index.
        Used by migration tooling to identify which docs still need their
        pages backfilled (e.g. after upgrading from chunks-only to dual index).
        """
        if not self.client.indices.exists(index=self.pages_index):
            return set()
        body = {
            "size": 0,
            "aggs": {"docs": {"terms": {"field": "doc_id", "size": 10_000}}},
        }
        resp = self.client.search(index=self.pages_index, body=body)
        return {b["key"] for b in resp["aggregations"]["docs"]["buckets"]}
