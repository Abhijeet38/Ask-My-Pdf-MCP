"""Orchestrate full ingestion of a single PDF: extract → chunk → embed → write."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..manifest import DocumentEntry, compute_doc_id, manifest
from ..config import settings
from ..store.client import OpenSearchStore
from .chunker import chunk_blocks
from .embed import Embedder
from .pdf_extract import extract_blocks, extract_pages

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    doc_name: str
    pages: int
    chunks: int
    elapsed_seconds: float
    skipped: bool = False
    error: str | None = None


def ingest_pdf(
    pdf_path: Path,
    *,
    store: OpenSearchStore,
    force: bool = False,
) -> IngestResult:
    """Ingest one PDF end-to-end.

    Returns an IngestResult; populates the manifest on success.
    Raises only on hard infrastructure errors (e.g. OpenSearch unreachable);
    parse-level errors are returned in result.error.
    """
    start = time.time()
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        return IngestResult(
            doc_id="",
            doc_name=pdf_path.name,
            pages=0,
            chunks=0,
            elapsed_seconds=0,
            error=f"file not found: {pdf_path}",
        )

    doc_id = compute_doc_id(pdf_path)
    doc_name = pdf_path.name

    # Register in manifest if not already there
    entry = manifest.get(doc_id)
    if entry is None:
        entry = DocumentEntry(
            doc_id=doc_id, name=doc_name, path=pdf_path, size_bytes=pdf_path.stat().st_size
        )
        manifest.register(entry)

    # Skip if already indexed and not forcing
    if entry.indexed and not force:
        return IngestResult(
            doc_id=doc_id,
            doc_name=doc_name,
            pages=entry.pages,
            chunks=entry.chunks,
            elapsed_seconds=0,
            skipped=True,
        )

    if force:
        store.delete_doc(doc_id)
        manifest.mark_unindexed(doc_id)

    try:
        log.info("Extracting %s", doc_name)
        blocks, total_pages = extract_blocks(pdf_path)

        log.info(
            "Chunking %s (%d blocks across %d pages)", doc_name, len(blocks), total_pages
        )
        chunks = chunk_blocks(
            blocks,
            doc_id=doc_id,
            doc_name=doc_name,
            chunk_tokens=settings.chunk_tokens,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            err = "no chunks produced from PDF (empty or unreadable)"
            return IngestResult(doc_id, doc_name, total_pages, 0, time.time() - start, error=err)

        log.info("Embedding %d chunks", len(chunks))
        embedder = Embedder.instance()
        vectors = embedder.embed_passages(c.text for c in chunks)

        log.info("Writing %d chunks to OpenSearch", len(chunks))
        store.bulk_index(chunks, vectors)

        # Also populate the pages index for accurate literal counting
        # (overlap-free, used by keyword_search). Cheap — no embeddings.
        page_texts = extract_pages(pdf_path)
        if page_texts:
            log.info("Writing %d page records to %s", len(page_texts), store.pages_index)
            store.bulk_index_pages(doc_id=doc_id, doc_name=doc_name, page_texts=page_texts)

        manifest.mark_indexed(doc_id, pages=total_pages, chunks=len(chunks))
        return IngestResult(
            doc_id=doc_id,
            doc_name=doc_name,
            pages=total_pages,
            chunks=len(chunks),
            elapsed_seconds=round(time.time() - start, 2),
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Ingestion failed for %s", doc_name)
        return IngestResult(
            doc_id=doc_id,
            doc_name=doc_name,
            pages=0,
            chunks=0,
            elapsed_seconds=round(time.time() - start, 2),
            error=str(e),
        )
