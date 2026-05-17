"""ingest_document and ingest_all — drive on-demand ingestion."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..ingest.pipeline import IngestResult, ingest_pdf
from ..manifest import manifest
from ._state import AppState


def _resolve_doc(doc_id_or_name: str) -> Path | None:
    """Find a discovered PDF by doc_id or by file name."""
    e = manifest.get(doc_id_or_name)
    if e is None:
        e = manifest.get_by_name(doc_id_or_name)
    return e.path if e else None


def ingest_document(doc_id_or_name: str, force: bool = False) -> dict[str, Any]:
    """Ingest a single discovered PDF.

    Args:
        doc_id_or_name: A doc_id from list_documents OR a PDF file name.
        force: Re-ingest even if already indexed.

    Returns: IngestResult as a dict.
    """
    state = AppState.instance()
    pdf_path = _resolve_doc(doc_id_or_name)
    if pdf_path is None:
        # also accept arbitrary path argument
        candidate = Path(doc_id_or_name).expanduser().resolve()
        if candidate.exists() and candidate.suffix.lower() == ".pdf":
            pdf_path = candidate
        else:
            return {
                "error": f"unknown doc_id or name: {doc_id_or_name!r}. "
                         "Check list_documents() for valid identifiers."
            }

    result: IngestResult = ingest_pdf(pdf_path, store=state.store, force=force)
    out = asdict(result)
    return out


def ingest_all(force: bool = False) -> dict[str, Any]:
    """Ingest every discovered, not-yet-indexed PDF (or every PDF if force=True)."""
    state = AppState.instance()
    results: list[dict[str, Any]] = []
    indexed = skipped = errors = 0

    for entry in manifest.all():
        if entry.indexed and not force:
            skipped += 1
            continue
        r = ingest_pdf(entry.path, store=state.store, force=force)
        results.append(asdict(r))
        if r.error:
            errors += 1
        else:
            indexed += 1

    return {
        "indexed": indexed,
        "skipped": skipped,
        "errors": errors,
        "results": results,
    }
