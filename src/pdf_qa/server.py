"""FastMCP server for the PDF Q&A toolkit.

Seven tools are registered:

  query_documents          - grounded Q&A with source attribution (the main one)
  keyword_search           - exact / regex search across the corpus
  count_occurrences        - exact term count by re-reading the PDF (rawdict)
  list_documents           - inventory of discovered + indexed PDFs
  ingest_document          - on-demand ingestion of a specific PDF
  ingest_all               - bulk ingestion of every discovered PDF
  get_document_metadata    - per-doc page count, word count, authors, title

Run with:
    python -m pdf_qa
or:
    pdf-qa-server   (when installed via `pip install -e .`)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import tools as _tools
from .config import settings
from .manifest import manifest
from .tools._state import AppState

log = logging.getLogger("pdf_qa.server")

mcp = FastMCP("pdf-qa")


# ---------------------------------------------------------------------------
# Tools — thin wrappers so MCP introspection sees clean signatures + docstrings.
# ---------------------------------------------------------------------------

@mcp.tool()
def query_documents(
    question: str,
    doc_filter: list[str] | None = None,
    top_k: int = 0,
) -> dict[str, Any]:
    """Answer a natural-language question grounded in the indexed PDFs.

    Returns a synthesized answer with inline citations of the form
    [doc_name p.N], plus a `sources` list of the chunks the model saw.

    Args:
        question: The natural-language question.
        doc_filter: Optional list of doc_ids or doc names to restrict search to.
        top_k: How many chunks to retrieve. 0 means use the default (5).
    """
    return _tools.query_documents(question, doc_filter=doc_filter, top_k=top_k)


@mcp.tool()
def keyword_search(
    term: str,
    doc_filter: list[str] | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    max_hits: int = 10,
) -> dict[str, Any]:
    """Search every indexed chunk for exact (or regex) matches of `term`.

    Use this for occurrence-count and 'on which page does X appear' questions
    where vector search is not appropriate. Returns total_occurrences (across
    the whole corpus / filtered docs) and up to max_hits page-tagged snippets.
    """
    return _tools.keyword_search(
        term,
        doc_filter=doc_filter,
        regex=regex,
        case_sensitive=case_sensitive,
        max_hits=max_hits,
    )


@mcp.tool()
def count_occurrences(
    doc_name: str,
    term: str,
    case_sensitive: bool = False,
    whole_word: bool = True,
) -> dict[str, Any]:
    """Count exact occurrences of a term in a document by re-reading the PDF.

    Uses per-glyph (rawdict) extraction for maximum accuracy — matches what
    Acrobat's Find reports. Use this when you need a precise count for a
    specific document (e.g., "how many times does X appear in doc Y?").

    Args:
        doc_name: Exact document name (as shown by list_documents).
        term: The term to count, e.g. 'WikiText-2'.
        case_sensitive: Match case exactly. Default False.
        whole_word: Require word boundaries — 'cat' won't match 'category'. Default True.
    """
    return _tools.count_occurrences(
        doc_name, term, case_sensitive=case_sensitive, whole_word=whole_word
    )


@mcp.tool()
def list_documents() -> dict[str, Any]:
    """List every PDF the server knows about, with indexed status."""
    return _tools.list_documents()


@mcp.tool()
def ingest_document(doc_id_or_name: str, force: bool = False) -> dict[str, Any]:
    """Ingest a discovered PDF on demand.

    Args:
        doc_id_or_name: A doc_id (from list_documents) or a file name. An
            absolute path also works.
        force: Re-index even if already indexed.
    """
    return _tools.ingest_document(doc_id_or_name, force=force)


@mcp.tool()
def ingest_all(force: bool = False) -> dict[str, Any]:
    """Ingest every discovered PDF that is not yet indexed."""
    return _tools.ingest_all(force=force)


@mcp.tool()
def get_document_metadata(doc_id_or_name: str) -> dict[str, Any]:
    """Return metadata (pages, chunks, size, indexed status) for one PDF.

    Includes: true page count, word count, title, authors, last_author,
    and a first-page excerpt. Use for questions like "how many pages",
    "how many words", "who is the last author", or "what is the title".
    """
    return _tools.get_document_metadata(doc_id_or_name)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    # Direct logs to stderr so they don't pollute the MCP stdio channel.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _bootstrap(eager: bool) -> None:
    state = AppState.instance()
    state.store.health()  # fail fast if OpenSearch is unreachable
    n_new, n_indexed = state.reconcile_manifest()
    log.info(
        "Manifest reconciled: discovered=%d, already_indexed=%d, total_known=%d",
        n_new,
        n_indexed,
        len(manifest),
    )
    log.info("LLM provider: %s", settings.llm_provider)
    log.info("Embedding model: %s", settings.embedding_model)
    log.info("OpenSearch index: %s @ %s:%d", settings.os_index, settings.os_host, settings.os_port)

    if eager:
        not_indexed = [e for e in manifest.all() if not e.indexed]
        if not_indexed:
            log.info("Eager mode: ingesting %d unindexed PDFs", len(not_indexed))
            _tools.ingest_all(force=False)
        else:
            log.info("Eager mode: nothing to ingest")


def run(eager: bool = True) -> None:
    """Entry point used by __main__ and the `pdf-qa-server` console script."""
    _setup_logging()
    _bootstrap(eager=eager)
    log.info("Starting MCP server on stdio")
    mcp.run()
