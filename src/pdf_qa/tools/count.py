"""count_occurrences — exact term count by re-reading the PDF with rawdict.

Backs questions like "how many times does X appear" where the keyword_search
tool (which queries the pages index) might miss edge cases or where the user
wants a direct PDF-level count without depending on index freshness.
"""

from __future__ import annotations

import re
from typing import Any

import fitz  # PyMuPDF

from ..ingest.pdf_extract import _rawdict_text
from ..manifest import manifest


def count_occurrences(
    doc_name: str,
    term: str,
    case_sensitive: bool = False,
    whole_word: bool = True,
) -> dict[str, Any]:
    """Count literal occurrences of a term in a document.

    Re-reads the PDF using per-glyph (rawdict) extraction for maximum
    accuracy — matches what Acrobat's Find would report.

    Args:
        doc_name: Exact document name (as shown by list_documents).
        term: The term to count, e.g. 'WikiText-2'.
        case_sensitive: Match case exactly. Default False.
        whole_word: Require word boundaries. Default True.

    Returns: dict with count, per-page breakdown, and metadata.
    """
    if not term:
        return {"error": "term is required"}

    e = manifest.get(doc_name) or manifest.get_by_name(doc_name)
    if e is None:
        return {"error": f"unknown doc: {doc_name!r}"}

    if not e.path.exists():
        return {"error": f"file not found: {e.path}"}

    # Build regex pattern
    pattern = re.escape(term)
    if whole_word:
        pattern = rf"\b{pattern}\b"
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(pattern, flags)

    # Extract text page-by-page using rawdict (glyph-level)
    doc = fitz.open(str(e.path))
    total = 0
    per_page: dict[int, int] = {}

    for page_index in range(doc.page_count):
        page = doc[page_index]
        text = _rawdict_text(page)
        count = len(compiled.findall(text))
        if count > 0:
            per_page[page_index + 1] = count
            total += count

    doc.close()

    return {
        "doc_name": e.name,
        "term": term,
        "case_sensitive": case_sensitive,
        "whole_word": whole_word,
        "count": total,
        "pages_with_matches": per_page,
    }
