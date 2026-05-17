"""get_document_metadata — title, page count, indexed status, authors, etc."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from ..manifest import manifest


def get_document_metadata(doc_id_or_name: str) -> dict[str, Any]:
    """Return metadata (pages, chunks, size, indexed status) for one PDF."""
    if not doc_id_or_name:
        return {"error": "doc_id_or_name is required"}

    e = manifest.get(doc_id_or_name) or manifest.get_by_name(doc_id_or_name)
    if e is None:
        return {"error": f"unknown doc: {doc_id_or_name!r}"}

    result: dict[str, Any] = {
        "doc_id": e.doc_id,
        "name": e.name,
        "path": str(e.path),
        "size_bytes": e.size_bytes,
        "size_kb": round(e.size_bytes / 1024, 1),
        "pages": e.pages,
        "chunks": e.chunks,
        "indexed": e.indexed,
    }

    # Extract PDF-level metadata (title, authors, true page count) from the file.
    pdf_meta = _extract_pdf_metadata(e.path)
    result.update(pdf_meta)

    return result


def _extract_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """Extract title, authors, page count, word count, and first-page excerpt."""
    meta: dict[str, Any] = {}
    if not pdf_path.exists():
        return meta

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return meta

    # True page count from the PDF itself
    meta["pages"] = doc.page_count

    # Word count: extract all text and count whitespace-separated tokens
    full_text_parts: list[str] = []
    for page in doc:
        full_text_parts.append(page.get_text("text") or "")
    full_text = "\n".join(full_text_parts)
    words = full_text.split()
    meta["word_count"] = len(words)

    # PDF-level metadata
    pdf_info = doc.metadata or {}
    if pdf_info.get("title"):
        meta["title"] = pdf_info["title"]
    if pdf_info.get("author"):
        meta["pdf_author_field"] = pdf_info["author"]

    # First-page text extraction for author parsing
    if doc.page_count > 0:
        first_page_text = doc[0].get_text("text") or ""
        meta["first_page_excerpt"] = first_page_text[:1500]

        # Try to extract authors from first page
        authors = _parse_authors_from_first_page(first_page_text)
        if authors:
            meta["authors"] = authors
            meta["last_author"] = authors[-1]

    doc.close()
    return meta


# Patterns that indicate a line is NOT an author name
_NON_AUTHOR_KEYWORDS = [
    "university", "institute", "department", "inc.", "usa", "uk", "ca,",
    "proceedings", "conference", "abstract", "@", "http", "association",
    "computational", "pages", "arxiv", "seattle", "irvine", "allen",
    "intelligence", "supported", "funded",
]

# Superscript/marker characters commonly attached to author names
_MARKER_RE = re.compile(r"[∗†§‡¶\*]+")


def _parse_authors_from_first_page(text: str) -> list[str]:
    """Heuristic: extract author names from the first page of an academic paper.

    Strategy: find lines that look like person names (2-4 capitalized words,
    no affiliation keywords, no long sentences). Skip the title block and
    stop at abstract/introduction.
    """
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return []

    author_candidates: list[str] = []

    for line in lines[:30]:
        stripped = line.strip()
        if not stripped:
            continue

        # Stop at abstract/intro markers
        if re.match(r"^(Abstract|Introduction|1\s+Introduction|1\.|ABSTRACT)", stripped):
            break

        # Skip lines that look like conference headers, page numbers, or titles
        if re.match(r"^(Proceedings|In Proc|arXiv|\d{4,}$)", stripped):
            continue

        # Clean markers
        cleaned = _MARKER_RE.sub("", stripped).strip()
        # Remove trailing numbers (footnote refs)
        cleaned = re.sub(r"\s*\d+$", "", cleaned).strip()

        if not cleaned:
            continue

        # Skip if contains affiliation/non-author keywords
        lower = cleaned.lower()
        if any(kw in lower for kw in _NON_AUTHOR_KEYWORDS):
            continue

        # Author name heuristic: 2-5 words, mostly capitalized, looks like
        # "FirstName [Middle] LastName [Suffix]"
        words = cleaned.split()
        if not (2 <= len(words) <= 5):
            continue

        # Most words should start with uppercase
        cap_count = sum(1 for w in words if w[0:1].isupper())
        if cap_count < len(words) * 0.6:
            continue

        # Reject if it looks like a title (contains colons, or common title words)
        if ":" in cleaned or any(
            w.lower() in ("using", "for", "with", "from", "the", "and", "of", "a", "an", "in", "on", "to")
            for w in words
        ):
            continue

        # Reject if it's too long to be a name (>40 chars usually means a title)
        if len(cleaned) > 40:
            continue

        author_candidates.append(cleaned)

    # Deduplicate while preserving order
    seen: set[str] = set()
    authors: list[str] = []
    for a in author_candidates:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            authors.append(a)

    return authors
