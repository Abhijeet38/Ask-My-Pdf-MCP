"""keyword_search — literal/regex search backing 'how many times does X
appear' and 'on which page does X start' question shapes.
"""

from __future__ import annotations

from typing import Any

from ..store.search import keyword_search as _keyword_search
from ._state import AppState


def keyword_search(
    term: str,
    doc_filter: list[str] | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    max_hits: int = 10,
) -> dict[str, Any]:
    """Search every indexed chunk for exact (or regex) matches of `term`.

    Args:
        term: The literal phrase or regex pattern.
        doc_filter: Optional list of doc_ids or doc_names.
        regex: Treat `term` as a regular expression (OpenSearch regexp + Python re).
        case_sensitive: Default False.
        max_hits: Maximum sample hits to return; total_occurrences counts across
            ALL chunks regardless of this cap.

    Returns: dict with total_occurrences, sample hits with page+snippet, and
        the set of distinct documents that matched.
    """
    if not term:
        return {"term": "", "total_occurrences": 0, "hits": [], "docs_matched": []}

    store = AppState.instance().store
    sample_hits, total = _keyword_search(
        store,
        term=term,
        doc_filter=doc_filter or None,
        regex=regex,
        case_sensitive=case_sensitive,
        max_hits=max_hits,
    )

    docs_matched = sorted({h.doc_name for h in sample_hits})
    return {
        "term": term,
        "regex": regex,
        "case_sensitive": case_sensitive,
        "total_occurrences": total,
        "matching_chunks": len(sample_hits),
        "docs_matched": docs_matched,
        "hits": [
            {
                "doc_id": h.doc_id,
                "doc_name": h.doc_name,
                "page": h.page,
                "score": h.score,
                "snippet": h.snippet,
            }
            for h in sample_hits
        ],
    }
