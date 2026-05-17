"""Search-time helpers: hybrid retrieval (BM25 ⊕ k-NN) with reciprocal-rank
fusion, plus a literal/keyword search path that backs the keyword_search tool
and the "how many times does X appear" question type.

We do RRF client-side rather than relying on OpenSearch hybrid search
pipelines so the code is portable across OpenSearch versions and
self-explanatory in the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .client import OpenSearchStore


@dataclass
class SearchHit:
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    kind: str
    text: str
    score: float  # fused score (after RRF) or raw bm25 score


@dataclass
class KeywordHit:
    doc_id: str
    doc_name: str
    page: int
    snippet: str
    score: float


# ---------------------------------------------------------------------------
# Hybrid retrieval (vector + lexical) with RRF
# ---------------------------------------------------------------------------

def hybrid_search(
    store: OpenSearchStore,
    *,
    query_text: str,
    query_vector: np.ndarray,
    top_k: int = 5,
    doc_filter: list[str] | None = None,
    candidates_per_branch: int = 20,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Fetch top candidates from BM25 and k-NN, fuse with RRF, return top_k."""
    filters = _build_filters(doc_filter)
    bm25_hits = _bm25(store, query_text, candidates_per_branch, filters)
    knn_hits = _knn(store, query_vector, candidates_per_branch, filters)

    fused = _reciprocal_rank_fusion(bm25_hits, knn_hits, rrf_k=rrf_k)
    return fused[:top_k]


def _bm25(
    store: OpenSearchStore, query: str, size: int, filters: list[dict]
) -> list[SearchHit]:
    body = {
        "size": size,
        "query": _wrap_with_filters(
            {"match": {"text": {"query": query, "operator": "or"}}},
            filters,
        ),
    }
    resp = store.client.search(index=store.index, body=body)
    return [_hit_from_source(h) for h in resp["hits"]["hits"]]


def _knn(
    store: OpenSearchStore, vector: np.ndarray, size: int, filters: list[dict]
) -> list[SearchHit]:
    knn_clause = {
        "knn": {"embedding": {"vector": vector.tolist(), "k": size}}
    }
    if filters:
        # OpenSearch k-NN supports an inline filter under the knn clause
        knn_clause["knn"]["embedding"]["filter"] = {"bool": {"must": filters}}
    body = {"size": size, "query": knn_clause}
    resp = store.client.search(index=store.index, body=body)
    return [_hit_from_source(h) for h in resp["hits"]["hits"]]


def _reciprocal_rank_fusion(
    *rankings: list[SearchHit], rrf_k: int = 60
) -> list[SearchHit]:
    """Combine multiple rankings into one ordered list using RRF."""
    fused_score: dict[str, float] = {}
    by_id: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = f"{hit.doc_id}:{hit.chunk_index}"
            fused_score[key] = fused_score.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_id[key] = hit
    out: list[SearchHit] = []
    for key in sorted(fused_score, key=lambda k: fused_score[k], reverse=True):
        h = by_id[key]
        out.append(
            SearchHit(
                doc_id=h.doc_id,
                doc_name=h.doc_name,
                page=h.page,
                chunk_index=h.chunk_index,
                kind=h.kind,
                text=h.text,
                score=round(fused_score[key], 6),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Keyword / literal search (backs keyword_search tool)
# ---------------------------------------------------------------------------

def keyword_search(
    store: OpenSearchStore,
    *,
    term: str,
    doc_filter: list[str] | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    max_hits: int = 50,
) -> tuple[list[KeywordHit], int]:
    """Return (sample_hits, total_match_count) where total counts literal
    matches of `term` across all PAGES (overlap-free, accurate).

    Reads from the pages index (one record per page) rather than the chunks
    index — this avoids the chunk-overlap double-count bug. The chunks index
    is still used for hybrid retrieval in `hybrid_search()`.
    """
    filters = _build_filters(doc_filter)
    if regex:
        match_clause = {"regexp": {"text": {"value": term, "case_insensitive": not case_sensitive}}}
    else:
        match_clause = {"match_phrase": {"text": term}}

    body = {
        "size": max_hits,
        "query": _wrap_with_filters(match_clause, filters),
        "highlight": {
            "fields": {"text": {"fragment_size": 240, "number_of_fragments": 1}},
            "pre_tags": ["<<"],
            "post_tags": [">>"],
        },
    }

    resp = store.client.search(index=store.pages_index, body=body)
    sample_hits: list[KeywordHit] = []
    pattern = _compile_pattern(term, regex=regex, case_sensitive=case_sensitive)
    total_occurrences = 0

    for h in resp["hits"]["hits"]:
        src = h["_source"]
        # Pages don't overlap — counting per-page and summing is exact.
        page_count = len(pattern.findall(src["text"])) if pattern else 0
        total_occurrences += page_count
        snippet = "(no preview)"
        if "highlight" in h and "text" in h["highlight"]:
            snippet = h["highlight"]["text"][0]
        sample_hits.append(
            KeywordHit(
                doc_id=src["doc_id"],
                doc_name=src["doc_name"],
                page=src["page"],
                snippet=snippet,
                score=h["_score"],
            )
        )

    # If max_hits cut us off, scroll the rest just to count occurrences.
    total_pages = resp["hits"]["total"]["value"]
    if total_pages > max_hits and pattern is not None:
        for src in _scroll_remaining(store, body["query"], skip=max_hits):
            total_occurrences += len(pattern.findall(src["text"]))

    return sample_hits, total_occurrences


def _scroll_remaining(store: OpenSearchStore, query: dict, skip: int):
    """Yield _source dicts beyond the initial response, used only for counting."""
    resp = store.client.search(
        index=store.pages_index,
        body={"size": 500, "from": skip, "query": query, "_source": ["text"]},
    )
    for h in resp["hits"]["hits"]:
        yield h["_source"]


def _compile_pattern(term: str, *, regex: bool, case_sensitive: bool):
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        try:
            return re.compile(term, flags)
        except re.error:
            return None
    return re.compile(re.escape(term), flags)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filters(doc_filter: list[str] | None) -> list[dict]:
    if not doc_filter:
        return []
    # filter accepts either doc_id values or doc_name values; we support both.
    return [{"bool": {"should": [
        {"terms": {"doc_id": doc_filter}},
        {"terms": {"doc_name": doc_filter}},
    ], "minimum_should_match": 1}}]


def _wrap_with_filters(must_clause: dict, filters: list[dict]) -> dict:
    if not filters:
        return must_clause
    return {"bool": {"must": [must_clause], "filter": filters}}


def _hit_from_source(h: dict) -> SearchHit:
    src = h["_source"]
    return SearchHit(
        doc_id=src["doc_id"],
        doc_name=src["doc_name"],
        page=src["page"],
        chunk_index=src["chunk_index"],
        kind=src.get("kind", "text"),
        text=src["text"],
        score=h.get("_score", 0.0) or 0.0,
    )
