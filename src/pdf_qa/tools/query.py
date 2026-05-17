"""query_documents — primary Q&A tool.

Pipeline: embed question → hybrid retrieve (BM25 ⊕ k-NN with RRF) → build
prompt → call LLM → return grounded answer + per-source attribution.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import settings
from ..prompts import SYSTEM_PROMPT, Chunk as PromptChunk, build_user_prompt
from ..store.search import hybrid_search
from ._state import AppState

log = logging.getLogger(__name__)


def query_documents(
    question: str,
    doc_filter: list[str] | None = None,
    top_k: int = 0,
) -> dict[str, Any]:
    """Answer `question` using indexed PDFs.

    Args:
        question: A natural-language question.
        doc_filter: Optional list of doc_ids or doc_names to restrict the search to.
        top_k: How many chunks to retrieve. Defaults to settings.top_k.

    Returns: dict with the synthesized answer, sources, and bookkeeping.
    """
    if not question or not question.strip():
        return {"answer": "", "sources": [], "error": "empty question"}

    state = AppState.instance()
    k = top_k if top_k > 0 else settings.top_k

    # 1. Embed question
    q_vec = state.embedder.embed_query(question)

    # 2. Hybrid retrieve
    hits = hybrid_search(
        state.store,
        query_text=question,
        query_vector=q_vec,
        top_k=k,
        doc_filter=doc_filter or None,
    )

    if not hits:
        return {
            "answer": "The provided documents do not contain enough information to answer this.",
            "sources": [],
            "retrieved_chunks_count": 0,
            "llm_provider": state.llm.name,
        }

    # 3. Build prompt
    prompt_chunks = [
        PromptChunk(doc_name=h.doc_name, page=h.page, text=h.text, score=h.score)
        for h in hits
    ]
    user_prompt = build_user_prompt(question, prompt_chunks)

    # 4. Call LLM
    answer = state.llm.generate(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=1024)

    # 5. Format sources for the response
    sources = [
        {
            "doc_id": h.doc_id,
            "doc_name": h.doc_name,
            "page": h.page,
            "kind": h.kind,
            "score": h.score,
            "snippet": _snippet(h.text),
        }
        for h in hits
    ]
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks_count": len(hits),
        "llm_provider": state.llm.name,
    }


def _snippet(text: str, max_chars: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[: max_chars - 1] + "…"
