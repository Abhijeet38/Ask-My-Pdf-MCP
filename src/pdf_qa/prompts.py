"""Prompt templates used by the QA tool.

Kept in a single file so the prompt format is auditable in one place and
easy to tune without grepping the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = """You are a research assistant answering questions strictly from \
the provided PDF excerpts. Follow these rules:

1. Use ONLY the information in the CONTEXT block. If the answer is not present, \
respond exactly: "The provided documents do not contain enough information to answer this."
2. Do not invent facts, numbers, or citations.
3. When you state a fact, cite the source inline using the format [doc_name p.N], \
matching the labels shown in the context. Multiple citations: [doc_a p.3][doc_b p.7].
4. If the question requires synthesizing across documents, do so and cite each.
5. Keep answers concise — a few sentences for narrative questions, a single value \
for numeric questions. Show calculation steps when arithmetic is involved.
"""


@dataclass
class Chunk:
    """Minimal view of an indexed chunk passed to the prompt builder."""

    doc_name: str
    page: int
    text: str
    score: float = 0.0


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    """Assemble CONTEXT + QUESTION block for the LLM."""
    if not chunks:
        return (
            f"CONTEXT:\n(no relevant excerpts retrieved)\n\nQUESTION:\n{question}\n\n"
            "ANSWER:"
        )

    blocks: list[str] = []
    for i, ch in enumerate(chunks, 1):
        label = f"[{ch.doc_name} p.{ch.page}]"
        blocks.append(f"--- excerpt {i} {label} ---\n{ch.text.strip()}")
    context = "\n\n".join(blocks)

    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
