"""Token-aware page-aligned chunker.

Design rules:
- A chunk is associated with exactly one page. Chunks never span pages.
  This keeps source attribution exact for the QA tool.
- Tables emitted by pdf_extract are kept whole when possible (a table
  longer than the chunk budget is split, but only at row boundaries).
- Token counts are estimated with tiktoken's cl100k_base. The exact
  embedding tokenizer differs (BGE uses BERT WordPiece) but cl100k is
  close enough for sizing and avoids loading a second tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from .pdf_extract import PageBlock


# Singleton tokenizer — cheap to load but no need to repeat.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    kind: str       # "text" | "table"
    text: str
    n_tokens: int


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def chunk_blocks(
    blocks: list[PageBlock],
    *,
    doc_id: str,
    doc_name: str,
    chunk_tokens: int = 400,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    """Split blocks into ≤chunk_tokens-sized chunks, page-anchored."""
    chunks: list[Chunk] = []
    counter = 0

    for block in blocks:
        pieces = _split_block(block.text, block.kind, chunk_tokens, chunk_overlap)
        for piece in pieces:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=block.page,
                    chunk_index=counter,
                    kind=block.kind,
                    text=piece,
                    n_tokens=count_tokens(piece),
                )
            )
            counter += 1
    return chunks


def _split_block(text: str, kind: str, chunk_tokens: int, chunk_overlap: int) -> list[str]:
    """Split a single block at sensible boundaries."""
    if not text.strip():
        return []

    n = count_tokens(text)
    if n <= chunk_tokens:
        return [text]

    # Tables: split at row boundaries to keep header context.
    if kind == "table":
        return _split_markdown_table(text, chunk_tokens)

    # Text: split at paragraph then sentence boundaries.
    return _split_text_with_overlap(text, chunk_tokens, chunk_overlap)


def _split_text_with_overlap(text: str, target: int, overlap: int) -> list[str]:
    """Greedy paragraph→sentence packing with token-overlap window."""
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    # First pack paragraphs into chunks
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for p in paragraphs:
        ptok = count_tokens(p)
        if ptok > target:
            # paragraph alone exceeds target — sentence-split it
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_tokens = [], 0
            chunks.extend(_split_long_paragraph(p, target))
            continue
        if cur_tokens + ptok > target:
            chunks.append("\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += ptok
    if cur:
        chunks.append("\n".join(cur))

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    # Add token overlap between adjacent chunks (tail of i prepended to i+1)
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _ENC.encode(chunks[i - 1], disallowed_special=())
        tail = _ENC.decode(prev_tokens[-overlap:]) if len(prev_tokens) > overlap else chunks[i - 1]
        out.append((tail + "\n" + chunks[i]).strip())
    return out


def _split_long_paragraph(p: str, target: int) -> list[str]:
    """Break a paragraph that's larger than `target` at sentence/period
    boundaries; fall back to token slicing if no sentence breaks exist.
    """
    # crude sentence split — sufficient for our chunking purpose
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sentence in _split_sentences(p):
        stok = count_tokens(sentence)
        if stok > target:
            # last resort — slice by tokens
            tokens = _ENC.encode(sentence, disallowed_special=())
            for i in range(0, len(tokens), target):
                parts.append(_ENC.decode(tokens[i : i + target]))
            continue
        if buf_tokens + stok > target:
            parts.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sentence)
        buf_tokens += stok
    if buf:
        parts.append(" ".join(buf))
    return parts


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for tok in text.replace("\n", " ").split(" "):
        cur.append(tok)
        if tok.endswith((".", "!", "?")) and len(cur) > 3:
            out.append(" ".join(cur).strip())
            cur = []
    if cur:
        out.append(" ".join(cur).strip())
    return [s for s in out if s]


def _split_markdown_table(md: str, target: int) -> list[str]:
    """Split an oversized markdown table at row boundaries, keeping header."""
    lines = md.splitlines()
    if len(lines) < 3:
        return [md]
    header_block = "\n".join(lines[:2])  # header + separator row
    header_tokens = count_tokens(header_block)

    rows = lines[2:]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = header_tokens
    for row in rows:
        rt = count_tokens(row) + 1
        if cur_tokens + rt > target and cur:
            chunks.append(header_block + "\n" + "\n".join(cur))
            cur, cur_tokens = [], header_tokens
        cur.append(row)
        cur_tokens += rt
    if cur:
        chunks.append(header_block + "\n" + "\n".join(cur))
    return chunks
