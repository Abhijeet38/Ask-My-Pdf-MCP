"""Unit tests for the chunker. No external dependencies."""

from __future__ import annotations

from pdf_qa.ingest.chunker import (
    _split_markdown_table,
    _split_text_with_overlap,
    chunk_blocks,
    count_tokens,
)
from pdf_qa.ingest.pdf_extract import PageBlock


def test_count_tokens_returns_positive_integer():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_split_text_smaller_than_target_returns_single_chunk():
    text = "Short paragraph. Two sentences."
    out = _split_text_with_overlap(text, target=400, overlap=50)
    assert len(out) == 1


def test_split_text_with_long_input_produces_multiple_chunks():
    paragraph = "This is a sentence about machine learning. " * 100
    out = _split_text_with_overlap(paragraph, target=200, overlap=20)
    assert len(out) >= 2
    # Each chunk should be at or under target+overlap headroom
    for chunk in out:
        assert count_tokens(chunk) <= 200 + 20 + 50  # allow for sentence-boundary slack


def test_split_markdown_table_preserves_header_in_each_chunk():
    rows = ["| a | b |", "| - | - |"] + [f"| r{i} | v{i} |" for i in range(50)]
    md = "\n".join(rows)
    pieces = _split_markdown_table(md, target=80)
    assert len(pieces) >= 2
    for piece in pieces:
        assert piece.splitlines()[:2] == ["| a | b |", "| - | - |"]


def test_chunk_blocks_assigns_page_numbers():
    blocks = [
        PageBlock(page=1, text="Some content for page one. " * 5, kind="text"),
        PageBlock(page=2, text="Some content for page two. " * 5, kind="text"),
    ]
    chunks = chunk_blocks(blocks, doc_id="d", doc_name="d.pdf", chunk_tokens=400, chunk_overlap=20)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.doc_id == "d" for c in chunks)
    # chunk_index is monotonic
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
