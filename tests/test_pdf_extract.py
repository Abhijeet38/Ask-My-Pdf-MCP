"""Integration test for pdf_extract using a real bundled PDF."""

from __future__ import annotations

from pathlib import Path

from pdf_qa.ingest.pdf_extract import extract_blocks


def test_extract_blocks_produces_text_blocks(small_pdf: Path):
    blocks, total_pages = extract_blocks(small_pdf)
    assert total_pages > 0
    assert len(blocks) > 0
    # Every block should have a non-empty text body and a valid page number
    for b in blocks:
        assert b.text.strip()
        assert 1 <= b.page <= total_pages
        assert b.kind in {"text", "table"}


def test_extract_blocks_returns_distinct_pages(small_pdf: Path):
    blocks, total_pages = extract_blocks(small_pdf)
    if total_pages > 1:
        pages_seen = {b.page for b in blocks}
        assert len(pages_seen) > 1, "expected text on more than one page"
