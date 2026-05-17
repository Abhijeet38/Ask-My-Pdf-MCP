"""PDF parsing → page-anchored text blocks.

Uses PyMuPDF (fitz) for fast text extraction with native page numbers, and
attempts table detection via `Page.find_tables()` (PyMuPDF >= 1.23). Tables
are flattened to markdown and emitted as their own blocks so downstream
chunking treats them as cohesive units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


@dataclass
class PageBlock:
    """One contiguous piece of text with a single page anchor.

    A page can produce multiple blocks (e.g. body text + 1-2 tables).
    """

    page: int  # 1-indexed
    text: str
    kind: str = "text"  # "text" | "table"


def extract_blocks(pdf_path: Path) -> tuple[list[PageBlock], int]:
    """Return (blocks, total_pages).

    Tables (when detected) are emitted as separate `kind="table"` blocks
    rendered as markdown so the embedder sees them as a single unit.
    """
    doc = fitz.open(str(pdf_path))
    blocks: list[PageBlock] = []
    total_pages = doc.page_count

    for page_index in range(total_pages):
        page = doc[page_index]
        page_num = page_index + 1

        # ---- text -------------------------------------------------------
        # `text` is faster than blocks/dict and good enough; we don't
        # need layout reconstruction here.
        text = page.get_text("text") or ""
        text = _normalize(text)
        if text:
            blocks.append(PageBlock(page=page_num, text=text, kind="text"))

        # ---- tables -----------------------------------------------------
        try:
            tables = page.find_tables()
        except Exception:  # noqa: BLE001
            tables = None

        if tables:
            try:
                # PyMuPDF returns a TableFinder whose .tables is a list
                tbl_list = getattr(tables, "tables", []) or []
            except Exception:  # noqa: BLE001
                tbl_list = []
            for tbl in tbl_list:
                try:
                    md = _table_to_markdown(tbl.extract())
                except Exception as e:  # noqa: BLE001
                    log.debug("table extract failed on %s p.%d: %s", pdf_path.name, page_num, e)
                    continue
                if md:
                    blocks.append(PageBlock(page=page_num, text=md, kind="table"))

    doc.close()
    return blocks, total_pages


def extract_pages(pdf_path: Path) -> dict[int, str]:
    """Return {page_number: raw_normalized_text} for every page.

    Used by the pages index to support overlap-free literal counting in
    keyword_search. Uses PyMuPDF's `rawdict` mode (per-glyph extraction)
    rather than the default `text` mode so that terms split across spans
    by footnote markers, kerning, or layout (e.g. "WikiText-2" with a
    superscript footnote between the "-" and "2") are still counted.

    The default `text` mode interpolates whitespace between spans, which
    inflates the apparent gap and breaks literal `match_phrase` and
    Python `re` matching of compound tokens. Walking `rawdict` and
    concatenating only the actual glyph chars (no inserted whitespace
    between spans) recovers the visible text stream as PDF readers see it.
    """
    doc = fitz.open(str(pdf_path))
    pages: dict[int, str] = {}
    for page_index in range(doc.page_count):
        page = doc[page_index]
        text = _rawdict_text(page)
        if text:
            pages[page_index + 1] = text
    doc.close()
    return pages


def _rawdict_text(page) -> str:
    """Concatenate rawdict glyphs without interpolating whitespace between
    spans. Inserts a space between *lines* unless the previous line ended
    with a hyphen (so "WikiText-\\n2" → "WikiText-2", but "neural\\nnetworks"
    → "neural networks"). Collapses runs of whitespace.
    """
    blocks_out: list[str] = []
    for blk in page.get_text("rawdict").get("blocks", []):
        if blk.get("type", 0) != 0:  # 0 = text, 1 = image
            continue
        line_strs: list[str] = []
        for line in blk.get("lines", []):
            line_chars = [
                ch.get("c", "")
                for span in line.get("spans", [])
                for ch in span.get("chars", [])
            ]
            line_strs.append("".join(line_chars))
        # Join lines: no space if previous line ends with a hyphen
        joined: list[str] = []
        for i, line in enumerate(line_strs):
            if not line.strip():
                continue
            if not joined:
                joined.append(line)
            elif joined[-1].rstrip().endswith("-"):
                joined.append(line)
            else:
                joined.append(" " + line)
        if joined:
            blocks_out.append("".join(joined))
    text = "\n".join(blocks_out)
    # Collapse runs of horizontal whitespace; keep newlines as paragraph breaks
    import re as _re
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n[ \t]*", "\n", text)
    return text.strip()


def _normalize(text: str) -> str:
    """Light cleanup: collapse runs of whitespace, drop empty lines."""
    out_lines: list[str] = []
    for line in text.splitlines():
        s = " ".join(line.split())
        if s:
            out_lines.append(s)
    return "\n".join(out_lines)


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render a 2D list-of-rows as a GitHub-flavored markdown table.

    Empty / None cells become empty strings. Skip tables with zero or one
    populated rows (likely false positives from the table finder).
    """
    if not rows or len(rows) < 2:
        return ""
    cleaned: list[list[str]] = []
    width = max(len(r) for r in rows)
    for r in rows:
        padded = list(r) + [None] * (width - len(r))
        cleaned.append(["" if c is None else str(c).strip().replace("\n", " ") for c in padded])

    header = cleaned[0]
    body = cleaned[1:]
    if not any(cell for cell in header):
        # use the first non-empty row as a header
        for i, row in enumerate(body):
            if any(cell for cell in row):
                header = row
                body = body[i + 1 :]
                break

    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
