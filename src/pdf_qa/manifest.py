"""In-memory registry of discovered + indexed documents.

The manifest is the single source of truth for "what does the server know about?".
It is rebuilt at startup by scanning the data directory and reconciling against
OpenSearch (which docs already have chunks indexed).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path


def compute_doc_id(pdf_path: Path) -> str:
    """Stable identifier derived from file content (first 64 KB is sufficient
    to disambiguate without paying full-file hashing cost on large PDFs).
    """
    h = hashlib.sha1()
    h.update(pdf_path.name.encode())
    with pdf_path.open("rb") as f:
        h.update(f.read(64 * 1024))
    return h.hexdigest()[:16]


@dataclass
class DocumentEntry:
    doc_id: str
    name: str
    path: Path
    size_bytes: int
    pages: int = 0
    indexed: bool = False
    chunks: int = 0


class Manifest:
    """Thread-safe registry. Single instance owned by the server."""

    def __init__(self) -> None:
        self._docs: dict[str, DocumentEntry] = {}
        self._lock = threading.Lock()

    # ---- discovery ---------------------------------------------------------
    def register(self, entry: DocumentEntry) -> None:
        with self._lock:
            self._docs[entry.doc_id] = entry

    def discover(self, data_dir: Path) -> list[DocumentEntry]:
        """Walk data_dir, register every *.pdf as discovered (indexed=False).
        Existing entries are kept (so already-indexed docs stay marked).
        Returns the list of newly-discovered entries.
        """
        new: list[DocumentEntry] = []
        if not data_dir.exists():
            return new
        for pdf_path in sorted(data_dir.rglob("*.pdf")):
            try:
                stat = pdf_path.stat()
            except OSError:
                continue
            doc_id = compute_doc_id(pdf_path)
            with self._lock:
                if doc_id in self._docs:
                    continue
                entry = DocumentEntry(
                    doc_id=doc_id,
                    name=pdf_path.name,
                    path=pdf_path,
                    size_bytes=stat.st_size,
                )
                self._docs[doc_id] = entry
                new.append(entry)
        return new

    # ---- indexing state ----------------------------------------------------
    def mark_indexed(self, doc_id: str, *, pages: int, chunks: int) -> None:
        with self._lock:
            if doc_id in self._docs:
                e = self._docs[doc_id]
                e.indexed = True
                e.pages = pages
                e.chunks = chunks

    def mark_unindexed(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._docs:
                self._docs[doc_id].indexed = False
                self._docs[doc_id].chunks = 0

    # ---- accessors --------------------------------------------------------
    def get(self, doc_id: str) -> DocumentEntry | None:
        with self._lock:
            return self._docs.get(doc_id)

    def get_by_name(self, name: str) -> DocumentEntry | None:
        """Resolve by file name (case-insensitive). Returns first match."""
        n = name.lower()
        with self._lock:
            for e in self._docs.values():
                if e.name.lower() == n or e.name.lower().startswith(n):
                    return e
        return None

    def all(self) -> list[DocumentEntry]:
        with self._lock:
            return list(self._docs.values())

    def indexed_only(self) -> list[DocumentEntry]:
        with self._lock:
            return [e for e in self._docs.values() if e.indexed]

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)


# Singleton manifest used across the server.
manifest = Manifest()
