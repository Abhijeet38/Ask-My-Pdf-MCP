"""list_documents — show every doc the server knows about + indexed status."""

from __future__ import annotations

from typing import Any

from ..manifest import manifest


def list_documents() -> dict[str, Any]:
    """Return all known documents with their indexed status."""
    docs = []
    for e in manifest.all():
        docs.append(
            {
                "doc_id": e.doc_id,
                "name": e.name,
                "indexed": e.indexed,
                "pages": e.pages,
                "chunks": e.chunks,
                "size_kb": round(e.size_bytes / 1024, 1),
                "path": str(e.path),
            }
        )
    return {
        "total": len(docs),
        "indexed": sum(1 for d in docs if d["indexed"]),
        "documents": docs,
    }
