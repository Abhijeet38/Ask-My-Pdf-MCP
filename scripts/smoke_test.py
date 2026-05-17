"""Verify the full stack works: OpenSearch is reachable, embedder loads,
LLM creds are valid, and a simple query produces an answer.

Usage:
    python scripts/smoke_test.py "What is this corpus about?"
"""

from __future__ import annotations

import json
import logging
import sys

from pdf_qa.tools._state import AppState
from pdf_qa.tools.query import query_documents


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = argv if argv is not None else sys.argv[1:]
    question = (args[0] if args else "What topics are covered in the indexed documents?")

    print(f"\n--- Smoke test ---")
    print(f"Question: {question}\n")

    state = AppState.instance()
    state.store.health()
    print(f"OpenSearch healthy.")

    state.reconcile_manifest()
    indexed = sum(1 for e in state.store.client.cat.indices(index=state.store.index, format="json"))
    print(f"OpenSearch index ready.")

    print(f"Loading embedder ({state.embedder.dim}-dim)...")
    _ = state.embedder.embed_query("ping")
    print(f"Embedder OK.")

    print(f"Initializing LLM ({state.llm.name})...")
    _ = state.llm  # triggers lazy init
    print(f"LLM OK.\n")

    print(f"Running query_documents...\n")
    result = query_documents(question)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("answer") else 1


if __name__ == "__main__":
    raise SystemExit(main())
