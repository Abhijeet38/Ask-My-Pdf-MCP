"""End-to-end smoke test.

Requires:
    - OpenSearch reachable at OS_HOST:OS_PORT
    - LLM_PROVIDER=stub (set automatically below; no creds required)

Run only when integration test prerequisites are present:
    pytest -k integration tests/test_query_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _force_stub_provider():
    os.environ["LLM_PROVIDER"] = "stub"
    yield


def _opensearch_reachable() -> bool:
    try:
        import socket

        host = os.environ.get("OS_HOST", "localhost")
        port = int(os.environ.get("OS_PORT", "9200"))
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _opensearch_reachable(), reason="OpenSearch not running")
def test_end_to_end_with_stub(small_pdf: Path):
    # Re-create settings now that env is set
    from pdf_qa import config as cfg

    cfg.settings = cfg.Settings()

    from pdf_qa.tools._state import AppState
    from pdf_qa.tools.ingest import ingest_document
    from pdf_qa.tools.query import query_documents

    state = AppState.instance()
    state.store.reset_index()
    state.reconcile_manifest()

    # Ingest the bundled small PDF
    result = ingest_document(small_pdf.name, force=True)
    assert result.get("error") is None, result
    assert result["chunks"] > 0

    # Query it
    response = query_documents("What is this document about?")
    assert response["retrieved_chunks_count"] > 0
    assert response["answer"]
    assert response["llm_provider"] == "stub"
    assert response["sources"], "expected non-empty sources"
    src0 = response["sources"][0]
    assert "doc_name" in src0 and "page" in src0
