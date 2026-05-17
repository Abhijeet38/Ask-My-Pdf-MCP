"""One-shot: backfill the pages index for every PDF already in the manifest.

Use this after upgrading from chunks-only to chunks+pages dual-index. It
only re-extracts page text — no re-embedding, no re-chunking.
"""

from __future__ import annotations

import logging
import sys
import time

from pdf_qa.ingest.pdf_extract import extract_pages
from pdf_qa.manifest import manifest
from pdf_qa.tools._state import AppState


def main() -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("backfill_pages")

    state = AppState.instance()
    store = state.store               # ensures both indices exist
    state.reconcile_manifest()

    have = store.pages_indexed_doc_ids()
    log.info("Pages already populated for: %d docs", len(have))

    n_built = 0
    for entry in manifest.all():
        if entry.doc_id in have:
            log.info("  ✓ skip %s (pages already present)", entry.name)
            continue
        if not entry.path.exists():
            log.warning("  ✗ skip %s (source missing: %s)", entry.name, entry.path)
            continue
        t = time.time()
        page_texts = extract_pages(entry.path)
        if not page_texts:
            log.warning("  ✗ %s produced 0 pages", entry.name)
            continue
        store.bulk_index_pages(
            doc_id=entry.doc_id, doc_name=entry.name, page_texts=page_texts
        )
        log.info(
            "  + %s: %d pages in %.1fs",
            entry.name, len(page_texts), time.time() - t,
        )
        n_built += 1

    print(f"\nBackfilled pages for {n_built} doc(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
