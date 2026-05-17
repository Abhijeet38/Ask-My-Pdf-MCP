"""Bulk-ingest every PDF in DATA_DIR into OpenSearch, then exit.

Useful for pre-warming the index outside the MCP server, e.g. in CI or before
benchmarking. Equivalent to calling the `ingest_all` MCP tool and then
shutting down.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from pdf_qa import config as _config_module  # to allow --data-dir override
from pdf_qa.tools._state import AppState
from pdf_qa.tools.ingest import ingest_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdf-qa-ingest")
    parser.add_argument(
        "--data-dir",
        help="Directory of PDFs to ingest (overrides DATA_DIR env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even already-indexed PDFs",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the OpenSearch index first",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir
        # Re-create settings so DATA_DIR takes effect
        _config_module.settings = _config_module.Settings()

    state = AppState.instance()
    if args.reset:
        state.store.reset_index()
    state.store.health()
    state.reconcile_manifest()

    result = ingest_all(force=args.force)
    print(f"\n=== Ingestion summary ===")
    print(f"  indexed: {result['indexed']}")
    print(f"  skipped: {result['skipped']}")
    print(f"  errors:  {result['errors']}")
    for r in result["results"]:
        flag = "✓" if not r.get("error") else "✗"
        print(
            f"  {flag} {r['doc_name']:50s}  pages={r['pages']:>4}  chunks={r['chunks']:>5}  "
            f"{r['elapsed_seconds']:>5.1f}s"
            + (f"  ERROR: {r['error']}" if r.get("error") else "")
        )
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
