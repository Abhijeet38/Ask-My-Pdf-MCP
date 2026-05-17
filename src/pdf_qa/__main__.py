"""Module entry point.

    python -m pdf_qa             # discover PDFs, eagerly ingest unindexed ones, serve
    python -m pdf_qa --lazy      # serve immediately; agent must call ingest_*
    python -m pdf_qa --reset     # drop and recreate the OpenSearch index
"""

from __future__ import annotations

import argparse
import logging
import sys

from .server import run as run_server
from .tools._state import AppState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdf-qa-server")
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Skip startup ingestion. Agent triggers indexing via ingest_document/ingest_all.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the OpenSearch index before starting.",
    )
    args = parser.parse_args(argv)

    if args.reset:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        AppState.instance().store.reset_index()

    run_server(eager=not args.lazy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
