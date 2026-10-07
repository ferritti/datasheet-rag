"""Search the indexed chunks from the command line.

Embeds the question, finds the k nearest chunks of one chunking strategy and
prints them with their datasheet, page and cosine similarity. When the
question names devices (e.g. "STM32F401"), only their datasheets are
searched; --doc overrides this. The database must be running
(docker compose up -d --wait) and indexed (scripts/index_chunks.py).

Usage (from the repository root):
    python scripts/search.py "What is the maximum VDD of the STM32F401?"
    python scripts/search.py "What is the maximum VDD of the STM32F401?" --doc all
    python scripts/search.py "Flash memory size" --doc stm32f411re --strategy fixed --k 10 --full
"""

import argparse
import textwrap

from datasheet_rag.indexing.store import connect, count_chunks
from datasheet_rag.ingestion.chunking import STRATEGIES
from datasheet_rag.retrieval.dense import devices_in, search

PREVIEW_LINES = 4  # lines of each chunk shown without --full
PREVIEW_WIDTH = 120


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=STRATEGIES, default="structured", help="chunking strategy (default: structured)")
    parser.add_argument("--k", type=int, default=5, help="number of chunks to return (default: 5)")
    parser.add_argument("--doc", help="datasheet to search, e.g. stm32f411re, or 'all' "
                                      "(default: those of the devices the question names, otherwise all)")
    parser.add_argument("--full", action="store_true", help="print the whole text of each chunk")
    args = parser.parse_args()

    with connect() as conn:
        stored = sorted({doc for strategy, doc in count_chunks(conn) if strategy == args.strategy})
        if not stored:
            raise SystemExit(f"no {args.strategy!r} chunks in the database: run scripts/index_chunks.py first")
        if args.doc is None:
            docs = devices_in(args.question, stored)
        elif args.doc == "all":
            docs = []
        elif args.doc in stored:
            docs = [args.doc]
        else:
            raise SystemExit(f"unknown datasheet {args.doc!r}, expected one of: all, {', '.join(stored)}")
        hits = search(conn, args.question, args.strategy, args.k, docs)

    print(f"Searched {', '.join(docs) if docs else 'all datasheets'} ({args.strategy} chunks)\n")
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        print(f"{rank:>2}. score {hit.score:.3f}  {chunk.doc} page {chunk.page}  ({chunk.chunk_id})")
        lines = chunk.text.splitlines()
        if args.full:
            shown = lines
        else:
            shown = [textwrap.shorten(line, PREVIEW_WIDTH, placeholder=" ...") for line in lines[:PREVIEW_LINES]]
        for line in shown:
            print(f"      {line}")
        if len(lines) > len(shown):
            print(f"      [{len(lines) - len(shown)} more lines]")
        print()


if __name__ == "__main__":
    main()
