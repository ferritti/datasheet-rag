"""Embed the chunks and store them in PostgreSQL (pgvector).

Reads data/processed/chunks_<strategy>/<doc>.jsonl (written by build_chunks.py)
and, for each file, replaces that datasheet's chunks of that strategy in the
`chunks` table. The database must be running (docker compose up -d --wait).

Usage (from the repository root):
    python scripts/index_chunks.py
    python scripts/index_chunks.py --strategy structured
"""

import argparse
import time
from pathlib import Path

from datasheet_rag.indexing.embeddings import MODEL_NAME, embed_passages
from datasheet_rag.indexing.store import connect, count_chunks, create_schema, replace_chunks
from datasheet_rag.ingestion.chunking import STRATEGIES
from datasheet_rag.ingestion.chunks import load_chunks

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all", help="chunking strategy (default: all)")
    args = parser.parse_args()
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    with connect() as conn:
        create_schema(conn)
        print(f"Embedding with {MODEL_NAME}")
        for strategy in strategies:
            paths = sorted((PROCESSED_DIR / f"chunks_{strategy}").glob("*.jsonl"))
            if not paths:
                raise SystemExit(f"no chunks for {strategy!r} in {PROCESSED_DIR}: run scripts/build_chunks.py first")
            for path in paths:
                start = time.perf_counter()
                chunks = load_chunks(path)
                embeddings = embed_passages([chunk.text for chunk in chunks])
                replace_chunks(conn, strategy, path.stem, chunks, embeddings)
                print(f"  {strategy:<10} {path.stem:<12} {len(chunks):>4} chunks in {time.perf_counter() - start:.0f} s")

        print("\nStored chunks:")
        for (strategy, doc), n in count_chunks(conn).items():
            print(f"  {strategy:<10} {doc:<12} {n:>4}")


if __name__ == "__main__":
    main()
