"""Chunks and their embeddings in PostgreSQL, using the pgvector extension.

One table holds the chunks of both chunking strategies, told apart by the
`strategy` column, so the evaluation can search either of them. The SQL is
written out here rather than hidden behind a vector-store library, so the
schema and the queries are visible (and hybrid search will need them later).

There is deliberately no approximate (HNSW) index on the embeddings. With one
it was tried: a search restricted to one datasheet returned 1 row instead of
5, because the index first takes its nearest candidates over all chunks and
the WHERE clause then drops most of them. Exact search over the ~8,000 chunks
takes about 2 ms and is always right; an index (with pgvector's iterative
scans) only pays off at hundreds of thousands of chunks.
"""

import os
from collections.abc import Sequence

import numpy as np
import psycopg
from dotenv import find_dotenv, load_dotenv
from pgvector.psycopg import register_vector

from datasheet_rag.indexing.embeddings import DIMENSIONS
from datasheet_rag.ingestion.chunks import Chunk

SCHEMA = f"""
    CREATE TABLE IF NOT EXISTS chunks (
        strategy  text    NOT NULL,
        chunk_id  text    NOT NULL,
        doc       text    NOT NULL,
        page      integer NOT NULL,
        kind      text    NOT NULL,
        section   text    NOT NULL,
        title     text,
        text      text    NOT NULL,
        embedding vector({DIMENSIONS}) NOT NULL,
        PRIMARY KEY (strategy, chunk_id)
    )
"""


def connect() -> psycopg.Connection:
    """Connect with the settings in .env (see .env.example), in autocommit mode."""
    # .env is looked up from the current directory upwards, so scripts run from the repository root find it.
    load_dotenv(find_dotenv(usecwd=True))
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        autocommit=True,
    )
    # register_vector() needs the extension to exist in the database.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def replace_chunks(
    conn: psycopg.Connection, strategy: str, doc: str, chunks: Sequence[Chunk], embeddings: np.ndarray
) -> None:
    """Replace the chunks of one document and strategy with new ones."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    rows = [
        (strategy, c.chunk_id, c.doc, c.page, c.kind, c.section, c.title, c.text, embedding)
        for c, embedding in zip(chunks, embeddings)
    ]
    # One transaction: if anything fails, the old chunks are still there.
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE strategy = %s AND doc = %s", (strategy, doc))
        cur.executemany(
            "INSERT INTO chunks (strategy, chunk_id, doc, page, kind, section, title, text, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )


def count_chunks(conn: psycopg.Connection) -> dict[tuple[str, str], int]:
    """Number of stored chunks per (strategy, doc)."""
    rows = conn.execute("SELECT strategy, doc, count(*) FROM chunks GROUP BY strategy, doc ORDER BY strategy, doc")
    return {(strategy, doc): n for strategy, doc, n in rows}
