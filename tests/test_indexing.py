"""Tests for saving chunks and storing them in PostgreSQL.

The database tests use the `conn` fixture of conftest.py: a temporary schema in
the database of docker-compose.yml, skipped when it is not running. Embeddings
are made-up vectors: the tests do not load the embedding model.
"""

import numpy as np
import pytest

from datasheet_rag.indexing.embeddings import DIMENSIONS, QUERY_INSTRUCTION, query_text
from datasheet_rag.indexing.store import count_chunks, replace_chunks
from datasheet_rag.ingestion.chunks import Chunk, load_chunks, save_chunks


def chunk(doc: str, page: int, n: int) -> Chunk:
    return Chunk(chunk_id=f"{doc}-p{page}-text-{n}", doc=doc, page=page, kind="text", section="6.3 Operating conditions",
                 title=None, text=f"{doc.upper()} datasheet | 6.3 Operating conditions\nParagraph {n} (µA, –40 °C)")


def vectors(n: int) -> np.ndarray:
    return np.eye(n, DIMENSIONS, dtype=np.float32)  # n distinct unit vectors


def test_chunks_survive_a_jsonl_round_trip(tmp_path):
    chunks = [chunk("stm32f401re", 60, 1), Chunk("stm32f401re-p60-table14-1", "stm32f401re", 60, "table", "",
                                                 "Table 14. General operating conditions", "Symbol: VDD | Max: 3.6")]
    path = tmp_path / "chunks" / "stm32f401re.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks


def test_questions_get_the_bge_query_instruction():
    assert query_text("Maximum VDD?") == QUERY_INSTRUCTION + "Maximum VDD?"


def test_replace_chunks_replaces_only_that_document_and_strategy(conn):
    replace_chunks(conn, "structured", "stm32f401re", [chunk("stm32f401re", 60, n) for n in (1, 2, 3)], vectors(3))
    replace_chunks(conn, "structured", "stm32f411re", [chunk("stm32f411re", 61, 1)], vectors(1))
    replace_chunks(conn, "fixed", "stm32f401re", [chunk("stm32f401re", 60, 1)], vectors(1))
    # Rebuilding stm32f401re's structured chunks drops its old ones and nothing else.
    replace_chunks(conn, "structured", "stm32f401re", [chunk("stm32f401re", 61, 9)], vectors(1))
    assert count_chunks(conn) == {
        ("fixed", "stm32f401re"): 1,
        ("structured", "stm32f401re"): 1,
        ("structured", "stm32f411re"): 1,
    }
    [(text,)] = conn.execute("SELECT text FROM chunks WHERE strategy = 'structured' AND doc = 'stm32f401re'")
    assert text.endswith("Paragraph 9 (µA, –40 °C)")  # stored as written, non-ASCII included


def test_replace_chunks_rejects_mismatched_embeddings(conn):
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        replace_chunks(conn, "structured", "stm32f401re", [chunk("stm32f401re", 60, 1), chunk("stm32f401re", 60, 2)],
                       vectors(1))
