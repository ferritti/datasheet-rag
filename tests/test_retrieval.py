"""Tests for dense retrieval: device names in questions, and the vector search.

The search tests use the `conn` fixture of conftest.py (a temporary schema in
the database of docker-compose.yml, skipped when it is not running) and
made-up vectors, so the embedding model is not loaded.
"""

import numpy as np
import pytest

from datasheet_rag.indexing.embeddings import DIMENSIONS
from datasheet_rag.indexing.store import replace_chunks
from datasheet_rag.ingestion.chunks import Chunk
from datasheet_rag.retrieval.dense import devices_in, search_by_vector

DOCS = ["stm32f401re", "stm32f407vg", "stm32f410rb", "stm32f411re", "stm32f412zg", "stm32f429zi", "stm32f446re"]


@pytest.mark.parametrize(("question", "expected"), [
    ("What is the maximum VDD of the STM32F401?", ["stm32f401re"]),
    ("Maximum VDD of the stm32f401re?", ["stm32f401re"]),
    ("Flash memory size of the STM32F401xE", ["stm32f401re"]),
    ("Does the F446 have a USB OTG?", ["stm32f446re"]),
    ("Compare the F429 and the STM32F411", ["stm32f429zi", "stm32f411re"]),
    ("Is the F401 the same as the STM32F401RE?", ["stm32f401re"]),  # named twice, listed once
    ("Which STM32F4 devices have an FPU?", []),  # the series name: no device
    ("Maximum frequency of the STM32F405?", []),  # no datasheet of its own in the corpus
    ("Reset value 0xF401", []),  # not a device name
])
def test_devices_in_finds_the_datasheets_of_named_devices(question, expected):
    assert devices_in(question, DOCS) == expected


def unit(*weights: float) -> np.ndarray:
    """A unit vector whose first coordinates are proportional to `weights`, the others 0."""
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[:len(weights)] = weights
    return vector / np.linalg.norm(vector)


def chunk(doc: str, page: int) -> Chunk:
    return Chunk(f"{doc}-p{page}-text-1", doc, page, "text", "", None, f"{doc.upper()} datasheet\nPage {page}")


@pytest.fixture
def indexed(conn):
    replace_chunks(conn, "structured", "stm32f401re", [chunk("stm32f401re", p) for p in (10, 11, 12)],
                   np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]))
    replace_chunks(conn, "structured", "stm32f411re", [chunk("stm32f411re", 10)], np.stack([unit(1, 0.1)]))
    replace_chunks(conn, "fixed", "stm32f401re", [chunk("stm32f401re", 10)], np.stack([unit(1)]))
    return conn


QUERY = unit(1, 0.5)


def test_search_returns_the_nearest_chunks_first_with_their_cosine_similarity(indexed):
    hits = search_by_vector(indexed, QUERY, "structured", k=3)
    assert [hit.chunk.chunk_id for hit in hits] == ["stm32f411re-p10-text-1", "stm32f401re-p10-text-1",
                                                    "stm32f401re-p11-text-1"]
    assert [hit.score for hit in hits] == pytest.approx([QUERY @ unit(1, 0.1), QUERY @ unit(1), QUERY @ unit(0, 1)],
                                                        abs=1e-5)
    assert hits[1].chunk == chunk("stm32f401re", 10)  # every field comes back


def test_search_restricted_to_a_datasheet_returns_all_its_chunks(indexed):
    # Fewer chunks than k: an approximate index would have returned even fewer (indexing/store.py).
    hits = search_by_vector(indexed, QUERY, "structured", k=5, docs=["stm32f401re"])
    assert [(hit.chunk.doc, hit.chunk.page) for hit in hits] == [("stm32f401re", 10), ("stm32f401re", 11),
                                                                 ("stm32f401re", 12)]


def test_search_only_returns_chunks_of_the_given_strategy(indexed):
    hits = search_by_vector(indexed, QUERY, "fixed", k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["stm32f401re-p10-text-1"]
    assert hits[0].score == pytest.approx(QUERY @ unit(1), abs=1e-5)
