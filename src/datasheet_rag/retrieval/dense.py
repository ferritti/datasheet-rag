"""Dense retrieval: the chunks whose embeddings are nearest to the question's.

The search is exact (indexing/store.py explains why there is no approximate
index) and ranks chunks by cosine distance, pgvector's `<=>` operator. The
score returned is the cosine similarity, 1 - distance: 1 for the same
direction, 0 for unrelated.

The datasheets in the corpus share most of their text, and a question's
embedding hardly changes between "STM32F401" and "STM32F411", so a question
about one device also finds chunks of the others. devices_in() finds the
datasheets of the devices a question names, and the search can be restricted
to them.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg

from datasheet_rag.indexing.embeddings import embed_query
from datasheet_rag.ingestion.chunking import Strategy
from datasheet_rag.ingestion.chunks import Chunk

# A part number in a question: "STM32F401", "STM32F401RE", "stm32f401xe" or
# just "F401". What follows the digits (package, memory size) is ignored, since
# the corpus has one datasheet per part number.
PART_NUMBER = re.compile(r"\b(?:STM32)?(F4\d\d)(?!\d)", re.IGNORECASE)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity to the question, at most 1


def devices_in(question: str, docs: Collection[str]) -> list[str]:
    """The datasheets, among `docs`, of the devices a question names, in the order it names them.

    'What is the maximum VDD of the STM32F401?' -> ['stm32f401re']. The series
    name "STM32F4" names no device. A part number without a datasheet of its
    own in the corpus, such as STM32F405 (described in the STM32F407
    datasheet), matches nothing.
    """
    found = []
    for part in PART_NUMBER.findall(question):
        for doc in docs:
            if doc.startswith("stm32" + part.lower()) and doc not in found:
                found.append(doc)
    return found


def search_by_vector(
    conn: psycopg.Connection, vector: np.ndarray, strategy: Strategy, k: int = 5, docs: Sequence[str] = ()
) -> list[Hit]:
    """The k chunks of one strategy nearest to `vector`, nearest first, only from `docs` if any are given."""
    params = {"vector": vector, "strategy": strategy, "k": k}
    doc_filter = ""
    if docs:
        doc_filter = "AND doc = ANY(%(docs)s)"
        params["docs"] = list(docs)
    rows = conn.execute(
        f"""
        SELECT chunk_id, doc, page, kind, section, title, text, embedding <=> %(vector)s AS distance
        FROM chunks
        WHERE strategy = %(strategy)s {doc_filter}
        ORDER BY distance
        LIMIT %(k)s
        """,
        params,
    )
    return [
        Hit(Chunk(chunk_id, doc, page, kind, section, title, text), score=1 - distance)
        for chunk_id, doc, page, kind, section, title, text, distance in rows
    ]


def search(
    conn: psycopg.Connection, question: str, strategy: Strategy = "structured", k: int = 5, docs: Sequence[str] = ()
) -> list[Hit]:
    """The k chunks nearest to a question; see search_by_vector()."""
    return search_by_vector(conn, embed_query(question), strategy, k, docs)
