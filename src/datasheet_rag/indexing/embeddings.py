"""The embedding model, for chunks (passages) and questions (queries).

BAAI/bge-small-en-v1.5 was chosen by measuring the 7,911 chunks with each
candidate's tokenizer (docs/indexing.md): at its 512-token limit it truncates
1.7% of the structured chunks, like bge-base-en-v1.5, but is 2.6 times faster
and has half the dimensions; all-MiniLM-L6-v2 (256 tokens) would truncate
about half of them.
"""

from collections.abc import Sequence
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
# bge models are trained to match a query starting with this instruction
# against passages embedded as they are.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@cache
def load_model() -> SentenceTransformer:
    # Loaded once per process; sentence-transformers picks the device itself
    # (the Apple GPU through MPS when available, otherwise the CPU).
    return SentenceTransformer(MODEL_NAME)


def query_text(question: str) -> str:
    return QUERY_INSTRUCTION + question


def embed_passages(texts: Sequence[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """One unit-length vector per text, shape (len(texts), DIMENSIONS)."""
    # Unit length makes cosine similarity a plain dot product, as the bge
    # models recommend.
    return load_model().encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show_progress
    )


def embed_query(question: str) -> np.ndarray:
    """A unit-length vector for a question, shape (DIMENSIONS,)."""
    return load_model().encode(query_text(question), normalize_embeddings=True)
