"""Build the chunks of a datasheet with one of the two chunking strategies.

- "structured": table chunks (one line per row, with caption, section and
  footnotes) plus text chunks (running text grouped by section), see
  table_chunks.py and text_chunks.py;
- "fixed": the baseline, fixed-size windows of each page's raw text, see
  fixed_chunks.py.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from datasheet_rag.ingestion.chunks import Chunk, doc_id
from datasheet_rag.ingestion.fixed_chunks import fixed_chunks
from datasheet_rag.ingestion.pdf_parser import OutlineEntry, PageContent, parse_pdf, read_outline
from datasheet_rag.ingestion.sections import block_sections
from datasheet_rag.ingestion.table_chunks import table_chunks
from datasheet_rag.ingestion.text_chunks import text_chunks

Strategy = Literal["structured", "fixed"]
STRATEGIES: tuple[Strategy, ...] = ("structured", "fixed")


def chunk_pages(doc: str, pages: Sequence[PageContent], outline: Sequence[OutlineEntry], strategy: Strategy) -> list[Chunk]:
    """Chunks of already parsed pages, in page order (so one parse can serve both strategies)."""
    if strategy == "fixed":
        return fixed_chunks(doc, pages)
    if strategy == "structured":
        sections = block_sections(pages, outline)
        chunks = table_chunks(doc, pages, sections) + text_chunks(doc, pages, sections)
        return sorted(chunks, key=lambda chunk: chunk.page)
    raise ValueError(f"unknown chunking strategy {strategy!r}, expected one of {STRATEGIES}")


def build_chunks(pdf_path: str | Path, strategy: Strategy = "structured") -> list[Chunk]:
    """Parse a datasheet and chunk it."""
    return chunk_pages(doc_id(pdf_path), parse_pdf(pdf_path), read_outline(pdf_path), strategy)
