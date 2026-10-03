"""Fixed-size chunks: the baseline the structure-aware chunks are compared with.

Each page's raw text (PyMuPDF, in file order, whitespace collapsed) is cut into
windows of up to MAX_CHUNK_CHARS characters at spaces, each window repeating
the last OVERLAP_CHARS of the previous one so that a sentence cut in two is
whole in one of them. The baseline keeps the two rules that hold for every
chunk (one page per chunk, device name first) and nothing else: no table
handling, no sections, no cleanup, no skipped pages. Comparing it with the
structured chunks measures what the structure adds.
"""

from collections.abc import Sequence

from datasheet_rag.ingestion.chunks import MAX_CHUNK_CHARS, Chunk, device_name
from datasheet_rag.ingestion.pdf_parser import PageContent

OVERLAP_CHARS = 150


def windows(text: str, size: int, overlap: int) -> list[str]:
    """Cut `text` into pieces of at most `size` characters, at spaces, overlapping by about `overlap`."""
    if not text:
        return []
    pieces, start = [], 0
    while start + size < len(text):
        end = text.rfind(" ", start, start + size + 1)
        if end <= start + overlap:  # no space far enough in: cut inside the word
            end = start + size
        pieces.append(text[start:end])
        # The next piece starts at the first word that begins in the last `overlap` characters.
        space = text.find(" ", end - overlap, end)
        start = space + 1 if space != -1 else end
    pieces.append(text[start:])
    return pieces


def fixed_chunks(
    doc: str, pages: Sequence[PageContent], max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS
) -> list[Chunk]:
    prefix = f"{device_name(doc)} datasheet"
    chunks = []
    for page in pages:
        text = " ".join(page.text.split())
        for k, piece in enumerate(windows(text, max_chars - len(prefix) - 1, overlap), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}-p{page.page}-fixed-{k}",
                    doc=doc,
                    page=page.page,
                    kind="text",
                    section="",
                    title=None,
                    text=f"{prefix}\n{piece}",
                )
            )
    return chunks
