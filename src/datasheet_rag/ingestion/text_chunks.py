"""Turn the running text of a datasheet into chunks.

The text of a page is its blocks in reading order, minus what is not running
text or is already in a table chunk:
- the running header and footer;
- captioned tables with their caption and footnotes (in the table chunks), and
  the "tables" found inside figures;
- the text inside figures (axis labels, pin names of pinout drawings), from the
  "Figure N." caption down to ST's figure ID ("MS31151V4", "ai15062b"); the
  caption itself is kept. 418 of the 521 figures in the corpus have the ID; the
  others keep their text;
- unreadable text from fonts without a Unicode mapping;
- section headings, which go into the chunk prefix instead.
Pages of the table of contents and the final legal notice are skipped.

Blocks are grouped by section and packed into chunks of up to MAX_CHUNK_CHARS
that never cross a page or a section boundary.
"""

import re
from collections.abc import Mapping, Sequence

from datasheet_rag.ingestion.chunks import MAX_CHUNK_CHARS, Chunk, device_name, normalize_text
from datasheet_rag.ingestion.pdf_parser import PageContent, TextBlock
from datasheet_rag.ingestion.sections import heading_key
from datasheet_rag.ingestion.table_chunks import find_captioned_tables

FIGURE_CAPTION = re.compile(r"^Figure \d+\.")
# The ID ST prints at the bottom right of each figure: MS31151V4, MSv31179V1,
# ai15062, ai15062b, ai15062V2.
FIGURE_ID = re.compile(r"^(MSv?\d{5}V\d+|ai\d{5}[A-Za-z]?(V\d+)?)$")


def is_skipped_page(page: PageContent) -> bool:
    # Contents and lists of tables and figures have dot leaders up to the page number.
    return ". . . ." in page.text or "IMPORTANT NOTICE" in page.text


def is_unreadable(text: str) -> bool:
    """Text from a font without a Unicode mapping comes out as control characters or "(cid:N)"."""
    return "(cid:" in text or any(ord(ch) < 32 and ch not in "\n\t" for ch in text)


def figure_spans(blocks: Sequence[TextBlock]) -> list[tuple[float, float]]:
    """Vertical spans (top, bottom) inside figures: from below each caption to the bottom of the figure ID."""
    captions = sorted((b for b in blocks if FIGURE_CAPTION.match(b.text) and not b.in_margin), key=lambda b: b.bbox[1])
    ids = [b for b in blocks if any(FIGURE_ID.match(line.strip()) for line in b.text.splitlines())]
    spans = []
    for k, caption in enumerate(captions):
        next_caption_top = captions[k + 1].bbox[1] if k + 1 < len(captions) else float("inf")
        below = [b for b in ids if caption.bbox[3] < b.bbox[1] < next_caption_top]
        if below:
            spans.append((caption.bbox[3], min(b.bbox[3] for b in below)))
    return spans


def center_inside(block: TextBlock, bbox: tuple[float, float, float, float]) -> bool:
    x = (block.bbox[0] + block.bbox[2]) / 2
    y = (block.bbox[1] + block.bbox[3]) / 2
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split a paragraph longer than max_chars at sentence ends."""
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return pieces + [current]


def pack(prefix: str, paragraphs: Sequence[str], max_chars: int) -> list[str]:
    """Chunk texts: the prefix line, then as many paragraphs (one per line) as fit."""
    budget = max_chars - len(prefix) - 1
    lines = [piece for paragraph in paragraphs for piece in split_long_text(paragraph, budget)]
    texts, current = [], []
    for line in lines:
        if current and len("\n".join([prefix, *current, line])) > max_chars:
            texts.append("\n".join([prefix, *current]))
            current = []
        current.append(line)
    if current:
        texts.append("\n".join([prefix, *current]))
    return texts


def text_chunks(
    doc: str,
    pages: Sequence[PageContent],
    sections: Mapping[int, Sequence[str]],
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunks for the running text of a document, in page order.

    `sections` comes from block_sections() on the same pages.
    """
    used_by_tables = {id(block) for part in find_captioned_tables(pages) for block in part.used_blocks}
    chunks = []
    for page in pages:
        if is_skipped_page(page):
            continue
        spans = figure_spans(page.blocks)
        # (section, paragraphs) groups, in reading order.
        groups: list[tuple[str, list[str]]] = []
        for block, section in zip(page.blocks, sections[page.page]):
            y = (block.bbox[1] + block.bbox[3]) / 2
            if (
                block.in_margin
                or id(block) in used_by_tables
                or any(center_inside(block, table.bbox) for table in page.tables)
                or any(top < y <= bottom for top, bottom in spans)
                or is_unreadable(block.text)
                or heading_key(block.text) == heading_key(section)
            ):
                continue
            if not groups or groups[-1][0] != section:
                groups.append((section, []))
            groups[-1][1].append(normalize_text(block.text))

        k = 0
        for section, paragraphs in groups:
            prefix = " | ".join(filter(None, [f"{device_name(doc)} datasheet", section]))
            for text in pack(prefix, paragraphs, max_chars):
                k += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc}-p{page.page}-text-{k}",
                        doc=doc,
                        page=page.page,
                        kind="text",
                        section=section,
                        title=None,
                        text=text,
                    )
                )
    return chunks
