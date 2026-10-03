"""Turn the tables of a datasheet into chunks.

Only tables with a "Table N." caption just above them are kept: every real
table in an ST datasheet has one (repeated as "(continued)" on each further
page), while the extractor also finds "tables" inside figures (pin squares of
pinout drawings, graph grids, timing diagrams), which have none.

Each table row becomes one line of "column: value" pairs, so a row stays
readable when a table is split over several chunks. A chunk holds as many rows
as fit in MAX_CHUNK_CHARS, plus the footnotes those rows refer to; a single row
with long footnotes can exceed it, since cutting either would lose meaning.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from datasheet_rag.ingestion.chunks import MAX_CHUNK_CHARS, Chunk, device_name
from datasheet_rag.ingestion.pdf_parser import ExtractedTable, PageContent, TextBlock

CAPTION = re.compile(r"^Table (\d+)\.\s+(.+)$", re.DOTALL)
NOTE = re.compile(r"^(\d+)\.\s+(.+)$", re.DOTALL)  # a footnote block: "1.\nGuaranteed by design."
NOTE_MARKER = re.compile(r"\((\d{1,2})\)")  # a footnote reference in a cell: "1.7(1)"
NOTE_REFERENCE = re.compile(r"\[note (\d+)\]")  # the same reference after clean_cell()

# Distances in points, measured on the captioned tables of the corpus: a
# caption ends 0-14 pt above its table; the first footnote starts 2-11 pt below
# the table and each further footnote 2-7 pt below the previous one, while the
# next ordinary paragraph or figure caption starts 15 pt or more below.
MAX_CAPTION_GAP = 40
MAX_NOTE_GAP = 12

# The revision history describes changes to the document, not the device, and
# names dozens of other tables ("Updated Table 14: ..."), so it would be
# retrieved for questions about them. Its rows are also up to 2,800 characters.
SKIPPED_TABLES = ("Document revision history",)


@dataclass
class CaptionedTable:
    number: int  # N in "Table N."
    caption: str  # cleaned caption without "(continued)", e.g. "Table 14. General operating conditions"
    table: ExtractedTable
    notes: dict[int, str] = field(default_factory=dict)  # footnotes printed below this part of the table


def clean_cell(text: str) -> str:
    """Normalise the text of a table cell or caption."""
    if "(cid:" in text:  # characters from a font without a Unicode mapping: unreadable
        return ""
    # Names wrapped after an underscore or a slash in narrow cells:
    # "USART2_\nCTS", "TIM2_CH1/\nTIM2_ETR".
    text = text.replace("_\n", "_").replace("/\n", "/")
    text = re.sub(r"[–−](?=\d)", "-", text)  # en dash or minus sign used as a minus: "–0.3"
    # Footnote references, so that "1.7(1)" is not read as 1.71.
    text = NOTE_MARKER.sub(r" [note \1]", text)
    return " ".join(text.split())  # other line breaks and repeated spaces


def find_caption(table: ExtractedTable, blocks: Sequence[TextBlock]) -> tuple[int, str] | None:
    """The table number and caption of the "Table N." block just above the table, if any."""
    x0, top, x1, _ = table.bbox
    candidates = []
    for block in blocks:
        match = CAPTION.match(block.text)
        gap = top - block.bbox[3]
        overlaps = block.bbox[0] < x1 and block.bbox[2] > x0
        if match and not block.in_margin and overlaps and -3 <= gap <= MAX_CAPTION_GAP:
            candidates.append((gap, int(match[1]), clean_cell(block.text)))
    if not candidates:
        return None
    _, number, caption = min(candidates)
    return number, caption.replace(" (continued)", "")


def read_notes(blocks: Sequence[TextBlock], start: float) -> dict[int, str]:
    """Numbered footnotes in `blocks` (sorted top to bottom), the first one starting just below `start`."""
    notes: dict[int, str] = {}
    previous_bottom = start
    for block in blocks:
        if block.bbox[1] - previous_bottom > MAX_NOTE_GAP:
            break
        match = NOTE.match(block.text)
        if match:
            notes[int(match[1])] = " ".join(match[2].split())
        elif notes:  # a long footnote continued in the next block
            last = max(notes)
            notes[last] += " " + " ".join(block.text.split())
        elif block.text != "Notes:":  # some footnote lists have a "Notes:" heading
            break
        previous_bottom = block.bbox[3]
    return notes


def find_notes(table: ExtractedTable, blocks: Sequence[TextBlock]) -> dict[int, str]:
    """The numbered footnotes printed right below the table."""
    x0, _, x1, bottom = table.bbox
    below = sorted(
        (b for b in blocks if not b.in_margin and b.bbox[1] >= bottom - 2 and b.bbox[0] < x1 and b.bbox[2] > x0),
        key=lambda b: b.bbox[1],
    )
    return read_notes(below, start=bottom)


def notes_at_page_top(blocks: Sequence[TextBlock]) -> dict[int, str]:
    """Footnotes that open a page: they belong to a table that ended at the bottom of the previous page."""
    body = sorted((b for b in blocks if not b.in_margin), key=lambda b: b.bbox[1])
    if not body or not (NOTE.match(body[0].text) or body[0].text == "Notes:"):
        return {}
    return read_notes(body, start=body[0].bbox[1])


def column_names(table: ExtractedTable) -> list[str]:
    """One name per column, joining the header rows: "Pin Number" + "LQFP64" -> "Pin Number LQFP64"."""
    header = table.filled_rows[: table.header_rows]
    names = []
    for j in range(len(header[0])):
        parts: list[str] = []
        for row in header:
            part = clean_cell(row[j])
            if part and part not in parts:  # a cell merged over header rows repeats its text
                parts.append(part)
        names.append(" ".join(parts) or f"Column {j + 1}")
    return names


def row_to_text(names: Sequence[str], row: Sequence[str]) -> str:
    """'Symbol: VDD | Parameter: Standard operating voltage | Min: 1.7 [note 1] | ...'."""
    pairs: list[tuple[str, str]] = []
    for name, cell in zip(names, row):
        value = clean_cell(cell)
        # Skip empty cells, and a pair that repeats the previous one: a header
        # cell merged over columns with no sub-header gives them the same name.
        if value and (name, value) not in pairs[-1:]:
            pairs.append((name, value))
    return " | ".join(f"{name}: {value}" for name, value in pairs)


def find_captioned_tables(pages: Sequence[PageContent]) -> list[CaptionedTable]:
    captioned = []
    for i, page in enumerate(pages):
        on_page = []
        for table in page.tables:
            found = find_caption(table, page.blocks)
            if found:
                number, caption = found
                on_page.append(CaptionedTable(number, caption, table, find_notes(table, page.blocks)))
        # When a table ends near the bottom of the page, its footnotes continue
        # at the top of the next page.
        next_page = pages[i + 1] if i + 1 < len(pages) else None
        if on_page and next_page and next_page.page == page.page + 1:
            lowest = max(on_page, key=lambda part: part.table.bbox[3])
            for number, note in notes_at_page_top(next_page.blocks).items():
                lowest.notes.setdefault(number, note)
        captioned.extend(on_page)
    return captioned


def notes_text(text: str, notes: dict[int, str]) -> str:
    """The footnotes that `text` refers to, as one line."""
    numbers = sorted({int(n) for n in NOTE_REFERENCE.findall(text)} & notes.keys())
    return "Notes: " + " ".join(f"{n}. {notes[n]}" for n in numbers) if numbers else ""


def chunk_text(prefix: str, rows: Sequence[str], notes: dict[int, str]) -> str:
    """The prefix line, one line per table row, then the footnotes these refer to."""
    body = "\n".join([prefix, *rows])
    return "\n".join(filter(None, [body, notes_text(body, notes)]))


def table_chunks(doc: str, pages: Sequence[PageContent], max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Chunks for every captioned table of a document, in page order.

    The pages must come from parse_pdf() with the pdfplumber table extractor,
    which provides the filled rows and header rows used here.
    """
    captioned = find_captioned_tables(pages)
    if any(part.table.filled_rows is None for part in captioned):
        raise ValueError("table chunks need tables from the pdfplumber extractor")
    # A table split over pages prints its footnotes below the last part only,
    # but the references appear in every part: collect them per table number.
    notes_by_table: dict[int, dict[int, str]] = {}
    for part in captioned:
        notes_by_table.setdefault(part.number, {}).update(part.notes)

    chunks = []
    for part in captioned:
        if part.caption.split(". ", 1)[1] in SKIPPED_TABLES:
            continue
        prefix = f"{device_name(doc)} datasheet | {part.caption}"
        notes = notes_by_table[part.number]
        names = column_names(part.table)
        rows = [row_to_text(names, row) for row in part.table.filled_rows[part.table.header_rows :]]
        rows = [row for row in rows if row]

        # Rows are added to a chunk while its text, footnotes included, fits.
        groups: list[list[str]] = []
        for row in rows:
            if groups and len(chunk_text(prefix, groups[-1] + [row], notes)) <= max_chars:
                groups[-1].append(row)
            else:
                groups.append([row])
        for k, group in enumerate(groups, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}-p{part.table.page}-table{part.number}-{k}",
                    doc=doc,
                    page=part.table.page,
                    kind="table",
                    section="",  # filled in from the PDF outline in a later step
                    title=part.caption,
                    text=chunk_text(prefix, group, notes),
                )
            )
    return chunks
