"""Which section of the datasheet each text block belongs to.

The PDF outline (bookmarks) lists every section heading with its page, and on
all 829 headings of the corpus the heading is printed as a text block of its
own with exactly the outline title. So a block belongs to the last heading
seen before it, reading pages and blocks in order; no guessing from font
sizes is needed.
"""

from collections.abc import Sequence

from datasheet_rag.ingestion.pdf_parser import OutlineEntry, PageContent


def heading_key(text: str) -> str:
    return " ".join(text.split()).lower()


def block_sections(pages: Sequence[PageContent], outline: Sequence[OutlineEntry]) -> dict[int, list[str]]:
    """For each page number, the section title of each of its blocks (same order as page.blocks).

    Blocks before the first heading of the document get "".
    """
    # The outline also lists tables and figures; only headings start sections.
    headings: dict[int, list[OutlineEntry]] = {}
    for entry in outline:
        if not entry.title.startswith(("Table ", "Figure ")):
            headings.setdefault(entry.page, []).append(entry)

    sections: dict[int, list[str]] = {}
    current = ""
    for page in pages:
        pending = {heading_key(entry.title): entry for entry in headings.get(page.page, [])}
        page_sections = []
        for block in page.blocks:
            entry = pending.pop(heading_key(block.text), None)
            if entry:
                current = " ".join(entry.title.split())
            page_sections.append(current)
        sections[page.page] = page_sections
    return sections
