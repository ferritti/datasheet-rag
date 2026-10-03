"""Extract page text and tables from a datasheet PDF.

Text always comes from PyMuPDF. Tables come from pdfplumber by default, which
handled the STM32 datasheet tables better (see docs/pdf_extraction.md). PyMuPDF
can still be used for tables, so the two extractors can be compared.

Page numbers are 1-based, matching what a PDF viewer shows. Citations and the
`page` field of the evaluation set use the same convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pdfplumber
import pymupdf
from pdfplumber.page import Page as PlumberPage
from pdfplumber.table import Table as PlumberTable

TableExtractor = Literal["pymupdf", "pdfplumber"]
TABLE_EXTRACTORS: tuple[TableExtractor, ...] = ("pymupdf", "pdfplumber")

# How pdfplumber turns the characters of a table cell into text, tuned on the
# STM32 datasheets (docs/pdf_extraction.md):
# - y_tolerance=4 (default 3): characters whose tops differ by less than this
#   are on the same line. Subscripts sit about 3.7 pt lower than their symbol,
#   so the default splits "VDD" into "V\nDD"; from 5 up, separate lines of
#   small text start to get mixed together.
# - char_dir_rotated="btt": rotated text (e.g. pin table headers) reads bottom
#   to top; the default order spells it backwards ("niP" for "Pin").
PDFPLUMBER_TEXT_SETTINGS: Mapping[str, Any] = {"y_tolerance": 4, "char_dir_rotated": "btt"}

# Bands of the printed (unrotated) page that hold the running header and
# footer, measured on all 1,274 pages of the corpus (all 842 pt high): header
# blocks end at y = 65-69 while the first body text ends at y >= 90; footer
# blocks start 95-104 pt above the bottom edge, body text at least 120 pt above.
HEADER_BOTTOM = 80
FOOTER_HEIGHT = 110

# (x0, top, x1, bottom) in PDF points, origin at the top-left corner of the page
# as displayed (that is, after any page rotation). pdfplumber and PyMuPDF's
# find_tables() use this convention; PyMuPDF's text positions are converted to it.
BBox = tuple[float, float, float, float]


@dataclass
class TextBlock:
    bbox: BBox
    text: str
    # True for the running header and footer (chapter title, device names,
    # document revision, page number), which repeat on every page.
    in_margin: bool


@dataclass
class ExtractedTable:
    page: int
    bbox: BBox
    # A cell is None where the extractor found no cell of its own, typically
    # inside merged cells. None is kept (not turned into "") so the report can
    # show where that happens.
    rows: list[list[str | None]]
    # Set only by the pdfplumber extractor, the one used for chunking:
    # filled_rows repeats a merged cell's text at every position it covers, and
    # header_rows counts the leading rows set in bold (ST's header rows).
    filled_rows: list[list[str]] | None = None
    header_rows: int = 0


@dataclass
class PageContent:
    page: int
    text: str
    tables: list[ExtractedTable] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)


def extract_text(page: pymupdf.Page) -> str:
    # Text in the order it is stored in the file, which in the ST datasheets is
    # the reading order. sort=True (order by position) was tried and is worse:
    # it interleaves the two columns of the cover page and, on landscape pages,
    # mixes the table caption with the table body.
    return page.get_text("text")


def extract_blocks(page: pymupdf.Page) -> list[TextBlock]:
    """The page's text blocks (groups of lines, roughly paragraphs) in file order."""
    page_height = page.cropbox.height  # height of the unrotated page
    blocks = []
    for x0, y0, x1, y1, text, _block_number, block_type in page.get_text("blocks"):
        if block_type != 0 or not text.strip():  # block_type 1 is an image
            continue
        # The margins are tested on the unrotated page: on landscape pages the
        # header and footer end up on the left and right of the displayed page.
        in_margin = y1 < HEADER_BOTTOM or y0 > page_height - FOOTER_HEIGHT
        # PyMuPDF gives text positions on the unrotated page; rotation_matrix
        # converts them to the displayed page, the coordinates of the tables.
        bbox = tuple(pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix)
        blocks.append(TextBlock(bbox=bbox, text=text.strip(), in_margin=in_margin))
    return blocks


def extract_tables_pymupdf(page: pymupdf.Page, page_number: int) -> list[ExtractedTable]:
    return [
        ExtractedTable(page=page_number, bbox=tuple(table.bbox), rows=table.extract())
        for table in page.find_tables().tables
    ]


def cell_grid(table: PlumberTable) -> tuple[list[float], list[float]]:
    """Column and row boundaries: every cell's left and top edge, plus the table's right and bottom."""
    xs = sorted({cell[0] for cell in table.cells}) + [table.bbox[2]]
    ys = sorted({cell[1] for cell in table.cells}) + [table.bbox[3]]
    return xs, ys


def fill_merged_cells(table: PlumberTable, rows: list[list[str | None]]) -> list[list[str]]:
    """Repeat each merged cell's text at every row and column position it covers.

    pdfplumber reports a merged cell once, at its top-left position, and None at
    the other positions it covers. The cell boxes tell which cell covers a
    position, so this works for cells merged across rows and across columns.
    """
    xs, ys = cell_grid(table)
    # Each cell's box -> the (row, column) where its text is reported.
    origin = {cell: (i, j) for i, row in enumerate(table.rows) for j, cell in enumerate(row.cells) if cell}
    filled = []
    for i, row in enumerate(table.rows):
        filled_row = []
        for j, cell in enumerate(row.cells):
            if cell is None:
                x, y = (xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2
                cell = next((c for c in table.cells if c[0] <= x <= c[2] and c[1] <= y <= c[3]), None)
            if cell is None:
                filled_row.append("")
            else:
                origin_row, origin_col = origin[cell]
                filled_row.append(rows[origin_row][origin_col] or "")
        filled.append(filled_row)
    return filled


def count_header_rows(page: PlumberPage, table: PlumberTable) -> int:
    """Number of header rows: the leading rows whose letters are all bold, and at least 1.

    Only letters are checked because header rows can hold regular digits and
    symbols ("-", an "Ω" from the Symbol font, footnote markers like "(12)").
    "All" rather than "most": some body rows start with a bold label followed
    by regular text, and are up to 82% bold. Every ST table, and every
    "continued" part of one, starts with a header row, hence the minimum of 1:
    it covers the rare header with regular letters (one table in the corpus
    has "Min Typ Max Unit" in regular type).
    """
    xs, ys = cell_grid(table)
    x0, top, x1, bottom = table.bbox
    letters = [c for c in page.chars if c["text"].isascii() and c["text"].isalpha() and x0 <= c["x0"] and c["x1"] <= x1]
    count = 0
    for row_top, row_bottom in zip(ys, ys[1:]):
        row_letters = [c for c in letters if row_top <= (c["top"] + c["bottom"]) / 2 < row_bottom]
        if not row_letters or not all("Bold" in c["fontname"] for c in row_letters):
            break
        count += 1
    return max(count, 1)


def extract_tables_pdfplumber(
    page: PlumberPage,
    page_number: int,
    text_settings: Mapping[str, Any] = PDFPLUMBER_TEXT_SETTINGS,
) -> list[ExtractedTable]:
    """Find the tables on a page; text_settings={} gives pdfplumber's defaults."""
    tables = []
    # find_tables() rather than extract_tables(): only the former also returns
    # each table's bounding box and cell boxes.
    for table in page.find_tables():
        rows = table.extract(**text_settings)
        tables.append(
            ExtractedTable(
                page=page_number,
                bbox=tuple(table.bbox),
                rows=rows,
                filled_rows=fill_merged_cells(table, rows),
                header_rows=count_header_rows(page, table),
            )
        )
    return tables


def parse_pdf(
    pdf_path: str | Path,
    pages: Sequence[int] | None = None,
    table_extractor: TableExtractor = "pdfplumber",
) -> list[PageContent]:
    """Parse the text and tables of a PDF, one PageContent per page.

    Args:
        pdf_path: path to the PDF file.
        pages: 1-based page numbers to parse; all pages if None.
        table_extractor: library used to find tables. One call uses a single
            extractor, so comparing them means calling this function once per
            extractor (which also lets the caller time each one).
    """
    if table_extractor not in TABLE_EXTRACTORS:
        raise ValueError(f"unknown table extractor {table_extractor!r}, expected one of {TABLE_EXTRACTORS}")

    # pdfplumber.open() is lazy: it only parses the pages that are accessed, so
    # opening it costs nothing when PyMuPDF is the table extractor.
    with pymupdf.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumber_doc:
        page_numbers = list(pages) if pages is not None else list(range(1, doc.page_count + 1))
        out_of_range = [p for p in page_numbers if not 1 <= p <= doc.page_count]
        if out_of_range:
            raise ValueError(f"{pdf_path}: pages {out_of_range} outside 1-{doc.page_count}")

        results = []
        for page_number in page_numbers:
            mupdf_page = doc[page_number - 1]
            if table_extractor == "pymupdf":
                tables = extract_tables_pymupdf(mupdf_page, page_number)
            else:
                plumber_page = plumber_doc.pages[page_number - 1]
                tables = extract_tables_pdfplumber(plumber_page, page_number)
                # pdfplumber caches every parsed character and line of a page;
                # freeing it keeps memory flat on 200-page datasheets.
                plumber_page.close()
            results.append(
                PageContent(
                    page=page_number,
                    text=extract_text(mupdf_page),
                    tables=tables,
                    blocks=extract_blocks(mupdf_page),
                )
            )
    return results
