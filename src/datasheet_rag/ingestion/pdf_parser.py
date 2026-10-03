"""Extract page text and tables from a datasheet PDF.

Text always comes from PyMuPDF. Tables can come from PyMuPDF or pdfplumber:
both are kept for now so we can compare them on the real STM32 datasheets and
keep the one that handles their tables better.

Page numbers are 1-based, matching what a PDF viewer shows. Citations and the
`page` field of the evaluation set use the same convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber
import pymupdf
from pdfplumber.page import Page as PlumberPage

TableExtractor = Literal["pymupdf", "pdfplumber"]
TABLE_EXTRACTORS: tuple[TableExtractor, ...] = ("pymupdf", "pdfplumber")

# (x0, top, x1, bottom) in PDF points, origin at the top-left corner of the page.
# Both libraries use this convention, so boxes from the two extractors can be
# compared directly.
BBox = tuple[float, float, float, float]


@dataclass
class ExtractedTable:
    page: int
    bbox: BBox
    # A cell is None where the extractor found no cell of its own, typically
    # inside merged cells. None is kept (not turned into "") so the report can
    # show where that happens.
    rows: list[list[str | None]]


@dataclass
class PageContent:
    page: int
    text: str
    tables: list[ExtractedTable] = field(default_factory=list)


def extract_text(page: pymupdf.Page) -> str:
    # sort=True orders text blocks by position (top to bottom, then left to
    # right) instead of the order they are stored in the file, which is not
    # always the reading order.
    return page.get_text("text", sort=True)


def extract_tables_pymupdf(page: pymupdf.Page, page_number: int) -> list[ExtractedTable]:
    return [
        ExtractedTable(page=page_number, bbox=tuple(table.bbox), rows=table.extract())
        for table in page.find_tables().tables
    ]


def extract_tables_pdfplumber(page: PlumberPage, page_number: int) -> list[ExtractedTable]:
    # find_tables() rather than extract_tables(): only the former also returns
    # each table's bounding box.
    return [
        ExtractedTable(page=page_number, bbox=tuple(table.bbox), rows=table.extract())
        for table in page.find_tables()
    ]


def parse_pdf(
    pdf_path: str | Path,
    pages: Sequence[int] | None = None,
    table_extractor: TableExtractor = "pymupdf",
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
            results.append(PageContent(page=page_number, text=extract_text(mupdf_page), tables=tables))
    return results
