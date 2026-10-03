"""Tests for the PDF parser, run on small PDFs generated on the fly.

The real datasheets are not in the repository, so the tests build PDFs with
PyMuPDF:
- sample_pdf has two pages: a title and a 3x3 table drawn with lines (like the
  tables in ST datasheets), then plain text only.
- cell_text_pdf has one table row with the three kinds of cell text that the
  extractors handle differently: a subscript, an underscore, rotated text.
"""

import pdfplumber
import pymupdf
import pytest

from datasheet_rag.ingestion.pdf_parser import TABLE_EXTRACTORS, extract_tables_pdfplumber, parse_pdf

TABLE_ROWS = [
    ["Symbol", "Min", "Max"],
    ["VDD", "1.7", "3.6"],
    ["VBAT", "1.65", "3.6"],
]
COL_WIDTH, ROW_HEIGHT = 100, 20
TABLE_BBOX = (72, 100, 72 + COL_WIDTH * len(TABLE_ROWS[0]), 100 + ROW_HEIGHT * len(TABLE_ROWS))
CELL_TEXT_BBOX = (72, 100, 372, 130)  # one row of three 100 x 30 pt cells


def draw_grid(page: pymupdf.Page, bbox: tuple[float, float, float, float], n_rows: int, n_cols: int) -> None:
    x0, y0, x1, y1 = bbox
    for i in range(n_rows + 1):
        y = y0 + i * (y1 - y0) / n_rows
        page.draw_line((x0, y), (x1, y))
    for j in range(n_cols + 1):
        x = x0 + j * (x1 - x0) / n_cols
        page.draw_line((x, y0), (x, y1))


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    x0, y0 = TABLE_BBOX[:2]
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 80), "Table 1. Operating conditions", fontsize=12)
        draw_grid(page, TABLE_BBOX, n_rows=len(TABLE_ROWS), n_cols=len(TABLE_ROWS[0]))
        for i, row in enumerate(TABLE_ROWS):
            for j, cell in enumerate(row):
                # insert_text places the text baseline, so 14 pt down sits inside the box.
                page.insert_text((x0 + j * COL_WIDTH + 5, y0 + i * ROW_HEIGHT + 14), cell, fontsize=10)
        doc.new_page().insert_text((72, 80), "Page without tables", fontsize=12)
        doc.save(path)
    return path


@pytest.fixture
def cell_text_pdf(tmp_path):
    path = tmp_path / "cell_text.pdf"
    x0, y0 = CELL_TEXT_BBOX[:2]
    with pymupdf.open() as doc:
        page = doc.new_page()
        draw_grid(page, CELL_TEXT_BBOX, n_rows=1, n_cols=3)
        # "VDD" with "DD" as a subscript, sized like in ST datasheets: a 9 pt
        # symbol and a 7.2 pt subscript whose top is about 3.7 pt lower.
        page.insert_text((x0 + 5, y0 + 15), "V", fontsize=9)
        page.insert_text((x0 + 5 + pymupdf.get_text_length("V", fontsize=9), y0 + 17.3), "DD", fontsize=7.2)
        page.insert_text((x0 + 105, y0 + 15), "PWR_CR", fontsize=9)
        # rotate=90 writes the text bottom to top, like the pin table headers.
        page.insert_text((x0 + 215, y0 + 25), "Pin", fontsize=9, rotate=90)
        doc.save(path)
    return path


def test_returns_one_page_content_per_page_numbered_from_one(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert [page.page for page in pages] == [1, 2]


def test_extracts_page_text(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert "Table 1. Operating conditions" in pages[0].text
    assert "VBAT" in pages[0].text
    assert "Page without tables" in pages[1].text


@pytest.mark.parametrize("extractor", TABLE_EXTRACTORS)
def test_extracts_table_cells_and_bbox(sample_pdf, extractor):
    pages = parse_pdf(sample_pdf, table_extractor=extractor)
    assert len(pages[0].tables) == 1
    table = pages[0].tables[0]
    assert table.page == 1
    assert table.rows == TABLE_ROWS
    assert table.bbox == pytest.approx(TABLE_BBOX, abs=1)
    assert pages[1].tables == []


@pytest.mark.parametrize("extractor", TABLE_EXTRACTORS)
def test_pages_argument_selects_pages_by_one_based_number(sample_pdf, extractor):
    pages = parse_pdf(sample_pdf, pages=[2], table_extractor=extractor)
    assert len(pages) == 1
    assert pages[0].page == 2
    assert "Page without tables" in pages[0].text


def test_rejects_pages_out_of_range(sample_pdf):
    with pytest.raises(ValueError, match=r"pages \[0, 3\] outside 1-2"):
        parse_pdf(sample_pdf, pages=[0, 1, 3])


def test_rejects_unknown_extractor(sample_pdf):
    with pytest.raises(ValueError, match="unknown table extractor"):
        parse_pdf(sample_pdf, table_extractor="camelot")


def test_default_extractor_keeps_subscripts_underscores_and_rotated_text(cell_text_pdf):
    [page] = parse_pdf(cell_text_pdf)
    assert page.tables[0].rows == [["VDD", "PWR_CR", "Pin"]]


# The next two tests pin down the library behaviour behind the choices in
# docs/pdf_extraction.md. If one starts failing after a library upgrade, the
# choice of extractor and settings should be checked again.


def test_pdfplumber_defaults_split_subscripts_and_reverse_rotated_text(cell_text_pdf):
    with pdfplumber.open(cell_text_pdf) as pdf:
        tables = extract_tables_pdfplumber(pdf.pages[0], page_number=1, text_settings={})
    assert tables[0].rows == [["V\nDD", "PWR_CR", "niP"]]


def test_pymupdf_moves_underscores_to_their_own_line(cell_text_pdf):
    [page] = parse_pdf(cell_text_pdf, table_extractor="pymupdf")
    assert page.tables[0].rows[0][1] == "PWR CR\n_"
