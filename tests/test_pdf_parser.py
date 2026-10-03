"""Tests for the PDF parser, run on a small PDF generated on the fly.

The real datasheets are not in the repository, so each test builds a two-page
PDF with PyMuPDF: page 1 has a title and a 3x3 table drawn with lines (like the
tables in ST datasheets), page 2 has plain text only.
"""

import pymupdf
import pytest

from datasheet_rag.ingestion.pdf_parser import TABLE_EXTRACTORS, parse_pdf

TABLE_ROWS = [
    ["Symbol", "Min", "Max"],
    ["VDD", "1.7", "3.6"],
    ["VBAT", "1.65", "3.6"],
]
TABLE_ORIGIN = (72, 100)
COL_WIDTH, ROW_HEIGHT = 100, 20
TABLE_BBOX = (
    TABLE_ORIGIN[0],
    TABLE_ORIGIN[1],
    TABLE_ORIGIN[0] + COL_WIDTH * len(TABLE_ROWS[0]),
    TABLE_ORIGIN[1] + ROW_HEIGHT * len(TABLE_ROWS),
)


def draw_table(page: pymupdf.Page, rows: list[list[str]]) -> None:
    """Draw a grid of lines with one text cell in each box."""
    x0, y0, x1, y1 = TABLE_BBOX
    for i in range(len(rows) + 1):
        y = y0 + i * ROW_HEIGHT
        page.draw_line((x0, y), (x1, y))
    for j in range(len(rows[0]) + 1):
        x = x0 + j * COL_WIDTH
        page.draw_line((x, y0), (x, y1))
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            # insert_text places the text baseline, so 14 pt down sits inside the box.
            page.insert_text((x0 + j * COL_WIDTH + 5, y0 + i * ROW_HEIGHT + 14), cell, fontsize=10)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 80), "Table 1. Operating conditions", fontsize=12)
        draw_table(page, TABLE_ROWS)
        doc.new_page().insert_text((72, 80), "Page without tables", fontsize=12)
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
