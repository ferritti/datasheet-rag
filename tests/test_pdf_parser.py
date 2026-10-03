"""Tests for the PDF parser, run on small PDFs generated on the fly.

The real datasheets are not in the repository, so the tests build PDFs with
PyMuPDF:
- sample_pdf has two pages: a title and a 3x3 table drawn with lines (like the
  tables in ST datasheets), then plain text only.
- cell_text_pdf has one table row with the three kinds of cell text that the
  extractors handle differently: a subscript, an underscore, rotated text.
- merged_table_pdf has a table with merged cells and two bold header rows.
- make_margin_pdf builds a page with a running header, a footer and body text,
  optionally rotated like the landscape pages of the datasheets.
"""

import pdfplumber
import pymupdf
import pytest

from datasheet_rag.ingestion.pdf_parser import (
    TABLE_EXTRACTORS,
    OutlineEntry,
    extract_tables_pdfplumber,
    parse_pdf,
    read_outline,
)

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


@pytest.fixture
def merged_table_pdf(tmp_path):
    """A 4x3 table: "Symbol" spans header rows 0-1, "Value" spans columns 1-2, "VDD" spans rows 2-3."""
    path = tmp_path / "merged_table.pdf"
    xs, ys = [72, 172, 272, 372], [100, 120, 140, 160, 180]
    with pymupdf.open() as doc:
        page = doc.new_page()
        for y in (100, 140, 180):
            page.draw_line((72, y), (372, y))
        for y in (120, 160):  # no line under "Symbol" and "VDD", which span two rows
            page.draw_line((172, y), (372, y))
        for x in (72, 172, 372):
            page.draw_line((x, 100), (x, 180))
        page.draw_line((272, 120), (272, 180))  # no line inside "Value", which spans two columns
        bold, regular = "hebo", "helv"  # PyMuPDF's names for Helvetica-Bold and Helvetica
        cells = {
            (0, 0): ("Symbol", bold), (0, 1): ("Value", bold), (1, 1): ("Min", bold), (1, 2): ("Max", bold),
            (2, 0): ("VDD", regular), (2, 1): ("1.7", regular), (2, 2): ("3.6", regular),
            (3, 1): ("1.8", regular), (3, 2): ("3.3", regular),
        }
        for (i, j), (text, font) in cells.items():
            page.insert_text((xs[j] + 5, ys[i] + 14), text, fontsize=9, fontname=font)
        doc.save(path)
    return path


def make_margin_pdf(path, rotation: int):
    """One A4 page with a running header, a footer and body text, rotated by `rotation` degrees."""
    with pymupdf.open() as doc:
        page = doc.new_page()  # 595 x 842 pt
        page.insert_text((72, 65), "Electrical characteristics", fontsize=9)
        page.insert_text((72, 750), "DS10086 Rev 5 60/137", fontsize=9)
        page.insert_text((200, 400), "Body text", fontsize=9)
        page.set_rotation(rotation)
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


def test_read_outline(tmp_path):
    path = tmp_path / "outline.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.set_toc([[1, "1 Introduction", 1], [2, "Table 1. Device summary", 2]])
        doc.save(path)
    assert read_outline(path) == [OutlineEntry(1, "1 Introduction", 1), OutlineEntry(2, "Table 1. Device summary", 2)]


def test_rejects_pages_out_of_range(sample_pdf):
    with pytest.raises(ValueError, match=r"pages \[0, 3\] outside 1-2"):
        parse_pdf(sample_pdf, pages=[0, 1, 3])


def test_rejects_unknown_extractor(sample_pdf):
    with pytest.raises(ValueError, match="unknown table extractor"):
        parse_pdf(sample_pdf, table_extractor="camelot")


def test_pdfplumber_fills_merged_cells_and_counts_bold_header_rows(merged_table_pdf):
    [page] = parse_pdf(merged_table_pdf)
    table = page.tables[0]
    assert table.rows == [
        ["Symbol", "Value", None],
        [None, "Min", "Max"],
        ["VDD", "1.7", "3.6"],
        [None, "1.8", "3.3"],
    ]
    assert table.filled_rows == [
        ["Symbol", "Value", "Value"],
        ["Symbol", "Min", "Max"],
        ["VDD", "1.7", "3.6"],
        ["VDD", "1.8", "3.3"],
    ]
    assert table.header_rows == 2


def test_body_row_with_a_bold_label_is_not_a_header_row(tmp_path):
    # Like the thermal characteristics tables: a bold label followed by regular
    # text in the same cell ("Thermal resistance" + " LQFP64").
    path = tmp_path / "bold_label.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        draw_grid(page, (72, 100, 372, 140), n_rows=2, n_cols=2)
        page.insert_text((77, 114), "Parameter", fontsize=9, fontname="hebo")
        page.insert_text((227, 114), "Value", fontsize=9, fontname="hebo")
        page.insert_text((77, 134), "Thermal resistance", fontsize=9, fontname="hebo")
        page.insert_text((77 + pymupdf.get_text_length("Thermal resistance ", fontname="hebo", fontsize=9), 134),
                         "LQFP64", fontsize=9, fontname="helv")
        page.insert_text((227, 134), "32", fontsize=9, fontname="helv")
        doc.save(path)
    [page] = parse_pdf(path)
    assert page.tables[0].header_rows == 1


def test_first_row_is_a_header_row_even_without_bold_text(sample_pdf):
    # sample_pdf's table is set in regular type only.
    pages = parse_pdf(sample_pdf)
    assert pages[0].tables[0].header_rows == 1


def test_blocks_flag_running_header_and_footer(tmp_path):
    [page] = parse_pdf(make_margin_pdf(tmp_path / "margins.pdf", rotation=0))
    assert {block.text: block.in_margin for block in page.blocks} == {
        "Electrical characteristics": True,
        "DS10086 Rev 5 60/137": True,
        "Body text": False,
    }


def test_blocks_on_rotated_page_use_displayed_coordinates(tmp_path):
    # On a landscape page PyMuPDF reports text on the unrotated page; the blocks
    # must use the displayed page's coordinates, like pdfplumber's tables, and
    # still recognise the header and footer (now on the left and right edges).
    path = make_margin_pdf(tmp_path / "rotated.pdf", rotation=90)
    [page] = parse_pdf(path)
    body = next(block for block in page.blocks if block.text == "Body text")
    with pdfplumber.open(path) as pdf:
        # keep_blank_chars keeps "Body text" as one phrase; pdfplumber spells
        # the rotated phrase backwards, which does not matter for its box.
        words = pdf.pages[0].extract_words(keep_blank_chars=True)
        [phrase] = [w for w in words if w["text"] in ("Body text", "txet ydoB")]
    # A few points of slack: PyMuPDF's box includes the font's space above and
    # below the letters; wrong coordinates would be off by hundreds of points.
    assert body.bbox == pytest.approx((phrase["x0"], phrase["top"], phrase["x1"], phrase["bottom"]), abs=3)
    assert [block.in_margin for block in page.blocks if block.text != "Body text"] == [True, True]


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
