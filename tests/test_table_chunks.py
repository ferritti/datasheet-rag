"""Tests for table chunking, on hand-built pages (no PDF needed).

Positions mimic the measurements on the ST datasheets: a caption 2 pt above
its table, footnotes 2-7 pt apart, the next paragraph 15 pt or more below.
"""

import pytest

from datasheet_rag.ingestion.pdf_parser import ExtractedTable, PageContent, TextBlock
from datasheet_rag.ingestion.table_chunks import (
    clean_cell,
    column_names,
    find_caption,
    find_notes,
    notes_at_page_top,
    row_to_text,
    table_chunks,
)

HEADER = ["Symbol", "Parameter", "Min", "Max", "Unit"]


def block(text: str, top: float, bottom: float, in_margin: bool = False) -> TextBlock:
    return TextBlock(bbox=(67, top, 528, bottom), text=text, in_margin=in_margin)


def table(page: int, top: float, bottom: float, filled_rows: list[list[str]], header_rows: int = 1) -> ExtractedTable:
    return ExtractedTable(page=page, bbox=(67, top, 528, bottom), rows=filled_rows, filled_rows=filled_rows,
                          header_rows=header_rows)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("1.7(1)", "1.7 [note 1]"),
        ("VDDA\n(2)(3)", "VDDA [note 2] [note 3]"),
        ("USART2_\nCTS", "USART2_CTS"),
        ("TIM2_CH1/\nTIM2_ETR", "TIM2_CH1/TIM2_ETR"),
        ("Power Scale3: Regulator ON,\nVOS[1:0] = 0x01", "Power Scale3: Regulator ON, VOS[1:0] = 0x01"),
        ("–0.3", "-0.3"),
        ("THCLK−0.5", "THCLK-0.5"),
        ("2.7 – 3.6", "2.7 – 3.6"),  # a dash between spaces is a range, not a minus sign
        ("(7 × 7 mm)", "(7 × 7 mm)"),  # only numbers in brackets are footnote references
        ("(cid:51)(cid:40)", ""),
    ],
)
def test_clean_cell(raw, clean):
    assert clean_cell(raw) == clean


def test_column_names_join_header_rows():
    pin_table = table(38, 100, 300, [
        ["Pin Number", "Pin Number", "Pin name", ""],
        ["LQFP64", "LQFP100", "Pin name", ""],
        ["1", "2", "PE2", "-"],
    ], header_rows=2)
    assert column_names(pin_table) == ["Pin Number LQFP64", "Pin Number LQFP100", "Pin name", "Column 4"]


def test_row_to_text_skips_empty_cells_and_repeated_pairs():
    # "millimeters" spans the two value columns with no sub-header, so both get
    # the same name and the merged cell's text: it is written once.
    names = ["Symbol", "millimeters", "millimeters", "Typ", "Max", "Note"]
    assert row_to_text(names, ["A", "1.0", "1.0", "-", "-", ""]) == "Symbol: A | millimeters: 1.0 | Typ: - | Max: -"


def test_find_caption_takes_the_table_caption_just_above():
    t = table(61, 103, 226, [HEADER])
    blocks = [
        block("Table 13. Thermal characteristics", 20, 30),  # too far above
        block("Table 14. General operating conditions (continued)", 90, 101),
        block("STM32F401xD STM32F401xE", 59, 70, in_margin=True),
    ]
    number, caption, caption_block = find_caption(t, blocks)
    assert (number, caption) == (14, "Table 14. General operating conditions")
    assert caption_block is blocks[1]


def test_find_caption_returns_none_for_tables_in_figures():
    t = table(36, 150, 600, [["PE2", "1"]])
    assert find_caption(t, [block("Figure 13. LQFP100 pinout", 130, 141)]) is None


def test_find_notes_reads_consecutive_footnotes_and_their_continuations():
    t = table(61, 103, 226, [HEADER])
    blocks = [
        block("1.\nGuaranteed by design.", 230, 240),
        block("2.\nWhen the ADC is used, refer", 243, 252),
        block("to Table 66.", 251, 260),  # a footnote continued in the next block
        block("Note:\nThis paragraph is not a footnote.", 275, 290),  # 15 pt below
    ]
    notes, note_blocks = find_notes(t, blocks)
    assert notes == {1: "Guaranteed by design.", 2: "When the ADC is used, refer to Table 66."}
    assert note_blocks == blocks[:3]


def test_notes_at_page_top():
    blocks = [
        block("DS10086 Rev 5 62/137", 743, 753, in_margin=True),
        block("Notes:", 90, 102),
        block("1.\nEvaluated by characterization.", 110, 121),
        block("2.\nGuaranteed by design.", 125, 135),
        block("6.3.2 VCAP_1/VCAP_2 external capacitors", 180, 194),
    ]
    notes, note_blocks = notes_at_page_top(blocks)
    assert notes == {1: "Evaluated by characterization.", 2: "Guaranteed by design."}
    assert note_blocks == blocks[1:4]
    assert notes_at_page_top([block("6.3.2 VCAP_1/VCAP_2 external capacitors", 96, 110)]) == ({}, [])


def test_table_split_over_pages_gets_its_footnotes_in_every_part():
    pages = [
        PageContent(page=60, text="", tables=[table(60, 159, 300, [
            HEADER,
            ["VDD", "Standard operating voltage", "1.7(1)", "3.6", "V"],
        ])], blocks=[block("Table 14. General operating conditions", 146, 157)]),
        PageContent(page=61, text="", tables=[table(61, 103, 200, [
            HEADER,
            ["TA", "Ambient temperature(2)", "–40", "85", "°C"],
        ])], blocks=[
            block("Table 14. General operating conditions (continued)", 90, 101),
            block("1.\nVDD minimum value with an external supervisor.", 204, 214),
            block("2.\nSee the thermal characteristics.", 217, 227),
        ]),
    ]
    chunks = table_chunks("stm32f401re", pages)
    assert [(c.chunk_id, c.page, c.kind, c.title) for c in chunks] == [
        ("stm32f401re-p60-table14-1", 60, "table", "Table 14. General operating conditions"),
        ("stm32f401re-p61-table14-1", 61, "table", "Table 14. General operating conditions"),
    ]
    assert chunks[0].text == (
        "STM32F401RE datasheet | Table 14. General operating conditions\n"
        "Symbol: VDD | Parameter: Standard operating voltage | Min: 1.7 [note 1] | Max: 3.6 | Unit: V\n"
        "Notes: 1. VDD minimum value with an external supervisor."
    )
    assert chunks[1].text.endswith("Notes: 2. See the thermal characteristics.")


def test_section_of_the_caption_goes_into_the_prefix():
    blocks = [block("6.3.1 General operating conditions", 120, 133), block("Table 14. General operating conditions", 146, 157)]
    pages = [PageContent(page=60, text="", blocks=blocks, tables=[table(60, 159, 300, [
        HEADER, ["VDD", "Standard operating voltage", "1.7", "3.6", "V"],
    ])])]
    sections = {60: ["6.3.1 General operating conditions", "6.3.1 General operating conditions"]}
    [chunk] = table_chunks("stm32f401re", pages, sections)
    assert chunk.section == "6.3.1 General operating conditions"
    assert chunk.text.startswith(
        "STM32F401RE datasheet | 6.3.1 General operating conditions | Table 14. General operating conditions\n"
    )


def test_long_tables_are_split_into_chunks_that_each_start_with_the_caption():
    rows = [HEADER] + [[f"V{i}", "Some voltage", "1.0", "2.0", "V"] for i in range(10)]
    pages = [PageContent(page=5, text="", tables=[table(5, 120, 700, rows)],
                         blocks=[block("Table 3. Voltages", 107, 118)])]
    chunks = table_chunks("stm32f401re", pages, max_chars=250)
    assert len(chunks) > 1
    assert all(c.text.startswith("STM32F401RE datasheet | Table 3. Voltages\nSymbol: V") for c in chunks)
    assert all(len(c.text) <= 250 for c in chunks)
    assert sum(c.text.count("Parameter: Some voltage") for c in chunks) == 10  # every row exactly once


def test_revision_history_is_skipped():
    pages = [PageContent(page=136, text="", tables=[table(136, 120, 400, [
        ["Date", "Revision", "Changes"],
        ["06-Mar-2015", "5", "Updated Table 14: Current characteristics."],
    ])], blocks=[block("Table 89. Document revision history", 107, 118)])]
    assert table_chunks("stm32f401re", pages) == []


def test_table_chunks_need_pdfplumber_tables():
    pymupdf_table = ExtractedTable(page=5, bbox=(67, 120, 528, 300), rows=[HEADER, HEADER])
    pages = [PageContent(page=5, text="", tables=[pymupdf_table], blocks=[block("Table 3. Voltages", 107, 118)])]
    with pytest.raises(ValueError, match="pdfplumber"):
        table_chunks("stm32f401re", pages)
