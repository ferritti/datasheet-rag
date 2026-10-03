"""Tests for text chunking, on hand-built pages (no PDF needed)."""

from datasheet_rag.ingestion.pdf_parser import ExtractedTable, PageContent, TextBlock
from datasheet_rag.ingestion.text_chunks import is_unreadable, pack, split_long_text, text_chunks

SECTION = "6.3.2 VCAP_1/VCAP_2 external capacitors"


def block(text: str, top: float, bottom: float, in_margin: bool = False) -> TextBlock:
    return TextBlock(bbox=(67, top, 528, bottom), text=text, in_margin=in_margin)


def test_keeps_running_text_and_drops_what_is_not():
    blocks = [
        block("Electrical characteristics\nSTM32F401xD STM32F401xE", 59, 70, in_margin=True),
        block("6.3.2 \nVCAP_1/VCAP_2 external capacitors", 90, 104),  # the heading: goes into the prefix
        block("Stabilization for the main regulator is achieved by connecting\nexternal capacitor CEXT.", 110, 130),
        block("Table 16. VCAP_1/VCAP_2 operating conditions", 140, 150),  # caption of a kept table
        block("Symbol Parameter Min Max", 155, 165),  # inside the table
        block("1.\nWhen bypassing the voltage regulator.", 204, 214),  # the table's footnote
        block("Figure 20. External capacitor CEXT", 240, 250),  # figure caption: kept
        block("ESR\nR Leak", 270, 300),  # label inside the figure
        block("MS19044V2", 320, 328),  # ST's figure ID: end of the figure
        block("1. Legend: ESR is the equivalent series resistance.", 335, 345),  # below the figure: kept
        block("3(\x16 3(\x14", 400, 410),  # unreadable font
        block("60/137\nDS10086 Rev 5", 743, 753, in_margin=True),
    ]
    table = ExtractedTable(page=62, bbox=(67, 152, 528, 200), rows=[["a"]], filled_rows=[["Symbol"], ["CEXT"]],
                           header_rows=1)
    pages = [PageContent(page=62, text="", tables=[table], blocks=blocks)]
    sections = {62: [""] + [SECTION] * (len(blocks) - 1)}
    [chunk] = text_chunks("stm32f401re", pages, sections)
    assert (chunk.chunk_id, chunk.page, chunk.kind, chunk.section) == ("stm32f401re-p62-text-1", 62, "text", SECTION)
    assert chunk.text == (
        f"STM32F401RE datasheet | {SECTION}\n"
        "Stabilization for the main regulator is achieved by connecting external capacitor CEXT.\n"
        "Figure 20. External capacitor CEXT\n"
        "1. Legend: ESR is the equivalent series resistance."
    )


def test_sections_and_pages_start_new_chunks():
    pages = [
        PageContent(page=20, text="", blocks=[block("Para A", 100, 110), block("Para B", 120, 130)]),
        PageContent(page=21, text="", blocks=[block("Para C", 100, 110)]),
    ]
    sections = {20: ["3.1 Clocks", "3.2 Boot modes"], 21: ["3.2 Boot modes"]}
    chunks = text_chunks("stm32f401re", pages, sections)
    assert [(c.chunk_id, c.text) for c in chunks] == [
        ("stm32f401re-p20-text-1", "STM32F401RE datasheet | 3.1 Clocks\nPara A"),
        ("stm32f401re-p20-text-2", "STM32F401RE datasheet | 3.2 Boot modes\nPara B"),
        ("stm32f401re-p21-text-1", "STM32F401RE datasheet | 3.2 Boot modes\nPara C"),
    ]


def test_contents_pages_are_skipped():
    pages = [PageContent(page=3, text="6.3.1 General operating conditions . . . . . . . 60",
                         blocks=[block("6.3.1 General operating conditions . . . . . . . 60", 100, 110)])]
    assert text_chunks("stm32f401re", pages, {3: [""]}) == []


def test_pack_fills_chunks_up_to_the_limit():
    texts = pack("PREFIX", ["a" * 40, "b" * 40, "c" * 40], max_chars=100)
    assert texts == ["PREFIX\n" + "a" * 40 + "\n" + "b" * 40, "PREFIX\n" + "c" * 40]


def test_long_paragraphs_are_split_at_sentence_ends():
    sentence = "The regulator needs an external capacitor. "
    pieces = split_long_text(sentence * 10, max_chars=100)
    assert all(len(piece) <= 100 for piece in pieces)
    assert " ".join(pieces) == (sentence * 10).strip()


def test_is_unreadable():
    assert is_unreadable("3(\x16 3(\x14")
    assert is_unreadable("(cid:51)(cid:40)")
    assert not is_unreadable("VDD = 1.7 to 3.6 V\n2 wait states")
