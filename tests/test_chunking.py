"""Tests for the fixed-size baseline and for building chunks from a PDF."""

import pymupdf
import pytest

from datasheet_rag.ingestion.chunking import build_chunks
from datasheet_rag.ingestion.fixed_chunks import fixed_chunks, windows
from datasheet_rag.ingestion.pdf_parser import PageContent

WORDS = " ".join(f"word{i}" for i in range(200))


def test_windows_respect_the_size_and_overlap_at_word_boundaries():
    pieces = windows(WORDS, size=100, overlap=30)
    assert all(len(piece) <= 100 for piece in pieces)
    assert all(piece == piece.strip() for piece in pieces)
    assert set(" ".join(pieces).split()) == set(WORDS.split())  # nothing lost
    for previous, piece in zip(pieces, pieces[1:]):
        assert piece.split()[0] in previous.split()  # each piece starts inside the previous one


def test_windows_of_short_and_empty_text():
    assert windows("short text", size=100, overlap=30) == ["short text"]
    assert windows("", size=100, overlap=30) == []


def test_fixed_chunks_never_cross_pages_and_start_with_the_device_name():
    pages = [PageContent(page=1, text=WORDS), PageContent(page=2, text="Last page.")]
    chunks = fixed_chunks("stm32f401re", pages, max_chars=200, overlap=30)
    assert all(len(chunk.text) <= 200 and chunk.text.startswith("STM32F401RE datasheet\n") for chunk in chunks)
    assert chunks[0].chunk_id == "stm32f401re-p1-fixed-1"
    assert chunks[-1].chunk_id == "stm32f401re-p2-fixed-1"
    assert chunks[-1].text == "STM32F401RE datasheet\nLast page."


@pytest.fixture
def datasheet_pdf(tmp_path):
    """A one-page 'datasheet': a section heading, a paragraph and a captioned 2x3 table."""
    path = tmp_path / "DS_stm32f401re.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 100), "6.3.1 General operating conditions", fontsize=11, fontname="hebo")
        page.insert_text((72, 130), "The device operates from 1.7 V to 3.6 V.", fontsize=9)
        page.insert_text((150, 165), "Table 14. General operating conditions", fontsize=9, fontname="hebo")
        for y in (170, 190, 210):
            page.draw_line((72, y), (372, y))
        for x in (72, 172, 272, 372):
            page.draw_line((x, 170), (x, 210))
        for j, (header, value) in enumerate([("Symbol", "VDD"), ("Min", "1.7"), ("Max", "3.6")]):
            page.insert_text((77 + 100 * j, 184), header, fontsize=9, fontname="hebo")
            page.insert_text((77 + 100 * j, 204), value, fontsize=9)
        # PyMuPDF requires the outline to start at level 1.
        doc.set_toc([[1, "6.3.1 General operating conditions", 1], [2, "Table 14. General operating conditions", 1]])
        doc.save(path)
    return path


def test_build_structured_chunks(datasheet_pdf):
    chunks = build_chunks(datasheet_pdf, "structured")
    assert [(chunk.chunk_id, chunk.kind) for chunk in chunks] == [
        ("stm32f401re-p1-table14-1", "table"),
        ("stm32f401re-p1-text-1", "text"),
    ]
    table, text = chunks
    assert table.text == (
        "STM32F401RE datasheet | 6.3.1 General operating conditions | Table 14. General operating conditions\n"
        "Symbol: VDD | Min: 1.7 | Max: 3.6"
    )
    assert text.text == (
        "STM32F401RE datasheet | 6.3.1 General operating conditions\nThe device operates from 1.7 V to 3.6 V."
    )


def test_build_fixed_chunks(datasheet_pdf):
    [chunk] = build_chunks(datasheet_pdf, "fixed")
    assert chunk.chunk_id == "stm32f401re-p1-fixed-1"
    assert "Table 14. General operating conditions" in chunk.text and "VDD" in chunk.text


def test_unknown_strategy(datasheet_pdf):
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        build_chunks(datasheet_pdf, "semantic")
