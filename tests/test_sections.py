"""Tests for assigning text blocks to outline sections."""

from datasheet_rag.ingestion.pdf_parser import OutlineEntry, PageContent, TextBlock
from datasheet_rag.ingestion.sections import block_sections


def page(number: int, *texts: str) -> PageContent:
    return PageContent(page=number, text="", blocks=[TextBlock((67, 100, 528, 110), t, False) for t in texts])


def test_blocks_belong_to_the_last_heading_before_them_across_pages():
    outline = [
        OutlineEntry(1, "6 Electrical characteristics", 59),
        OutlineEntry(2, "6.3 Operating conditions", 60),
        OutlineEntry(3, "6.3.1 General operating conditions", 60),
        OutlineEntry(3, "Table 14. General operating conditions", 60),  # tables do not start sections
    ]
    pages = [
        page(58, "Text before the first heading"),
        page(59, "6 \nElectrical characteristics", "Introduction to the chapter"),
        page(60, "End of the chapter introduction", "6.3 \nOperating conditions",
             "6.3.1 \nGeneral operating conditions", "Table 14. General operating conditions"),
        page(61, "Text continued on the next page"),
    ]
    assert block_sections(pages, outline) == {
        58: [""],
        59: ["6 Electrical characteristics", "6 Electrical characteristics"],
        60: ["6 Electrical characteristics", "6.3 Operating conditions",
             "6.3.1 General operating conditions", "6.3.1 General operating conditions"],
        61: ["6.3.1 General operating conditions"],
    }


def test_a_heading_only_starts_a_section_on_its_own_page():
    # The same words on another page (e.g. a cross-reference) are not the heading.
    outline = [OutlineEntry(2, "3.1 Clocks", 20)]
    pages = [page(19, "3.1 Clocks"), page(20, "3.1 Clocks", "Body")]
    assert block_sections(pages, outline) == {19: [""], 20: ["3.1 Clocks", "3.1 Clocks"]}
