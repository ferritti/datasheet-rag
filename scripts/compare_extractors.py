"""Compare PyMuPDF and pdfplumber table extraction on the datasheets.

Reproduces the numbers in docs/pdf_extraction.md. For each PDF it counts:
- captions: "Table N. ..." lines in the page text, a rough count of the real
  tables (a table split over pages has a "(continued)" caption on each page);
- caption pages without a table: pages with a caption where an extractor
  found no table at all;
- matched tables: found by both extractors (bounding boxes overlapping by more
  than half), and how many of them have exactly the same cells;
- tables found by only one extractor;
- cells where an extractor moves an underscore to a line of its own;
- cells that PDFPLUMBER_TEXT_SETTINGS changes compared with pdfplumber's
  defaults. --samples N prints N of them at random, to check them by eye.

Usage (from the repository root):
    python scripts/compare_extractors.py
    python scripts/compare_extractors.py data/raw/DS_stm32f401re.pdf --samples 25
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pdfplumber
import pymupdf

from datasheet_rag.ingestion.pdf_parser import (
    BBox,
    ExtractedTable,
    extract_tables_pdfplumber,
    extract_tables_pymupdf,
    extract_text,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CAPTION = re.compile(r"^\s*Table \d+\.\s+\S")
# A line that is only an underscore: what PyMuPDF makes of "PWR_CR" ("PWR CR\n_").
DETACHED_UNDERSCORE = re.compile(r"(^|\n)_($|\n)")
MATCH_IOU = 0.5


def find_captions(text: str) -> list[str]:
    # The list of tables at the start of a datasheet repeats every caption with
    # dot leaders up to the page number; those pages are not table pages.
    if ". . . ." in text:
        return []
    return [line.strip() for line in text.splitlines() if CAPTION.match(line)]


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two boxes: 1 for identical boxes, 0 if disjoint."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    intersection = max(width, 0) * max(height, 0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_tables(
    tables_a: list[ExtractedTable], tables_b: list[ExtractedTable]
) -> list[tuple[ExtractedTable, ExtractedTable]]:
    """Pair each table in tables_a with the unused table in tables_b that overlaps it most."""
    pairs, used = [], set()
    for table in tables_a:
        candidates = [(bbox_iou(table.bbox, other.bbox), j) for j, other in enumerate(tables_b) if j not in used]
        best_iou, best_j = max(candidates, default=(0.0, None))
        if best_iou > MATCH_IOU:
            used.add(best_j)
            pairs.append((table, tables_b[best_j]))
    return pairs


def cells(tables: list[ExtractedTable]) -> list[str | None]:
    return [cell for table in tables for row in table.rows for cell in row]


def count_detached_underscores(tables: list[ExtractedTable]) -> int:
    return sum(1 for cell in cells(tables) if cell and DETACHED_UNDERSCORE.search(cell))


def compare_pdf(pdf_path: Path) -> tuple[Counter, dict[str, list[int]], list[tuple[int, str, str]]]:
    """Return the counts, the caption pages without a table, and the cells changed by the settings."""
    counts = Counter()
    pages_without_table = {"pymupdf": [], "pdfplumber": []}
    changed_cells = []  # (page, text with pdfplumber defaults, text with our settings)

    with pymupdf.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumber_doc:
        for page_number in range(1, doc.page_count + 1):
            mupdf_page = doc[page_number - 1]
            plumber_page = plumber_doc.pages[page_number - 1]
            tables = {
                "pymupdf": extract_tables_pymupdf(mupdf_page, page_number),
                "pdfplumber": extract_tables_pdfplumber(plumber_page, page_number),
            }
            default_tables = extract_tables_pdfplumber(plumber_page, page_number, text_settings={})
            plumber_page.close()

            captions = find_captions(extract_text(mupdf_page))
            counts["pages"] += 1
            counts["captions"] += len(captions)
            counts["continued"] += sum("(continued)" in caption for caption in captions)
            for extractor, page_tables in tables.items():
                counts[f"tables {extractor}"] += len(page_tables)
                counts[f"detached _ {extractor}"] += count_detached_underscores(page_tables)
                if captions and not page_tables:
                    pages_without_table[extractor].append(page_number)

            pairs = match_tables(tables["pymupdf"], tables["pdfplumber"])
            counts["matched"] += len(pairs)
            counts["identical"] += sum(a.rows == b.rows for a, b in pairs)

            # find_tables() does not depend on the text settings, so the two
            # lists hold the same tables in the same order.
            for default_cell, cell in zip(cells(default_tables), cells(tables["pdfplumber"])):
                counts["cells pdfplumber"] += 1
                if default_cell != cell:
                    changed_cells.append((page_number, default_cell, cell))
    counts["changed by settings"] = len(changed_cells)
    return counts, pages_without_table, changed_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="*", type=Path, help="datasheet PDFs (default: all in data/raw)")
    parser.add_argument("--samples", type=int, default=0, help="print N random cells changed by the settings")
    args = parser.parse_args()
    pdf_paths = args.pdfs or sorted(RAW_DIR.glob("*.pdf"))

    pymupdf.no_recommend_layout()
    results = {pdf_path.stem: compare_pdf(pdf_path) for pdf_path in pdf_paths}
    total = sum((counts for counts, _, _ in results.values()), Counter())

    print("| Datasheet | Pages | Captions | Continued | Tables PyMuPDF | Tables pdfplumber "
          "| Matched | Identical | Only PyMuPDF | Only pdfplumber |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, c in [(stem, counts) for stem, (counts, _, _) in results.items()] + [("**Total**", total)]:
        print(f"| {name} | {c['pages']} | {c['captions']} | {c['continued']} | {c['tables pymupdf']} "
              f"| {c['tables pdfplumber']} | {c['matched']} | {c['identical']} "
              f"| {c['tables pymupdf'] - c['matched']} | {c['tables pdfplumber'] - c['matched']} |")

    print("\n| Datasheet | Cells (pdfplumber) | Detached `_` PyMuPDF | Detached `_` pdfplumber | Changed by settings |")
    print("|---|---|---|---|---|")
    for name, c in [(stem, counts) for stem, (counts, _, _) in results.items()] + [("**Total**", total)]:
        print(f"| {name} | {c['cells pdfplumber']} | {c['detached _ pymupdf']} "
              f"| {c['detached _ pdfplumber']} | {c['changed by settings']} |")

    print("\nCaption pages without a table (PyMuPDF / pdfplumber):")
    for stem, (_, without_table, _) in results.items():
        print(f"- {stem}: {without_table['pymupdf']} / {without_table['pdfplumber']}")

    if args.samples:
        all_changed = [(stem, *cell) for stem, (_, _, changed) in results.items() for cell in changed]
        print(f"\n{args.samples} random cells changed by the settings (default -> ours):")
        for stem, page, before, after in random.Random(0).sample(all_changed, min(args.samples, len(all_changed))):
            print(f"- {stem} p{page}: {before!r} -> {after!r}")


if __name__ == "__main__":
    main()
