"""Write a page-by-page report of the text and tables extracted from datasheet PDFs.

The report is for checking extraction quality by eye: for each page it shows the
raw text from PyMuPDF and the tables found by PyMuPDF and by pdfplumber,
optionally under an image of the page to compare against.

Usage (from the repository root):
    python scripts/inspect_pdf.py data/raw/DS_stm32f401re.pdf --pages 1-40 --render-pages

Output: data/reports/<pdf name>/report.md (+ pages/page_NNN.png with --render-pages).
"""

import argparse
import time
from pathlib import Path

import pymupdf

from datasheet_rag.ingestion.pdf_parser import (
    PDFPLUMBER_TEXT_SETTINGS,
    TABLE_EXTRACTORS,
    ExtractedTable,
    PageContent,
    parse_pdf,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
# High enough to read the small fonts used in datasheet tables, low enough to
# keep a page image around 150 KB.
PAGE_IMAGE_DPI = 110
EXTRACTOR_NAMES = {"pymupdf": "PyMuPDF", "pdfplumber": "pdfplumber"}


def parse_page_range(spec: str) -> list[int]:
    """Turn a spec like "1-3,7" into sorted, unique page numbers: [1, 2, 3, 7]."""
    pages = set()
    for part in spec.split(","):
        start, _, end = part.strip().partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return sorted(pages)


def format_cell(cell: str | None) -> str:
    # ∅ marks cells the extractor returned as None (usually inside merged
    # cells), to tell them apart from cells that exist but contain no text.
    if cell is None:
        return "∅"
    return cell.replace("<", "&lt;").replace("|", "\\|").replace("\n", "<br>")


def table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return "_(no rows)_"
    n_cols = max(len(row) for row in rows)
    # Column numbers as the header instead of the table's first row: whether
    # that row really is the header is part of what we want to check.
    lines = [
        "| " + " | ".join(str(i) for i in range(1, n_cols + 1)) + " |",
        "|" + "---|" * n_cols,
    ]
    for row in rows:
        cells = [format_cell(cell) for cell in row] + [""] * (n_cols - len(row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def empty_cell_ratio(tables: list[ExtractedTable]) -> float | None:
    """Share of cells that are None or blank, over all the given tables."""
    cells = [cell for table in tables for row in table.rows for cell in row]
    if not cells:
        return None
    return sum(1 for cell in cells if cell is None or not cell.strip()) / len(cells)


def format_ratio(ratio: float | None) -> str:
    return "" if ratio is None else f"{ratio:.0%}"


def table_section(index: int, table: ExtractedTable) -> list[str]:
    n_cols = max((len(row) for row in table.rows), default=0)
    bbox = ", ".join(f"{v:.0f}" for v in table.bbox)
    return [
        f"**Table {index}**: {len(table.rows)}×{n_cols}, bbox ({bbox}), "
        f"{format_ratio(empty_cell_ratio([table]))} empty cells",
        "",
        table_to_markdown(table.rows),
        "",
    ]


def page_section(page_number: int, text: str, tables: dict[str, list[ExtractedTable]], image: str | None) -> list[str]:
    lines = [f"## Page {page_number}", ""]
    if image:
        lines += [f"![Page {page_number}]({image})", ""]
    # <details> keeps long page texts collapsed so the tables stay easy to scan.
    lines += [
        f"<details><summary>Raw text ({len(text):,} chars)</summary>",
        "",
        "```text",
        text.rstrip(),
        "```",
        "",
        "</details>",
        "",
    ]
    for extractor, page_tables in tables.items():
        lines += [f"### {EXTRACTOR_NAMES[extractor]}: {len(page_tables)} tables", ""]
        for index, table in enumerate(page_tables, start=1):
            lines += table_section(index, table)
    return lines


def build_report(
    pdf_path: Path,
    page_count: int,
    pages_spec: str | None,
    results: dict[str, list[PageContent]],
    timings: dict[str, float],
    images: dict[int, str],
) -> str:
    mupdf_pages, plumber_pages = results["pymupdf"], results["pdfplumber"]
    times = ", ".join(f"{EXTRACTOR_NAMES[e]} {timings[e]:.1f} s" for e in TABLE_EXTRACTORS)
    totals = ", ".join(f"{EXTRACTOR_NAMES[e]} {sum(len(p.tables) for p in results[e])}" for e in TABLE_EXTRACTORS)
    lines = [
        f"# Extraction report: {pdf_path.stem}",
        "",
        f"- Source: `{pdf_path.name}` ({page_count} pages)",
        f"- Pages in this report: {pages_spec or 'all'} ({len(mupdf_pages)} pages)",
        "- Text: PyMuPDF `get_text()`, in the order stored in the file",
        f"- pdfplumber cell text settings: `{dict(PDFPLUMBER_TEXT_SETTINGS)}`",
        f"- Extraction time (text + tables): {times}",
        f"- Tables found: {totals}",
        "",
        "In tables, `∅` is a cell returned as None (usually part of a merged cell) "
        "and `<br>` a line break inside a cell. "
        "Diff is the number of tables found by pdfplumber minus those found by PyMuPDF.",
        "",
        "## Summary",
        "",
        "| Page | Text chars | Tables PyMuPDF | Tables pdfplumber | Diff | Empty cells PyMuPDF | Empty cells pdfplumber |",
        "|---|---|---|---|---|---|---|",
    ]
    for mupdf_page, plumber_page in zip(mupdf_pages, plumber_pages):
        diff = len(plumber_page.tables) - len(mupdf_page.tables)
        lines.append(
            f"| [{mupdf_page.page}](#page-{mupdf_page.page}) | {len(mupdf_page.text):,} "
            f"| {len(mupdf_page.tables)} | {len(plumber_page.tables)} | {f'{diff:+d}' if diff else ''} "
            f"| {format_ratio(empty_cell_ratio(mupdf_page.tables))} "
            f"| {format_ratio(empty_cell_ratio(plumber_page.tables))} |"
        )
    lines.append("")

    for mupdf_page, plumber_page in zip(mupdf_pages, plumber_pages):
        tables = {"pymupdf": mupdf_page.tables, "pdfplumber": plumber_page.tables}
        lines += page_section(mupdf_page.page, mupdf_page.text, tables, images.get(mupdf_page.page))
    return "\n".join(lines)


def render_page_images(pdf_path: Path, page_numbers: list[int], out_dir: Path) -> dict[int, str]:
    """Save one PNG per page; return each page's image path relative to the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    with pymupdf.open(pdf_path) as doc:
        for page_number in page_numbers:
            name = f"page_{page_number:03d}.png"
            doc[page_number - 1].get_pixmap(dpi=PAGE_IMAGE_DPI).save(str(out_dir / name))
            images[page_number] = f"{out_dir.name}/{name}"
    return images


def inspect_pdf(pdf_path: Path, pages_spec: str | None, render_pages: bool, out_root: Path) -> Path:
    pages = parse_page_range(pages_spec) if pages_spec else None

    # Each extractor gets its own parse_pdf call so it can be timed on its own.
    results, timings = {}, {}
    for extractor in TABLE_EXTRACTORS:
        start = time.perf_counter()
        results[extractor] = parse_pdf(pdf_path, pages, table_extractor=extractor)
        timings[extractor] = time.perf_counter() - start

    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    page_numbers = [page.page for page in results["pymupdf"]]
    images = render_page_images(pdf_path, page_numbers, out_dir / "pages") if render_pages else {}
    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count

    report_path = out_dir / "report.md"
    report = build_report(pdf_path, page_count, pages_spec, results, timings, images)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+", type=Path, help="datasheet PDF(s) to inspect")
    parser.add_argument("--pages", help='pages to include, e.g. "1-40" or "12,45-47" (default: all)')
    parser.add_argument("--render-pages", action="store_true", help="add an image of each page to the report")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR, help=f"output root (default: {REPORTS_DIR})")
    args = parser.parse_args()

    # Silence PyMuPDF's one-time suggestion to install its separate layout package.
    pymupdf.no_recommend_layout()
    for pdf_path in args.pdfs:
        report_path = inspect_pdf(pdf_path, args.pages, args.render_pages, args.out_dir)
        print(f"{pdf_path.name}: {report_path}")


if __name__ == "__main__":
    main()
