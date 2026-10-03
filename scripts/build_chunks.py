"""Chunk the datasheets and save the chunks for indexing.

For each chunking strategy and datasheet it writes
- data/processed/chunks_<strategy>/<doc>.jsonl: one chunk per line, the input
  of the indexing step (one file per datasheet, so rebuilding one datasheet
  leaves the others untouched);
- data/reports/<pdf name>/chunks_<strategy>.md: every chunk, to check by eye;
and prints a summary table per strategy.

Usage (from the repository root):
    python scripts/build_chunks.py
    python scripts/build_chunks.py data/raw/DS_stm32f401re.pdf --strategy structured
"""

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import pymupdf

from datasheet_rag.ingestion.chunking import STRATEGIES, chunk_pages
from datasheet_rag.ingestion.chunks import MAX_CHUNK_CHARS, Chunk, doc_id
from datasheet_rag.ingestion.pdf_parser import parse_pdf, read_outline

REPO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_DIR / "data" / "raw"
PROCESSED_DIR = REPO_DIR / "data" / "processed"
REPORTS_DIR = REPO_DIR / "data" / "reports"


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def preview(chunks: list[Chunk], title: str) -> str:
    lines = [f"# {title}", "", f"{len(chunks)} chunks.", ""]
    for chunk in chunks:
        details = [f"page {chunk.page}", chunk.kind, f"{len(chunk.text)} chars"] + ([chunk.section] if chunk.section else [])
        lines += [f"## {chunk.chunk_id}", "", " · ".join(details), "", "```text", chunk.text, "```", ""]
    return "\n".join(lines)


def summary_row(name: str, chunks: list[Chunk]) -> str:
    lengths = [len(chunk.text) for chunk in chunks]
    tables = sum(chunk.kind == "table" for chunk in chunks)
    return (f"| {name} | {len(chunks)} | {tables} | {len(chunks) - tables} | {statistics.median(lengths):.0f} "
            f"| {max(lengths)} | {sum(length > MAX_CHUNK_CHARS for length in lengths)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="*", type=Path, help="datasheet PDFs (default: all in data/raw)")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all", help="chunking strategy (default: all)")
    args = parser.parse_args()
    pdf_paths = args.pdfs or sorted(RAW_DIR.glob("*.pdf"))
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    pymupdf.no_recommend_layout()
    all_chunks: dict[str, list[Chunk]] = {strategy: [] for strategy in strategies}
    rows: dict[str, list[str]] = {strategy: [] for strategy in strategies}
    for pdf_path in pdf_paths:
        doc = doc_id(pdf_path)
        # One parse serves every strategy.
        pages, outline = parse_pdf(pdf_path), read_outline(pdf_path)
        for strategy in strategies:
            chunks = chunk_pages(doc, pages, outline, strategy)
            write_jsonl(chunks, PROCESSED_DIR / f"chunks_{strategy}" / f"{doc}.jsonl")
            report_path = REPORTS_DIR / pdf_path.stem / f"chunks_{strategy}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(preview(chunks, f"{doc}: {strategy} chunks"), encoding="utf-8")
            all_chunks[strategy] += chunks
            rows[strategy].append(summary_row(doc, chunks))

    for strategy in strategies:
        print(f"\n{strategy} chunks -> {PROCESSED_DIR / f'chunks_{strategy}'}/")
        print("| Datasheet | Chunks | Table | Text | Median chars | Max chars | Over limit |")
        print("|---|---|---|---|---|---|---|")
        print("\n".join(rows[strategy] + [summary_row("**Total**", all_chunks[strategy])]))


if __name__ == "__main__":
    main()
