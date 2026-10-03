# Chunking

Chunks are the units that get embedded, searched and cited. This page explains
how the datasheets are cut into chunks, with the measurements behind each
choice. Two strategies are built, so that the evaluation can compare them:

- **structured** (the main one): one kind of chunk for tables, one for running
  text, both aware of the document's structure;
- **fixed** (the baseline): fixed-size windows of each page's raw text.

The PDF parsing these build on is described in [pdf_extraction.md](pdf_extraction.md).

## Two rules for every chunk

1. **A chunk never crosses a page.** The evaluation counts a retrieved chunk
   as a hit when its `doc` and `page` match the ground truth (recall@5, MRR),
   so a chunk spanning two pages would make its citation ambiguous. A table
   split over pages becomes one chunk per page, each repeating the caption.
2. **A chunk starts with the device name.** The datasheets have near-identical
   tables (Table 14 of the STM32F401 and STM32F411 differ in a few numbers), and
   only the name lets keyword search and embeddings tell them apart.

## Structured chunks

### Table chunks ([`table_chunks.py`](../src/datasheet_rag/ingestion/table_chunks.py))

```
STM32F401RE datasheet | 6.3.1 General operating conditions | Table 14. General operating conditions
Symbol: fHCLK | Parameter: Internal AHB clock frequency | Conditions: Power Scale3: Regulator ON, VOS[1:0] bits in PWR_CR register = 0x01 | Min: 0 | Typ: - | Max: 60 | Unit: MHz
...
Symbol: VDD | Parameter: Standard operating voltage | Conditions: - | Min: 1.7 [note 1] | Typ: - | Max: 3.6 | Unit: V
Notes: 1. VDD/VDDA minimum value of 1.7 V with the use of an external power supply supervisor (refer to Section 3.14.2: Internal reset OFF).
```

- **Which tables.** Only those with a "Table N." caption just above them (0-14
  pt in the corpus; the limit is 40 pt). Every real ST table has one, repeated
  as "(continued)" on further pages, while the "tables" found inside figures
  (pin squares of pinouts, graph grids, timing diagrams) have none. This keeps
  995 table parts of the 3,006 tables pdfplumber finds; on F401RE every kept
  table matches a caption and every dropped one larger than 2,000 pt² was
  checked to be a figure.
- **One line per row, as `column: value` pairs**, so that a row stays readable
  when a table is split over several chunks. Rows are added to a chunk while it
  fits in 1,000 characters.
- **Column names** join the header rows (`Pin Number` + `LQFP64` →
  `Pin Number LQFP64`). Header rows are the leading rows whose letters are all
  bold, at least one: of the 995 captioned parts, 669 have one header row, 314
  two and 12 three (the current consumption tables).
- **Merged cells** repeat their text at every position they cover (e.g. the
  unit `MHz` over four rows), using pdfplumber's cell boxes.
- **Cleanup**: `1.7(1)` becomes `1.7 [note 1]` so it is not read as 1.71; en
  dashes and minus signs before a digit become `-`; line breaks after `_` or
  `/` are joined (`USART2_CTS`), others become spaces; cells in fonts without a
  Unicode mapping (`(cid:51)…`) are dropped.
- **Footnotes.** The numbered notes below a table (2-11 pt below it and 2-7 pt
  apart, while the next paragraph starts 15 pt or more below; the limit is 12
  pt) are collected per table number, because a table split over pages prints
  them under its last part only. When a table ends near the bottom of a page,
  its notes continue at the top of the next one, sometimes under a "Notes:"
  heading. Each chunk carries the notes its rows refer to: 98.1% of the 6,040
  references in table chunks find their note.
- **Skipped**: the document revision history, which describes the document,
  not the device, and names dozens of other tables ("Updated Table 14: …").

### Text chunks ([`text_chunks.py`](../src/datasheet_rag/ingestion/text_chunks.py))

```
STM32F401RE datasheet | 6.3.2 VCAP_1/VCAP_2 external capacitors
Stabilization for the main regulator is achieved by connecting external capacitor CEXT to the VCAP_1 and VCAP_2 pin. ...
CEXT is specified in Table 16.
Figure 20. External capacitor CEXT
1. Legend: ESR is the equivalent series resistance.
```

The text of a page is its text blocks in reading order, minus:

- the **running header and footer**, in fixed bands of the printed page
  (measured on all 1,274 pages);
- **captioned tables** with their caption and footnotes, already in the table
  chunks, and the "tables" found inside figures;
- the **text inside figures** (axis labels, pin names), from the "Figure N."
  caption down to the ID ST prints at the bottom of each figure (`MS31151V4`,
  `ai15062b`). The caption is kept. 418 of the 521 figures have the ID, which
  drops about 10,000 blocks of labels; the others, mostly package outline
  drawings, keep their text. Figure frames were tried first, but are found for
  only 30% of the figures;
- **unreadable text** from fonts without a Unicode mapping;
- **section headings**, which go into the prefix instead.

Pages of the table of contents and lists of tables and figures (61 pages) and
the final legal notice (7) are skipped.

**Sections come from the PDF outline** ([`sections.py`](../src/datasheet_rag/ingestion/sections.py)):
on all 829 headings of the corpus, the outline title is printed as a text block
of its own, so a block belongs to the last heading before it. Paragraphs are
grouped by section and packed into chunks of up to 1,000 characters, split at
sentence ends if a paragraph is longer; a chunk never crosses a section or a
page.

## Fixed-size baseline ([`fixed_chunks.py`](../src/datasheet_rag/ingestion/fixed_chunks.py))

Each page's raw text (whitespace collapsed) is cut at spaces into windows of up
to 1,000 characters, each repeating the last 150 characters of the previous
one. It follows the two rules above and nothing else: no table handling, no
sections, no cleanup, no skipped pages. Comparing it with the structured chunks
measures what the structure adds.

## Results

| Datasheet | Pages | Structured: table | Structured: text | Fixed |
|---|---|---|---|---|
| F401RE | 137 | 355 | 180 | 293 |
| F407VG | 206 | 579 | 215 | 442 |
| F410RB | 141 | 474 | 166 | 313 |
| F411RE | 151 | 412 | 186 | 322 |
| F412ZG | 202 | 665 | 216 | 431 |
| F429ZI | 240 | 731 | 234 | 492 |
| F446RE | 197 | 566 | 226 | 413 |
| **Total** | **1,274** | **3,782** | **1,423** | **2,706** |

Median length is 784 characters for structured chunks and 992 for fixed ones.
56 structured chunks (1.1%) exceed 1,000 characters: a single table row with
its footnotes, which is not cut.

## Known limitations

- Words wrapped inside narrow cells keep a space (`IDD_VBA T`, `EVENT OUT`):
  from the text alone a wrap cannot be told from a real space.
- Headers rotated over two lines keep their lines in reverse order
  (`type Pin` for "Pin type").
- 1.9% of footnote references have no note, mostly in the WLCSP mechanical
  data tables.
- 20% of the figures (no ID found) keep their labels.
- The cover page has no section in the six datasheets whose outline has no
  "Features" entry.
- PyMuPDF sometimes glues a link to the previous word (`given inTable 54`).

## Output and how to run

From the repository root, with the datasheets in `data/raw/`:

```bash
python scripts/build_chunks.py
```

This writes, for each strategy and datasheet, `data/processed/chunks_<strategy>/<doc>.jsonl`
(one chunk per line with the fields `chunk_id`, `doc`, `page`, `kind`,
`section`, `title`, `text`) and a preview of every chunk in
`data/reports/<pdf name>/chunks_<strategy>.md`. Both are gitignored, since they
contain datasheet text. In code, `build_chunks(pdf_path, strategy)` in
[`chunking.py`](../src/datasheet_rag/ingestion/chunking.py) returns the chunks
of one datasheet.
