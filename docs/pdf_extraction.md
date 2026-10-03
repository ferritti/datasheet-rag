# PDF extraction: choosing a table extractor

Most facts a user asks about in a datasheet (supply voltages, currents, timings,
pin functions) live in tables, so the first step of the pipeline is getting
tables out of the PDFs with their cell text intact. Two libraries in the stack
can find tables, PyMuPDF (`Page.find_tables`) and pdfplumber (`Page.find_tables`).
This page compares them on the corpus and explains the choice.

**Decision:** tables come from **pdfplumber** with two text settings
(`y_tolerance=4`, `char_dir_rotated="btt"`); page text and page images come from
**PyMuPDF**. See [Decision](#decision) for why.

## Corpus and method

- 7 STM32F4 datasheets (F401RE, F407VG, F410RB, F411RE, F412ZG, F429ZI, F446RE),
  1,274 pages in total.
- [`scripts/inspect_pdf.py`](../scripts/inspect_pdf.py) writes a page-by-page
  report (raw text, both extractors' tables, page image) to check results by eye.
- [`scripts/compare_extractors.py`](../scripts/compare_extractors.py) produces
  all the counts below.
- **Reference for "real" tables.** ST gives every table a caption, "Table N. Title",
  repeated as "(continued)" on each further page. The corpus has 1,014 captions
  (256 of them "continued"). This is not an exact table count, but a captioned
  page where an extractor finds nothing is a missed table.
- **Matching.** A table found by both extractors is a pair whose bounding boxes
  overlap with intersection over union > 0.5. Both libraries use the same
  coordinates (PDF points, origin top-left).

## Finding tables

| Datasheet | Pages | Captions | Continued | Tables PyMuPDF | Tables pdfplumber | Matched | Only PyMuPDF | Only pdfplumber |
|---|---|---|---|---|---|---|---|---|
| F401RE | 137 | 116 | 23 | 203 | 317 | 196 | 7 | 121 |
| F407VG | 206 | 157 | 55 | 314 | 496 | 308 | 6 | 188 |
| F410RB | 141 | 117 | 24 | 151 | 250 | 147 | 4 | 103 |
| F411RE | 151 | 119 | 23 | 201 | 319 | 197 | 4 | 122 |
| F412ZG | 202 | 152 | 32 | 263 | 471 | 259 | 4 | 212 |
| F429ZI | 240 | 186 | 52 | 446 | 653 | 439 | 7 | 214 |
| F446RE | 197 | 167 | 47 | 330 | 500 | 322 | 8 | 178 |
| **Total** | **1,274** | **1,014** | **256** | **1,908** | **3,006** | **1,868** | **40** | **1,138** |

- **Both extractors find the same real tables.** The only captioned pages without
  a table are the same for both: the "Ordering information scheme" in four
  datasheets, which is drawn without grid lines (its content is still in the
  page text).
- **The unmatched tables are parts of figures.** On F401RE, every unmatched table
  larger than 2,000 pt² (about 45 × 45 pt) is part of a figure, and the smaller
  ones are too small to be real tables: the pin squares of package pinout
  drawings (the LQFP100 pinout on F401RE p36 alone gives pdfplumber 86 "tables"),
  block diagrams, graphs with grid lines, the memory map, package drawings. pdfplumber produces many more
  of them, so a filter for figure fragments is needed (see [Open issues](#open-issues-for-chunking)).

## Text inside cells

This is where the two differ. Examples are from F401RE.

| Issue | Example | PyMuPDF | pdfplumber, defaults | pdfplumber, our settings |
|---|---|---|---|---|
| Underscore in a name | `PDR_ON` (p25) | `PDR ON set to VDD\n_` | correct | correct |
| Subscript | `VDD`, `fHCLK` (p60) | `V\nDD` | `V\nDD` | `VDD` |
| Rotated text | pin table headers (p38-44) | correct | reversed: `epyt\nniP` | words correct; a header rotated over two lines keeps its lines in reverse order: `type\nPin` |

- **Underscores.** PyMuPDF moves the underscore to a line of its own in 3,972
  cells across the corpus; pdfplumber in none. PyMuPDF's `text_y_tolerance`
  (tried 3 to 5) does not change this, and it cannot be repaired afterwards:
  `PDR ON set to VDD\n_` does not say which space the underscore belongs to.
- **Subscripts.** pdfplumber groups characters into lines when their tops differ
  by less than `y_tolerance` (default 3 pt). A subscript's top is about 3.7 pt
  below its symbol's (`V` at 9 pt, `DD` at 7.2 pt on p60), so the default splits
  `VDD` into `V\nDD`. With 4 the subscript joins its line; from 5 up, separate
  lines of small text start to mix (block diagram on p14:
  `Voltage\nregulator\n3.3 to 1 .2 V` becomes `3. r3 Ve ogto lut al1 ag .t2 eo rV`).
- **Rotated text.** pdfplumber reads rotated characters in the wrong order;
  `char_dir_rotated="btt"` (bottom to top) fixes the words.
- **Effect of the settings.** They change 6,233 of the 105,411 cells pdfplumber
  extracts. A random sample of 30 (`--samples 30`) was checked by eye: all are
  corrections, e.g. `V ≤ V ≤ V\nSS IN DD` → `VSS ≤ VIN ≤ VDD`,
  `001PFQL` → `LQFP100`, `f /RTCCLKfrequency ratio\nPCLK1` →
  `fPCLK1/RTCCLK frequency ratio`; one rotated `I/O structure` improves only in
  part (`structure\nO\n/\nI`).

**Speed** is not a deciding factor: each extractor takes 7 to 13 s per
datasheet including page text (report headers of `inspect_pdf.py`).

## Decision

pdfplumber with `PDFPLUMBER_TEXT_SETTINGS` (in
[`pdf_parser.py`](../src/datasheet_rag/ingestion/pdf_parser.py)) is the default
table extractor. Register, pin and signal names with underscores (`PWR_CR`,
`VCAP_1`, `USART2_TX`) and symbols with subscripts (`VDD`, `fHCLK`) are exactly
the terms users ask about and that keyword search (BM25) matches on. pdfplumber
gets both right with two settings, while PyMuPDF's underscore problem has
neither a setting nor a reliable repair. pdfplumber's extra false positives are
figure fragments, which have to be filtered anyway.

PyMuPDF stays for page text and page images, and as an alternative table
extractor (`table_extractor="pymupdf"`) for comparisons. The tests in
[`tests/test_pdf_parser.py`](../tests/test_pdf_parser.py) pin down the library
behaviour behind this choice on a synthetic PDF, so a library upgrade that
changes it makes a test fail.

## Page text: file order, not sorted

PyMuPDF can return page text in the order it is stored in the file (default) or
sorted by position (`sort=True`). In these datasheets the file order is the
reading order, and sorting makes things worse:

- on the two-column cover page (F401RE p1) it interleaves lines of the two columns;
- on landscape pages (page rotation 90°: 6 pages in F401RE, 14 in F429ZI, 9 in
  F446RE) it mixes the table caption with the table body.

## Open issues for chunking

1. **Figure fragments detected as tables**: filter them (e.g. by size, shape or
   missing text); the criterion still has to be measured.
2. **Tables split over pages** (256 "continued" captions): each part needs the
   caption and the header row.
3. **Merged cells** come out as `None`: values such as the unit (`V`, `MHz`)
   appear only in the first row of a group and need to be carried down.
4. **Values**: footnote markers stick to numbers (`1.7(1)`), and minus signs
   are en dashes (`–0.3`).
5. **Line breaks inside cells**: join without a space after `_` (`USART2_\nCTS`),
   with a space otherwise.
6. **Page text**: drop running headers and footers (`DS10086 Rev 5`, `60/137`);
   on pages with tables, use the extracted tables plus the text outside their
   bounding boxes, since the raw page text repeats the table in a flattened form;
   drop text from fonts without a Unicode mapping (`(cid:51)…`, e.g. the ball
   map on F401RE p37).

## Reproduce

From the repository root, with the datasheets in `data/raw/`:

```bash
python scripts/compare_extractors.py --samples 30
python scripts/inspect_pdf.py data/raw/DS_stm32f401re.pdf --render-pages
```
