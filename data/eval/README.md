# Evaluation data

- `questions.jsonl` is the benchmark. Every entry has been checked by hand
  against the PDF.
- `candidates.jsonl` holds proposed questions that have not been checked yet
  (`verified: false`). The evaluation does not use them.

One JSON object per line:

| Field      | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| `question` | The question, as a user would ask it.                                   |
| `answer`   | The short expected answer, e.g. `"3.6 V"`.                              |
| `doc`      | The datasheet, e.g. `"stm32f401re"` for `data/raw/DS_stm32f401re.pdf`.  |
| `page`     | The 1-based PDF page with the answer (the footer prints it as `60/137`).|
| `kind`     | Where the answer is on the page: `"table"` or `"text"`.                 |
| `source`   | The table caption or section heading that holds the answer.             |
| `verified` | Candidates only: `false` until checked.                                 |

## How the candidates were chosen

- Five per datasheet, three answered by a table and two by running text, so
  both kinds of chunk are measured.
- Facts were taken from the PDF page text, not from the chunks of either
  chunking strategy, so the questions do not favour one of them.
- Each answer appears on one page only (checked by searching the whole PDF),
  because retrieval is scored by matching `doc` and `page`.
- Questions are worded the way an engineer would ask them rather than copied
  from the datasheet, which would make retrieval look better than it is.
- Every question names the device, since the answers differ between devices.

## Verifying a candidate

Open `data/raw/DS_<doc>.pdf` at `page`, find `source` and check `answer`
against it. If it is right (or once corrected), move the line to
`questions.jsonl` without the `verified` field.
