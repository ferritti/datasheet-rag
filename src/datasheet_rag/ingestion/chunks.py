"""The chunk: the unit of text that gets embedded, searched and cited.

Two rules hold for every chunk, whatever strategy builds it:
- it comes from a single page, so a retrieved chunk cites exactly one page
  (the evaluation counts a hit when doc and page match the ground truth);
- its text starts with the device name, because the datasheets in the corpus
  have near-identical tables and only the name tells them apart.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Upper bound on a chunk's text length, to be checked against the embedding
# model's token limit once the model is chosen. A single table row longer than
# this still becomes one chunk.
MAX_CHUNK_CHARS = 1000


@dataclass
class Chunk:
    chunk_id: str  # e.g. "stm32f401re-p60-table14-1", stable across runs
    doc: str  # e.g. "stm32f401re", the same value as `doc` in data/eval/questions.jsonl
    page: int  # 1-based page number, used for the citation
    kind: Literal["table", "text"]
    section: str  # e.g. "6.3.1 General operating conditions"; "" if unknown
    title: str | None  # table caption, e.g. "Table 14. General operating conditions"
    text: str  # what gets embedded and searched


def normalize_text(text: str) -> str:
    """Join the lines of a cell or paragraph and normalise minus signs."""
    # Names wrapped after an underscore or a slash in narrow cells:
    # "USART2_\nCTS", "TIM2_CH1/\nTIM2_ETR".
    text = text.replace("_\n", "_").replace("/\n", "/")
    text = re.sub(r"[–−](?=\d)", "-", text)  # en dash or minus sign used as a minus: "–0.3"
    return " ".join(text.split())  # other line breaks and repeated spaces


def doc_id(pdf_path: str | Path) -> str:
    """'data/raw/DS_stm32f401re.pdf' -> 'stm32f401re'."""
    return Path(pdf_path).stem.removeprefix("DS_").lower()


def device_name(doc: str) -> str:
    """'stm32f401re' -> 'STM32F401RE'."""
    return doc.upper()


def save_chunks(chunks: Iterable[Chunk], path: str | Path) -> None:
    """Write chunks as JSON Lines: one chunk per line, with the Chunk fields as keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
