# Indexing and dense retrieval

This page explains how the chunks (see [chunking.md](chunking.md)) are embedded,
stored in PostgreSQL with pgvector and searched, with the measurements behind
each choice.

## Embedding model ([`embeddings.py`](../src/datasheet_rag/indexing/embeddings.py))

The first requirement is that a chunk fits in the model's input: the tokens
past the limit are silently dropped, so that part of the chunk can never be
found. Three common sentence-transformers models were measured on all 7,911
chunks (5,205 structured, 2,706 fixed), counting tokens with each model's own
tokenizer and timing the embedding of every chunk on the Apple GPU (MPS):

| Model | Dimensions | Token limit | Chunks over the limit (structured / fixed) | Time for all chunks |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 256 | 49.3% / 54.1% | ~30 s |
| **bge-small-en-v1.5** | 384 | 512 | 1.7% / 2.3% | ~83 s |
| bge-base-en-v1.5 | 768 | 512 | 1.7% / 2.3% | ~213 s |

Chunks have a median of 254 tokens (structured) and 263 (fixed), a 95th
percentile of 460 and 478, and a maximum of 572 and 570.

- **all-MiniLM-L6-v2 is ruled out**: it would cut half of the chunks.
- **The two bge models** share the tokenizer and the limit, so they truncate
  the same chunks. bge-small is 2.6 times faster and its vectors take half the
  space; whether bge-base retrieves better is left to the benchmark.
- **What truncation still costs.** The 87 structured chunks over 512 tokens
  are all table chunks, almost all from the *Alternate function mapping*
  table: rows of short pin-function names produce many tokens per character.
  What is lost is the last few tokens, usually the `AF15: EVENTOUT` column,
  which is the same in every row; in 3 chunks the cut reaches the notes.
  Shrinking every chunk to avoid this would cost more than it saves.

Two details of the bge models matter:

- **Questions get an instruction prefix**, `Represent this sentence for
  searching relevant passages: `, and chunks are embedded as they are. The bge
  models were trained this way for retrieval with short queries and long
  passages.
- **Vectors are normalized to unit length**, so the cosine similarity is a
  plain dot product.

## Storage ([`store.py`](../src/datasheet_rag/indexing/store.py))

PostgreSQL 17 with the pgvector 0.8 extension runs in Docker
([`docker-compose.yml`](../docker-compose.yml)); the connection settings are in
`.env` (see [`.env.example`](../.env.example)). One table holds the chunks of
both strategies:

```sql
CREATE TABLE chunks (
    strategy  text    NOT NULL,   -- "structured" or "fixed"
    chunk_id  text    NOT NULL,   -- e.g. "stm32f401re-p60-table14-1"
    doc       text    NOT NULL,
    page      integer NOT NULL,
    kind      text    NOT NULL,   -- "table" or "text"
    section   text    NOT NULL,
    title     text,
    text      text    NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (strategy, chunk_id)
);
```

- **Plain SQL through psycopg**, not a vector-store library, so the schema and
  the queries are visible. Hybrid search (keywords plus vectors) will need SQL
  of its own anyway.
- **Re-indexing replaces one datasheet of one strategy at a time**, deleting
  its old rows and inserting the new ones in one transaction. Running the
  indexing twice gives the same table, and a failure leaves the old chunks in
  place.

### No approximate index

The table has no HNSW (approximate nearest neighbour) index. One was tried,
and a search restricted to one datasheet returned 1 chunk instead of 5. The
index first finds its 40 nearest candidates (`hnsw.ef_search`) among all the
chunks, and only then does the `WHERE doc = ...` clause keep those of the
chosen datasheet, which here was one.

Without an index, PostgreSQL computes the distance to every chunk and keeps
the nearest ones, so the result is always exact. On this corpus that takes
about 9 ms over all 5,205 structured chunks and 2 ms within one datasheet. An
index (with pgvector's iterative scans, which fix the filtering problem) only
pays off at hundreds of thousands of chunks.

## Dense retrieval ([`dense.py`](../src/datasheet_rag/retrieval/dense.py))

`search()` embeds the question and returns the `k` chunks of one strategy with
the smallest cosine distance (pgvector's `<=>` operator). The score it reports
is the cosine similarity, `1 - distance`.

### Device filter

The datasheets share most of their text, and a question's embedding hardly
changes between "STM32F401" and "STM32F411". For *"What is the maximum
standard operating voltage VDD of the STM32F401?"*, 4 of the 5 nearest
structured chunks come from other datasheets (F446, F407, F410, F411).

`devices_in()` therefore looks for part numbers in the question (`STM32F401`,
`STM32F401RE`, `stm32f401xe` or just `F401`) and returns the datasheets whose
name starts with them. The search can then be restricted to those datasheets;
for the question above, all 5 results then come from the F401RE.

- The series name `STM32F4` names no device, so the question is searched in
  every datasheet.
- A question naming two devices searches both datasheets.
- A part number without a datasheet of its own in the corpus, such as
  STM32F405 (covered by the STM32F407 datasheet) or STM32F427 (covered by the
  STM32F429 one), matches nothing, so the question is searched everywhere.

### A known weakness

The filter fixes the confusion between devices, not the ranking within a
datasheet. The answer to the question above, `VDD | Standard operating
voltage | Max: 3.6 V`, is in chunk `stm32f401re-p60-table14-1`, which ranks
186th among the F401RE chunks. That chunk starts with four rows on clock
frequencies before the VDD row, so its embedding is mostly about clocks. The
fixed-size chunk of the same page ranks 5th. Smaller table chunks and hybrid
search (an exact match on "VDD" and "Standard operating voltage") are the two
fixes to compare on the benchmark.

## How to run

From the repository root:

```bash
docker compose up -d --wait
python scripts/build_chunks.py
python scripts/index_chunks.py
python scripts/search.py "What is the maximum standard operating voltage VDD of the STM32F401?"
```

`index_chunks.py` embeds the chunks written by `build_chunks.py` and stores
them, about 2 minutes for both strategies. `search.py` prints the nearest
chunks with their datasheet, page and score. Options: `--strategy fixed`,
`--k 10`, `--doc stm32f411re` or `--doc all` (to override the device filter),
and `--full` (whole chunk text).

The database tests (`pytest`) work in a temporary schema and are skipped when
the database is not running.
