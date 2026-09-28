"""Load and process GutenQA into the common RAG-experiments format.

GutenQA (https://huggingface.co/datasets/LumberChunker/GutenQA) ships 100 cleaned
public-domain books *already segmented into chunks* by LumberChunker, plus 3,000
needle-in-a-haystack questions. Each question points at the single gold chunk that
answers it, and carries a `Chunk Must Contain` excerpt of that chunk.

Because the books arrive pre-chunked, this script reconstructs each book by
concatenating its chunks in order, so that our own chunking methods can be applied
to the full document instead of inheriting LumberChunker's segmentation.

Outputs:
  data/documents/<book name>.txt   100 reconstructed books
  data/items.jsonl                 3,000 {question, answer, evidences} records

Evidence spans are character offsets into the reconstructed document, so that a
retrieved chunk can be scored by overlap regardless of how it was cut.

Requires: huggingface_hub, pandas, pyarrow
Run with: python datasets/gutenqa/load.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "LumberChunker/GutenQA"
CHUNKS_FILE = "gutenqa_chunks.parquet"
QUESTIONS_FILE = "questions.parquet"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ITEMS_PATH = DATA_DIR / "items.jsonl"

# Chunks are joined with a blank line, matching the paragraph breaks already
# present inside the chunks themselves.
CHUNK_SEPARATOR = "\n\n"

# The questions were generated from the books, so their quoting style does not
# always match the source (straight vs. curly quotes, hyphens vs. em dashes).
# These are folded away before locating an evidence excerpt.
CHAR_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‒": "-", "–": "-", "—": "-", "―": "-",
    "…": "...",
}


def fold(text):
    """Normalize `text` and return it with a map from folded to source offsets.

    The map has one entry per folded character plus a trailing sentinel, so a
    match found in the folded text can be reported as a span of the original.
    """
    folded = []
    offsets = []
    after_space = False
    for i, char in enumerate(text):
        if char.isspace():
            if not after_space:
                folded.append(" ")
                offsets.append(i)
                after_space = True
            continue
        after_space = False
        for out in CHAR_FOLD.get(char, char).lower():
            folded.append(out)
            offsets.append(i)
    offsets.append(len(text))
    return "".join(folded), offsets


def locate(excerpt, haystack, offset):
    """Find `excerpt` in `haystack`, returning (start, end) shifted by `offset`.

    Tries a verbatim match first, then a match on the folded text. Returns None
    if the excerpt cannot be located, which happens when the question generator
    paraphrased rather than quoted.
    """
    start = haystack.find(excerpt)
    if start != -1:
        return offset + start, offset + start + len(excerpt), "exact"

    folded_haystack, offsets = fold(haystack)
    folded_excerpt, _ = fold(excerpt)
    folded_excerpt = folded_excerpt.strip()
    if not folded_excerpt:
        return None
    start = folded_haystack.find(folded_excerpt)
    if start == -1:
        return None
    end = start + len(folded_excerpt)
    return offset + offsets[start], offset + offsets[end - 1] + 1, "folded"


def safe_filename(name):
    """Map a book name to a filename that is legal on every platform."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(". ")


def build_documents(chunks):
    """Reconstruct one document per book.

    Returns {book_id: (doc_id, text, {chunk_id: (start, end)})}, where the spans
    locate each source chunk inside the reconstructed text.
    """
    documents = {}
    for book_id, group in chunks.groupby("book_id", sort=True):
        group = group.sort_values("chunk_id")
        parts = []
        spans = {}
        cursor = 0
        for chunk_id, chunk in zip(group["chunk_id"], group["chunk"]):
            spans[int(chunk_id)] = (cursor, cursor + len(chunk))
            parts.append(chunk)
            cursor += len(chunk) + len(CHUNK_SEPARATOR)
        documents[int(book_id)] = (
            group["book_name"].iloc[0],
            CHUNK_SEPARATOR.join(parts),
            spans,
        )
    return documents


def build_items(questions, documents):
    """Turn each question into a {question, answer, evidences} record."""
    items = []
    counts = {"exact": 0, "folded": 0, "chunk": 0}
    for position, row in enumerate(questions.itertuples(index=False)):
        book_id = int(row.book_id)
        chunk_id = int(row.chunk_id)
        doc_id, text, spans = documents[book_id]

        chunk_start, chunk_end = spans[chunk_id]
        chunk_text = text[chunk_start:chunk_end]

        excerpt = row.chunk_must_contain
        span = locate(excerpt, chunk_text, chunk_start) if isinstance(excerpt, str) else None
        if span is None:
            # The gold chunk is itself the ground truth, so fall back to it
            # whole rather than dropping the question.
            start, end, match = chunk_start, chunk_end, "chunk"
        else:
            start, end, match = span
        counts[match] += 1

        items.append({
            "id": f"gutenqa-{position:04d}",
            "question": row.question,
            "answer": row.answer,
            "evidences": [{
                "doc_id": doc_id,
                "text": text[start:end],
                "start": start,
                "end": end,
                "match": match,
            }],
            "metadata": {
                "book_id": book_id,
                "source_chunk_id": chunk_id,
                "chunk_must_contain": excerpt if isinstance(excerpt, str) else None,
            },
        })
    return items, counts


def snake_case(frame):
    """The published columns are titled ("Book ID"); make them identifiers."""
    return frame.rename(columns=lambda name: name.strip().lower().replace(" ", "_"))


def main():
    print(f"Downloading {REPO_ID} ...")
    chunks = snake_case(pd.read_parquet(hf_hub_download(REPO_ID, CHUNKS_FILE, repo_type="dataset")))
    questions = snake_case(pd.read_parquet(hf_hub_download(REPO_ID, QUESTIONS_FILE, repo_type="dataset")))
    print(f"  {len(chunks):,} source chunks, {len(questions):,} questions")

    documents = build_documents(chunks)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for doc_id, text, _ in documents.values():
        path = DOCUMENTS_DIR / f"{safe_filename(doc_id)}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")
        written += 1
    lengths = sorted(len(text) for _, text, _ in documents.values())
    print(f"Wrote {written} documents to {DOCUMENTS_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} / max {lengths[-1]:,}")

    items, counts = build_items(questions, documents)
    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    print(f"  evidence located: {counts['exact']:,} verbatim, {counts['folded']:,} after folding, "
          f"{counts['chunk']:,} fell back to the whole gold chunk")


if __name__ == "__main__":
    main()
