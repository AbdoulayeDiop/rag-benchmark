"""Load and process SQuAD into the common RAG-experiments format.

SQuAD 1.1 (https://arxiv.org/abs/1606.05250) is extractive question answering
over Wikipedia: crowdworkers wrote questions about a paragraph and marked the
answer inside it, so every question has an exact character span. Development
questions carry up to six independent answers, one per annotator.

WHY 1.1 AND NOT 2.0: SQuAD 2.0 adds questions written to be unanswerable, but
unanswerable *from the single paragraph the annotator was shown*. This loader
rebuilds whole articles from their paragraphs, which invalidates that label --
the rest of the article may well answer the question. Measured on the grouped
corpus, 27% of 2.0's unanswerable questions had a near-duplicate answerable
question elsewhere in the same article, and with no gold answer the label cannot
be checked directly, only contradicted. SQuAD 1.1 has no such questions, so
nothing has to be discarded. For abstention, use TechQA (real forum questions no
technote resolves), Qasper or Natural Questions, where annotators judged the
whole document.

WHY THIS GROUPS PARAGRAPHS: SQuAD is distributed as individual paragraphs of
about 700 characters, far too short for chunking to mean anything. But every
paragraph carries the title of the article it came from, and the paragraphs under
one title are consecutive sections of that article. Joining them in order
rebuilds the article -- a median of 38 paragraphs and about 30k characters --
without refetching from the web or inventing anything. Answer offsets are
paragraph-relative, so they shift by the paragraph's position and stay exact.

Only plain text is written. SQuAD keeps no section headings, so there is no
structure to mark; the article title is the sole heading and would not make a
Markdown rendering worth comparing against the text one.

Outputs:
  data/documents/<article>.txt   one Wikipedia article per document
  data/items.jsonl               {question, answer, evidences} records

Requires: huggingface_hub, pandas, pyarrow
Run with: python datasets/squad/load.py [--max-documents 100]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "rajpurkar/squad"
SHARD = "plain_text/{split}-00000-of-00001.parquet"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ITEMS_PATH = DATA_DIR / "items.jsonl"

PARAGRAPH_SEPARATOR = "\n\n"


def document_id(title):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(". ")[:120] or "untitled"


def build_document(contexts):
    """Join an article's paragraphs, returning the text and each one's offset."""
    starts = {}
    at = 0
    for context in contexts:
        starts[context] = at
        at += len(context) + len(PARAGRAPH_SEPARATOR)
    return PARAGRAPH_SEPARATOR.join(contexts), starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents", type=int, default=100,
        help="Cap on the number of articles. Validation supplies 48 and the rest "
             "are filled from train, which keeps the corpus large enough that "
             "retrieval is not trivial. Pass 0 for no cap (490 articles).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "train"],
        choices=["validation", "train"],
        help="Splits to draw articles from, in order of preference.",
    )
    args = parser.parse_args()

    frames = []
    for split in args.splits:
        frame = pd.read_parquet(hf_hub_download(REPO_ID, SHARD.format(split=split), repo_type="dataset"))
        frame["split"] = split
        frames.append(frame)
        print(f"{split}: {len(frame):,} questions, {frame['title'].nunique()} articles")
    data = pd.concat(frames, ignore_index=True)

    # Articles are taken split by split, so validation is always included whole.
    titles = list(dict.fromkeys(data["title"]))
    if args.max_documents:
        titles = titles[:args.max_documents]
    data = data[data["title"].isin(set(titles))]
    print(f"Selected {len(titles)} articles covering {len(data):,} questions")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    lengths, paragraphs = [], []
    for title, group in data.groupby("title", sort=False):
        # dict.fromkeys keeps first-seen order, which is the article's own order.
        contexts = list(dict.fromkeys(group["context"]))
        text, starts = build_document(contexts)
        doc_id = document_id(title)
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
        documents[title] = (doc_id, text, starts)
        lengths.append(len(text))
        paragraphs.append(len(contexts))

    lengths.sort()
    paragraphs.sort()
    print(f"Wrote {len(lengths)} documents to {DOCUMENTS_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")
    print(f"  paragraphs per document: min {paragraphs[0]} / median "
          f"{paragraphs[len(paragraphs) // 2]} / max {paragraphs[-1]}")

    items = []
    located = mismatched = empty = 0
    for row in data.itertuples(index=False):
        doc_id, text, starts = documents[row.title]
        base = starts[row.context]

        evidences = []
        seen = set()
        answers = row.answers
        for answer_text, answer_start in zip(answers["text"], answers["answer_start"]):
            start = base + int(answer_start)
            end = start + len(answer_text)
            if text[start:end] != answer_text:
                # Should not happen: offsets are paragraph-relative by construction.
                mismatched += 1
                continue
            if (start, end) in seen:  # annotators often mark the same span
                continue
            seen.add((start, end))
            located += 1
            evidences.append({
                "doc_id": doc_id,
                "text": text[start:end],
                "start": start,
                "end": end,
                "match": "offset",
                "kind": "answer_span",
            })

        if not evidences:
            # SQuAD 1.1 has no unanswerable questions, so this means a broken row.
            empty += 1
            continue

        items.append({
            "id": f"squad-{row.id}",
            "question": row.question,
            "answer": answers["text"][0],
            "evidences": evidences,
            "metadata": {
                "doc_id": doc_id,
                "title": row.title,
                "split": row.split,
                "answer_type": "extractive",
                "answers": list(answers["text"]),
                "annotators": len(answers["text"]),
                # The paragraph the question was written against, which is the
                # gold passage a retriever should return.
                "context_span": {
                    "doc_id": doc_id,
                    "start": base,
                    "end": base + len(row.context),
                },
            },
        })

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    print(f"  answer spans located: {located:,} "
          f"({located / len(items):.1f} per question, from up to six annotators)")
    if mismatched:
        print(f"  {mismatched:,} offsets did not match their answer text")
    if empty:
        print(f"  {empty:,} questions dropped for having no usable answer span")


if __name__ == "__main__":
    main()
