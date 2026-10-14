"""Load and process TriviaQA into the common RAG-experiments format.

TriviaQA (https://huggingface.co/datasets/mandarjoshi/trivia_qa) pairs
trivia-enthusiast questions with evidence documents gathered independently of
the question writers. This script uses the `rc.wikipedia` configuration, whose
evidence is Wikipedia articles, rather than the web-search configuration whose
documents are scraped pages of uneven quality.

Split: validation. The test split ships `<unk>` as every answer -- it is held out
for the leaderboard -- so it cannot be used for evaluation here.

NOTE ON EVIDENCE: TriviaQA's annotation is document-level. A question is linked
to the Wikipedia pages it was matched against, not to a passage inside them.
A whole-document span would make the span metrics measure chunk size rather than
retrieval (`evidence_recall` would be the share of the article the chunks
cover), so, as for LiteraryQA, `evidences` is empty and the gold pages are named
in `metadata.doc_ids`: only `doc_hit` is computed on this dataset.

Because chunk-level scoring needs something finer, `metadata.answer_occurrences`
records where the answer string literally appears in each gold document -- 99.7%
of questions have at least one. That is distant supervision, inferred here and
not annotated by anyone, so it is kept out of `evidences` and should be treated
as an approximation of the supporting passage, not ground truth.

Articles are written as plain text only. The source has had its markup
stripped, and section headings survive only as short lines with a trailing
space: a Markdown rendering would have to guess them, and could not recover
their nesting, so none is written.

Outputs:
  data/documents/<page>.txt      one Wikipedia article per document
  data/items.jsonl               {question, answer, evidences} records

Requires: huggingface_hub, pandas, pyarrow
Run with: python datasets/triviaqa/load.py [--max-documents 1000]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "mandarjoshi/trivia_qa"
PARQUET = "rc.wikipedia/{split}-00000-of-00001.parquet"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ITEMS_PATH = DATA_DIR / "items.jsonl"

# Occurrences of the answer string in one document, beyond which the rest are
# not worth recording.
MAX_OCCURRENCES = 50


def select(frame, max_documents):
    """Take questions in a stable order until the document budget is spent.

    A question is kept only when every page it points at is inside the budget,
    so no item is left with partial evidence.
    """
    pages = {}
    rows = []
    for row in frame.sort_values("question_id").itertuples(index=False):
        entity = row.entity_pages
        filenames = list(entity["filename"])
        if not filenames:
            continue
        new = [name for name in filenames if name not in pages]
        if len(pages) + len(new) > max_documents:
            if new:  # taking this question would overshoot the budget
                continue
        for name, context in zip(filenames, entity["wiki_context"]):
            pages.setdefault(name, context)
        rows.append(row)
    return pages, rows


def answer_strings(answer):
    """Every surface form of the answer, longest first."""
    candidates = {text for text in list(answer["aliases"]) + [answer["value"]] if text}
    return sorted(candidates, key=len, reverse=True)


def find_occurrences(aliases, plain, doc_id):
    """Locate the longest answer alias that occurs in this document."""
    lowered = plain.lower()
    for alias in aliases:
        needle = alias.lower()
        starts = []
        at = lowered.find(needle)
        while at != -1 and len(starts) < MAX_OCCURRENCES:
            starts.append(at)
            at = lowered.find(needle, at + 1)
        if starts:
            return [{"doc_id": doc_id, "text": plain[start:start + len(alias)],
                     "start": start, "end": start + len(alias)} for start in starts]
    return []


def document_id(filename):
    """Turn `Andrew_Lloyd_Webber.txt` into a filesystem-safe document id."""
    stem = filename[:-4] if filename.endswith(".txt") else filename
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).rstrip(". ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents", type=int, default=1000,
        help="Cap on the number of Wikipedia articles. The full validation split "
             "has 10,033 pages totalling 234M chars, too large to index locally; "
             "the default of 1,000 matches the subset size used in the chunking "
             "literature. Pass 0 for no cap.",
    )
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    args = parser.parse_args()

    path = hf_hub_download(REPO_ID, PARQUET.format(split=args.split), repo_type="dataset")
    frame = pd.read_parquet(path)
    print(f"{args.split}: {len(frame):,} questions available")

    pages, rows = select(frame, args.max_documents or len(frame) * 100)
    print(f"Selected {len(pages):,} articles covering {len(rows):,} questions")

    # A previous build may have written articles this one no longer selects.
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DOCUMENTS_DIR.iterdir():
        stale.unlink()
    written = {}
    lengths = []
    for filename, plain in pages.items():
        doc_id = document_id(filename)
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(plain, encoding="utf-8", newline="\n")
        written[filename] = (doc_id, plain)
        lengths.append(len(plain))

    lengths.sort()
    print(f"Wrote {len(lengths):,} documents to {DOCUMENTS_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    items = []
    with_occurrences = 0
    for row in rows:
        aliases = answer_strings(row.answer)
        doc_ids, occurrences = [], []
        for filename in row.entity_pages["filename"]:
            doc_id, plain = written[filename]
            doc_ids.append(doc_id)
            occurrences.extend(find_occurrences(aliases, plain, doc_id))
        if occurrences:
            with_occurrences += 1
        items.append({
            "id": f"triviaqa-{row.question_id}",
            "question": row.question,
            "answer": row.answer["value"],
            # Document-level annotation only; see the note at the top.
            "evidences": [],
            "metadata": {
                "doc_ids": doc_ids,
                "split": args.split,
                "answer_type": row.answer["type"],
                "aliases": list(row.answer["aliases"]),
                "question_source": row.question_source,
                "answer_occurrences": occurrences,
            },
        })

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    links = sum(len(item["metadata"]["doc_ids"]) for item in items)
    print(f"  evidence is document-level: {links:,} document links, "
          f"{links / len(items):.1f} per question")
    print(f"  {with_occurrences:,} items ({100 * with_occurrences / len(items):.1f}%) have the answer "
          f"string located in a gold document (distant supervision)")


if __name__ == "__main__":
    main()
