"""Load and process TriviaQA into the common RAG-experiments format.

TriviaQA (https://huggingface.co/datasets/mandarjoshi/trivia_qa) pairs
trivia-enthusiast questions with evidence documents gathered independently of
the question writers. This script uses the `rc.wikipedia` configuration, whose
evidence is Wikipedia articles, rather than the web-search configuration whose
documents are scraped pages of uneven quality.

Split: validation. The test split ships `<unk>` as every answer -- it is held out
for the leaderboard -- so it cannot be used for evaluation here.

NOTE ON EVIDENCE: TriviaQA's annotation is document-level. A question is linked
to the Wikipedia pages it was matched against, not to a passage inside them, so
each entry in `evidences` spans a whole document and carries `text: null`
(the span is the document; repeating megabytes of article text per question
would bloat items.jsonl for nothing).

Because chunk-level scoring needs something finer, `metadata.answer_occurrences`
records where the answer string literally appears in each gold document -- 99.3%
of questions have at least one. That is distant supervision, inferred here and
not annotated by anyone, so it is kept out of `evidences` and should be treated
as an approximation of the supporting passage, not ground truth.

Wikipedia section headings survive in the source text as short lines with a
trailing space. They are detected heuristically and marked in the Markdown
rendering; the original nesting (h2 vs h3) is not recoverable, so every detected
heading is written at one level.

Outputs:
  data/documents/<page>.txt      one Wikipedia article per document
  data/documents_md/<page>.md    the same article with headings marked
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
MARKDOWN_DIR = DATA_DIR / "documents_md"
ITEMS_PATH = DATA_DIR / "items.jsonl"

BLOCK_SEPARATOR = "\n\n"
# A heading is a single short line kept on its own, with the trailing space left
# behind when the article's markup was stripped.
HEADING_LEVEL = "##"
HEADING_MAX_CHARS = 60
# Occurrences of the answer string in one document, beyond which the rest are
# not worth recording.
MAX_OCCURRENCES = 50


def is_heading(block):
    """Whether a block looks like a stripped Wikipedia section heading."""
    if "\n" in block or not block.endswith(" "):
        return False
    stripped = block.strip()
    return bool(stripped) and len(stripped) <= HEADING_MAX_CHARS and stripped[-1] not in ".?!,:;"


def render(context):
    """Split an article into blocks and render both versions.

    Returns (plain, markdown, spans) where spans[i] gives block i's position in
    each rendering, so a span found in one can be reported in the other.
    """
    blocks = context.split(BLOCK_SEPARATOR)
    plain_parts, markdown_parts, spans = [], [], []
    plain_at = markdown_at = 0
    for block in blocks:
        heading = is_heading(block)
        if heading:
            # "  Characteristics " becomes "## Characteristics", so the text
            # moves by the marker minus whatever leading space was dropped.
            content = block.strip()
            lead = len(block) - len(block.lstrip())
            rendered = f"{HEADING_LEVEL} {content}"
            content_start = plain_at + lead
            shift = markdown_at + len(HEADING_LEVEL) + 1 - content_start
        else:
            content = block
            rendered = block
            content_start = plain_at
            shift = markdown_at - plain_at
        spans.append({
            "plain": (plain_at, plain_at + len(block)),
            "markdown": (markdown_at, markdown_at + len(rendered)),
            "content": (content_start, content_start + len(content)),
            "shift": shift,
            "heading": heading,
        })
        plain_parts.append(block)
        markdown_parts.append(rendered)
        plain_at += len(block) + len(BLOCK_SEPARATOR)
        markdown_at += len(rendered) + len(BLOCK_SEPARATOR)
    # Joining the untouched blocks reproduces the source article exactly.
    return BLOCK_SEPARATOR.join(plain_parts), BLOCK_SEPARATOR.join(markdown_parts), spans


def to_markdown_span(start, end, spans):
    """Map a plain-text span onto the Markdown rendering.

    A block's text is unchanged between renderings -- only its position moves,
    and headings additionally lose surrounding whitespace to the marker -- so a
    span inside the block's content shifts by a fixed amount. Spans touching a
    heading's dropped whitespace have no counterpart and return None.
    """
    for span in spans:
        plain_start, plain_end = span["plain"]
        if plain_start <= start and end <= plain_end:
            content_start, content_end = span["content"]
            if content_start <= start and end <= content_end:
                return start + span["shift"], end + span["shift"]
            return None
    return None


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


def find_occurrences(aliases, plain, spans, doc_id):
    """Locate the longest answer alias that occurs in this document."""
    lowered = plain.lower()
    for alias in aliases:
        needle = alias.lower()
        starts = []
        at = lowered.find(needle)
        while at != -1 and len(starts) < MAX_OCCURRENCES:
            starts.append(at)
            at = lowered.find(needle, at + 1)
        if not starts:
            continue
        occurrences = []
        for start in starts:
            end = start + len(alias)
            markdown = to_markdown_span(start, end, spans)
            occurrences.append({
                "doc_id": doc_id,
                "text": plain[start:end],
                "start": start,
                "end": end,
                "markdown": (
                    {"start": markdown[0], "end": markdown[1]} if markdown else None
                ),
            })
        return occurrences
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

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    rendered = {}
    lengths = []
    headings = 0
    for filename, context in pages.items():
        doc_id = document_id(filename)
        plain, markdown, spans = render(context)
        assert plain == context, f"reconstruction changed {doc_id}"
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(plain, encoding="utf-8", newline="\n")
        (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        rendered[filename] = (doc_id, plain, spans, len(markdown))
        lengths.append(len(plain))
        headings += sum(1 for span in spans if span["heading"])

    lengths.sort()
    print(f"Wrote {len(lengths):,} documents to {DOCUMENTS_DIR} and {MARKDOWN_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")
    print(f"  {headings:,} section headings detected across the corpus")

    items = []
    with_occurrences = 0
    for row in rows:
        aliases = answer_strings(row.answer)
        evidences, occurrences = [], []
        for filename in row.entity_pages["filename"]:
            doc_id, plain, spans, markdown_length = rendered[filename]
            evidences.append({
                "doc_id": doc_id,
                "text": None,  # document-level annotation; see the note at the top
                "start": 0,
                "end": len(plain),
                "match": "document",
                "kind": "document",
                "markdown": {"start": 0, "end": markdown_length},
            })
            occurrences.extend(find_occurrences(aliases, plain, spans, doc_id))
        if occurrences:
            with_occurrences += 1
        items.append({
            "id": f"triviaqa-{row.question_id}",
            "question": row.question,
            "answer": row.answer["value"],
            "evidences": evidences,
            "metadata": {
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
    print(f"  evidence is document-level: {sum(len(i['evidences']) for i in items):,} "
          f"document links, {sum(len(i['evidences']) for i in items) / len(items):.1f} per question")
    print(f"  {with_occurrences:,} items ({100 * with_occurrences / len(items):.1f}%) have the answer "
          f"string located in a gold document (distant supervision)")


if __name__ == "__main__":
    main()
