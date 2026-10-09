"""Load and process SQuAD into the common RAG-experiments format.

SQuAD 1.1 (https://arxiv.org/abs/1606.05250) is extractive question answering
over Wikipedia: crowdworkers wrote questions about a paragraph and marked the
answer inside it, so every question has an exact character span. Development
questions carry up to six independent answers, one per annotator.

WHY 1.1 AND NOT 2.0: SQuAD 2.0 adds questions written to be unanswerable, but
unanswerable *from the single paragraph the annotator was shown*. This loader
puts each paragraph back into its whole article, which invalidates that label --
the rest of the article may well answer the question. Measured on the grouped
corpus, 27% of 2.0's unanswerable questions had a near-duplicate answerable
question elsewhere in the same article, and with no gold answer the label cannot
be checked directly, only contradicted. SQuAD 1.1 has no such questions, so
nothing has to be discarded. For abstention, use TechQA (real forum questions no
technote resolves), Qasper or Natural Questions, where annotators judged the
whole document.

WHY THIS REFETCHES THE ARTICLES: SQuAD is distributed as individual paragraphs of
about 700 characters, far too short for chunking to mean anything. Joining the
paragraphs that share a title does not rebuild the article. The paper (section
3) says paragraphs shorter than 500 characters were discarded, along with
images, figures and tables, so the joined text has holes; and the paragraphs
are not stored in article order (Warsaw and Nikola Tesla, for instance, are
shuffled). So this script fetches each article as it stood on REVISION_DATE,
builds the document from that revision's paragraphs, and keeps a question only
if its paragraph appears there verbatim (whitespace aside). Answer offsets are
paragraph-relative, so they shift by the paragraph's position and stay exact,
with no fuzzy matching anywhere.

The paper gives no snapshot date, and the paragraphs come from several
revisions: Super Bowl 50 needs a revision from after the February 2016 game,
while Normans and Nikola Tesla match best on late 2015. One date for every
article is a deliberate simplification. On 2016-06-01, the month SQuAD was
released, 82% of the 2,067 validation paragraphs appear verbatim and 79% of
the questions are kept. The loss is concentrated: Super Bowl 50, Oxygen, Kenya,
University of Chicago and Nikola Tesla keep under half their questions. The
script prints every article that keeps under half its paragraphs. Looser
matching was tried on Nikola Tesla: 86 of its 92 paragraphs reach a difflib
ratio of 0.9, and 93% of answers then map onto identical text, but the rest
land on edited wording ("killed through overwork" became "die through
overwork"), so it was not adopted.

PAGE MOVES: a title can belong to another page today than it did in 2016.
In January 2023 "Private school" and "Independent school" swapped histories, so
the page SQuAD used now lives under "Independent school", and following the
current redirect lands on a different 12 kB article that matches no paragraph.
Each title is therefore resolved through the move log -- every move away from it
after REVISION_DATE, followed to the end -- and the revision is read from the
page that held the title on that date, without following redirects. Eleven of
the 100 default articles live under another title today; for the swap, and for
"Financial crisis of 2007–08" (swapped again in 2025), the redirect would have
led to the wrong page.

Paragraphs are the revision's <p> elements, outside tables, with footnote
markers and edit links removed and whitespace collapsed. Keeping the page-number
citations SQuAD sometimes shows (":121,154") matched no extra paragraph.

Only plain text is written. SQuAD keeps no section headings, so there is no
structure to mark; the article title is the sole heading and would not make a
Markdown rendering worth comparing against the text one.

Outputs:
  data/documents/<article>.txt   one Wikipedia article per document
  data/items.jsonl               {question, answer, evidences} records

Requires: huggingface_hub, pandas, pyarrow, requests, beautifulsoup4
Run with: python datasets/squad/load.py [--max-documents 100] [--delay 0.3]
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download

REPO_ID = "rajpurkar/squad"
SHARD = "plain_text/{split}-00000-of-00001.parquet"
API = "https://en.wikipedia.org/w/api.php"
# Wikipedia requires a descriptive User-Agent; requests without one are refused.
USER_AGENT = "rag-experiments/0.1 (research dataset preparation; contact via repository)"
# The revision of every article is the one live at this moment.
REVISION_DATE = "2016-06-01T00:00:00Z"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ITEMS_PATH = DATA_DIR / "items.jsonl"
CACHE_DIR = ROOT / ".cache" / f"revisions-{REVISION_DATE[:10]}"

PARAGRAPH_SEPARATOR = "\n\n"


def document_id(title):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(". ")[:120] or "untitled"


def fold(text):
    """Collapse whitespace runs to one space and strip the ends.

    Returns the folded text and, for each index of `text` plus its end, the
    index it lands on in the folded text, so that offsets can be carried over.
    """
    folded, positions = [], []
    pending_space = False
    for char in text:
        if char.isspace():
            # A run becomes one space at len(folded), written before the next
            # character; leading and trailing runs are dropped.
            pending_space = bool(folded)
            positions.append(len(folded))
            continue
        if pending_space:
            folded.append(" ")
            pending_space = False
        positions.append(len(folded))
        folded.append(char)
    positions.append(len(folded))
    return "".join(folded), positions


def call(session, delay, params, attempts=5):
    """One API request, honouring the Retry-After of a 429."""
    for _ in range(attempts):
        time.sleep(delay)
        response = session.get(API, params={**params, "format": "json", "formatversion": 2}, timeout=120)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", 11)))
            continue
        response.raise_for_status()
        return response.json()
    raise requests.RequestException(f"rate limited after {attempts} attempts")


def resolve(session, delay, title):
    """Find the page that held `title` on REVISION_DATE and its revision then.

    Follows every move away from the title made after that date, so the answer is
    the page's title today. Returns a dict with `title`, `moves`, `revid` and
    `timestamp`, or None if the page had no revision by that date.
    """
    current, since, after, moves = title, REVISION_DATE, 0, []
    while True:
        events = call(session, delay, {
            "action": "query", "list": "logevents", "letype": "move", "letitle": current,
            "lestart": since, "ledir": "newer", "lelimit": 10,
        })["query"]["logevents"]
        # A title swap logs its moves within one second, the other page leaving
        # the title just before ours arrives ("2007–2008 financial crisis",
        # January 2025), so only a move logged after our arrival counts.
        event = next((e for e in events if e["logid"] > after), None)
        if event is None:
            break
        target = event["params"]["target_title"]
        moves.append({"timestamp": event["timestamp"], "from": current, "to": target})
        current, since, after = target, event["timestamp"], event["logid"]

    # No `redirects`: a redirect points at whichever page holds a title today.
    pages = call(session, delay, {
        "action": "query", "prop": "revisions", "titles": current, "rvlimit": 1,
        "rvstart": REVISION_DATE, "rvdir": "older", "rvprop": "ids|timestamp",
    })["query"]["pages"]
    revisions = pages[0].get("revisions") if pages else None
    if not revisions:
        return None
    return {"title": current, "moves": moves,
            "revid": revisions[0]["revid"], "timestamp": revisions[0]["timestamp"]}


def fetch(session, delay, title):
    """The resolved revision of an article and its rendered HTML, cached on disk."""
    cached = CACHE_DIR / f"{document_id(title)}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    revision = resolve(session, delay, title)
    if revision is not None:
        revision["html"] = call(session, delay, {
            "action": "parse", "oldid": revision["revid"], "prop": "text",
        })["parse"]["text"]
    cached.write_text(json.dumps(revision, ensure_ascii=False), encoding="utf-8", newline="\n")
    return revision


def paragraphs(html):
    """The article's body paragraphs, as SQuAD extracted them.

    SQuAD stripped tables, and footnote markers never appear in its paragraphs.
    """
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("sup.reference, .mw-editsection"):
        element.decompose()
    blocks = []
    for paragraph in soup.find_all("p"):
        if paragraph.find_parent("table"):
            continue
        text = fold(paragraph.get_text())[0]
        if text:
            blocks.append(text)
    return blocks


def build_document(blocks):
    """Join paragraphs, returning the text and each paragraph's first offset."""
    starts = {}
    at = 0
    for block in blocks:
        starts.setdefault(block, at)
        at += len(block) + len(PARAGRAPH_SEPARATOR)
    return PARAGRAPH_SEPARATOR.join(blocks), starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents", type=int, default=100,
        help="Cap on the number of articles kept. Validation supplies 48 and the "
             "rest are filled from train, which keeps the corpus large enough that "
             "retrieval is not trivial. An article none of whose paragraphs is "
             "found is skipped and the next one taken. Pass 0 for no cap (490 "
             "articles, one fetch each).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "train"],
        choices=["validation", "train"],
        help="Splits to draw articles from, in order of preference.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between Wikipedia requests (cached articles are free).",
    )
    args = parser.parse_args()

    frames = []
    for split in args.splits:
        frame = pd.read_parquet(hf_hub_download(REPO_ID, SHARD.format(split=split), repo_type="dataset"))
        frame["split"] = split
        frames.append(frame)
        print(f"{split}: {len(frame):,} questions, {frame['title'].nunique()} articles")
    data = pd.concat(frames, ignore_index=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # Articles are taken split by split, so validation is always included whole.
    documents = {}
    stats = dict.fromkeys(["paragraphs", "paragraphs_found", "questions", "questions_kept"], 0)
    weak, skipped, moved = [], [], []
    for title, group in data.groupby("title", sort=False):
        if args.max_documents and len(documents) == args.max_documents:
            break
        name = urllib.parse.unquote(title).replace("_", " ")
        revision = fetch(session, args.delay, name)
        contexts = list(dict.fromkeys(group["context"]))
        stats["paragraphs"] += len(contexts)
        stats["questions"] += len(group)
        if revision is None:
            skipped.append(f"{title} (no revision by {REVISION_DATE[:10]})")
            continue
        text, starts = build_document(paragraphs(revision["html"]))
        found = {context: starts[fold(context)[0]] for context in contexts if fold(context)[0] in starts}
        kept = int(group["context"].isin(found).sum())
        stats["paragraphs_found"] += len(found)
        stats["questions_kept"] += kept
        if revision["title"] != name:
            moved.append(f"{title} -> {revision['title']}")
        if not found:
            skipped.append(f"{title} (no paragraph found)")
            continue
        if len(found) < len(contexts) / 2:
            weak.append(f"{title}: {len(found)}/{len(contexts)} paragraphs, {kept}/{len(group)} questions")
        documents[title] = (document_id(title), text, found, revision)

    # A previous build may have written articles this one no longer keeps.
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DOCUMENTS_DIR.glob("*.txt"):
        stale.unlink()
    lengths, counts = [], []
    for doc_id, text, _, _ in documents.values():
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
        lengths.append(len(text))
        counts.append(text.count(PARAGRAPH_SEPARATOR) + 1)

    lengths.sort()
    counts.sort()
    print(f"Wrote {len(lengths)} documents to {DOCUMENTS_DIR} "
          f"(revisions live on {REVISION_DATE[:10]})")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")
    print(f"  paragraphs per document: min {counts[0]} / median "
          f"{counts[len(counts) // 2]} / max {counts[-1]}")
    print(f"  SQuAD paragraphs found verbatim: {stats['paragraphs_found']:,} of "
          f"{stats['paragraphs']:,} ({stats['paragraphs_found'] / stats['paragraphs']:.0%}); "
          f"questions kept: {stats['questions_kept']:,} of {stats['questions']:,} "
          f"({stats['questions_kept'] / stats['questions']:.0%})")
    for label, titles in [("moved since", moved), ("under half the paragraphs found", weak),
                          ("skipped", skipped)]:
        if titles:
            print(f"  {len(titles)} articles {label}:")
            for line in titles:
                print(f"    {line}")

    items = []
    located = mismatched = empty = 0
    for row in data[data["title"].isin(documents)].itertuples(index=False):
        doc_id, text, found, revision = documents[row.title]
        if row.context not in found:
            continue
        base = found[row.context]
        context, positions = fold(row.context)

        evidences = []
        seen = set()
        answers = row.answers
        for answer_text, answer_start in zip(answers["text"], answers["answer_start"]):
            answer_start = int(answer_start)
            start = base + positions[answer_start]
            end = base + positions[answer_start + len(answer_text)]
            if text[start:end] != fold(answer_text)[0]:
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
                "wikipedia_title": revision["title"],
                "revision_id": revision["revid"],
                "answer_type": "extractive",
                "answers": list(answers["text"]),
                "annotators": len(answers["text"]),
                # The paragraph the question was written against, which is the
                # gold passage a retriever should return.
                "context_span": {
                    "doc_id": doc_id,
                    "start": base,
                    "end": base + len(context),
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
