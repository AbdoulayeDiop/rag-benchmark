"""Load and process PoQuAD into the common RAG-experiments format.

PoQuAD (https://github.com/weed478/poquad) is the Polish Question Answering
Dataset, modelled on SQuAD 2.0: extractive answer spans plus deliberately
unanswerable questions, over Polish Wikipedia, with an extra generative answer
layer. It is the only non-English dataset in this repository.

WHY THIS SCRIPT REBUILDS THE DOCUMENTS: as distributed, every PoQuAD record is a
single paragraph of about 930 characters (964 of them in dev, 7,708 in train).
That is a passage collection, not a document collection -- chunking has nothing
to do at that size. Each record does carry the URL of the Wikipedia article its
paragraph was taken from, so this script fetches the full article and relocates
the answer spans into it, producing documents of roughly 24k characters that
still carry exact character-level evidence.

The cost is drift: the paragraphs were extracted in 2022 and Wikipedia has moved
on, so a paragraph that no longer appears in its article cannot be relocated and
its questions are dropped. The script reports exactly how many were lost.

Answer spans are the evidence. For unanswerable questions `evidences` is empty by
construction, and `metadata.context_span` records the passage the question was
written against, which stays useful for checking whether retrieval finds the
right region even when no answer exists.

Outputs:
  data/documents/<article>.txt      one Wikipedia article per document
  data/documents_md/<article>.md    the same article with Markdown headings
  data/items.jsonl                  {question, answer, evidences} records

Requires: requests
Run with: python datasets/poquad/load.py [--splits dev train]
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests

SOURCE = "https://raw.githubusercontent.com/weed478/poquad/master/poquad_{split}.json"
API = "https://pl.wikipedia.org/w/api.php"
# Wikipedia requires a descriptive User-Agent; requests without one are refused.
USER_AGENT = "rag-experiments/0.1 (research dataset preparation; contact via repository)"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MARKDOWN_DIR = DATA_DIR / "documents_md"
ITEMS_PATH = DATA_DIR / "items.jsonl"
CACHE_DIR = ROOT / ".cache"

BLOCK_SEPARATOR = "\n\n"
# MediaWiki plain-text extracts keep section headings as "== Name ==", where the
# number of equals signs is the real heading level.
HEADING_PATTERN = re.compile(r"^(=+)\s*(.+?)\s*\1$")


def fold(text):
    """Collapse whitespace, returning the folded text and an offset map."""
    folded, offsets = [], []
    after_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not after_space:
                folded.append(" ")
                offsets.append(index)
                after_space = True
            continue
        after_space = False
        folded.append(char)
        offsets.append(index)
    offsets.append(len(text))
    return "".join(folded), offsets


def find(needle, haystack, folded_haystack, offsets, base=0):
    """Locate `needle`, verbatim then on the folded text. Returns (start, end, how)."""
    at = haystack.find(needle)
    if at != -1:
        return base + at, base + at + len(needle), "exact"
    target = re.sub(r"\s+", " ", needle).strip()
    if not target:
        return None
    at = folded_haystack.find(target)
    if at == -1:
        return None
    end = at + len(target)
    return base + offsets[at], base + offsets[end - 1] + 1, "folded"


def article_title(url):
    """The Wikipedia title a PoQuAD record points at."""
    return urllib.parse.unquote(url.rsplit("/", 1)[-1]).replace("_", " ")


def safe_name(title):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).rstrip(". ")[:120]


def fetch_extract(session, title, delay, attempts=5):
    """Download an article's plain-text extract, cached on disk.

    Whole-article extracts are expensive to render, so Wikipedia rate-limits them
    hard: `exlimit` is forced to 1 (no batching) and roughly one request in ten
    comes back 429 with a Retry-After header. Honouring that header is what keeps
    this to a few seconds per article -- guessing a backoff instead turns it into
    about a minute per article.
    """
    cached = CACHE_DIR / f"{safe_name(title)}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    for _ in range(attempts):
        time.sleep(delay)
        response = session.get(API, params={
            "action": "query", "prop": "extracts", "explaintext": 1,
            "titles": title, "format": "json", "redirects": 1, "formatversion": 2,
        }, timeout=60)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", 11)))
            continue
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        extract = pages[0].get("extract", "") if pages else ""
        cached.write_text(extract, encoding="utf-8")
        return extract
    raise requests.RequestException(f"rate limited after {attempts} attempts")


def render(extract):
    """Split an extract into blocks and render plain text and Markdown.

    Headings keep their text in both files and their level only in Markdown, so
    the two renderings hold the same words and differ only in structure.
    """
    plain_parts, markdown_parts, spans = [], [], []
    plain_at = markdown_at = 0
    for block in re.split(r"\n{2,}", extract):
        block = block.strip()
        if not block:
            continue
        match = HEADING_PATTERN.match(block)
        if match:
            level = min(len(match.group(1)), 6)
            content = match.group(2)
            rendered = f"{'#' * level} {content}"
            shift = markdown_at + level + 1 - plain_at
        else:
            content = block
            rendered = block
            shift = markdown_at - plain_at
        spans.append({
            "plain": (plain_at, plain_at + len(content)),
            "markdown": (markdown_at, markdown_at + len(rendered)),
            "shift": shift,
        })
        plain_parts.append(content)
        markdown_parts.append(rendered)
        plain_at += len(content) + len(BLOCK_SEPARATOR)
        markdown_at += len(rendered) + len(BLOCK_SEPARATOR)
    return BLOCK_SEPARATOR.join(plain_parts), BLOCK_SEPARATOR.join(markdown_parts), spans


def to_markdown_span(start, end, spans):
    """Map a plain-text span onto the Markdown rendering."""
    for span in spans:
        plain_start, plain_end = span["plain"]
        if plain_start <= start and end <= plain_end:
            return start + span["shift"], end + span["shift"]
    return None


def build(split, session, delay, stats):
    """Fetch articles for one split and turn its questions into items."""
    payload = requests.get(SOURCE.format(split=split), timeout=300).json()["data"]
    print(f"{split}: {len(payload)} records, "
          f"{sum(len(p['qas']) for a in payload for p in a['paragraphs'])} questions")

    documents = {}
    items = []
    for record in payload:
        title = article_title(record["url"])
        doc_id = safe_name(title)

        if doc_id not in documents:
            try:
                extract = fetch_extract(session, title, delay)
            except requests.RequestException as error:
                print(f"  ! {title}: {error}")
                stats["fetch_failed"] += 1
                continue
            if not extract.strip():
                stats["fetch_empty"] += 1
                continue
            plain, markdown, spans = render(extract)
            folded, offsets = fold(plain)
            documents[doc_id] = (plain, markdown, spans, folded, offsets)

        plain, markdown, spans, folded, offsets = documents[doc_id]

        for paragraph in record["paragraphs"]:
            context = paragraph["context"]
            located = find(context, plain, folded, offsets)
            if located is None:
                # The article moved on since PoQuAD was built.
                stats["context_lost"] += 1
                stats["questions_lost"] += len(paragraph["qas"])
                continue
            context_start, context_end, _ = located
            stats["context_found"] += 1
            region = plain[context_start:context_end]
            region_folded, region_offsets = fold(region)

            for index, qa in enumerate(paragraph["qas"]):
                evidences = []
                answers = qa.get("answers") or []
                for answer in answers:
                    span = find(answer["text"], region, region_folded, region_offsets, context_start)
                    if span is None:
                        stats["answer_lost"] += 1
                        continue
                    start, end, how = span
                    markdown_span = to_markdown_span(start, end, spans)
                    stats["answer_found"] += 1
                    evidences.append({
                        "doc_id": doc_id,
                        "text": plain[start:end],
                        "start": start,
                        "end": end,
                        "match": how,
                        "kind": "answer_span",
                        "markdown": (
                            {"start": markdown_span[0], "end": markdown_span[1]}
                            if markdown_span else None
                        ),
                    })

                impossible = bool(qa.get("is_impossible"))
                context_markdown = to_markdown_span(context_start, context_end, spans)
                items.append({
                    "id": f"poquad-{split}-{record['id']}-{index:02d}",
                    "question": qa["question"],
                    "answer": "Unanswerable" if impossible else (
                        answers[0]["text"] if answers else "Unanswerable"
                    ),
                    "evidences": [] if impossible else evidences,
                    "metadata": {
                        "doc_id": doc_id,
                        "title": record["title"],
                        "url": record["url"],
                        "split": split,
                        "language": "pl",
                        "answer_type": "unanswerable" if impossible else "extractive",
                        "generative_answer": (
                            answers[0].get("generative_answer") if answers else None
                        ),
                        "answers": [answer["text"] for answer in answers],
                        # The passage the question was written against, relocated
                        # into the full article.
                        "context_span": {
                            "doc_id": doc_id,
                            "start": context_start,
                            "end": context_end,
                            "markdown": (
                                {"start": context_markdown[0], "end": context_markdown[1]}
                                if context_markdown else None
                            ),
                        },
                    },
                })
    return documents, items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["dev"], choices=["dev", "train"],
        help="Splits to build. Defaults to dev (964 articles); train is 7,708 "
             "articles and so 7,708 Wikipedia requests. The test split is not public.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between Wikipedia requests (cached articles are free).",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    stats = dict.fromkeys(
        ["fetch_failed", "fetch_empty", "context_found", "context_lost",
         "questions_lost", "answer_found", "answer_lost"], 0
    )
    all_documents, all_items = {}, []
    for split in args.splits:
        documents, items = build(split, session, args.delay, stats)
        all_documents.update(documents)
        all_items.extend(items)

    lengths = []
    for doc_id, (plain, markdown, _, _, _) in all_documents.items():
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(plain, encoding="utf-8", newline="\n")
        (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        lengths.append(len(plain))
    lengths.sort()
    print(f"Wrote {len(lengths):,} documents to {DOCUMENTS_DIR} and {MARKDOWN_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in all_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    unanswerable = sum(1 for item in all_items if item["metadata"]["answer_type"] == "unanswerable")
    print(f"Wrote {len(all_items):,} items to {ITEMS_PATH}")
    print(f"  {len(all_items) - unanswerable:,} answerable, {unanswerable:,} unanswerable")
    print(f"  passages relocated into their article: {stats['context_found']:,}; "
          f"lost to article drift: {stats['context_lost']:,} "
          f"(dropping {stats['questions_lost']:,} questions)")
    print(f"  answer spans located: {stats['answer_found']:,}; not found inside the passage: "
          f"{stats['answer_lost']:,}")
    if stats["fetch_failed"] or stats["fetch_empty"]:
        print(f"  articles unavailable: {stats['fetch_failed']} failed, {stats['fetch_empty']} empty")


if __name__ == "__main__":
    main()
