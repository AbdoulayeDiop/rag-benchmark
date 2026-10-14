"""Load and process PoQuAD into the common RAG-experiments format.

PoQuAD (https://github.com/weed478/poquad) is the Polish Question Answering
Dataset, modelled on SQuAD 2.0: extractive answer spans plus deliberately
unanswerable questions, over Polish Wikipedia, with an extra generative answer
layer. It is the only non-English dataset in this repository.

WHY THIS REFETCHES THE ARTICLES: as distributed, every PoQuAD record is a single
passage of about 930 characters (964 of them in dev, 7,708 in train). That is a
passage collection, not a document collection -- chunking has nothing to do at
that size. Each record carries the URL of the Wikipedia article its passage was
taken from, so this script fetches the article and relocates the passage and its
answer spans into it.

WHICH REVISION: the articles have been edited since PoQuAD was built, and
fetching them as they are today loses every passage that was edited since: 379
of 959 dev passages, and 1,837 questions, when this script fetched live
extracts. So each article is fetched as it stood on REVISION_DATE, the date of
the last data update in the PoQuAD repository (1 September 2022). The records
give no revision of their own, and the passages were written over the months
before that date, so one date for every article is a simplification, as it is
for SQuAD.

PAGE MOVES: as in the SQuAD loader, a title is resolved through the move log --
every move away from it after REVISION_DATE, followed to the end -- and the
revision is read from the page that held the title on that date, without
following redirects, which point at whichever page holds a title today.

Paragraphs are the revision's <p> elements, outside tables, with footnote
markers and edit links removed and whitespace collapsed, joined by a blank line.
A passage is located on the document with all whitespace collapsed, so a
passage that spans two paragraphs is found too, and its offsets are carried
back to the document. Answer offsets are passage-relative, so they shift by the
passage's position and stay exact.

Only plain text is written. Section headings are not <p> elements, so the
document holds none, and there is no structure for a Markdown rendering to mark.

EVIDENCE IS THE PASSAGE, NOT THE ANSWER: an item's one evidence span is the
passage its question was written against, relocated into the article, and the
answer spans inside it are kept in `metadata.answer_spans`, as for SQuAD. An
answer of a few words made `evidence_recall` nearly the same as `hit`, credited
a chunk that holds the answer words but cuts off the sentence that makes them an
answer, and, with several annotators' answers pooled, counted as all needed
spans of which any one is enough. The passage is what a retriever should return.

Only answerable questions are kept: the
experiments score retrieval, and an unanswerable question has nothing to
retrieve. An article that keeps no question is not written.

Outputs:
  data/documents/<article>.txt      one Wikipedia article per document
  data/items.jsonl                  {question, answer, evidences} records; the
                                    evidence is the passage, the answer spans
                                    are in metadata.answer_spans

Requires: requests, beautifulsoup4, tqdm
Run with: python datasets/poquad/load.py [--splits dev train] [--delay 0.3]
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SOURCE = "https://raw.githubusercontent.com/weed478/poquad/master/poquad_{split}.json"
API = "https://pl.wikipedia.org/w/api.php"
# Wikipedia requires a descriptive User-Agent; requests without one are refused.
USER_AGENT = "rag-experiments/0.1 (research dataset preparation; contact via repository)"
# The revision of every article is the one live at this moment: the last data
# update of the PoQuAD repository.
REVISION_DATE = "2022-09-01T00:00:00Z"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ITEMS_PATH = DATA_DIR / "items.jsonl"
CACHE_DIR = ROOT / ".cache"
REVISIONS_DIR = CACHE_DIR / f"revisions-{REVISION_DATE[:10]}"

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


def unfold(text):
    """The folded text of `text`, and the index in `text` of each folded character.

    The inverse of `fold`'s positions: a folded span `[a, b)` that starts and
    ends on a character other than a space is `text[back[a]:back[b - 1] + 1]`.
    """
    folded, positions = fold(text)
    back = [0] * len(folded)
    for index, char in enumerate(text):
        if not char.isspace():
            back[positions[index]] = index
    return folded, back


def source(split):
    """PoQuAD's records for one split, downloaded once."""
    cached = CACHE_DIR / f"poquad_{split}.json"
    if not cached.exists():
        response = requests.get(SOURCE.format(split=split), timeout=300)
        response.raise_for_status()
        cached.write_bytes(response.content)
    return json.loads(cached.read_text(encoding="utf-8"))["data"]


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
        # the title just before ours arrives, so only a move logged after our
        # arrival counts.
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
    cached = REVISIONS_DIR / f"{document_id(title)}.json"
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
    """The article's body paragraphs: <p> elements outside tables, without footnote markers."""
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


def article_title(url):
    """The Wikipedia title a PoQuAD record points at."""
    return urllib.parse.unquote(url.rsplit("/", 1)[-1]).replace("_", " ")


def build(records, split, session, delay, stats, documents, skipped, weak, moved):
    """Fetch the articles of one split's records and turn its questions into items.

    `documents` maps an article title to `(doc_id, text, revision)` and is
    shared between splits, so an article is fetched and written once.
    """
    by_title = {}
    for record in records:
        by_title.setdefault(article_title(record["url"]), []).append(record)

    items = []
    bar = tqdm(by_title.items(), desc=f"{split} articles", unit="article")
    for title, group in bar:
        passages = [(record, paragraph) for record in group for paragraph in record["paragraphs"]]
        stats["passages"] += len(passages)
        stats["questions"] += sum(len(paragraph["qas"]) for _, paragraph in passages)
        if title not in documents:
            revision = fetch(session, delay, title)
            if revision is None:
                skipped.append(f"{title} (no revision by {REVISION_DATE[:10]})")
                tqdm.write(f"  skipped {skipped[-1]}")
                continue
            text = PARAGRAPH_SEPARATOR.join(paragraphs(revision["html"]))
            documents[title] = (document_id(title), text, revision, unfold(text), [])
            if revision["title"] != title:
                moved.append(f"{title} -> {revision['title']}")
        doc_id, text, revision, (folded, back), _ = documents[title]

        found = 0
        for record, paragraph in passages:
            context, positions = fold(paragraph["context"])
            at = folded.find(context) if context else -1
            if at == -1:
                continue
            found += 1
            stats["passages_found"] += 1
            context_start, context_end = back[at], back[at + len(context) - 1] + 1

            for index, qa in enumerate(paragraph["qas"]):
                answers = qa.get("answers") or []
                if qa.get("is_impossible") or not answers:
                    stats["unanswerable_dropped"] += 1
                    continue
                stats["questions_kept"] += 1
                answer_spans = []
                seen = set()
                for answer in answers:
                    answer_text = fold(answer["text"])[0]
                    start = positions[answer["answer_start"]]
                    end = positions[answer["answer_start"] + len(answer["text"])]
                    if not answer_text or folded[at + start:at + end] != answer_text:
                        stats["answers_mismatched"] += 1
                        continue
                    span = (back[at + start], back[at + end - 1] + 1)
                    if span in seen:
                        continue
                    seen.add(span)
                    stats["answers_located"] += 1
                    answer_spans.append({
                        "doc_id": doc_id,
                        "text": text[span[0]:span[1]],
                        "start": span[0],
                        "end": span[1],
                        "match": "offset",
                        "kind": "answer_span",
                    })
                if not answer_spans:
                    # An answerable question whose answer did not survive the
                    # relocation: it cannot be scored, so it is left out.
                    stats["questions_kept"] -= 1
                    stats["answerable_dropped"] += 1
                    continue

                items.append({
                    "id": f"poquad-{split}-{record['id']}-{index:02d}",
                    "question": qa["question"],
                    "answer": answers[0]["text"],
                    # The passage the question was written against is the
                    # evidence, not the answer spans inside it (see the module
                    # docstring).
                    "evidences": [{
                        "doc_id": doc_id,
                        "text": text[context_start:context_end],
                        "start": context_start,
                        "end": context_end,
                        "match": "folded",
                        "kind": "passage",
                    }],
                    "metadata": {
                        "doc_id": doc_id,
                        "title": record["title"],
                        "url": record["url"],
                        "split": split,
                        "language": "pl",
                        "wikipedia_title": revision["title"],
                        "revision_id": revision["revid"],
                        "answer_type": "extractive",
                        "generative_answer": answers[0].get("generative_answer"),
                        "answers": [answer["text"] for answer in answers],
                        "answer_spans": answer_spans,
                    },
                })
        documents[title][4].append(found)
        if not found:
            skipped.append(f"{title} (no passage found)")
            tqdm.write(f"  skipped {skipped[-1]}")
        elif found < len(passages) / 2:
            weak.append(f"{title}: {found}/{len(passages)} passages")
        bar.set_postfix(passages=f"{stats['passages_found']}/{stats['passages']}",
                        questions=f"{stats['questions_kept']}/{stats['questions']}",
                        skipped=len(skipped))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["dev"], choices=["dev", "train"],
        help="Splits to build. Defaults to dev (947 articles); train is 6,701 "
             "articles, each three Wikipedia requests. The test split is not public.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between Wikipedia requests (cached articles are free).",
    )
    args = parser.parse_args()

    REVISIONS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    stats = dict.fromkeys(["passages", "passages_found", "questions", "questions_kept",
                           "answers_located", "answers_mismatched", "answerable_dropped",
                           "unanswerable_dropped"], 0)
    documents, skipped, weak, moved = {}, [], [], []
    items = []
    for split in args.splits:
        records = source(split)
        print(f"{split}: {len(records):,} passages, "
              f"{sum(len(p['qas']) for r in records for p in r['paragraphs']):,} questions")
        items += build(records, split, session, args.delay, stats, documents, skipped, weak,
                       moved)

    # An article is written only if it keeps a question.
    used = {item["metadata"]["doc_id"] for item in items}
    kept = {title: entry for title, entry in documents.items() if entry[0] in used}
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DOCUMENTS_DIR.glob("*.txt"):
        stale.unlink()
    lengths = []
    for doc_id, text, _, _, _ in kept.values():
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
        lengths.append(len(text))

    for item in items:
        document = (DOCUMENTS_DIR / f"{item['metadata']['doc_id']}.txt").read_text(encoding="utf-8")
        for evidence in item["evidences"] + item["metadata"]["answer_spans"]:
            assert document[evidence["start"]:evidence["end"]] == evidence["text"], item["id"]

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    lengths.sort()
    print(f"Wrote {len(lengths):,} documents to {DOCUMENTS_DIR} "
          f"(revisions live on {REVISION_DATE[:10]})")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")
    print(f"  passages found: {stats['passages_found']:,} of {stats['passages']:,} "
          f"({stats['passages_found'] / stats['passages']:.0%}); questions kept: "
          f"{stats['questions_kept']:,} of {stats['questions']:,} "
          f"({stats['questions_kept'] / stats['questions']:.0%})")
    print(f"Wrote {len(items):,} answerable items to {ITEMS_PATH}; "
          f"{stats['unanswerable_dropped']:,} unanswerable questions left out")
    print(f"  evidence: the question's passage; answer spans located: {stats['answers_located']:,}; not matching their "
          f"offset: {stats['answers_mismatched']:,}, leaving "
          f"{stats['answerable_dropped']:,} answerable questions without a span (dropped)")
    for label, titles in [("moved since", moved), ("under half the passages found", weak),
                          ("skipped", skipped)]:
        if titles:
            print(f"  {len(titles)} articles {label}:")
            for line in titles[:40]:
                print(f"    {line}")
            if len(titles) > 40:
                print(f"    ... and {len(titles) - 40} more")


if __name__ == "__main__":
    main()
