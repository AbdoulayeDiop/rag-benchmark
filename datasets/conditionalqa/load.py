"""Load and process ConditionalQA into the common RAG-experiments format.

ConditionalQA (https://github.com/haitian-sun/ConditionalQA) asks questions about
UK government policy pages. Each question comes with a scenario describing the
asker's situation, and answers are often conditional -- true only when certain
clauses apply -- which is why a question can carry several answers, each with its
own list of conditions.

It is the best-annotated source in this repository: evidence is given as the
exact page elements supporting an answer, and all 10,284 evidence entries match a
document element verbatim, so every span resolves without any fuzzy matching.

The documents ship as lists of HTML elements (<h1>, <p>, <li>, <tr>), so all
three renderings are produced from the real markup rather than reconstructed:
plain text, Markdown, and HTML.

NOTE ON THE QUERY: a ConditionalQA question is meaningless without its scenario --
"How long will it be before I hear back from the court?" says nothing about who is
asking or what for. `question` therefore holds the scenario and the question
joined, which is what should be embedded; `metadata.scenario` and
`metadata.original_question` keep the two parts separately.

Splits: train and dev. The test split ships without answers.

Outputs:
  data/documents/<page>.txt      one policy page per document
  data/documents_md/<page>.md    the same page as Markdown
  data/documents_html/<page>.html   the same page as HTML
  data/items.jsonl               {question, answer, evidences} records

Requires: requests, markdownify
Run with: python datasets/conditionalqa/load.py [--splits train dev]
"""

import argparse
import html as html_module
import json
import re
import urllib.parse
from pathlib import Path

import requests
from markdownify import markdownify

SOURCE = "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/{name}"
DOCUMENTS_FILE = "documents.json"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MARKDOWN_DIR = DATA_DIR / "documents_md"
HTML_DIR = DATA_DIR / "documents_html"
ITEMS_PATH = DATA_DIR / "items.jsonl"
CACHE_DIR = ROOT / ".cache"

BLOCK_SEPARATOR = "\n\n"
ELEMENT_PATTERN = re.compile(r"^<(\w+)>(.*)</\1>$", re.DOTALL)
# Table rows store their cells separated by a pipe.
CELL_SEPARATOR = " | "


def fetch(name):
    """Download (once) and parse one of the dataset's JSON files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / name
    if not cached.exists():
        print(f"Downloading {name} ...")
        response = requests.get(SOURCE.format(name=name), timeout=300)
        response.raise_for_status()
        cached.write_bytes(response.content)
    return json.loads(cached.read_text(encoding="utf-8"))


def parse(element):
    """Split a stored element into its tag and inner text."""
    match = ELEMENT_PATTERN.match(element.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return "p", element.strip()


def as_markdown(tag, text, element):
    """Render one element as Markdown.

    markdownify does the conversion, so any inline markup inside an element is
    handled properly rather than by hand. Table rows are the one exception: the
    dataset stores their cells as pipe-separated text inside a bare <tr>, with no
    <td> markup for markdownify to see, so they are formatted here.
    """
    if tag == "tr":
        cells = [cell.strip() for cell in text.split(CELL_SEPARATOR)]
        return "| " + " | ".join(cells) + " |"
    # Escaping is left off: backslashes before markdown specials are noise to a
    # retriever. markdownify still collapses runs of spaces, so a Markdown span
    # is not always character-identical to its plain-text twin -- which is why
    # each rendering carries its own span text below.
    return markdownify(
        element, heading_style="ATX",
        escape_asterisks=False, escape_underscores=False, escape_misc=False,
    ).strip()


def document_id(url):
    """`https://www.gov.uk/child-tax-credit` -> `child-tax-credit`."""
    path = urllib.parse.urlparse(url).path.strip("/").replace("/", "_")
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", path) or "index"


def render(document):
    """Build all three renderings, recording each element's span in each.

    Returns (plain, markdown, html, spans) where spans[i] locates element i.
    """
    title = document["title"]
    plain_parts, markdown_parts, html_parts, spans = [], [], [], []

    header = "\n".join([
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        f"<title>{html_module.escape(title)}</title>",
        f'<link rel="canonical" href="{html_module.escape(document["url"])}">',
        "</head>", "<body>", "",
    ])
    html_parts.append(header)
    plain_at = markdown_at = 0
    html_at = len(header)
    in_table = False

    for element in document["contents"]:
        tag, text = parse(element)
        rendered = as_markdown(tag, text, element)

        # <tr> is only valid inside a table, so wrap consecutive runs of rows.
        if tag == "tr" and not in_table:
            html_parts.append("<table>\n")
            html_at += len("<table>\n")
            in_table = True
        elif tag != "tr" and in_table:
            html_parts.append("</table>\n")
            html_at += len("</table>\n")
            in_table = False

        line = f"{element.strip()}\n"
        spans.append({
            "plain": (plain_at, plain_at + len(text)),
            "markdown": (markdown_at, markdown_at + len(rendered)),
            "html": (html_at, html_at + len(element.strip())),
        })
        plain_parts.append(text)
        markdown_parts.append(rendered)
        html_parts.append(line)
        plain_at += len(text) + len(BLOCK_SEPARATOR)
        markdown_at += len(rendered) + len(BLOCK_SEPARATOR)
        html_at += len(line)

    if in_table:
        html_parts.append("</table>\n")
    html_parts.append("</body>\n</html>\n")
    return (
        BLOCK_SEPARATOR.join(plain_parts),
        BLOCK_SEPARATOR.join(markdown_parts),
        "".join(html_parts),
        spans,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev"], choices=["train", "dev"],
        help="Splits to build. The test split has no answers and is excluded.",
    )
    args = parser.parse_args()

    documents = fetch(DOCUMENTS_FILE)
    print(f"{len(documents)} policy pages")

    for directory in (DOCUMENTS_DIR, MARKDOWN_DIR, HTML_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    rendered = {}
    lengths = []
    for document in documents:
        doc_id = document_id(document["url"])
        plain, markdown, html, spans = render(document)
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(plain, encoding="utf-8", newline="\n")
        (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        (HTML_DIR / f"{doc_id}.html").write_text(html, encoding="utf-8", newline="\n")
        # Evidence names an element by its text, so remember where each sits.
        # A page can repeat an element ("<p>Yes</p>"); such evidence is
        # ambiguous and is resolved to the first occurrence.
        index, repeated = {}, set()
        for position, element in enumerate(document["contents"]):
            if element in index:
                repeated.add(element)
            else:
                index[element] = position
        rendered[document["url"]] = (doc_id, plain, markdown, html, spans, index, repeated)
        lengths.append(len(plain))

    lengths.sort()
    print(f"Wrote {len(lengths)} documents to {DOCUMENTS_DIR}, {MARKDOWN_DIR} and {HTML_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    items = []
    located = ambiguous = unlocated = 0
    for split in args.splits:
        records = fetch(f"{split}.json")
        print(f"{split}: {len(records)} questions")
        for record in records:
            doc_id, plain, markdown, html, spans, index, repeated = rendered[record["url"]]

            evidences = []
            for element in record["evidences"]:
                position = index.get(element)
                if position is None:
                    unlocated += 1
                    continue
                if element in repeated:
                    ambiguous += 1
                span = spans[position]
                located += 1
                evidences.append({
                    "doc_id": doc_id,
                    "text": plain[span["plain"][0]:span["plain"][1]],
                    "start": span["plain"][0],
                    "end": span["plain"][1],
                    "match": "element",
                    "kind": "element",
                    "markdown": {
                        "text": markdown[span["markdown"][0]:span["markdown"][1]],
                        "start": span["markdown"][0],
                        "end": span["markdown"][1],
                    },
                    "html": {
                        "text": html[span["html"][0]:span["html"][1]],
                        "start": span["html"][0],
                        "end": span["html"][1],
                    },
                })

            answers = record.get("answers") or []
            unanswerable = bool(record.get("not_answerable"))
            items.append({
                "id": f"conditionalqa-{record['id']}",
                # The question as written is underspecified on its own ("How long
                # will it be before I hear back?") and only means anything with
                # the scenario, so the two are joined into the query to use.
                "question": f"{record['scenario']} {record['question']}".strip(),
                "answer": "Unanswerable" if unanswerable or not answers else answers[0][0],
                "evidences": evidences,
                "metadata": {
                    "doc_id": doc_id,
                    "split": split,
                    "url": record["url"],
                    "scenario": record["scenario"],
                    "original_question": record["question"],
                    "answer_type": "unanswerable" if unanswerable else (
                        "yes_no" if answers and answers[0][0].lower() in ("yes", "no") else "extractive"
                    ),
                    "answers": [
                        {"text": answer[0], "conditions": answer[1]} for answer in answers
                    ],
                    "conditional": any(answer[1] for answer in answers),
                },
            })

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    unanswerable = sum(1 for item in items if item["metadata"]["answer_type"] == "unanswerable")
    conditional = sum(1 for item in items if item["metadata"]["conditional"])
    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    print(f"  {unanswerable:,} unanswerable, {conditional:,} with conditional answers")
    print(f"  evidence spans located: {located:,}; unlocated: {unlocated:,}; "
          f"ambiguous (element repeated on the page, first used): {ambiguous:,}")
    print(f"  {sum(1 for i in items if not i['evidences']):,} items carry no evidence")


if __name__ == "__main__":
    main()
