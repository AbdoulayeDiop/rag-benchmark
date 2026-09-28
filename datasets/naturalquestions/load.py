"""Load and process Natural Questions into the common RAG-experiments format.

Natural Questions (https://aclanthology.org/Q19-1026/) pairs real Google search
queries with the Wikipedia page a user was shown, annotated by five people each.
Every annotator marks a *long answer* -- the paragraph, list or table that
answers the question -- and optionally a *short answer* span inside it, so the
dataset carries human evidence at two granularities on genuinely long documents.

Spans are given as token indices, so this script never has to match text to find
them: it builds the document from the token stream and keeps a token-to-character
map, which makes every span exact by construction.

DOCUMENT RECONSTRUCTION: each page arrives as a token stream where HTML tags are
tokens flagged `is_html`. The body text is the non-HTML tokens; the tags tell us
where paragraphs, headings, list items and table rows begin. The raw HTML is also
available but is five to seven times larger than the text, most of it Wikipedia
navigation, scripts and footers. Dumping it would make the HTML rendering hold
quite different content from the other two and break the comparison they exist
for, so the HTML rendering is rebuilt from the same blocks instead.

Split: validation. NQ's test set is held out for the leaderboard. Only the first
shard is used -- it holds 1,089 documents, already more than the document budget.

Outputs:
  data/documents/<page>.txt      one Wikipedia page per document
  data/documents_md/<page>.md    the same page as Markdown
  data/documents_html/<page>.html   the same page as clean HTML
  data/items.jsonl               {question, answer, evidences} records

Requires: huggingface_hub, pandas, pyarrow
Run with: python datasets/naturalquestions/load.py [--max-documents 300]
"""

import argparse
import html as html_module
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "google-research-datasets/natural_questions"
SHARD = "dev/validation-00000-of-00007.parquet"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MARKDOWN_DIR = DATA_DIR / "documents_md"
HTML_DIR = DATA_DIR / "documents_html"
ITEMS_PATH = DATA_DIR / "items.jsonl"

BLOCK_SEPARATOR = "\n\n"
CELL_SEPARATOR = " | "
# Tags that start a new block, mapped to the kind of block they start.
OPENING = {
    "<p>": ("p", 0), "<li>": ("li", 0), "<tr>": ("tr", 0), "<dd>": ("p", 0),
    "<dt>": ("p", 0), "<blockquote>": ("p", 0),
    **{f"<h{level}>": ("h", level) for level in range(1, 7)},
}
CLOSING = {f"</{tag.strip('<>')}>" for tag in OPENING}
# Inside a table row these separate cells rather than starting a block.
CELL_TAGS = {"<td>", "<th>"}


def build_blocks(tokens):
    """Group the token stream into blocks, remembering which tokens are where.

    Returns a list of blocks, each {kind, level, tokens: [(index, text), ...]}.
    HTML tokens are dropped from the text but decide where blocks begin.
    """
    blocks = []
    current = None

    def flush():
        nonlocal current
        if current and current["tokens"]:
            blocks.append(current)
        current = None

    for index, (token, is_html) in enumerate(zip(tokens["token"], tokens["is_html"])):
        if not is_html:
            if current is None:
                current = {"kind": "p", "level": 0, "tokens": [], "cells": False}
            current["tokens"].append((index, token))
            continue

        tag = token.lower()
        if tag in OPENING:
            flush()
            kind, level = OPENING[tag]
            current = {"kind": kind, "level": level, "tokens": [], "cells": kind == "tr"}
        elif tag in CLOSING:
            flush()
        elif tag in CELL_TAGS and current is not None and current["cells"] and current["tokens"]:
            # Mark a cell boundary inside the current row.
            current["tokens"].append((None, CELL_SEPARATOR.strip()))
    flush()
    return blocks


def join(block):
    """The block's plain text, and each token's offset inside it."""
    parts = []
    offsets = {}
    at = 0
    for index, token in block["tokens"]:
        if parts:
            at += 1  # the space joining tokens
        if index is not None:
            offsets[index] = (at, at + len(token))
        parts.append(token)
        at += len(token)
    return " ".join(parts), offsets


def render(tokens):
    """Build all three renderings plus a token-to-character map for the text."""
    blocks = build_blocks(tokens)
    plain_parts, markdown_parts, html_parts, spans = [], [], [], []
    token_map = {}

    header_lines = ["<!DOCTYPE html>", '<html lang="en">', "<head>",
                    '<meta charset="utf-8">', "</head>", "<body>", ""]
    html_header = "\n".join(header_lines)
    html_parts.append(html_header)

    plain_at = markdown_at = 0
    html_at = len(html_header)

    for block in blocks:
        text, offsets = join(block)
        if not text.strip():
            continue

        kind, level = block["kind"], block["level"]
        if kind == "h":
            markdown = f"{'#' * min(level, 6)} {text}"
            tag = f"h{min(level, 6)}"
        elif kind == "li":
            markdown = f"- {text}"
            tag = "li"
        elif kind == "tr":
            markdown = "| " + " | ".join(
                cell.strip() for cell in text.split(CELL_SEPARATOR)
            ) + " |"
            tag = "tr"
        else:
            markdown = text
            tag = "p"

        # Only &, < and > need escaping in element content; escaping quotes too
        # would rewrite most paragraphs for nothing.
        element = f"<{tag}>{escape(text)}</{tag}>\n"
        markdown_offset = markdown.find(text)
        for index, (start, end) in offsets.items():
            token_map[index] = (plain_at + start, plain_at + end)
        spans.append({
            "plain": (plain_at, plain_at + len(text)),
            # A table row reorders its cells, so its text may not appear in the
            # Markdown line verbatim; such a block has no Markdown offset.
            "markdown_shift": (
                markdown_at + markdown_offset - plain_at if markdown_offset >= 0 else None
            ),
            "html_text_start": html_at + len(f"<{tag}>"),
        })

        plain_parts.append(text)
        markdown_parts.append(markdown)
        html_parts.append(element)
        plain_at += len(text) + len(BLOCK_SEPARATOR)
        markdown_at += len(markdown) + len(BLOCK_SEPARATOR)
        html_at += len(element)

    html_parts.append("</body>\n</html>\n")
    return (
        BLOCK_SEPARATOR.join(plain_parts),
        BLOCK_SEPARATOR.join(markdown_parts),
        "".join(html_parts),
        spans,
        token_map,
    )


def span_for(start_token, end_token, token_map):
    """Character span covering the annotated token range, or None.

    `end_token` is exclusive, and the range can include HTML tokens that carry no
    text, so the real edges are the first and last mapped tokens inside it.
    """
    present = [index for index in range(start_token, end_token) if index in token_map]
    if not present:
        return None
    return token_map[present[0]][0], token_map[present[-1]][1]


def escape(text):
    """Escape a text node. Quotes only matter inside attributes, so leave them."""
    return html_module.escape(text, quote=False)


def other_spans(start, end, spans, plain):
    """Map a plain-text span onto the Markdown and HTML renderings.

    Escaping expands the text, so an HTML offset is found by escaping the part of
    the block that precedes the span rather than assuming a constant shift.
    """
    def block_at(position):
        for span in spans:
            block_start, block_end = span["plain"]
            if block_start <= position <= block_end:
                return span
        return None

    # A long answer can cover a whole table or list, so it may begin in one block
    # and end in another; each edge is mapped through its own block.
    first, last = block_at(start), block_at(end)
    if first is None or last is None:
        return None, None

    markdown = None
    if first["markdown_shift"] is not None and last["markdown_shift"] is not None:
        markdown_start = start + first["markdown_shift"]
        markdown_end = end + last["markdown_shift"]
        if markdown_end >= markdown_start:
            markdown = (markdown_start, markdown_end)

    html_start = first["html_text_start"] + len(escape(plain[first["plain"][0]:start]))
    html_end = last["html_text_start"] + len(escape(plain[last["plain"][0]:end]))
    html = (html_start, html_end) if html_end >= html_start else None
    return markdown, html


def document_id(title):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(". ")[:120] or "untitled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents", type=int, default=300,
        help="Cap on the number of Wikipedia pages. The shard holds 1,089; the "
             "default of 300 matches the subset size used in the chunking "
             "literature. Pass 0 for no cap.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(hf_hub_download(REPO_ID, SHARD, repo_type="dataset"))
    print(f"validation shard: {len(frame):,} questions")

    for directory in (DOCUMENTS_DIR, MARKDOWN_DIR, HTML_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    documents = {}
    items = []
    counts = {"long": 0, "short": 0, "unlocated": 0, "no_html": 0, "no_markdown": 0}
    lengths = []

    for row in frame.itertuples(index=False):
        title = row.document["title"]
        doc_id = document_id(title)
        if doc_id not in documents:
            if args.max_documents and len(documents) >= args.max_documents:
                continue
            plain, markdown, html, spans, token_map = render(row.document["tokens"])
            if not plain.strip():
                continue
            (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(plain, encoding="utf-8", newline="\n")
            (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
            (HTML_DIR / f"{doc_id}.html").write_text(html, encoding="utf-8", newline="\n")
            documents[doc_id] = (plain, markdown, html, spans, token_map)
            lengths.append(len(plain))

        plain, markdown, html, spans, token_map = documents[doc_id]
        annotations = row.annotations

        evidences = []
        seen = set()

        def add(start_token, end_token, kind):
            span = span_for(int(start_token), int(end_token), token_map)
            if span is None:
                counts["unlocated"] += 1
                return
            if (span, kind) in seen:
                return
            seen.add((span, kind))
            start, end = span
            markdown_span, html_span = other_spans(start, end, spans, plain)
            if html_span is None:
                counts["no_html"] += 1
            if markdown_span is None:
                counts["no_markdown"] += 1
            counts["long" if kind == "long_answer" else "short"] += 1
            evidences.append({
                "doc_id": doc_id,
                "text": plain[start:end],
                "start": start,
                "end": end,
                "match": "token_index",
                "kind": kind,
                "markdown": (
                    {"text": markdown[markdown_span[0]:markdown_span[1]],
                     "start": markdown_span[0], "end": markdown_span[1]}
                    if markdown_span else None
                ),
                "html": (
                    {"text": html[html_span[0]:html_span[1]],
                     "start": html_span[0], "end": html_span[1]}
                    if html_span else None
                ),
            })

        short_texts = []
        for long_answer in annotations["long_answer"]:
            if long_answer["start_token"] >= 0:
                add(long_answer["start_token"], long_answer["end_token"], "long_answer")
        for short in annotations["short_answers"]:
            for start_token, end_token, text in zip(
                short["start_token"], short["end_token"], short["text"]
            ):
                add(start_token, end_token, "short_answer")
                short_texts.append(text)

        yes_no = [y for y in annotations["yes_no_answer"] if y != -1]
        if short_texts:
            answer, answer_type = short_texts[0], "short_answer"
        elif yes_no:
            answer, answer_type = ("Yes" if yes_no[0] == 1 else "No"), "yes_no"
        elif evidences:
            answer, answer_type = evidences[0]["text"], "long_answer"
        else:
            answer, answer_type = "Unanswerable", "unanswerable"

        items.append({
            "id": f"naturalquestions-{row.id}",
            "question": row.question["text"],
            "answer": answer,
            "evidences": evidences,
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "url": row.document["url"],
                "split": "validation",
                "answer_type": answer_type,
                "short_answers": short_texts,
                # Five people annotate each question; a question counts as
                # answerable when at least two of them marked a long answer.
                "annotators_with_long_answer": sum(
                    1 for la in annotations["long_answer"] if la["start_token"] >= 0
                ),
            },
        })

    lengths.sort()
    print(f"Wrote {len(lengths)} documents to {DOCUMENTS_DIR}, {MARKDOWN_DIR} and {HTML_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    unanswerable = sum(1 for i in items if i["metadata"]["answer_type"] == "unanswerable")
    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    print(f"  {len(items) - unanswerable:,} answerable, {unanswerable:,} unanswerable "
          f"(no annotator found a passage)")
    print(f"  evidence spans: {counts['long']:,} long answers, {counts['short']:,} short answers")
    if counts["unlocated"]:
        print(f"  {counts['unlocated']:,} annotated ranges held no body text (tables stripped of cells)")
    if counts["no_markdown"]:
        print(f"  {counts['no_markdown']:,} spans sit in table rows whose cells are "
              f"reordered in Markdown and have no Markdown offset")
    if counts["no_html"]:
        print(f"  {counts['no_html']:,} spans have no HTML offset")


if __name__ == "__main__":
    main()
