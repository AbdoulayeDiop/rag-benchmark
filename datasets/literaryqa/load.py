"""Load and process LiteraryQA into the common RAG-experiments format.

LiteraryQA (https://huggingface.co/datasets/sapienzanlp/LiteraryQA) is a cleaned
subset of NarrativeQA: crowdsourced questions over full-length literary works,
with the books themselves left out of the distribution for copyright reasons.
Only the annotations and a list of Project Gutenberg URLs are published, so the
books have to be downloaded and cleaned locally.

The documented one-liner, `load_dataset("sapienzanlp/LiteraryQA")`, does not work:
the Hub repository is script-based (so it needs `trust_remote_code=True`), and the
published script declares an empty `_LOCAL_FILES`, which makes its
`_split_generators` raise a KeyError before it downloads anything. Instead, this
script drives the authors' own cleaning code, vendored under `vendor/` from
github.com/SapienzaNLP/LiteraryQA at the commit recorded in `vendor/COMMIT`, and
mirrors the order of operations in their `scripts/download_and_clean_books.py`.

NOTE ON EVIDENCE: LiteraryQA carries no span annotations -- answers are free-form
and were written from a summary, not quoted from the book. Every item therefore
has an empty `evidences` list. It is usable for end-to-end answer quality, not for
retrieval precision, and it is the long-document counterpart to datasets like
GutenQA that do have spans.

Because the source is HTML with real heading tags, each book is written three
ways -- plain text, Markdown and HTML -- so that structure-aware chunking can be
compared against flat text. All three contain the same cleaned content; they
differ only in whether heading levels are marked.

Outputs:
  data/documents/<gutenberg id>.txt     one cleaned book per document
  data/documents_md/<gutenberg id>.md   the same book with Markdown headings
  data/documents_html/<gutenberg id>.html   the same book as headings/paragraphs
  data/items.jsonl                      {question, answer, evidences} records

Requires: huggingface_hub, requests, tqdm, chardet, beautifulsoup4[html5lib], ftfy, loguru
Run with: python datasets/literaryqa/load.py [--splits test train validation]
"""

import argparse
import csv
import html as html_module
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))

from literaryqa.clean import detect_encoding_and_read, extract_raw_text, remove_gutenberg_info  # noqa: E402
from literaryqa.download import download_htm_from_gutenberg  # noqa: E402

from structure import HEADING_TAGS, extract_blocks  # noqa: E402

REPO_ID = "sapienzanlp/LiteraryQA"
URLS_FILE = "literaryqa_annotations/literaryqa_urls.tsv"
ANNOTATIONS_FILE = "literaryqa_annotations/{split}.jsonl"

DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MARKDOWN_DIR = DATA_DIR / "documents_md"
HTML_DIR = DATA_DIR / "documents_html"
ITEMS_PATH = DATA_DIR / "items.jsonl"
# Raw and intermediate files are kept outside data/ so that data/ holds only the
# processed dataset. Downloaded HTML is cached here, so re-runs are cheap.
CACHE_DIR = ROOT / ".cache"


def normalize(text):
    """The `normalize=True` branch of the upstream `clean_and_save`."""
    return (
        text.replace("--", "—")
        .replace("——", "—")
        .translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    )


def clean_book(book_id, html):
    """Apply the upstream cleaning pipeline to one book's HTML.

    This is `literaryqa.clean.clean_and_save` inlined, because that function
    writes with the platform default encoding -- which mangles or fails on
    non-cp1252 characters on Windows -- and returns nothing. The cleaning steps
    below are unchanged from upstream, including `normalize=True`.
    """
    return normalize(
        remove_gutenberg_info(raw_text=extract_raw_text(html), gt_id=book_id, log_file=None)
    )


def tagged_lines(blocks):
    """Flatten (tag, text) blocks into the (tag, line) list the cleaner sees.

    `extract_raw_text` joins blocks with a blank line and `remove_gutenberg_info`
    then works line by line, so the blank separators have to be reproduced for
    the line sequence to match what upstream filters.
    """
    lines = []
    for index, (tag, text) in enumerate(blocks):
        if index:
            lines.append((None, ""))
        lines.extend((tag, line) for line in text.split("\n"))
    return lines


def retag(cleaned, lines):
    """Re-attach heading tags to the lines that survived the cleaner.

    `remove_gutenberg_info` only drops and strips lines, never rewrites them, so
    its output is a subsequence of the stripped input. Walking both in order
    recovers which tag each surviving line came from.
    """
    kept = []
    cursor = 0
    for line in cleaned.split("\n"):
        while cursor < len(lines) and lines[cursor][1].strip() != line:
            cursor += 1
        if cursor < len(lines):
            kept.append((lines[cursor][0], line))
            cursor += 1
        else:  # alignment lost; treat the remainder as body text
            kept.append((None, line))
    return kept


def render_markdown(kept):
    """Heading lines become ATX headings; everything else stays as written."""
    parts = []
    for tag, line in kept:
        if tag in HEADING_TAGS:
            parts.append(f"{'#' * int(tag[1])} {line}")
        else:
            parts.append(line)
    return normalize("\n\n".join(parts))


def render_html(book_id, title, kept):
    """A minimal document keeping only the heading/paragraph structure."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html_module.escape(title)}</title>",
        f'<meta name="gutenberg-id" content="{html_module.escape(book_id)}">',
        "</head>",
        "<body>",
    ]
    for tag, line in kept:
        name = tag if tag in HEADING_TAGS else "p"
        parts.append(f"<{name}>{html_module.escape(normalize(line))}</{name}>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def load_urls(splits):
    """Read the published URL table, keeping only the requested splits."""
    path = hf_hub_download(REPO_ID, URLS_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if row["split"] in splits]


def load_annotations(splits):
    """Read the question/answer annotations for the requested splits."""
    records = []
    for split in splits:
        path = hf_hub_download(REPO_ID, ANNOTATIONS_FILE.format(split=split), repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
    return records


def build_documents(rows, titles, delay):
    """Download and clean one book per row, in all three renderings.

    Returns ({gutenberg_id: doc_id}, failed ids, books whose structure could not
    be recovered and so got plain text only).
    """
    for directory in (DOCUMENTS_DIR, MARKDOWN_DIR, HTML_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    written = {}
    failed = []
    flat = []
    for row in (bar := tqdm(rows, desc="Books", unit="book")):
        book_id, split = row["book_id"], row["split"]
        destination = DOCUMENTS_DIR / f"{book_id}.txt"

        cached_html = CACHE_DIR / split / f"{book_id}.htm"
        if not cached_html.exists():
            time.sleep(delay)  # Gutenberg's mirrors are volunteer-run; do not hammer them.
        if download_htm_from_gutenberg(
            book_id=book_id, split=split, save_dir=CACHE_DIR, log_dir=CACHE_DIR / "logs", pbar=bar
        ) is None:
            failed.append(book_id)
            continue

        html = detect_encoding_and_read(cached_html)
        text = clean_book(book_id, html)
        destination.write_text(text, encoding="utf-8", newline="\n")
        written[book_id] = destination.stem

        # The structured renderings are only written when the adapted extractor
        # provably reproduces the vendored text, so the .txt stays canonical.
        blocks = extract_blocks(html)
        if "\n\n".join(block for _, block in blocks) != extract_raw_text(html):
            flat.append(book_id)
            continue
        kept = retag(
            remove_gutenberg_info(raw_text=extract_raw_text(html), gt_id=book_id, log_file=None),
            tagged_lines(blocks),
        )
        title = titles.get(book_id, book_id)
        (MARKDOWN_DIR / f"{book_id}.md").write_text(render_markdown(kept), encoding="utf-8", newline="\n")
        (HTML_DIR / f"{book_id}.html").write_text(
            render_html(book_id, title, kept), encoding="utf-8", newline="\n"
        )
    return written, failed, flat


def build_items(records, documents):
    """Turn each annotated question into a {question, answer, evidences} record."""
    items = []
    skipped = 0
    for record in records:
        book_id = record["gutenberg_id"]
        doc_id = documents.get(book_id)
        if doc_id is None:  # book could not be downloaded
            skipped += len(record["qas"])
            continue
        for index, qa in enumerate(record["qas"]):
            answers = list(qa["answers"])
            items.append({
                "id": f"literaryqa-{record['document_id']}-{index:03d}",
                "question": qa["question"],
                "answer": answers[0],
                # LiteraryQA has no span annotations; see the note at the top.
                "evidences": [],
                "metadata": {
                    "doc_id": doc_id,
                    "gutenberg_id": book_id,
                    "title": record["title"],
                    "split": record["split"],
                    "answers": answers,
                },
            })
    return items, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["test"], choices=["train", "validation", "test"],
        help="Splits to build. Defaults to test (138 books), the split used to report "
             "LiteraryQA document statistics; all three total 650 books.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait before each Gutenberg download (cached books are free).",
    )
    args = parser.parse_args()

    rows = load_urls(set(args.splits))
    print(f"{len(rows)} books listed for split(s) {', '.join(sorted(args.splits))}")

    records = load_annotations(args.splits)
    titles = {record["gutenberg_id"]: record["title"] for record in records}

    documents, failed, flat = build_documents(rows, titles, args.delay)
    lengths = sorted(len((DOCUMENTS_DIR / f"{b}.txt").read_text(encoding="utf-8")) for b in documents)
    print(f"Wrote {len(documents)} documents to {DOCUMENTS_DIR}, {MARKDOWN_DIR} and {HTML_DIR}")
    if lengths:
        print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} / max {lengths[-1]:,}")
    if failed:
        print(f"  {len(failed)} books could not be downloaded from any mirror: {', '.join(failed)}")
    if flat:
        print(f"  {len(flat)} books got plain text only (structure extraction diverged): {', '.join(flat)}")

    items, skipped = build_items(records, documents)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    if skipped:
        print(f"  {skipped:,} questions dropped because their book is missing")
    print("  evidences are empty for every item: LiteraryQA has no span annotations")


if __name__ == "__main__":
    main()
