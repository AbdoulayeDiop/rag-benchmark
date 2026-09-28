"""Load and process Qasper into the common RAG-experiments format.

Qasper (https://allenai.org/data/qasper) is question answering over NLP research
papers: questions were written by readers who saw only title and abstract, then
answered by experts who read the full paper and marked the paragraphs that
support each answer. That evidence annotation is what makes it useful here --
retrieval can be scored directly, not just end-to-end answer quality.

The Hub repository `allenai/qasper` is a loader script, so `load_dataset` would
need `trust_remote_code=True`. This script skips that and downloads the original
archives from AI2's S3 bucket, which is where the Hub script points anyway.

Qasper records section names with a " ::: " separator marking nesting, so each
paper is written twice: as plain text, and as Markdown with real heading levels.
Chunkers can then be compared with and without structure. Every evidence span is
resolved in both renderings, since 222 evidence annotations in the test split
are themselves section headings and so differ between the two.

Figure and table captions are appended under a trailing "Figures and Tables"
heading, because 459 evidence annotations point at a caption
("FLOAT SELECTED: ...") and the source keeps captions in a separate field with
no position information.

Outputs:
  data/documents/<arxiv id>.txt      one paper per document, plain text
  data/documents_md/<arxiv id>.md    the same paper with Markdown headings
  data/items.jsonl                   {question, answer, evidences} records

Requires: requests
Run with: python datasets/qasper/load.py [--splits test train dev]
"""

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import requests

ARCHIVES = {
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    "train": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
}
MEMBER = "qasper-{split}-v0.3.json"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MARKDOWN_DIR = DATA_DIR / "documents_md"
ITEMS_PATH = DATA_DIR / "items.jsonl"
CACHE_DIR = ROOT / ".cache"

BLOCK_SEPARATOR = "\n\n"
# Qasper nests section names as "Parent ::: Child ::: Grandchild".
LEVEL_SEPARATOR = " ::: "
CAPTIONS_HEADING = "Figures and Tables"
# Evidence pointing at a figure or table is stored with this prefix, followed by
# the caption verbatim.
FLOAT_PREFIX = "FLOAT SELECTED: "


def squeeze(text):
    """Collapse runs of whitespace, for matching text that differs only in layout."""
    return re.sub(r"\s+", " ", text).strip()


def fetch(split):
    """Download (once) and return the parsed JSON for one split."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    member = MEMBER.format(split=split)
    cached = CACHE_DIR / member
    if not cached.exists():
        url = ARCHIVES[split]
        archive = CACHE_DIR / url.rsplit("/", 1)[-1]
        if not archive.exists():
            print(f"Downloading {url} ...")
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            archive.write_bytes(response.content)
        with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
            cached.write_bytes(tar.extractfile(member).read())
    return json.loads(cached.read_text(encoding="utf-8"))


def heading(name, level):
    """A heading block: plain text keeps the source name, Markdown gets its level.

    Only the leaf is kept in Markdown, since the ancestors are already present as
    the enclosing headings.
    """
    return {
        "plain": name,
        "markdown": f"{'#' * level} {name.split(LEVEL_SEPARATOR)[-1]}",
        "is_heading": True,
    }


def paragraph(text):
    """A body block, rendered identically in both versions."""
    return {"plain": text, "markdown": text, "is_heading": False}


def build_blocks(paper):
    """Break one paper into the ordered blocks both renderings are built from."""
    blocks = [heading(paper["title"], 1)]
    if paper.get("abstract"):
        blocks.append(heading("Abstract", 2))
        blocks.append(paragraph(paper["abstract"]))

    for section in paper["full_text"]:
        name = section.get("section_name")
        if name:
            # Depth 1 sits under the title, which is the only level-1 heading.
            blocks.append(heading(name, min(name.count(LEVEL_SEPARATOR) + 2, 6)))
        blocks.extend(paragraph(text) for text in section["paragraphs"] if text and text.strip())

    captions = [figure["caption"] for figure in paper.get("figures_and_tables", []) if figure.get("caption")]
    if captions:
        blocks.append(heading(CAPTIONS_HEADING, 2))
        blocks.extend(paragraph(caption) for caption in captions)
    return [block for block in blocks if block["plain"] and block["plain"].strip()]


def render(blocks):
    """Join blocks into both documents, recording where each block landed."""
    plain_parts, markdown_parts, spans = [], [], []
    plain_at = markdown_at = 0
    for block in blocks:
        spans.append({
            "plain": (plain_at, plain_at + len(block["plain"])),
            "markdown": (markdown_at, markdown_at + len(block["markdown"])),
        })
        plain_parts.append(block["plain"])
        markdown_parts.append(block["markdown"])
        plain_at += len(block["plain"]) + len(BLOCK_SEPARATOR)
        markdown_at += len(block["markdown"]) + len(BLOCK_SEPARATOR)
    return BLOCK_SEPARATOR.join(plain_parts), BLOCK_SEPARATOR.join(markdown_parts), spans


def locate(needle, blocks, spans):
    """Find `needle` among the blocks, returning spans into both renderings.

    Returns (plain_span, markdown_span, how) or None. Evidence is annotated as
    whole paragraphs, so a block-level match is the normal case; the offset math
    only applies within the one block that matched.
    """
    for index, block in enumerate(blocks):
        if block["plain"] == needle:
            return spans[index]["plain"], spans[index]["markdown"], "exact"

    for how, transform in (("exact", lambda s: s), ("folded", squeeze)):
        target = transform(needle)
        if not target:
            continue
        for index, block in enumerate(blocks):
            offset = transform(block["plain"]).find(target)
            if offset == -1:
                continue
            plain_start, plain_end = spans[index]["plain"]
            markdown_span = spans[index]["markdown"]
            if how == "exact" and not block["is_heading"]:
                # Body blocks are byte-identical in both files, so the offset
                # inside the block carries over unchanged.
                plain_span = (plain_start + offset, plain_start + offset + len(target))
                shift = markdown_span[0] - plain_start
                return plain_span, (plain_span[0] + shift, plain_span[1] + shift), how
            # Headings differ between renderings, and a folded match has no
            # reliable offset inside the block: fall back to the whole block.
            return (plain_start, plain_end), markdown_span, how
    return None


def format_answer(answer):
    """Render one annotator's answer, following the official Qasper evaluator."""
    if answer["unanswerable"]:
        return "Unanswerable", "unanswerable"
    if answer["extractive_spans"]:
        return "; ".join(span.strip() for span in answer["extractive_spans"]), "extractive"
    if answer["free_form_answer"]:
        return answer["free_form_answer"], "abstractive"
    if answer["yes_no"] is not None:
        return ("Yes" if answer["yes_no"] else "No"), "yes_no"
    return "Unanswerable", "unanswerable"


def build_items(paper_id, paper, split, blocks, plain, markdown, spans, counts):
    """One item per question, merging the evidence of every annotator."""
    items = []
    for question in paper["qas"]:
        annotations = [format_answer(a["answer"]) for a in question["answers"]]
        # Prefer an annotator who could answer; fall back to "Unanswerable" only
        # when every annotator marked it so.
        answer, answer_type = next(
            ((a, t) for a, t in annotations if t != "unanswerable"), annotations[0]
        )

        evidences = []
        seen = set()
        for annotation in question["answers"]:
            for excerpt in annotation["answer"]["evidence"]:
                is_caption = excerpt.startswith(FLOAT_PREFIX)
                needle = excerpt[len(FLOAT_PREFIX):] if is_caption else excerpt
                if needle in seen:
                    continue
                seen.add(needle)
                found = locate(needle, blocks, spans)
                if found is None:
                    counts["unlocated"] += 1
                    continue
                (start, end), (md_start, md_end), how = found
                counts[how] += 1
                evidences.append({
                    "doc_id": paper_id,
                    "text": plain[start:end],
                    "start": start,
                    "end": end,
                    "match": how,
                    "kind": "figure_or_table" if is_caption else "paragraph",
                    "markdown": {
                        "text": markdown[md_start:md_end],
                        "start": md_start,
                        "end": md_end,
                    },
                })

        items.append({
            "id": f"qasper-{question['question_id']}",
            "question": question["question"].strip(),
            "answer": answer,
            "evidences": evidences,
            "metadata": {
                "doc_id": paper_id,
                "title": paper["title"],
                "split": split,
                "answer_type": answer_type,
                "answers": [a for a, _ in annotations],
                "annotators": len(annotations),
            },
        })
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["test"], choices=["test", "train", "dev"],
        help="Splits to build. Defaults to test (416 papers), the split whose "
             "statistics are quoted in the chunking literature.",
    )
    args = parser.parse_args()

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    all_items = []
    counts = {"exact": 0, "folded": 0, "unlocated": 0}
    lengths = []

    for split in args.splits:
        papers = fetch(split)
        print(f"{split}: {len(papers)} papers, {sum(len(p['qas']) for p in papers.values())} questions")
        for paper_id, paper in papers.items():
            blocks = build_blocks(paper)
            plain, markdown, spans = render(blocks)
            (DOCUMENTS_DIR / f"{paper_id}.txt").write_text(plain, encoding="utf-8", newline="\n")
            (MARKDOWN_DIR / f"{paper_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
            lengths.append(len(plain))
            all_items.extend(
                build_items(paper_id, paper, split, blocks, plain, markdown, spans, counts)
            )

    lengths.sort()
    print(f"Wrote {len(lengths)} documents to {DOCUMENTS_DIR} and {MARKDOWN_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in all_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    evidenced = sum(1 for item in all_items if item["evidences"])
    print(f"Wrote {len(all_items):,} items to {ITEMS_PATH}")
    print(f"  {evidenced:,} carry evidence; {len(all_items) - evidenced:,} do not "
          f"(mostly questions every annotator marked unanswerable)")
    print(f"  evidence spans located: {counts['exact']:,} verbatim, {counts['folded']:,} "
          f"after whitespace folding, {counts['unlocated']:,} unlocated")


if __name__ == "__main__":
    main()
