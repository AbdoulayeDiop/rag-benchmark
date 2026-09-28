"""Load and process TechQA into the common RAG-experiments format.

TechQA (https://arxiv.org/abs/1911.02984) is technical support QA: real questions
from IBM developer forums, answered by an excerpt from an IBM technote, with a
large technote corpus to retrieve from. Its unanswerable questions are genuine --
forum questions no technote resolves -- rather than constructed distractors.

WHICH RELEASE: this uses NVIDIA's TechQA-RAG-Eval, a reduced repackaging built for
RAG evaluation (910 questions, 28,481 technotes, 46MB). The three alternatives
were worse fits:
  * `rojagtap/tech-qa` keeps only the single gold document per question, so there
    is no corpus to retrieve from and retrieval cannot be scored at all.
  * The original `PrimeQA/TechQA` archive is 2.96GB, almost all of it the full
    800k-technote corpus that would be subsampled to a fraction of a percent here.
  * IBM's own distribution requires registration.

EVIDENCE: NVIDIA's repackaging dropped the original START_OFFSET/END_OFFSET
fields, keeping the answer as text. The answer *is* the annotator-marked excerpt,
so locating that text in the gold technote recovers the human annotation rather
than inventing one -- 99% of answerable questions match verbatim. This is not the
same as TriviaQA's answer-occurrence heuristic, where a short entity string may
occur incidentally; here the answer is a multi-sentence excerpt.

Technotes carry consistent upper-case section headings (ABSTRACT, CONTENT,
RESOLVING THE PROBLEM, ...), which the Markdown rendering marks. They are flat --
there is no nesting to recover -- so all render at one level. Note that technotes
quote shell snippets, so some body lines start with "#" and will look like
headings to a Markdown parser even though this script did not mark them; the .txt
rendering is the one free of that ambiguity.

Outputs:
  data/documents/<technote>.txt      one technote per document
  data/documents_md/<technote>.md    the same technote with headings marked
  data/items.jsonl                   {question, answer, evidences} records

Requires: huggingface_hub
Run with: python datasets/techqa/load.py [--max-documents 2000]
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/TechQA-RAG-Eval"
QUESTIONS_FILE = "train.json"
CORPUS_FILE = "corpus.zip"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MARKDOWN_DIR = DATA_DIR / "documents_md"
ITEMS_PATH = DATA_DIR / "items.jsonl"

BLOCK_SEPARATOR = "\n\n"
TITLE_PREFIX = "Title: "
# Section headings are written in upper case on a line of their own. Two- and
# three-letter runs ("UP") are fragments of body text, not headings.
HEADING_MIN_CHARS = 4
HEADING_MAX_CHARS = 60
HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9 ()/,.'&:-]+$")


def squeeze(text):
    return re.sub(r"\s+", " ", text).strip()


def is_heading(block):
    """Whether a block is a technote section heading."""
    if "\n" in block:
        return False
    stripped = block.strip()
    if not HEADING_MIN_CHARS <= len(stripped) <= HEADING_MAX_CHARS:
        return False
    return bool(HEADING_PATTERN.match(stripped)) and stripped == stripped.upper()


def render(text):
    """Render a technote as plain text and Markdown, recording block spans.

    The plain rendering is the corpus file unchanged, so it stays byte-identical
    to the gold context the dataset ships.
    """
    spans = []
    markdown_parts = []
    plain_at = markdown_at = 0
    for block in text.split(BLOCK_SEPARATOR):
        stripped = block.strip()
        if stripped.startswith(TITLE_PREFIX):
            rendered = f"# {stripped[len(TITLE_PREFIX):].strip()}"
        elif is_heading(block):
            rendered = f"## {stripped}"
        else:
            rendered = block
        # Where the block's own text sits, ignoring any marker added in front.
        lead = len(block) - len(block.lstrip())
        content_start = plain_at + lead
        offset = rendered.find(stripped) if stripped else 0
        spans.append({
            "plain": (plain_at, plain_at + len(block)),
            "content": (content_start, content_start + len(stripped)),
            "shift": markdown_at + offset - content_start,
            "markdown": (markdown_at, markdown_at + len(rendered)),
        })
        markdown_parts.append(rendered)
        plain_at += len(block) + len(BLOCK_SEPARATOR)
        markdown_at += len(rendered) + len(BLOCK_SEPARATOR)
    return BLOCK_SEPARATOR.join(markdown_parts), spans


def to_markdown_span(start, end, spans):
    """Map a plain-text span onto the Markdown rendering.

    Answer excerpts routinely run across several blocks, so the ends are mapped
    independently: each block shifts by a fixed amount, and only blocks that
    became headings shift differently from their neighbours. An end that lands in
    a block's leading whitespace has no counterpart, and the span is reported as
    unmapped rather than guessed.
    """
    def locate_edge(position):
        for span in spans:
            block_start, block_end = span["plain"]
            if block_start <= position <= block_end:
                content_start, content_end = span["content"]
                if content_start <= position <= content_end:
                    return position + span["shift"]
                return None
        return None

    mapped_start = locate_edge(start)
    mapped_end = locate_edge(end)
    if mapped_start is None or mapped_end is None or mapped_end < mapped_start:
        return None
    return mapped_start, mapped_end


def locate(answer, text):
    """Find the annotated answer excerpt in its technote.

    A few excerpts differ from the source only in how whitespace was collapsed.
    Those are matched on the text with all whitespace removed, keeping an index
    of where each surviving character came from so real offsets can be reported.
    """
    start = text.find(answer)
    if start != -1:
        return start, start + len(answer), "exact"

    positions = [index for index, char in enumerate(text) if not char.isspace()]
    compact = "".join(text[index] for index in positions)
    target = re.sub(r"\s+", "", answer)
    if not target:
        return None
    at = compact.find(target)
    if at == -1:
        return None
    return positions[at], positions[at + len(target) - 1] + 1, "folded"


def document_id(filename):
    stem = filename[:-4] if filename.endswith(".txt") else filename
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents", type=int, default=2000,
        help="Cap on corpus size. Every gold technote is always included; the "
             "rest of the budget is filled with distractors, in a stable order. "
             "The full corpus is 28,481 technotes. Pass 0 for no cap.",
    )
    args = parser.parse_args()

    questions = json.loads(
        Path(hf_hub_download(REPO_ID, QUESTIONS_FILE, repo_type="dataset")).read_text(encoding="utf-8")
    )
    archive = zipfile.ZipFile(hf_hub_download(REPO_ID, CORPUS_FILE, repo_type="dataset"))
    available = sorted(name for name in archive.namelist() if name.endswith(".txt"))
    print(f"{len(questions)} questions, {len(available):,} technotes in the corpus")

    gold = {context["filename"] for question in questions for context in question["contexts"]}
    selected = [name for name in available if name.split("/")[-1] in gold]
    if args.max_documents:
        distractors = [name for name in available if name.split("/")[-1] not in gold]
        selected += distractors[:max(0, args.max_documents - len(selected))]
    else:
        selected = available
    print(f"Selected {len(selected):,} technotes ({len(gold):,} gold + "
          f"{len(selected) - len(gold):,} distractors)")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    lengths = []
    headings = 0
    for name in selected:
        filename = name.split("/")[-1]
        text = archive.read(name).decode("utf-8", errors="replace")
        doc_id = document_id(filename)
        markdown, spans = render(text)
        (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
        (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        documents[filename] = (doc_id, text, markdown, spans)
        lengths.append(len(text))
        headings += sum(1 for span in spans if markdown[span["markdown"][0]:].startswith("#"))

    lengths.sort()
    print(f"Wrote {len(lengths):,} documents to {DOCUMENTS_DIR} and {MARKDOWN_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    items = []
    counts = {"exact": 0, "folded": 0, "unlocated": 0, "no_markdown": 0, "empty_answer": 0}
    for question in questions:
        evidences = []
        # Two questions (DEV_Q014, DEV_Q094) are flagged answerable but ship an
        # empty answer; treat them as unanswerable rather than emitting a
        # zero-length span.
        empty_answer = not (question["answer"] or "").strip()
        if empty_answer:
            counts["empty_answer"] += 1
        for context in question["contexts"] if not empty_answer else []:
            doc_id, text, markdown, spans = documents[context["filename"]]
            found = locate(question["answer"], text)
            if found is None:
                counts["unlocated"] += 1
                continue
            start, end, how = found
            counts[how] += 1
            markdown_span = to_markdown_span(start, end, spans)
            if markdown_span is None:
                counts["no_markdown"] += 1
            evidences.append({
                "doc_id": doc_id,
                "text": text[start:end],
                "start": start,
                "end": end,
                "match": how,
                "kind": "answer_span",
                "markdown": (
                    {"text": markdown[markdown_span[0]:markdown_span[1]],
                     "start": markdown_span[0], "end": markdown_span[1]}
                    if markdown_span else None
                ),
            })

        impossible = bool(question["is_impossible"]) or empty_answer
        items.append({
            "id": f"techqa-{question['id']}",
            # Forum questions are a title followed by the poster's description.
            "question": question["question"].strip(),
            "answer": "Unanswerable" if impossible else question["answer"],
            "evidences": evidences,
            "metadata": {
                "answer_type": "unanswerable" if impossible else "extractive",
                "gold_documents": [
                    documents[c["filename"]][0] for c in question["contexts"]
                ],
                "domain": "ibm_technical_support",
            },
        })

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    unanswerable = sum(1 for item in items if item["metadata"]["answer_type"] == "unanswerable")
    print(f"Wrote {len(items):,} items to {ITEMS_PATH}")
    print(f"  {len(items) - unanswerable:,} answerable, {unanswerable:,} unanswerable "
          f"(genuine: no technote resolves them)")
    print(f"  answer spans recovered: {counts['exact']:,} verbatim, {counts['folded']:,} "
          f"after whitespace folding, {counts['unlocated']:,} not found in the gold technote")
    if counts["empty_answer"]:
        print(f"  {counts['empty_answer']} questions flagged answerable ship an empty "
              f"answer and are recorded as unanswerable")
    if counts["no_markdown"]:
        print(f"  {counts['no_markdown']:,} spans cross block boundaries and have no "
              f"Markdown counterpart")


if __name__ == "__main__":
    main()
