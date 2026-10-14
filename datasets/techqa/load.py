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

SPLIT: NVIDIA's `train.json` holds the original train (600) and dev (310)
questions together; the id prefix (`TRAIN_`, `DEV_`) tells them apart. dev, the
smallest, is the default. Only answerable questions whose answer was located are
kept: the experiments score retrieval, and an unanswerable question has nothing
to retrieve.

CORPUS: only the gold technotes of the kept questions -- those their contexts
name -- are written; no distractors are added.

Technotes are written as the corpus ships them, in plain text only. A Markdown
rendering was tried and dropped: its headings had to be guessed from upper-case
lines, and technotes quote shell snippets whose lines start with "#", which a
Markdown parser reads as headings.

Outputs:
  data/documents/<technote>.txt      one technote per document
  data/items.jsonl                   {question, answer, evidences} records

Requires: huggingface_hub
Run with: python datasets/techqa/load.py [--splits dev train]
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
ITEMS_PATH = DATA_DIR / "items.jsonl"


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
        "--splits", nargs="+", default=["dev"], choices=["dev", "train"],
        help="Original splits to build, told apart by the question id. Defaults to "
             "dev (310 questions), the smallest; train has 600.",
    )
    args = parser.parse_args()

    questions = json.loads(
        Path(hf_hub_download(REPO_ID, QUESTIONS_FILE, repo_type="dataset")).read_text(encoding="utf-8")
    )
    prefixes = tuple(f"{split.upper()}_" for split in args.splits)
    selected = [question for question in questions if question["id"].startswith(prefixes)]
    archive = zipfile.ZipFile(hf_hub_download(REPO_ID, CORPUS_FILE, repo_type="dataset"))
    paths = {name.split("/")[-1]: name for name in archive.namelist() if name.endswith(".txt")}
    print(f"{len(questions)} questions, {len(selected)} in split(s) {', '.join(args.splits)}; "
          f"{len(paths):,} technotes in the corpus")

    documents = {}
    items = []
    counts = {"exact": 0, "folded": 0, "unlocated": 0, "unanswerable": 0, "empty_answer": 0,
              "no_span": 0}
    for question in selected:
        # Two questions (DEV_Q014, DEV_Q094) are flagged answerable but ship an
        # empty answer; they are left out with the unanswerable ones rather than
        # given a zero-length span.
        if question["is_impossible"]:
            counts["unanswerable"] += 1
            continue
        if not (question["answer"] or "").strip():
            counts["empty_answer"] += 1
            continue
        for context in question["contexts"]:
            if context["filename"] not in documents:
                text = archive.read(paths[context["filename"]]).decode("utf-8", errors="replace")
                documents[context["filename"]] = (document_id(context["filename"]), text)
        evidences = []
        for context in question["contexts"]:
            doc_id, text = documents[context["filename"]]
            found = locate(question["answer"], text)
            if found is None:
                counts["unlocated"] += 1
                continue
            start, end, how = found
            counts[how] += 1
            evidences.append({
                "doc_id": doc_id,
                "text": text[start:end],
                "start": start,
                "end": end,
                "match": how,
                "kind": "answer_span",
            })

        if not evidences:
            # The answer is in none of its technotes: nothing to score it on.
            counts["no_span"] += 1
            continue
        items.append({
            "id": f"techqa-{question['id']}",
            # Forum questions are a title followed by the poster's description.
            "question": question["question"].strip(),
            "answer": question["answer"],
            "evidences": evidences,
            "metadata": {
                "answer_type": "extractive",
                "gold_documents": [
                    documents[c["filename"]][0] for c in question["contexts"]
                ],
                "domain": "ibm_technical_support",
            },
        })

    # Only the gold technotes of the kept questions; a previous build may have
    # written others.
    used = {doc_id for item in items for doc_id in item["metadata"]["gold_documents"]}
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DOCUMENTS_DIR.iterdir():
        stale.unlink()
    lengths = []
    for doc_id, text in documents.values():
        if doc_id in used:
            (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
            lengths.append(len(text))
    lengths.sort()
    print(f"Wrote {len(lengths):,} gold technotes to {DOCUMENTS_DIR}")
    print(f"  chars per document: min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} "
          f"/ avg {sum(lengths) // len(lengths):,} / max {lengths[-1]:,}")

    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items):,} answerable items to {ITEMS_PATH}")
    print(f"  left out: {counts['unanswerable']:,} unanswerable (genuine: no technote "
          f"resolves them), {counts['empty_answer']} flagged answerable with an empty answer, "
          f"{counts['no_span']} whose answer is in none of its technotes")
    print(f"  answer spans recovered: {counts['exact']:,} verbatim, {counts['folded']:,} "
          f"after whitespace folding, {counts['unlocated']:,} not found in a gold technote")


if __name__ == "__main__":
    main()
