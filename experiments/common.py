"""What the experiment scripts share: the layout on disk, configuration files, the datasets.

Each stage writes under its own tree, one directory level per choice made so
far, so a path reads as the pipeline that produced it:

    chunks/<dataset>/<rendering>/<chunking>/<augmentation>/                     ingest.py
    indexes/<dataset>/<rendering>/<chunking>/<augmentation>/<embedding>/        ingest.py
    results/<dataset>/<rendering>/<chunking>/<augmentation>/<embedding>/<retrieval>/            retrieve.py
    results/<dataset>/<rendering>/<chunking>/<augmentation>/<embedding>/<retrieval>/<rerank>/   rerank.py

Every level is named `<method>-<param>=<value>-...` with the parameters in
alphabetical order, and holds a `config.json` with everything that decided its
content. The name is for reading and selecting (`squad/*/sentence-*/**`); the
config is the record, and a directory is refused when it was started with a
different one, so a run stopped anywhere resumes instead of mixing two set-ups.

A configuration file is JSON. A stage lists alternatives, `{"method": ...,
"params": {...}}`, and a list inside `params` is a sweep: one alternative per
value, every combination -- except for a parameter whose own value is a list
(`html`'s `headings`). Any entry, and the file, may carry a `description`,
which stands in for a comment and is otherwise ignored. Nothing comes from the
environment: a run is what its file says.
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import traceback
from pathlib import Path

from rag.chunking import Chunk

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
EXPERIMENTS = ROOT / "experiments"
CHUNKS = EXPERIMENTS / "chunks"
INDEXES = EXPERIMENTS / "indexes"
RESULTS = EXPERIMENTS / "results"

# Directory depth of each kind of run under its tree.
CHUNKS_DEPTH, INDEX_DEPTH, RETRIEVAL_DEPTH, RERANK_DEPTH = 4, 5, 6, 7

# Where each rendering's files live under datasets/<name>/data/.
RENDERINGS = {"txt": ("documents", ".txt"),
              "md": ("documents_md", ".md"),
              "html": ("documents_html", ".html")}

# The key an evidence holds its spans in a rendering under; plain text uses
# the evidence's own `start` and `end`.
RENDERING_KEYS = {"md": "markdown", "html": "html"}


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as file:
        return [json.loads(line) for line in file]


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def normalized(value):
    """`value` through JSON and back, so that it compares equal to its copy on disk.

    A tuple, such as a tuple of heading names, is read back as a list.
    """
    return json.loads(json.dumps(value))


def start_run(directory, config, started):
    """Write `config` into a run directory, refusing one `started` with another.

    A directory with nothing done yet holds nothing the old configuration
    produced, so a run that failed at once can be started again changed.
    """
    path = Path(directory) / "config.json"
    if started and path.exists() and read_json(path) != config:
        raise ValueError(f"{directory} was started with a different configuration; delete it "
                         f"to start again. On disk: {read_json(path)}")
    write_json(path, config)


def load_config(path, allowed):
    """Read a configuration file, refusing a top-level key not in `allowed`."""
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    check_keys(config, allowed, path)
    return config


def check_keys(entry, allowed, where):
    unknown = [key for key in entry if key not in (*allowed, "description")]
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")


def sweep(params, fixed=()):
    """Every parameter set the lists in `params` stand for, one per combination.

    A key in `fixed` takes a list as its value and is not swept.
    """
    swept = [key for key, value in params.items() if isinstance(value, list) and key not in fixed]
    return [{**params, **dict(zip(swept, values))}
            for values in itertools.product(*(params[key] for key in swept))]


def is_none(method):
    return method in (None, "", "none")


def as_list(value):
    return [value] if isinstance(value, str) else list(value or [])


def name(method, params=None):
    """A directory name: `method`, then each parameter as `-key=value`, alphabetically."""
    text = method + "".join(f"-{key}={value}" for key, value in sorted((params or {}).items()))
    return re.sub(r"[^A-Za-z0-9._+=-]", "_", text)


def relative(path, root):
    """A run's path under its tree, with forward slashes: how configs and summaries name it."""
    return Path(path).relative_to(root).as_posix()


def resolve(root, targets, depths, where):
    """The run directories under `root` that `targets` select.

    `targets` is a path under `root`, a glob pattern (`squad/*/sentence-*/**`),
    "all", or a list of them. Only directories at one of `depths` below
    `root` and holding a `config.json` are kept. A target that selects nothing
    is an error, so a typo does not pass for an empty experiment.
    """
    selected = []
    for target in as_list(targets):
        patterns = ["/".join(["*"] * depth) for depth in depths] if target == "all" else [target]
        found = [path for pattern in patterns for path in sorted(root.glob(pattern))
                 if path.is_dir() and len(path.relative_to(root).parts) in depths
                 and (path / "config.json").exists()]
        if not found:
            raise ValueError(f"{where}: {target!r} selects nothing under {root}")
        selected += [path for path in found if path not in selected]
    return selected


def check_dataset(dataset, where):
    """Refuse a dataset whose loader has not been run.

    Exact names only: Windows would open datasets/SQuAD as datasets/squad, but
    the runs would be filed under another name.
    """
    if dataset not in os.listdir(DATASETS) or not (DATASETS / dataset / "data").is_dir():
        raise ValueError(f"{where}: datasets/{dataset}/data does not exist (names are "
                         f"case-sensitive); run its loader first")


def has_rendering(dataset, rendering):
    return (DATASETS / dataset / "data" / RENDERINGS[rendering][0]).is_dir()


def read_documents(dataset, rendering="txt", max_documents=None):
    """Yield `(doc_id, text)` for a dataset's documents in one rendering, by name.

    `max_documents` keeps the first that many, in the same order every time, so
    a run started on a few documents can be extended to more.
    """
    folder, extension = RENDERINGS[rendering]
    directory = DATASETS / dataset / "data" / folder
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} does not exist: run datasets/{dataset}/load.py first, and check "
            f"that {dataset} has a '{rendering}' rendering")
    for path in sorted(directory.glob(f"*{extension}"))[:max_documents]:
        # newline="" reads the file as it was written. The default would turn
        # a stray "\r\n" into "\n" and shift every offset after it, and the
        # evidence spans are offsets into the file as it is.
        with open(path, encoding="utf-8", newline="") as file:
            yield path.stem, file.read()


def load_chunks(chunks_directory):
    """Read a chunks run's chunks, in document order: the order their vectors are stored in."""
    return [Chunk(**record) for record in read_jsonl(Path(chunks_directory) / "chunks.jsonl")]


def index_chunks(index_directory):
    """The chunks directory an index was built from."""
    return CHUNKS / read_json(Path(index_directory) / "config.json")["chunks"]


def evidence_spans(item, rendering):
    """The item's evidence as `(doc_id, start, end)` in `rendering`.

    Returns None when the item has evidence but none of it in this rendering,
    as distinct from an item that has none at all (an empty list).
    """
    spans = []
    for evidence in item["evidences"]:
        span = evidence if rendering == "txt" else evidence.get(RENDERING_KEYS[rendering])
        if span:
            spans.append((evidence["doc_id"], span["start"], span["end"]))
    return None if item["evidences"] and not spans else spans


def item_documents(item):
    """The documents an item's answer is in: those of its evidence, or the one it names."""
    documents = sorted({evidence["doc_id"] for evidence in item["evidences"]})
    if not documents and item["metadata"].get("doc_id"):
        documents = [item["metadata"]["doc_id"]]
    return documents


def sample_order(item_id):
    """The key questions are sorted by, so that `max_items` picks the same sample every time.

    A hash of the id: a larger sample contains a smaller one.
    """
    return hashlib.sha1(item_id.encode()).hexdigest()


def load_items(dataset, rendering, indexed):
    """The dataset's questions an index over the documents `indexed` can be scored on.

    Returns them in sample order, each with its `question`, `answer`,
    `spans` and `documents`, and a count of those left out and why. A question
    tied to no document is left out, and so is one whose documents are not all
    in the index -- an index over `max_documents` holds part of a corpus, and a
    question about a document it lacks could never be answered -- and one whose
    evidence is not in the index's rendering.
    """
    items, skipped = [], {"no_document": 0, "not_indexed": 0, "not_in_rendering": 0}
    for item in read_jsonl(DATASETS / dataset / "data" / "items.jsonl"):
        documents = item_documents(item)
        spans = evidence_spans(item, rendering)
        if not documents:
            skipped["no_document"] += 1
        elif not indexed.issuperset(documents):
            skipped["not_indexed"] += 1
        elif spans is None:
            skipped["not_in_rendering"] += 1
        else:
            items.append({"id": item["id"], "question": item["question"],
                          "answer": item.get("answer"), "documents": documents, "spans": spans})
    items.sort(key=lambda item: sample_order(item["id"]))
    return items, skipped


def run_each(runs, describe, run, dry_run=False, keep_going=False):
    """List `runs`, then call `run` on each unless `dry_run`.

    `describe(run)` names a run in the listing and in errors. With
    `keep_going`, a failure is printed and the next run started, and the
    failures are listed at the end with a non-zero exit; without it, the first
    one stops everything.
    """
    for number, entry in enumerate(runs, start=1):
        print(f"{number:>3}. {describe(entry)}")
    if dry_run:
        return
    failed = []
    for number, entry in enumerate(runs, start=1):
        print(f"\n=== [{number}/{len(runs)}] {describe(entry)}")
        try:
            run(entry)
        except Exception as error:
            if not keep_going:
                raise
            traceback.print_exc()
            failed.append((entry, error))
    if failed:
        print(f"\n{len(failed)} of {len(runs)} runs failed:")
        for entry, error in failed:
            print(f"  {describe(entry)}: {error}")
        raise SystemExit(1)


def parse_arguments(description):
    """The command line every experiment script takes: a configuration file and two flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="a JSON configuration file (see the module docstring)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the file and list the runs, without running them")
    parser.add_argument("--keep-going", action="store_true",
                        help="go on to the next run when one fails, and list failures at the end")
    return parser.parse_args()
