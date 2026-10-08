"""Retrieval: the chunks of each index that its dataset's questions retrieve, best first.

    .venv\\Scripts\\python.exe -m experiments.retrieve experiments/configs/retrieve_dense.json

The configuration file names the indexes and the retrieval methods, and every
pair of the two is run:

    {
      "index": "squad/txt/*/none/bge-m3",
      "retrieval": [{"method": "dense", "params": {"top_k": 50}},
                    {"method": "hybrid", "params": {"top_k": [20, 50]}}],
      "max_items": 2000
    }

`index` is a path under `indexes/` (`squad/txt/sentence-max_tokens=256-overlap=0/none/bge-m3`),
a glob pattern over such paths, "all", or a list of them. A retrieval `method`
is dense, sparse or hybrid (`rag.retrieval`), and `top_k` (50 if left out) is
how many chunks each retriever returns; hybrid fuses two such lists and keeps
`top_k`. `max_items` takes that many questions in the sample order (see
`common.load_items`); left out, all of them.

Each run writes

    results/<index>/<method>-top_k=<k>/config.json     records the index as it stood, and the method.
    results/<index>/<method>-top_k=<k>/ranking.jsonl   holds one line per question.

A line of `ranking.jsonl` holds the question's `id` and text, its `ranking` --
`[doc_id, start, end, chunk index, score]` per chunk, best first, which is
enough to score the spans and to find each chunk's text in the index's
`chunks.jsonl` -- and its `cost`: `retrieval_seconds` and the query's
`embedding_tokens`. Queries are embedded a batch at a time, so a question's
share of that request is the batch's divided evenly; the search itself is
timed per question.

Questions are retrieved a batch at a time and appended as each batch
finishes, so a run stopped anywhere continues after the questions already
there, and a larger `max_items` extends a smaller one. The index's document
and chunk counts are part of `config.json`, so a run is not extended after its
index has grown.
"""

import itertools
import time

from tqdm import tqdm

from rag.embedding import BATCH_SIZE
from rag.ingestion import load_bm25, open_vector_store
from rag.llm import get_client, usage, usage_since
from rag.retrieval import METHODS, RRF_K, embed_queries, retrieve

from experiments.common import (INDEX_DEPTH, INDEXES, RESULTS, append_jsonl, check_keys,
                                index_chunks, load_chunks, load_config, load_items, name,
                                parse_arguments, read_json, read_jsonl, relative, resolve, run_each,
                                start_run, sweep)

CONFIG_KEYS = ("index", "retrieval", "max_items")

# Chunks each retriever returns when a configuration does not say.
TOP_K = 50


def retrieval_options(entry, where):
    """The retrieval alternatives one entry stands for, its sweeps expanded."""
    check_keys(entry, ("method", "params"), where)
    if entry.get("method") not in METHODS:
        raise ValueError(f"{where}: unknown retrieval method {entry.get('method')!r}; choose "
                         f"from {METHODS}")
    check_keys(entry.get("params", {}), ("top_k",), f"{where}, params")
    return [{"method": entry["method"], "top_k": params.get("top_k", TOP_K)}
            for params in sweep(entry.get("params", {}))]


def open_stores(index_directory, method, top_k, chunks):
    """Open what `method` searches in an index: `(vector store, BM25 retriever)`.

    The one a method does not use is None. A collection that holds fewer
    vectors than the index has `chunks` is refused: it was not finished, and
    the chunks it lacks could never be found.
    """
    config = read_json(index_directory / "config.json")
    store = bm25 = None
    if method in ("dense", "hybrid"):
        model = config["embedding"]["model"]
        store = open_vector_store(index_directory / "chroma", model)
        stored = store._collection.count()
        if stored != chunks:
            raise ValueError(f"{index_directory} has {chunks} chunks but {stored} {model} "
                             f"vectors: run experiments.ingest again")
    if method in ("sparse", "hybrid"):
        if not (index_directory / "bm25").is_dir():
            raise ValueError(f"{index_directory} has no BM25 index: run experiments.ingest again")
        language = read_json(index_chunks(index_directory) / "config.json")["language"]
        bm25 = load_bm25(index_directory / "bm25", language, top_k)
        # bm25s refuses a k larger than its corpus, which an index over a few
        # documents can have.
        bm25.similarity_top_k = min(top_k, len(bm25.corpus))
    return store, bm25


def _ranking(results):
    """A ranking as stored: `[doc_id, start, end, chunk index, score]` per chunk."""
    return [[chunk.doc_id, chunk.start, chunk.end, chunk.index, round(float(value), 6)]
            for chunk, value in results]


def run_retrieval(index_directory, method, top_k, max_items=None):
    """Retrieve for one index with one method, continuing what is on disk; return the directory."""
    index_config = read_json(index_directory / "config.json")
    chunks_directory = index_chunks(index_directory)
    chunks_config = read_json(chunks_directory / "config.json")
    documents = read_jsonl(chunks_directory / "documents.jsonl")
    chunks = load_chunks(chunks_directory)
    directory = RESULTS / relative(index_directory, INDEXES) / name(method, {"top_k": top_k})
    directory.mkdir(parents=True, exist_ok=True)
    config = {"stage": "retrieval", "index": relative(index_directory, INDEXES),
              "index_config": index_config, "chunks_config": chunks_config,
              "documents": len(documents), "chunks": len(chunks),
              "method": method, "top_k": top_k, "rrf_k": RRF_K if method == "hybrid" else None}
    done = {record["id"] for record in read_jsonl(directory / "ranking.jsonl")}
    start_run(directory, config, started=bool(done))

    items, skipped = load_items(chunks_config["dataset"], chunks_config["rendering"],
                                {chunk.doc_id for chunk in chunks})
    items = items[:max_items]
    todo = [item for item in items if item["id"] not in done]
    print(f"{relative(directory, RESULTS)}: {len(items)} questions ({len(done)} done), "
          f"left out: {skipped}")
    if not todo:
        return directory

    store, bm25 = open_stores(index_directory, method, top_k, len(chunks))
    model = index_config["embedding"]["model"]
    client = get_client()
    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="retrieving", unit="batch"):
        batch = todo[start:start + BATCH_SIZE]
        questions = [item["question"] for item in batch]
        before, started = usage(), time.perf_counter()
        vectors = embed_queries(questions, model, client) if store else [None] * len(batch)
        # The batch's one request, shared evenly between its questions.
        shared_seconds = (time.perf_counter() - started) / len(batch)
        shared_tokens = usage_since(before).get("embedding_tokens", 0) / len(batch)
        records = []
        for item, vector in zip(batch, vectors):
            started = time.perf_counter()
            results = retrieve(item["question"], store, bm25, method, top_k, vector=vector)
            records.append({"id": item["id"], "question": item["question"],
                            "ranking": _ranking(results),
                            "cost": {"retrieval_seconds": round(time.perf_counter() - started
                                                                + shared_seconds, 6),
                                     "embedding_tokens": round(shared_tokens, 3)}})
        append_jsonl(directory / "ranking.jsonl", records)
    return directory


def main():
    arguments = parse_arguments(__doc__.splitlines()[0])
    config = load_config(arguments.config, CONFIG_KEYS)
    indexes = resolve(INDEXES, config.get("index"), (INDEX_DEPTH,), arguments.config)
    entries = config.get("retrieval") or []
    if not entries:
        raise ValueError(f"{arguments.config}: list at least one retrieval entry")
    options = [option for number, entry in enumerate(entries, start=1)
               for option in retrieval_options(entry, f"{arguments.config}, retrieval {number}")]
    run_each(list(itertools.product(indexes, options)),
             lambda run: f"{relative(run[0], INDEXES)} / "
                         f"{name(run[1]['method'], {'top_k': run[1]['top_k']})}",
             lambda run: run_retrieval(run[0], run[1]["method"], run[1]["top_k"],
                                       config.get("max_items")),
             arguments.dry_run, arguments.keep_going)


if __name__ == "__main__":
    main()
