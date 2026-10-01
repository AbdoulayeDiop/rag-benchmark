"""Evaluation: how well an index's retrieval finds each question's evidence.

One run scores one index (`indexes/<dataset>/<run>/`, built by `ingest.py`)
under one retrieval set-up -- method, embedding model, reranker -- against the
dataset's questions, and writes

    results/<dataset>/<run>/<set-up>/config.json     what was evaluated
    results/<dataset>/<run>/<set-up>/items.jsonl     one line per question: its spans and rankings
    results/<dataset>/<run>/<set-up>/summary.json    the metrics, averaged

Only retrieval is scored, for now: chunking changes what is retrieved, and an
answer generated from it measures the generator and its judge as much as the
chunks. The metrics compare the retrieved chunks' character spans with the
evidence spans the loaders recorded, so they call no model, cost nothing beyond
the retrieval itself, and come out the same every time:

    hit@k            a chunk among the first k overlaps an evidence span
    mrr@10           1 / rank of the first such chunk, 0 beyond 10
    ndcg@k           binary relevance (the chunk overlaps evidence); the ideal
                     ranking puts every chunk of the index that does first
    precision@k      share of the characters of the first k chunks that are evidence
    recall@k         share of the evidence characters the first k chunks cover
    f1@k             harmonic mean of the two
    doc_hit@k        a chunk among the first k is from an evidence document

Precision and recall are counted in characters -- Chroma's chunking
evaluation (Smith and Troynikov, 2024) counts the same in tokens -- so they
share a unit and their F1 means something. Precision is what keeps the comparison fair to small chunks: the
other metrics favour large ones -- a method that cuts a book into four pieces
hits the evidence at k = 1 by covering a quarter of the book -- and precision
pays for that quarter. Where evidence is a short answer span, as in SQuAD,
precision is small for any chunk size; it compares methods, it is not good
or bad in itself.

Each metric is computed twice, on the retrievers' ranking (`retrieved`) and on
the reranker's (`reranked`), and over two groups of questions kept apart:
`evidence`, those with evidence spans in the run's rendering, and
`no_evidence`, those without spans but tied to a document -- unanswerable
questions in a single-document dataset, and all of LiteraryQA, whose loader
records no spans -- which get only `doc_hit`. A question
with neither is left out, and so is one whose documents are not all in the
index: an index over `--max-documents` holds part of a corpus, and a question
about a document it lacks could never be answered.

A dataset's evidence is not all alike, which the per-dataset tables should
note. SQuAD's several spans are the answers of several annotators, each enough
alone, and Qasper's merge the evidence of several annotators, so `recall`
there undercounts an answerable context. ConditionalQA's are the page elements
an answer and its conditions rest on together, which `recall` measures and
`hit` does not. TriviaQA's evidence is a whole document, so `hit` is
`doc_hit` there and `precision` and `recall` mean little. GutenQA's "chunk" matches (12%)
are the gold chunk of the source dataset, longer than the answer itself.

The questions are taken in an order fixed by a hash of their ids, so
`--max-items` picks the same sample every time, and a larger one contains a
smaller. Questions are retrieved and reranked a batch at a time and appended
to `items.jsonl` as each batch finishes; a run stopped anywhere continues
after the questions already there. `items.jsonl` keeps the rankings, not the
metrics, and `summary.json` is recomputed from it at the end of every run.
"""

import argparse
import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from embedding import BATCH_SIZE, load_tokenizer
from ingest import ROOT, _append_jsonl, _read_json, _read_jsonl, _write_json, load_chunks
from llm import get_client
from rerank import rerank
from retrieval import METHODS, RRF_K, embed_queries, open_stores, retrieve

# The cut-offs every metric is reported at.
KS = (1, 3, 5, 10)

# The key an evidence holds its spans in a rendering under; plain text uses
# the evidence's own `start` and `end`.
RENDERING_KEYS = {"md": "markdown", "html": "html"}


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


def load_items(dataset, rendering, indexed):
    """The dataset's questions an index over the documents `indexed` can be scored on.

    Returns them in the order `--max-items` samples from (by a hash of the
    id), each with its `spans` and `documents`, and a count of those left out
    and why.
    """
    path = ROOT / "datasets" / dataset / "data" / "items.jsonl"
    items, skipped = [], {"no_document": 0, "not_indexed": 0, "not_in_rendering": 0}
    for item in _read_jsonl(path):
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
                          "documents": documents, "spans": spans})
    items.sort(key=lambda item: hashlib.sha1(item["id"].encode()).hexdigest())
    return items, skipped


def _merge(intervals):
    """Union of `(start, end)` intervals, as sorted disjoint intervals."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _intersection(first, second):
    """Characters two sets of sorted disjoint intervals have in common."""
    total = i = j = 0
    while i < len(first) and j < len(second):
        total += max(0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return total


def _intervals(spans):
    """`[doc_id, start, end, ...]` spans as each document's merged intervals."""
    by_document = {}
    for span in spans:
        by_document.setdefault(span[0], []).append((span[1], span[2]))
    return {doc_id: _merge(intervals) for doc_id, intervals in by_document.items()}


def _length(intervals_by_document):
    """Characters in a document-to-intervals mapping."""
    return sum(end - start for intervals in intervals_by_document.values()
               for start, end in intervals)


def _overlaps(chunk, spans):
    doc_id, start, end = chunk[:3]
    return any(doc_id == span_doc and start < span_end and span_start < end
               for span_doc, span_start, span_end in spans)


def relevant_chunks(spans, chunks_by_document):
    """How many chunks of the index overlap an evidence span: the ideal ranking's length."""
    return sum(_overlaps((doc_id, chunk.start, chunk.end), spans)
               for doc_id in {span[0] for span in spans}
               for chunk in chunks_by_document.get(doc_id, ()))


def score(ranking, record, ks=KS):
    """The metrics of one ranking -- `[doc_id, start, end, ...]` lists -- for one record."""
    spans = [tuple(span) for span in record["spans"]]
    documents = set(record["documents"])
    metrics = {}
    for k in ks:
        top = ranking[:k]
        metrics[f"doc_hit@{k}"] = float(any(chunk[0] in documents for chunk in top))
    if not spans:
        return metrics
    relevant = [_overlaps(chunk, spans) for chunk in ranking]
    first = next((rank for rank, hit in enumerate(relevant[:10], start=1) if hit), None)
    metrics["mrr@10"] = 1 / first if first else 0.0
    # Evidence spans overlap (SQuAD's annotators often mark the same answer),
    # and so do chunks cut with an overlap: both sides are counted as their
    # union, per document, so no character counts twice.
    evidence = _intervals(spans)
    evidence_chars = _length(evidence)
    for k in ks:
        metrics[f"hit@{k}"] = float(any(relevant[:k]))
        dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(relevant[:k], start=1) if hit)
        ideal = sum(1 / math.log2(rank + 1)
                    for rank in range(1, min(k, record["relevant_chunks"]) + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
        retrieved = _intervals(ranking[:k])
        covered = sum(_intersection(intervals, retrieved.get(doc, []))
                      for doc, intervals in evidence.items())
        precision = covered / _length(retrieved) if retrieved else 0.0
        recall = covered / evidence_chars if evidence_chars else 0.0
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"f1@{k}"] = (2 * precision * recall / (precision + recall)
                              if precision + recall else 0.0)
    return metrics


def summarize(records, ks=KS):
    """Average the metrics of every record, per stage and per group of questions."""
    stages = [stage for stage in ("retrieved", "reranked") if records and stage in records[0]]
    summary = {"items": {"evidence": sum(bool(r["spans"]) for r in records),
                         "no_evidence": sum(not r["spans"] for r in records)},
               "metrics": {}}
    for stage in stages:
        summary["metrics"][stage] = {}
        for group, members in (("evidence", [r for r in records if r["spans"]]),
                               ("no_evidence", [r for r in records if not r["spans"]])):
            if not members:
                continue
            scores = [score(record[stage], record, ks) for record in members]
            summary["metrics"][stage][group] = {
                name: sum(s[name] for s in scores) / len(scores) for name in scores[0]}
    return summary


def build_config(index_directory, method="hybrid", embedding_model=None, top_k=50,
                 rerank_model=None, rerank_depth=50, rerank_max_tokens=None):
    """Describe an evaluation: the index as it stands and the retrieval set-up.

    The index's document and chunk counts are part of it, so results are not
    appended to after the index has grown; `--max-items` is not, since a
    larger sample extends a smaller one.
    """
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method {method!r}; choose from {METHODS}")
    index_directory = Path(index_directory)
    documents = _read_jsonl(index_directory / "documents.jsonl")
    config = {
        "index": index_directory.name,
        "index_config": _read_json(index_directory / "config.json"),
        "documents": len(documents),
        "chunks": sum(document["chunks"] for document in documents),
        "retrieval": method,
        "embedding_model": embedding_model if method != "sparse" else None,
        "top_k": top_k,
        "rrf_k": RRF_K if method == "hybrid" else None,
        "rerank_model": rerank_model,
        "rerank_depth": rerank_depth if rerank_model else None,
        "rerank_max_tokens": rerank_max_tokens if rerank_model else None,
    }
    return json.loads(json.dumps(config))


def setup_name(config):
    """Directory name for an evaluation set-up: method, embedding model, reranker."""
    name = config["retrieval"]
    if config["embedding_model"]:
        name += f"-{config['embedding_model']}"
    if config["rerank_model"]:
        name += f"+{config['rerank_model']}"
    return re.sub(r"[^A-Za-z0-9._+=-]", "_", name)


def _ranking(results):
    """A ranking as stored: `[doc_id, start, end, chunk index, score]` per chunk."""
    return [[chunk.doc_id, chunk.start, chunk.end, chunk.index, round(float(value), 6)]
            for chunk, value in results]


def evaluate(dataset, index, method="hybrid", embedding_model=None, top_k=50, rerank_model=None,
             rerank_depth=50, rerank_tokenizer=None, rerank_max_tokens=None, max_items=None,
             name=None, workers=4):
    """Evaluate one index under one retrieval set-up; return the summary.

    `index` is the run's directory name under `indexes/<dataset>/`. `top_k`
    chunks are retrieved per retriever (hybrid fuses two such lists and keeps
    `top_k`), and the first `rerank_depth` of them are reranked by
    `rerank_model`, if one is named, `workers` questions at a time. A chunk too
    long for the reranker is cut to `rerank_max_tokens`, counted by
    `rerank_tokenizer` (see `rerank.rerank`). `max_items` evaluates that many
    questions of the sample order; None, all of them.
    """
    index_directory = ROOT / "indexes" / dataset / index
    if not (index_directory / "config.json").exists():
        raise FileNotFoundError(f"{index_directory} is not an index: run ingest.py first")
    config = build_config(index_directory, method, embedding_model, top_k, rerank_model,
                          rerank_depth, rerank_max_tokens)
    results_directory = ROOT / "results" / dataset / index / (name or setup_name(config))
    results_directory.mkdir(parents=True, exist_ok=True)
    config_path = results_directory / "config.json"
    done = {record["id"] for record in _read_jsonl(results_directory / "items.jsonl")}
    if done and _read_json(config_path) != config:
        raise ValueError(f"{results_directory} was evaluated with a different configuration or "
                         f"against a smaller index; pass another --name or delete it. "
                         f"On disk: {_read_json(config_path)}")
    _write_json(config_path, config)

    rendering = config["index_config"]["rendering"]
    chunks_by_document = {}
    for chunk in load_chunks(index_directory):
        chunks_by_document.setdefault(chunk.doc_id, []).append(chunk)
    items, skipped = load_items(dataset, rendering, set(chunks_by_document))
    items = items[:max_items]
    todo = [item for item in items if item["id"] not in done]
    print(f"{dataset}/{index}: {len(items)} questions to score ({len(done)} done), "
          f"left out: {skipped}")

    store, bm25 = open_stores(index_directory, method, embedding_model, top_k)
    client = get_client()
    tokenizer = load_tokenizer(rerank_tokenizer) if rerank_model and rerank_max_tokens else None

    def rerank_one(pair):
        item, retrieved = pair
        return rerank(item["question"], retrieved[:rerank_depth], rerank_model, client,
                      tokenizer, rerank_max_tokens)

    with ThreadPoolExecutor(workers) as pool:
        for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="evaluating", unit="batch"):
            batch = todo[start:start + BATCH_SIZE]
            questions = [item["question"] for item in batch]
            vectors = (embed_queries(questions, embedding_model, client) if store
                       else [None] * len(batch))
            retrieved = [retrieve(question, vector, store, bm25, method, top_k)
                         for question, vector in zip(questions, vectors)]
            reranked = (list(pool.map(rerank_one, zip(batch, retrieved))) if rerank_model
                        else [None] * len(batch))
            records = []
            for item, first, second in zip(batch, retrieved, reranked):
                record = {**item, "relevant_chunks": relevant_chunks(
                    [tuple(span) for span in item["spans"]], chunks_by_document),
                    "retrieved": _ranking(first)}
                if second is not None:
                    record["reranked"] = _ranking(second)
                records.append(record)
            _append_jsonl(results_directory / "items.jsonl", records)

    selected = {item["id"] for item in items}
    records = [record for record in _read_jsonl(results_directory / "items.jsonl")
               if record["id"] in selected]
    lengths = [chunk.end - chunk.start for chunks in chunks_by_document.values()
               for chunk in chunks]
    summary = {"dataset": dataset, "index": index, "setup": results_directory.name,
               "mean_chunk_chars": sum(lengths) / len(lengths) if lengths else 0,
               "skipped": skipped, **summarize(records)}
    _write_json(results_directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="a directory of indexes/")
    parser.add_argument("index", help="the run to evaluate: a directory of indexes/<dataset>/")
    parser.add_argument("--retrieval", default="hybrid", choices=METHODS)
    parser.add_argument("--top-k", type=int, default=50,
                        help="chunks retrieved per retriever; hybrid fuses and keeps as many")
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL"),
                        help="the model the index was embedded with, for dense and hybrid; "
                             "defaults to EMBEDDING_MODEL")
    parser.add_argument("--rerank-model", default=os.environ.get("RERANK_MODEL"),
                        help="cross-encoder served at the endpoint's /rerank; defaults to "
                             "RERANK_MODEL")
    parser.add_argument("--no-rerank", action="store_true",
                        help="score the retrievers' ranking alone, even if RERANK_MODEL is set")
    parser.add_argument("--rerank-depth", type=int, default=50,
                        help="how many of the retrieved chunks are reranked")
    parser.add_argument("--rerank-max-tokens", type=int,
                        default=os.environ.get("RERANK_MAX_TOKENS"),
                        help="the reranker's limit on a query and chunk together; longer chunks "
                             "are cut to fit (8192 for bge-reranker-v2-m3). Defaults to "
                             "RERANK_MAX_TOKENS; unset, nothing is cut")
    parser.add_argument("--rerank-tokenizer", default=os.environ.get("RERANK_TOKENIZER"),
                        help="Hugging Face repository of the reranker's tokenizer (BAAI/bge-m3); "
                             "defaults to RERANK_TOKENIZER")
    parser.add_argument("--max-items", type=int, help="questions to score, from a fixed sample "
                                                      "order; all of them by default")
    parser.add_argument("--name", help="directory under results/<dataset>/<index>/")
    parser.add_argument("--workers", type=int, default=4, help="rerank requests in flight at once")
    arguments = parser.parse_args()
    rerank_model = None if arguments.no_rerank else arguments.rerank_model
    if rerank_model and arguments.rerank_max_tokens and not arguments.rerank_tokenizer:
        parser.error("--rerank-max-tokens counts with the reranker's own tokenizer: "
                     "set --rerank-tokenizer or RERANK_TOKENIZER as well")
    summary = evaluate(arguments.dataset, arguments.index, arguments.retrieval,
                       arguments.embedding_model, arguments.top_k, rerank_model,
                       arguments.rerank_depth, arguments.rerank_tokenizer,
                       arguments.rerank_max_tokens, arguments.max_items, arguments.name,
                       arguments.workers)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
