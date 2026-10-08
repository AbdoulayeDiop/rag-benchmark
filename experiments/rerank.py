"""Reranking: the chunks a retrieval run found for each question, reordered by a cross-encoder.

    .venv\\Scripts\\python.exe -m experiments.rerank experiments/configs/rerank.json

The configuration file names the retrieval runs and the rerankers, and every
pair of the two is run:

    {
      "retrieval": "squad/txt/*/none/bge-m3/dense-top_k=50",
      "reranking": [{"method": "cross_encoder",
                     "params": {"model": "bge-reranker-v2-m3", "top_k": 10,
                                "max_tokens": 8192, "tokenizer": "BAAI/bge-m3"}}],
      "max_items": 2000,
      "workers": 16
    }

`retrieval` is a path under `results/` written by `retrieve.py`, a glob
pattern over such paths, "all", or a list of them. The only method is
`cross_encoder` (`rag.rerank`): `model` is served at the endpoint's `/rerank`,
every retrieved chunk is scored and the best `top_k` (10 if left out) kept,
and a chunk too long for the model's pair limit is cut to `max_tokens`,
counted by `tokenizer`. `max_items` reranks the first that many questions of
the retrieval run; left out, all of them. `workers` is how many questions are
reranked at once (16).

Each run writes, under the retrieval run's directory,

    <retrieval>/<model>-<params>/config.json     records the retrieval run's config, and the reranker.
    <retrieval>/<model>-<params>/ranking.jsonl   holds one line per question.

in the shape `retrieve.py` writes: the question's `id` and text, its
`ranking`, and its `cost` -- the retrieval's, with `rerank_seconds` and
`rerank_tokens` added, so the line holds what the whole ranking cost. A chunk
is scored as its `get_text_to_embed`, read from the index's `chunks.jsonl`.
A run resumes after the questions already there, and refuses to go on if the
retrieval run under it was started again with another configuration.
"""

import itertools
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from rag.embedding import load_tokenizer
from rag.llm import get_client, usage, usage_since
from rag.rerank import rerank

from experiments.common import (INDEXES, RESULTS, RETRIEVAL_DEPTH, append_jsonl, check_keys,
                                index_chunks, is_none, load_chunks, load_config, name,
                                parse_arguments, read_json, read_jsonl, relative, resolve, run_each,
                                start_run, sweep)

CONFIG_KEYS = ("retrieval", "reranking", "max_items", "workers")

# The reranking methods implemented; a rule-based one would join them here.
RERANKERS = ("cross_encoder",)

# Chunks kept after reranking when a configuration does not say.
TOP_K = 10

# Questions reranked a batch at a time, each batch appended once it is done.
BATCH_SIZE = 64


def reranking_options(entry, where):
    """The reranking alternatives one entry stands for, its sweeps expanded."""
    check_keys(entry, ("method", "params"), where)
    method = entry.get("method")
    if is_none(method) or method not in RERANKERS:
        raise ValueError(f"{where}: unknown reranking method {method!r}; choose from "
                         f"{RERANKERS}")
    check_keys(entry.get("params", {}), ("model", "top_k", "max_tokens", "tokenizer"),
               f"{where}, params")
    options = []
    for params in sweep(entry.get("params", {})):
        if not params.get("model"):
            raise ValueError(f"{where}: {method} needs params.model")
        if params.get("max_tokens") and not params.get("tokenizer"):
            raise ValueError(f"{where}: max_tokens is counted with the reranker's tokenizer; "
                             f"set params.tokenizer too")
        options.append({"method": method, "model": params["model"],
                        "top_k": params.get("top_k", TOP_K),
                        "max_tokens": params.get("max_tokens"),
                        "tokenizer": params.get("tokenizer")})
    return options


def option_name(option):
    """A reranking run's directory name: the model, then `max_tokens` (if set) and `top_k`."""
    params = {"top_k": option["top_k"]}
    if option["max_tokens"]:
        params["max_tokens"] = option["max_tokens"]
    return name(option["model"], params)


def run_rerank(retrieval_directory, option, max_items=None, workers=16):
    """Rerank one retrieval run's questions with one reranker, continuing what is on disk."""
    retrieval_config = read_json(retrieval_directory / "config.json")
    directory = retrieval_directory / option_name(option)
    directory.mkdir(exist_ok=True)
    config = {"stage": "rerank", "retrieval": relative(retrieval_directory, RESULTS),
              "retrieval_config": retrieval_config, **option}
    done = {record["id"] for record in read_jsonl(directory / "ranking.jsonl")}
    start_run(directory, config, started=bool(done))

    records = read_jsonl(retrieval_directory / "ranking.jsonl")[:max_items]
    todo = [record for record in records if record["id"] not in done]
    print(f"{relative(directory, RESULTS)}: {len(records)} questions ({len(done)} done)")
    if not todo:
        return directory

    index_directory = INDEXES / retrieval_config["index"]
    chunks = {(chunk.doc_id, chunk.index): chunk
              for chunk in load_chunks(index_chunks(index_directory))}
    client = get_client()
    tokenizer = load_tokenizer(option["tokenizer"]) if option["max_tokens"] else None

    def rerank_one(record):
        """Rerank one question's chunks; the cost is its thread's."""
        before, started = usage(thread=True), time.perf_counter()
        retrieved = [chunks[(entry[0], entry[3])] for entry in record["ranking"]]
        reranked = rerank(record["question"], retrieved, option["model"], client, tokenizer,
                          option["max_tokens"], rerank_top_k=option["top_k"])
        spent = usage_since(before, thread=True)
        return {"id": record["id"], "question": record["question"],
                "ranking": [[chunk.doc_id, chunk.start, chunk.end, chunk.index,
                             round(float(score), 6)] for chunk, score in reranked],
                "cost": {**record.get("cost", {}),
                         "rerank_seconds": round(time.perf_counter() - started, 6),
                         "rerank_tokens": spent.get("rerank_tokens", 0)}}

    with ThreadPoolExecutor(workers) as pool:
        for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="reranking", unit="batch"):
            append_jsonl(directory / "ranking.jsonl",
                         list(pool.map(rerank_one, todo[start:start + BATCH_SIZE])))
    return directory


def main():
    arguments = parse_arguments(__doc__.splitlines()[0])
    config = load_config(arguments.config, CONFIG_KEYS)
    retrievals = [path for path in resolve(RESULTS, config.get("retrieval"), (RETRIEVAL_DEPTH,),
                                           arguments.config)
                  if read_json(path / "config.json").get("stage") == "retrieval"]
    entries = config.get("reranking") or []
    if not entries:
        raise ValueError(f"{arguments.config}: list at least one reranking entry")
    options = [option for number, entry in enumerate(entries, start=1)
               for option in reranking_options(entry, f"{arguments.config}, reranking {number}")]
    # The endpoint used here reranks 64 questions in 10.7 s with 4 requests in
    # flight, 5.4 s with 8, 4.5 s with 16 and 4.3 s with 32: 16 is about where
    # it stops scaling.
    workers = config.get("workers", 16)
    run_each(list(itertools.product(retrievals, options)),
             lambda run: f"{relative(run[0], RESULTS)} / {option_name(run[1])}",
             lambda run: run_rerank(run[0], run[1], config.get("max_items"), workers),
             arguments.dry_run, arguments.keep_going)


if __name__ == "__main__":
    main()
