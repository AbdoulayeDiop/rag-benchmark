"""Evaluation: how well the rankings of a retrieval or reranking run find each question's evidence.

    .venv\\Scripts\\python.exe -m experiments.evaluate experiments/configs/evaluate.json

The configuration file names the runs to score, and optionally a RAGAS judge:

    {
      "results": "squad/**",
      "ragas": {"judge_model": "gpt-oss-120b", "k": 5, "max_items": 200,
                "requests_per_minute": 10, "workers": 8}
    }

`results` is a path under `results/` written by `retrieve.py` or `rerank.py`,
a glob pattern over such paths, "all", or a list of them. Each run is scored
on every question in its `ranking.jsonl`, and everything goes into the run's
own directory:

    <run>/evaluation/summary.json                  holds the span metrics and the RAGAS ones.
    <run>/evaluation/ragas-<judge>-k=<k>.jsonl     holds the judge's verdict on each question.

The span metrics (`rag.evaluation`) call no model and are recomputed from the
rankings at every run. They are computed over two groups of questions kept
apart: `evidence`, those with evidence spans in the index's rendering, and
`no_evidence`, those without spans but tied to a document -- unanswerable
questions in a single-document dataset, and all of LiteraryQA, whose loader
records no spans -- which get only `doc_hit`.

A dataset's evidence is not all alike, which the per-dataset tables should
note. SQuAD's several spans are the answers of several annotators, each enough
alone, and Qasper's merge the evidence of several annotators, so
`evidence_recall` there undercounts an answerable context. ConditionalQA's are
the page elements an answer and its conditions rest on together, which
`evidence_recall` measures and `hit` does not. TriviaQA's evidence is a whole
document, so `hit` is `doc_hit` there and `evidence_recall` means little.
GutenQA's "chunk" matches (12%) are the gold chunk of the source dataset,
longer than the answer itself.

The summary also gives the mean `cost` of a question, from its line in
`ranking.jsonl`, and the medians of the seconds. A question ranked before costs
were recorded has none, and `questions` says how many do.

With `ragas`, the first `k` chunks (5) of each ranking are judged by
`judge_model` (`rag.judge`), on a sample: the first `max_items` (200) questions
of the sample order that have evidence and a reference answer, since RAGAS
needs one. `workers` questions (8) are judged at once, their calls kept to the
judge's `requests_per_minute` at the endpoint (10 for gpt-oss-120b); left out,
they are not spaced. A chunk is shown to the judge as `build_text_for_llm`
gives it: each augmentation headed by its metadata key, then `original_text`
(tags stripped for HTML), so the judge can tell what the document says from
what was generated about it. Verdicts are appended a batch of questions at a
time, so a stopped run continues after the questions already judged; a
question whose judging failed is kept with its error and left out of the
means, and judged again the next time the run is evaluated. The judge and `k` name the file, and the summary holds one entry per
pair under `ragas`. At about a minute a question under the judge's limit, it
is meant for a few runs, not a grid.
"""

import asyncio
from statistics import mean, median

from tqdm import tqdm

from rag.chunking.base import build_text_for_llm
from rag.evaluation import mean_scores, relevant_chunks, score

from experiments.common import (INDEXES, RERANK_DEPTH, RESULTS, RETRIEVAL_DEPTH, append_jsonl,
                                check_keys, index_chunks, load_chunks, load_config, load_items,
                                name, parse_arguments, read_json, read_jsonl, relative, resolve,
                                run_each, sample_order, write_json)

CONFIG_KEYS = ("results", "ragas")
RAGAS_KEYS = ("judge_model", "k", "max_items", "requests_per_minute", "workers",
              "max_output_tokens")


def summarize_cost(records):
    """The mean cost of a question, and the median of its seconds, over the records that have one."""
    costs = [record["cost"] for record in records if "cost" in record]
    if not costs:
        return {}
    summary = {"questions": len(costs)}
    for key in sorted({key for cost in costs for key in cost}):
        values = [cost.get(key, 0) for cost in costs]
        summary[key] = round(mean(values), 6)
        if key.endswith("_seconds"):
            summary[key.replace("_seconds", "_seconds_median")] = round(median(values), 6)
    return summary


def _setup(directory):
    """A run's configuration, and the retrieval run's (its own, for a retrieval run)."""
    config = read_json(directory / "config.json")
    return config, (config["retrieval_config"] if config["stage"] == "rerank" else config)


def score_spans(directory):
    """Score one run's rankings against the evidence spans; return the summary."""
    config, retrieval = _setup(directory)
    chunks_config = retrieval["chunks_config"]
    chunks_by_document = {}
    for chunk in load_chunks(index_chunks(INDEXES / retrieval["index"])):
        chunks_by_document.setdefault(chunk.doc_id, []).append(chunk)
    items, skipped = load_items(chunks_config["dataset"], chunks_config["rendering"],
                                set(chunks_by_document))
    items = {item["id"]: item for item in items}
    records = read_jsonl(directory / "ranking.jsonl")
    if not records:
        raise FileNotFoundError(f"{directory} holds no rankings yet")

    groups = {"evidence": [], "no_evidence": []}
    for record in records:
        item = items[record["id"]]
        relevant = relevant_chunks([tuple(span) for span in item["spans"]], chunks_by_document)
        groups["evidence" if item["spans"] else "no_evidence"].append(
            score(record["ranking"], item["spans"], item["documents"], relevant))
    lengths = [chunk.end - chunk.start for chunks in chunks_by_document.values()
               for chunk in chunks]
    reranking = ({key: config[key] for key in ("method", "model", "top_k", "max_tokens")}
                 if config["stage"] == "rerank" else None)
    return {
        "dataset": chunks_config["dataset"],
        "run": relative(directory, RESULTS),
        "index": retrieval["index"],
        "setup": relative(directory, RESULTS / retrieval["index"]),
        "chunking": chunks_config,
        "embedding": retrieval["index_config"]["embedding"],
        "retrieval": {"method": retrieval["method"], "top_k": retrieval["top_k"]},
        "reranking": reranking,
        "documents": retrieval["documents"],
        "chunks": retrieval["chunks"],
        "mean_chunk_chars": sum(lengths) / len(lengths) if lengths else 0,
        "skipped": skipped,
        "items": {group: len(scores) for group, scores in groups.items()},
        "metrics": {group: mean_scores(scores) for group, scores in groups.items() if scores},
        "cost": summarize_cost(records),
    }


def judge_run(directory, judge_model, k=5, max_items=200, requests_per_minute=None, workers=8,
              max_output_tokens=2048):
    """Judge one run's rankings with RAGAS, continuing what is on disk; return the summary entry."""
    # Imported here: ragas is slow to load, and most evaluations do not use it.
    from rag.judge import judge, judge_metrics, summarize

    _, retrieval = _setup(directory)
    chunks_directory = index_chunks(INDEXES / retrieval["index"])
    chunks = {(chunk.doc_id, chunk.index): chunk for chunk in load_chunks(chunks_directory)}
    chunks_config = retrieval["chunks_config"]
    items, _ = load_items(chunks_config["dataset"], chunks_config["rendering"],
                          {doc_id for doc_id, _ in chunks})
    items = {item["id"]: item for item in items}
    records = sorted((record for record in read_jsonl(directory / "ranking.jsonl")
                      if items[record["id"]]["spans"] and items[record["id"]]["answer"]),
                     key=lambda record: sample_order(record["id"]))[:max_items]
    path = directory / "evaluation" / f"{name(f'ragas-{judge_model}', {'k': k})}.jsonl"
    path.parent.mkdir(exist_ok=True)
    # A question whose judging failed is judged again: its verdict is dropped
    # from the file, through a copy so that a stopped run loses none of the others.
    verdicts = read_jsonl(path)
    kept = [record for record in verdicts if "error" not in record]
    if len(kept) < len(verdicts):
        print(f"{len(verdicts) - len(kept)} failed questions in {path} are judged again")
        partial = path.with_suffix(".tmp")
        partial.unlink(missing_ok=True)
        append_jsonl(partial, kept)
        partial.replace(path)
    done = {record["id"] for record in kept}
    todo = [record for record in records if record["id"] not in done]
    print(f"{len(records)} questions to judge ({len(done)} done) by {judge_model}, top {k}")

    def context(entry):
        chunk = chunks[(entry[0], entry[3])]
        return build_text_for_llm(chunk.original_text, chunk.metadata)

    async def run():
        precision, recall = judge_metrics(judge_model, requests_per_minute, max_output_tokens)
        semaphore = asyncio.Semaphore(workers)

        async def one(record):
            async with semaphore:
                contexts = [context(entry) for entry in record["ranking"][:k]]
                return {"id": record["id"],
                        **await judge(record["question"], items[record["id"]]["answer"],
                                      contexts, precision, recall)}

        # A batch is written when its last question is judged, so a stopped
        # run loses at most one batch.
        batch_size = workers * 4
        for start in tqdm(range(0, len(todo), batch_size), desc="judging", unit="batch"):
            append_jsonl(path, await asyncio.gather(*map(one, todo[start:start + batch_size])))

    if todo:
        asyncio.run(run())
    selected = {record["id"] for record in records}
    judged = [record for record in read_jsonl(path) if record["id"] in selected]
    entry = {"judge_model": judge_model, "k": k, "context": "build_text_for_llm",
             **summarize(judged, k)}
    if entry["failed"]:
        print(f"{entry['failed']} questions could not be judged; their errors are in {path}")
    return entry


def evaluate(directory, ragas=None):
    """Score one run, and judge it if `ragas` names a judge; write and return the summary."""
    path = directory / "evaluation" / "summary.json"
    previous = read_json(path) if path.exists() else {}
    summary = {**score_spans(directory), "ragas": previous.get("ragas", {})}
    write_json(path, summary)
    if ragas:
        k = ragas.get("k", 5)
        summary["ragas"][f"{ragas['judge_model']}-k={k}"] = judge_run(
            directory, ragas["judge_model"], k, ragas.get("max_items", 200),
            ragas.get("requests_per_minute"), ragas.get("workers", 8),
            ragas.get("max_output_tokens", 2048))
        write_json(path, summary)
    print(f"{summary['run']}: {summary['metrics'].get('evidence', summary['metrics'].get('no_evidence'))}")
    return summary


def main():
    arguments = parse_arguments(__doc__.splitlines()[0])
    config = load_config(arguments.config, CONFIG_KEYS)
    ragas = config.get("ragas")
    if ragas is not None:
        check_keys(ragas, RAGAS_KEYS, f"{arguments.config}, ragas")
        if not ragas.get("judge_model"):
            raise ValueError(f"{arguments.config}, ragas: name the judge in judge_model")
    runs = [path for path in resolve(RESULTS, config.get("results"),
                                     (RETRIEVAL_DEPTH, RERANK_DEPTH), arguments.config)
            if (path / "ranking.jsonl").exists()]
    run_each(runs, lambda run: relative(run, RESULTS), lambda run: evaluate(run, ragas),
             arguments.dry_run, arguments.keep_going)


if __name__ == "__main__":
    main()
