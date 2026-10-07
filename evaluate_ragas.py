"""RAGAS context precision and context recall of an evaluated retrieval set-up.

`evaluate.py` scores retrieval against the evidence spans, which calls no
model but trusts the spans: where they are loose (Qasper's merged annotators,
TriviaQA's whole documents, GutenQA's gold chunks) or where another chunk says
the same thing as the marked one, the span metrics can be wrong. RAGAS's two
retrieval metrics judge the retrieved text itself, against the question's
reference answer, with an LLM:

    context_precision@k  ContextPrecision (with reference): each of the first k
                         chunks is judged useful or not in arriving at the
                         reference answer, one call per chunk; the score is
                         the average precision of those verdicts, so useful
                         chunks count for more the higher they rank
    context_recall@k     ContextRecall: the reference answer is split into
                         statements and each is judged attributable to the
                         first k chunks or not, one call per question; the
                         score is the share attributed

The metrics come from the `ragas` library (`ragas.metrics.collections`, its
instructor-based implementation), not rewritten here, so the figures are
RAGAS's own and comparable with published ones. They are run on the rankings
`evaluate.py` stored, so retrieval is not repeated. A chunk is judged as
`build_text_for_llm` gives it: each augmentation headed by its metadata key,
then `original_text` (tags stripped for HTML), so the judge can tell what the
document says from what was generated about it.

They cost what the span metrics do not. One question takes k + 1 judge calls;
on SQuAD with 256-token chunks and k = 5, a question took 10.8 s for precision
and 3.3 s for recall with gpt-oss-120b (10.3 s and 2.7 s with
mistral-small-3-2-24b-instruct-2506), each call some 1,000-1,500 input tokens
with the prompt's examples. Under the endpoint's 128,000 input tokens a minute,
that is on the order of ten questions a minute, so a set-up is judged on a
sample: the first `--max-items` (200) of the order `evaluate.py` samples in,
which is contained in its own sample.

The judge is lenient. In that trial, "What does the original manuscript show?"
(SQuAD, about Luther's hymn; reference "multiple revisions") was given
precision and recall of 1.0 by both judges for five chunks of other articles,
the first about the manuscript of Harper Lee's *Go Set a Watchman*, none of
which mentions a revision; the span metrics scored the same ranking 0. SQuAD's
questions take their passage for granted, and a judge shown only the question
and a short answer accepts a chunk on the same topic. Read the two kinds of
metric together, not one in place of the other.

Only questions with evidence spans are judged: the rest are unanswerable or
have no span to have found (LiteraryQA), and RAGAS needs a reference. The
judge's verdicts are written a batch of questions at a time to

    results/<dataset>/<index>/<set-up>/ragas-<judge>-k=<k>/config.json
    results/<dataset>/<index>/<set-up>/ragas-<judge>-k=<k>/items.jsonl
    results/<dataset>/<index>/<set-up>/ragas-<judge>-k=<k>/summary.json

and a stopped run continues after the questions already judged. A question
whose judging fails (an answer that does not parse after instructor's own
retries) is recorded with its error and left out of the means, and the
summary counts it, rather than stopping hours of judging for one question.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from ragas.llms.base import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall
from tqdm import tqdm

from chunking.base import build_text_for_llm
from ingest import ROOT, _append_jsonl, _read_json, _read_jsonl, _write_json, load_chunks

load_dotenv(override=True)

METRICS = ("context_precision", "context_recall")


def judge_directory(results_directory, judge_model, k):
    """Where a set-up's RAGAS results go: one directory per judge and cut-off."""
    return results_directory / re.sub(r"[^A-Za-z0-9._+=-]", "_", f"ragas-{judge_model}-k={k}")


def _rate_limited(error):
    """Whether `error`, or one it was raised from, is the endpoint's rate limit.

    Instructor wraps the endpoint's error in its own, so the causes are read,
    and its message too, which keeps the status code when the chain does not.
    """
    while error is not None:
        if isinstance(error, RateLimitError) or "Error code: 429" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def pace(llm, requests_per_minute=None, attempts=5):
    """Make `llm`'s calls keep to `requests_per_minute`, and retry one the endpoint refuses.

    The endpoint limits a judge by requests as well as by tokens: gpt-oss-120b
    is allowed 10 requests a minute, and a question takes k + 1 of them, so
    sending them as fast as they come only collects refusals. Calls are spaced
    60 / `requests_per_minute` seconds apart, across every question in flight;
    None leaves them unspaced. A refused call waits a minute and is sent again,
    up to `attempts` times, so a refusal costs one call, not a question's.
    """
    agenerate, lock = llm.agenerate, asyncio.Lock()
    interval = 60 / requests_per_minute if requests_per_minute else 0
    last = [-interval]

    async def paced(*arguments, **options):
        for attempt in range(attempts):
            async with lock:
                wait = last[0] + interval - asyncio.get_running_loop().time()
                if wait > 0:
                    await asyncio.sleep(wait)
                last[0] = asyncio.get_running_loop().time()
            try:
                return await agenerate(*arguments, **options)
            except Exception as error:
                if not _rate_limited(error) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(60)

    llm.agenerate = paced
    return llm


async def judge(question, reference, contexts, precision, recall):
    """Context precision and recall of `contexts` for one question, or the error that stopped it."""
    try:
        scores = {
            "context_precision": (await precision.ascore(
                user_input=question, reference=reference, retrieved_contexts=contexts)).value,
            "context_recall": (await recall.ascore(
                user_input=question, reference=reference, retrieved_contexts=contexts)).value,
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:500]}
    # RAGAS adds 1e-10 to the precision's denominator, and gives NaN recall
    # for an answer it split into no statements.
    return {name: None if value is None or math.isnan(value) else round(value, 6)
            for name, value in scores.items()}


def summarize(records, k):
    """Mean of each metric over the questions judged, with the counts behind it."""
    summary = {"items": len(records),
               "failed": sum("error" in record for record in records)}
    for name in METRICS:
        values = [record[name] for record in records if record.get(name) is not None]
        summary[f"{name}@{k}"] = sum(values) / len(values) if values else None
        summary[f"{name}_items"] = len(values)
    return summary


def evaluate_ragas(dataset, index, setup, judge_model, k=5, max_items=200, stage=None,
                   requests_per_minute=None, workers=8, max_output_tokens=2048):
    """Judge one evaluated set-up's rankings with RAGAS; return the summary.

    `setup` is the directory `evaluate.py` wrote under
    `results/<dataset>/<index>/`. The first `k` chunks of each question's
    `stage` ranking -- "reranked" if the set-up has a reranker, "retrieved"
    otherwise, by default -- are judged by `judge_model`, `workers` questions
    at a time, their calls kept to `requests_per_minute` (see `pace`).
    `max_items` questions are judged, the first of the sample order that have
    evidence; None, all of them. `max_output_tokens` leaves a reasoning judge
    (gpt-oss-120b) room to think before it answers.
    """
    results_directory = ROOT / "results" / dataset / index / setup
    records = _read_jsonl(results_directory / "items.jsonl")
    if not records:
        raise FileNotFoundError(f"{results_directory} holds no rankings: run evaluate.py first")
    stage = stage or ("reranked" if "reranked" in records[0] else "retrieved")
    if stage not in records[0]:
        raise ValueError(f"{results_directory} has no '{stage}' ranking")

    config = {"judge_model": judge_model, "k": k, "stage": stage,
              "metrics": ["ContextPrecision", "ContextRecall"],
              "context": "build_text_for_llm",
              "evaluation": _read_json(results_directory / "config.json")}
    directory = judge_directory(results_directory, judge_model, k)
    directory.mkdir(exist_ok=True)
    done = {record["id"] for record in _read_jsonl(directory / "items.jsonl")}
    if done and _read_json(directory / "config.json") != config:
        raise ValueError(f"{directory} was judged with a different configuration, or the "
                         f"evaluation under it has changed; delete it to judge again")
    _write_json(directory / "config.json", config)

    # The order evaluate.py samples in, by a hash of the id, so that a larger
    # --max-items extends a smaller one.
    answers = {item["id"]: item["answer"]
               for item in _read_jsonl(ROOT / "datasets" / dataset / "data" / "items.jsonl")}
    records = sorted((record for record in records if record["spans"] and answers.get(record["id"])),
                     key=lambda record: hashlib.sha1(record["id"].encode()).hexdigest())
    records = records[:max_items]
    todo = [record for record in records if record["id"] not in done]
    print(f"{dataset}/{index}/{setup}: {len(records)} questions to judge ({len(done)} done) "
          f"by {judge_model}, top {k} of '{stage}'")

    chunks = {(chunk.doc_id, chunk.index): chunk for chunk in load_chunks(ROOT / "indexes" / dataset / index)}

    def context(entry):
        chunk = chunks[(entry[0], entry[3])]
        return build_text_for_llm(chunk.original_text, chunk.metadata)

    async def run():
        # No retries in the client: its own, a second apart, would spend the
        # judge's requests a minute on refusals. `pace` retries instead.
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""),
                             base_url=os.getenv("OPENAI_API_BASE"), max_retries=0)
        llm = pace(llm_factory(judge_model, client=client, temperature=0.0,
                               max_tokens=max_output_tokens), requests_per_minute)
        precision, recall = ContextPrecision(llm=llm), ContextRecall(llm=llm)
        semaphore = asyncio.Semaphore(workers)

        async def one(record):
            async with semaphore:
                contexts = [context(entry) for entry in record[stage][:k]]
                return {"id": record["id"], **await judge(record["question"], answers[record["id"]],
                                                         contexts, precision, recall)}

        # A batch is written when its last question is judged, so a stopped
        # run loses at most one batch.
        batch_size = workers * 4
        for start in tqdm(range(0, len(todo), batch_size), desc="judging", unit="batch"):
            batch = todo[start:start + batch_size]
            _append_jsonl(directory / "items.jsonl", await asyncio.gather(*map(one, batch)))

    if todo:
        asyncio.run(run())
    selected = {record["id"] for record in records}
    judged = [record for record in _read_jsonl(directory / "items.jsonl")
              if record["id"] in selected]
    summary = {"dataset": dataset, "index": index, "setup": setup, **config,
               **summarize(judged, k)}
    del summary["evaluation"]
    _write_json(directory / "summary.json", summary)
    if summary["failed"]:
        print(f"{summary['failed']} questions could not be judged; their errors are in "
              f"{directory / 'items.jsonl'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="a subdirectory of results/")
    parser.add_argument("index", help="a subdirectory of results/<dataset>/")
    parser.add_argument("setup", help="the evaluated set-up: a subdirectory of "
                                      "results/<dataset>/<index>/, as dense-bge-m3-top_k=50")
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"),
                        help="the LLM that judges; defaults to JUDGE_MODEL")
    parser.add_argument("--k", type=int, default=5, help="chunks judged per question")
    parser.add_argument("--max-items", type=int, default=200,
                        help="questions judged, from evaluate.py's sample order; 0 for all")
    parser.add_argument("--stage", choices=("retrieved", "reranked"),
                        help="the ranking judged; the reranked one if there is one by default")
    parser.add_argument("--requests-per-minute", type=float,
                        help="the judge's request limit at the endpoint (10 for gpt-oss-120b); "
                             "calls are spaced to keep to it. Unset, they are not spaced")
    parser.add_argument("--workers", type=int, default=8, help="questions judged at once")
    arguments = parser.parse_args()
    if not arguments.judge_model:
        parser.error("name the judge: --judge-model or JUDGE_MODEL")
    summary = evaluate_ragas(arguments.dataset, arguments.index, arguments.setup,
                             arguments.judge_model, arguments.k, arguments.max_items or None,
                             arguments.stage, arguments.requests_per_minute,
                             arguments.workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
