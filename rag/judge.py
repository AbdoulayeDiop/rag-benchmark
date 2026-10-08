"""RAGAS context precision and context recall: retrieved text judged by an LLM.

The span metrics of `evaluation` call no model but trust the spans: where they
are loose (Qasper's merged annotators, TriviaQA's whole documents, GutenQA's
gold chunks) or where another chunk says the same thing as the marked one,
they can be wrong. RAGAS's two retrieval metrics judge the retrieved text
itself, against the question's reference answer, with an LLM:

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
RAGAS's own and comparable with published ones.

They cost what the span metrics do not. One question takes k + 1 judge calls;
on SQuAD with 256-token chunks and k = 5, a question took 10.8 s for precision
and 3.3 s for recall with gpt-oss-120b (10.3 s and 2.7 s with
mistral-small-3-2-24b-instruct-2506), each call some 1,000-1,500 input tokens
with the prompt's examples. Under the endpoint's 128,000 input tokens a minute,
that is on the order of ten questions a minute.

The judge is lenient. In that trial, "What does the original manuscript show?"
(SQuAD, about Luther's hymn; reference "multiple revisions") was given
precision and recall of 1.0 by both judges for five chunks of other articles,
the first about the manuscript of Harper Lee's *Go Set a Watchman*, none of
which mentions a revision; the span metrics scored the same ranking 0. SQuAD's
questions take their passage for granted, and a judge shown only the question
and a short answer accepts a chunk on the same topic. Read the two kinds of
metric together, not one in place of the other.
"""

import asyncio
import math
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from ragas.llms.base import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall

load_dotenv(override=True)

METRICS = ("context_precision", "context_recall")


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


def judge_metrics(model, requests_per_minute=None, max_output_tokens=2048):
    """The RAGAS `(ContextPrecision, ContextRecall)` pair judged by `model`.

    Call it inside the event loop that will run them. `max_output_tokens`
    leaves a reasoning judge (gpt-oss-120b) room to think before it answers.
    """
    # No retries in the client: its own, a second apart, would spend the
    # judge's requests a minute on refusals. `pace` retries instead.
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""),
                         base_url=os.getenv("OPENAI_API_BASE"), max_retries=0)
    llm = pace(llm_factory(model, client=client, temperature=0.0, max_tokens=max_output_tokens),
               requests_per_minute)
    return ContextPrecision(llm=llm), ContextRecall(llm=llm)


async def judge(question, reference, contexts, precision, recall):
    """Context precision and recall of `contexts` for one question, or the error that stopped it.

    A question whose judging fails (an answer that does not parse after
    instructor's own retries) returns `{"error": ...}` rather than raising, so
    hours of judging are not stopped for one question.
    """
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
