"""The LLM client, and the one call every generated method makes.

Shared by `chunking.lumberchunker` and `augmentation`: the plain OpenAI client
against any OpenAI-compatible endpoint, with the same handling of rate limits
and empty answers wherever a model is asked for something.

No model is named here or anywhere else in the library: every function that
calls one takes `model`, the name the endpoint serves it under. Which models a
run uses is the experimenter's choice, and a default would make a result depend
on a name buried in the code, and break on any endpoint that does not serve it.
Prefer a model that is not a reasoning one: the methods give small output
budgets, which a reasoning model spends before it answers.

Every client built here also counts what its calls cost, so the methods can be
compared by cost as well as by score. The endpoint reports the tokens of each
chat, embedding and rerank answer in its `usage`, and a hook on the HTTP
client adds them up as the answers arrive, which reaches calls made inside
other libraries too -- the semantic chunkers embed through LlamaIndex's client,
built with `count_usage` as its hook. `usage()` reads the totals and
`usage_since` what a stretch of work added to them. Tokens are the cost to
compare for the LLM methods: their time is mostly spent under the endpoint's
128,000 input tokens a minute, and the minutes waited out after a refusal are
counted apart, as `rate_limit_seconds`.
"""

import os
import threading
import time
from collections import Counter
from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI, RateLimitError

load_dotenv()
# The routes whose answers report tokens, and the name each is counted under.
ROUTES = {"/chat/completions": "llm", "/embeddings": "embedding", "/rerank": "rerank"}

# Totals for the whole process, and for each thread: the reranker scores many
# questions at once, one per thread, and each question's share is its thread's.
_lock = threading.Lock()
_totals = Counter()
_local = threading.local()


def _count(**amounts):
    with _lock:
        _totals.update(amounts)
    if not hasattr(_local, "totals"):
        _local.totals = Counter()
    _local.totals.update(amounts)


def count_usage(response):
    """Add the tokens an answer of the endpoint reports to the totals.

    An httpx response hook: `get_client` installs it, and a client built
    elsewhere can be given it as `event_hooks={"response": [count_usage]}`.
    LLM calls count `llm_input_tokens` and `llm_output_tokens`, embeddings
    `embedding_tokens` and reranking `rerank_tokens`, each with its
    `..._requests`.
    """
    kind = next((name for route, name in ROUTES.items()
                 if response.request.url.path.endswith(route)), None)
    if kind is None or response.status_code != 200:
        return
    response.read()
    usage = response.json().get("usage") or {}
    if kind == "llm":
        _count(llm_input_tokens=usage.get("prompt_tokens", 0),
               llm_output_tokens=usage.get("completion_tokens", 0), llm_requests=1)
    else:
        _count(**{f"{kind}_tokens": usage.get("prompt_tokens", 0), f"{kind}_requests": 1})


def usage(thread=False):
    """The tokens, requests and rate-limit waits counted so far, as a Counter.

    For the whole process, or with `thread`, only the calling thread's.
    """
    if thread:
        return Counter(getattr(_local, "totals", Counter()))
    with _lock:
        return Counter(_totals)


def usage_since(before, thread=False):
    """What was counted since `before`, a `usage()` taken earlier, without the zeros."""
    after = usage(thread)
    after.subtract(before)
    return {name: round(value, 3) if isinstance(value, float) else value
            for name, value in sorted(after.items()) if value}


def wait_for_rate_limit(seconds=60):
    """Wait out a rate-limit refusal, and count the wait as `rate_limit_seconds`."""
    time.sleep(seconds)
    _count(rate_limit_seconds=float(seconds))


def get_client(api_key=None, api_base=None, proxy=None, timeout=60.0, **kwargs):
    """Build an OpenAI client for any OpenAI-compatible endpoint.

    `api_key` and `api_base` default to OPENAI_API_KEY and OPENAI_API_BASE.
    `proxy` is an http(s) proxy URL, as in `chunking.semantic.openai_embedding`.
    """
    options = {"proxy": proxy, "timeout": timeout} if proxy else {}
    http_client = DefaultHttpxClient(event_hooks={"response": [count_usage]}, **options)
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=api_base or os.getenv("OPENAI_API_BASE"),
        http_client=http_client,
        **kwargs,
    )


def call_llm(prompt, model, client=None, max_output_tokens=512, attempts=5,
             system=None, temperature=0.0, response_format=None):
    """Send `prompt` as a single user message and return the model's answer.

    `client` defaults to `get_client()`. `system` is an optional system message
    sent ahead of the prompt, and `response_format` is passed to the endpoint
    as it is, as in `{"type": "json_object"}`.

    A rate-limited call waits a minute and is tried again, up to `attempts`
    times. The endpoint used here counts input tokens per minute, so a minute's
    wait is what clears its limit; one with a longer window needs more
    `attempts`. Anything else raises, and so does an empty answer:
    an experiment that quietly mixes augmented and bare chunks measures nothing.
    """
    client = client or get_client()
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    options = {"response_format": response_format} if response_format else {}
    for attempt in range(attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_output_tokens,
                messages=messages,
                **options,
            )
            break
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            wait_for_rate_limit()
    answer = (completion.choices[0].message.content or "").strip() if completion.choices else ""
    if not answer:
        raise ValueError(f"{model} returned an empty answer")
    return answer
