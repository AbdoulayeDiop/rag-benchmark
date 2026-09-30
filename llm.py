"""The LLM client, and the one call every generated method makes.

Shared by `chunking.lumberchunker` and `augmentation`: the plain OpenAI client
against any OpenAI-compatible endpoint, with the same handling of rate limits
and empty answers wherever a model is asked for something.
"""

import os
import time

import httpx
from openai import OpenAI, RateLimitError

# What the generated methods call unless told otherwise: served by the
# configured endpoint, 128,000 tokens of context, and not a reasoning model, so
# the output budget is spent on the answer.
DEFAULT_MODEL = "openweight-large"


def get_client(api_key=None, api_base=None, proxy=None, timeout=60.0, **kwargs):
    """Build an OpenAI client for any OpenAI-compatible endpoint.

    `api_key` and `api_base` default to OPENAI_API_KEY and OPENAI_API_BASE.
    `proxy` is an http(s) proxy URL, as in `chunking.semantic.openai_embedding`.
    """
    http_client = httpx.Client(proxy=proxy, timeout=timeout) if proxy else None
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=api_base or os.getenv("OPENAI_API_BASE"),
        http_client=http_client,
        **kwargs,
    )


def call_llm(prompt, model=DEFAULT_MODEL, client=None, max_output_tokens=512, attempts=5,
             system=None, temperature=0.0, response_format=None):
    """Send `prompt` as a single user message and return the model's answer.

    `client` defaults to `get_client()`. `system` is an optional system message
    sent ahead of the prompt, and `response_format` is passed to the endpoint
    as it is, as in `{"type": "json_object"}`.

    A rate-limited call waits a minute and is tried again, up to `attempts`
    times -- the configured endpoint counts input tokens per minute, so waiting
    is all there is to do. Anything else raises, and so does an empty answer:
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
            time.sleep(60)
    answer = (completion.choices[0].message.content or "").strip() if completion.choices else ""
    if not answer:
        raise ValueError(f"{model} returned an empty answer")
    return answer
