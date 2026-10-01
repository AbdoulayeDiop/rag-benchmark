"""Embedding: what a chunk is turned into a vector from, and the call that does it.

The step between augmentation and retrieval. What it embeds is a chunk's
`get_text_to_embed`: the span, or the augmented text, with any HTML tags
already stripped (see `chunking.base.build_text_to_embed`). Sparse retrieval
indexes the same text, so BM25 and the vectors are compared on equal input.

The plain OpenAI client is used, as in `augmentation`, rather than the
LlamaIndex embedding class the semantic chunkers take. That class sends every
text with its newlines replaced by spaces, so a vector would not be of the
node's own text; it retries a rate limit for one minute at most, against an
endpoint that counts tokens per minute; and it neither splits nor truncates a
request the endpoint refuses.

A text over the model's input limit is cut before it is sent, counted in
tokens. The size-bounded chunkers never produce one, but `semantic`, `tiled`
and `lumberchunker` have no upper bound, and a generated context in front of a
chunk adds to its length. Counting is exact only with the model's own
tokenizer (`load_tokenizer("BAAI/bge-m3")` for bge-m3); without one, tiktoken's
cl100k_base -- the tokenizer the chunkers budget in -- stands in, and a text it
lets through can still be over the model's limit. Measured on a GutenQA book,
8,192 tiktoken tokens came to 9,161-9,372 bge-m3 tokens, so with tiktoken the
limit needs a margin of some 15%.

What follows was measured on the endpoint these experiments ran against, with
bge-m3; another endpoint or model may behave differently. Vectors have 1,024
dimensions and come back unit-length. An input over 8,192 of the model's own
tokens, special tokens included, is refused with 413 rather than truncated --
bge-m3's tokenizer, loaded locally, counts exactly as the server does, and a
text cut to 8,192 is accepted. A request of more than 64 texts is refused with
413 too, and an empty input with 400.
"""

import time

import tiktoken
from openai import APIStatusError, RateLimitError
from tokenizers import Tokenizer

from llm import get_client

# Texts sent per request: the most the endpoint used here accepts. Many accept
# more; pass `batch_size` to `ingest.embed_corpus` to send more at once.
BATCH_SIZE = 64


def load_tokenizer(name=None):
    """Load the tokenizer an embedding model's input limit is counted in.

    `name` is a Hugging Face repository, as in "BAAI/bge-m3", whose tokenizer
    is downloaded once and cached. None gives tiktoken's cl100k_base, the
    tokenizer the chunkers size by, which only approximates any other model's.
    """
    return tiktoken.get_encoding("cl100k_base") if name is None else Tokenizer.from_pretrained(name)


def truncate_to_tokens(text, tokenizer, max_tokens):
    """Cut `text` to at most `max_tokens` tokens; return it and whether it was cut.

    The cut falls on a token boundary and keeps the text's beginning. With a
    Hugging Face tokenizer, the special tokens the model adds (<s> and </s>
    for bge-m3) count towards `max_tokens`, as they do on the server.
    """
    if isinstance(tokenizer, tiktoken.Encoding):
        tokens = tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text, False
        # A token can end inside a multi-byte character; the partial character
        # is dropped, so what is left is still a prefix of `text`.
        return tokenizer.decode_bytes(tokens[:max_tokens]).decode("utf-8", errors="ignore"), True

    encoding = tokenizer.encode(text)
    if len(encoding.ids) <= max_tokens:
        return text, False
    content = [offsets for offsets, special in zip(encoding.offsets, encoding.special_tokens_mask)
               if not special]
    budget = max_tokens - (len(encoding.ids) - len(content))
    return text[:content[budget - 1][1]], True


def _request(texts, model, client, attempts):
    """One embedding request, waiting out rate limits as `call_llm` does."""
    for attempt in range(attempts):
        try:
            response = client.embeddings.create(model=model, input=texts)
            break
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(60)
        except APIStatusError as error:
            if error.status_code == 413:
                raise ValueError(
                    f"{model} refused a request as too large: more texts than the endpoint "
                    f"accepts in one request, or a text over the model's limit -- set "
                    f"--embedding-max-tokens, and --embedding-tokenizer for an exact count"
                ) from error
            raise
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def embed_batch(texts, model, client=None, tokenizer=None, max_tokens=None, attempts=5):
    """Embed `texts`, at most `BATCH_SIZE` of them, in one request, with `model`.

    With `max_tokens`, a text longer than that is cut to fit first, counted by
    `tokenizer` (see `load_tokenizer`; None means tiktoken). Returns the
    vectors in order, and the positions in `texts` of those that were cut: a
    truncated chunk is embedded as less than it covers, and its results should
    be read with that in mind.
    """
    client = client or get_client()
    truncated = []
    if max_tokens is not None:
        tokenizer = tokenizer or load_tokenizer()
        cut_texts = []
        for position, text in enumerate(texts):
            text, was_cut = truncate_to_tokens(text, tokenizer, max_tokens)
            cut_texts.append(text)
            if was_cut:
                truncated.append(position)
        texts = cut_texts
    return _request(texts, model, client, attempts), truncated
