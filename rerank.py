"""Re-ranking: the retrieved chunks reordered by a cross-encoder.

The step after retrieval. A cross-encoder reads the query and one chunk
together and scores the pair, which is too slow to run over a corpus but
sharper than either retriever on the few dozen chunks they return. The
experiments here use bge-reranker-v2-m3, bge-m3's companion reranker.

The model is reached through the endpoint's `/rerank` route (Cohere's and
Jina's shape: a query, a list of documents, a relevance score per document),
which the OpenAI client has no method for; its generic `post` is used, so the
key, base URL and rate-limit errors are those of every other call here. What a
chunk is scored as is its `get_text_to_embed` -- the text it was embedded and
BM25-indexed as -- so an augmentation reaches the reranker too, and an HTML
chunk arrives without its tags.

What follows was measured on the endpoint these experiments ran against.
Scores are sigmoid outputs in [0, 1], one per document, independent of the
other documents in the request, so a long list can be scored in several
requests. A request of more than 64 documents is refused with 413, and so is
a pair -- query and document, special tokens included -- of more than 8,192
tokens: the endpoint does not truncate. The 40,000-character chunks of a
GutenQA run come to 9,700-10,800 tokens (five books measured), so a chunk is cut to fit, counted with
bge-m3's tokenizer, which the reranker shares. That count is exact: a pair cut
to 8,192 by `fit_documents` is accepted, and one cut to 8,193 refused.
"""

import time

import httpx
from openai import RateLimitError

from chunking import get_text_to_embed
from embedding import truncate_to_tokens
from llm import get_client

# Documents per request: the most the endpoint used here accepts.
BATCH_SIZE = 64

# Special tokens an XLM-RoBERTa pair adds around its two texts:
# <s> query </s></s> document </s>.
PAIR_SPECIAL_TOKENS = 4


def fit_documents(query, documents, tokenizer, max_tokens):
    """Cut each document so that it and `query` make a pair under `max_tokens`.

    `tokenizer` is a Hugging Face tokenizer (`embedding.load_tokenizer`).
    Returns the documents, cut where needed, and the positions of those that
    were cut.
    """
    query_tokens = len(tokenizer.encode(query, add_special_tokens=False).ids)
    # `truncate_to_tokens` counts a document's own two special tokens; the
    # pair has two more.
    budget = max_tokens - query_tokens - (PAIR_SPECIAL_TOKENS - 2)
    fitted, cut = [], []
    for position, document in enumerate(documents):
        document, was_cut = truncate_to_tokens(document, tokenizer, budget)
        fitted.append(document)
        if was_cut:
            cut.append(position)
    return fitted, cut


def _request(query, documents, model, client, attempts):
    """Score `documents` against `query` in one request; scores in document order."""
    for attempt in range(attempts):
        try:
            response = client.post("/rerank", cast_to=httpx.Response,
                                   body={"model": model, "query": query, "documents": documents})
            break
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(60)
    scores = [None] * len(documents)
    for result in response.json()["results"]:
        scores[result["index"]] = result["relevance_score"]
    if None in scores:
        raise ValueError(f"{model} returned {sum(s is not None for s in scores)} scores "
                         f"for {len(documents)} documents")
    return scores


def rerank(query, chunks, model, client=None, tokenizer=None, max_tokens=None, attempts=5):
    """Reorder `chunks`, `(Chunk, score)` pairs or chunks, by `model`'s relevance to `query`.

    Returns `(Chunk, relevance)` pairs, most relevant first. With
    `max_tokens`, a chunk whose pair with the query would reach it is cut to
    fit first, counted by `tokenizer` (a Hugging Face tokenizer, required
    then); without it, such a chunk makes the request fail. Rate limits are
    waited out as in `call_llm`.
    """
    client = client or get_client()
    chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
    if not chunks:
        return []
    documents = [get_text_to_embed(chunk) for chunk in chunks]
    if max_tokens is not None:
        if tokenizer is None:
            raise ValueError("cutting a chunk to the reranker's limit needs its tokenizer")
        documents, _ = fit_documents(query, documents, tokenizer, max_tokens)
    scores = []
    for start in range(0, len(documents), BATCH_SIZE):
        scores += _request(query, documents[start:start + BATCH_SIZE], model, client, attempts)
    # sorted() is stable: equal scores keep the retriever's order.
    return sorted(zip(chunks, scores), key=lambda pair: -pair[1])
