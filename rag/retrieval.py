"""Retrieval: the chunks of an index that answer a query, best first.

Three methods over the two stores of `ingestion` (`open_vector_store`,
`load_bm25`):

    dense     the query embedded by the run's embedding model, searched in its
              Chroma collection by cosine
    sparse    the query searched in its BM25 index
    hybrid    both lists merged by reciprocal rank fusion

All three return `(Chunk, score)` pairs, the chunk rebuilt by `node_to_chunk`
from the node either store hands back, so a result carries the span it is
scored on.

Queries are embedded with `embed_batch`, as the chunks were, not through a
LlamaIndex embedding class -- the one `QueryFusionRetriever` and
`VectorIndexRetriever` expect -- which would send the query with its newlines
replaced and so not as the chunks were sent. The Chroma store is queried with
the vector directly, and the fusion is written out here: with one query and no
generated rewrites, `QueryFusionRetriever` in "reciprocal_rerank" mode is the
same few lines behind an LLM-shaped interface.

Reciprocal rank fusion (Cormack et al., 2009) scores a chunk by the sum of
1 / (k + rank) over the lists it appears in. It uses ranks only, so BM25's
unbounded scores and cosine similarities need no normalising against each
other, and k = 60 is the value of the paper, kept by every implementation
since; it flattens the difference between the first few ranks so that a chunk
both retrievers place well beats one only a single retriever places first.
"""

from llama_index.core.vector_stores.types import VectorStoreQuery

from .embedding import BATCH_SIZE, embed_batch
from .ingestion import node_to_chunk
from .llm import get_client

METHODS = ("dense", "sparse", "hybrid")

# The k of reciprocal rank fusion; see the module docstring.
RRF_K = 60


def embed_queries(queries, model, client=None):
    """Embed `queries` with `model`, `BATCH_SIZE` to a request; return the vectors in order.

    Questions are a sentence or two, far below any model's input limit, so
    nothing is cut.
    """
    client = client or get_client()
    vectors = []
    for start in range(0, len(queries), BATCH_SIZE):
        vectors += embed_batch(queries[start:start + BATCH_SIZE], model, client)[0]
    return vectors


def dense(store, vector, top_k=50):
    """The `top_k` chunks nearest `vector` in a vector store, with their cosine similarity."""
    result = store.query(VectorStoreQuery(query_embedding=vector, similarity_top_k=top_k))
    return [(node_to_chunk(node), similarity)
            for node, similarity in zip(result.nodes, result.similarities)]


def sparse(bm25, query):
    """The chunks BM25 ranks highest for `query`, as many as the retriever's `similarity_top_k`."""
    return [(node_to_chunk(result), result.score) for result in bm25.retrieve(query)]


def reciprocal_rank_fusion(rankings, top_k=50, k=RRF_K):
    """Merge ranked lists of `(Chunk, score)` into one, by reciprocal rank fusion.

    A chunk's new score is the sum of 1 / (k + rank) over the lists that hold
    it, rank counted from 1. Ties keep the order the chunk was first met in,
    which puts the first list's chunks ahead. Returns the best `top_k`.
    """
    fused = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            key = (chunk.doc_id, chunk.index)
            previous = fused.get(key, (chunk, 0.0))
            fused[key] = (previous[0], previous[1] + 1 / (k + rank))
    return sorted(fused.values(), key=lambda pair: -pair[1])[:top_k]


def retrieve(query, store, bm25, method="hybrid", top_k=50, embedding_model=None, vector=None,
             client=None):
    """Retrieve the best `top_k` chunks for `query` with `method`.

    `store` is a vector store (`ingestion.open_vector_store`) and `bm25` a
    BM25Retriever (`ingestion.load_bm25`), whichever `method` searches.
    Dense and hybrid embed the query with `embedding_model`, the model the
    index was embedded with, through `client` (default `get_client()`). `vector` is the query's
    embedding already made: many queries are embedded far faster together
    (`embed_queries`, 64 a request) than one request each, so a caller with
    many passes them in and `embedding_model` is not needed. Hybrid takes
    `top_k` from each retriever before fusing, so that it sees as deep into
    each list as either does alone.
    """
    if method != "sparse" and vector is None:
        if not embedding_model:
            raise ValueError(f"{method} retrieval embeds the query: name the embedding model")
        vector = embed_queries([query], embedding_model, client)[0]
    if method == "dense":
        return dense(store, vector, top_k)
    if method == "sparse":
        return sparse(bm25, query)[:top_k]
    return reciprocal_rank_fusion([dense(store, vector, top_k), sparse(bm25, query)], top_k)
