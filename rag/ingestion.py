"""Ingestion: chunks turned into the two stores retrieval searches.

Dense and sparse retrieval read the two stores through LlamaIndex:
`ChromaVectorStore` over a persistent Chroma collection, and `BM25Retriever`
(bm25s underneath) saved with `persist`. Both hold the same nodes -- one per
chunk, with `get_text_to_embed` as its text -- so the two are compared on
equal input. A node's metadata is the whole chunk -- span, `original_text` and
everything the augmentations generated -- so a retrieved node is all retrieval
and evaluation need, and `node_to_chunk` turns it back into a `Chunk`. A node's
id is `<doc_id>:<chunk index>`.

Chroma's index is HNSW, so dense search is approximate: a chunking method's
score includes the index's own recall. At the sizes here (GutenQA, the largest
corpus, is some 45,000 chunks at 256 tokens) that is small, but it is not zero.

The LlamaIndex stores are used for storage only. Vectors still come from
`embed_batch`, which cuts a text over the model's limit to fit before sending
it, and are handed to the store on the nodes. Vectors are added a batch at a
time, in chunk order, so a collection's size is where an interrupted pass
continues from.

Nothing here knows where a dataset or a run lives: every function takes the
directory it reads or writes. `experiments/ingest.py` decides the layout.
"""

import json
import re
import time
from pathlib import Path

import chromadb
import Stemmer
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from .chunking import Chunk, build_text_to_embed, get_text_to_embed
from .embedding import BATCH_SIZE, embed_batch, load_tokenizer
from .llm import get_client

# Node metadata keys reserved for the chunk's fields and the stores; chunk
# metadata may not use them, and `node_to_chunk` leaves them out of it.
# `truncated` is written by `embed_nodes` on every node it adds to Chroma --
# True where `embed_batch` had to cut the text to fit the model -- and BM25
# nodes never carry it. `json_encoded_keys` names the values stored as JSON
# (see `to_nodes`).
RESERVED_METADATA_KEYS = ("doc_id", "chunk_index", "start", "end", "original_text", "truncated",
                          "json_encoded_keys")

# Snowball stemmer and bm25s stop-word list for a pysbd language code. Polish
# has neither -- PyStemmer ships no Polish algorithm and bm25s no Polish list --
# so PoQuAD is indexed on whole words, with nothing removed.
BM25_LANGUAGES = {"en": "english"}


def to_nodes(chunks):
    """Turn chunks into the LlamaIndex nodes both stores hold.

    A node's text is what the chunk is embedded as, and its metadata is the
    rest of the chunk: the span, `original_text`, and every key of the chunk's
    own metadata -- heading path, title, summary, context, keywords, questions.
    A retrieved node is therefore enough on its own, and `node_to_chunk`
    rebuilds the `Chunk`.

    Chroma accepts only flat values (strings, numbers, booleans), so a list or
    a dict -- the generated questions -- is stored as a JSON string, and
    `json_encoded_keys` names which ones, so that reading them back is exact
    rather than a guess at which strings happen to look like JSON.

    All of it is excluded from the text LlamaIndex shows an encoder:
    BM25Retriever indexes that text, and would otherwise match a query on
    "start: 1935" or count a chunk's words twice.
    """
    nodes = []
    for chunk in chunks:
        reserved = [key for key in chunk.metadata if key in RESERVED_METADATA_KEYS]
        if reserved:
            raise ValueError(f"chunk metadata uses {reserved}, which name node fields")
        metadata = {"doc_id": chunk.doc_id, "chunk_index": chunk.index,
                    "start": chunk.start, "end": chunk.end,
                    "original_text": chunk.original_text}
        encoded = []
        for key, value in chunk.metadata.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
                encoded.append(key)
            metadata[key] = value
        if encoded:
            metadata["json_encoded_keys"] = ",".join(encoded)
        excluded = list(metadata) + ["truncated"]
        nodes.append(TextNode(id_=f"{chunk.doc_id}:{chunk.index}",
                              text=get_text_to_embed(chunk), metadata=metadata,
                              excluded_embed_metadata_keys=excluded,
                              excluded_llm_metadata_keys=excluded))
    return nodes


def node_to_chunk(node):
    """Rebuild the `Chunk` a node returned by either store was made from.

    Takes a node or the `NodeWithScore` a retriever returns.

    `text_to_embed` is the node's text, so it is not stored twice; it is
    rebuilt from `original_text` and the metadata, as `augment_chunk` and
    `to_chunks` built it. A chunk embedded as its span gets None.
    """
    node = getattr(node, "node", node)
    metadata = dict(node.metadata)
    for key in filter(None, metadata.pop("json_encoded_keys", "").split(",")):
        metadata[key] = json.loads(metadata[key])
    original_text = metadata["original_text"]
    chunk_metadata = {key: value for key, value in metadata.items()
                      if key not in RESERVED_METADATA_KEYS}
    text_to_embed = build_text_to_embed(original_text, chunk_metadata)
    # The node's text is the same thing; rebuilding it rather than reading it
    # is the check that the metadata is complete.
    assert text_to_embed == node.text, f"node {node.node_id} does not rebuild its own text"
    return Chunk(doc_id=metadata["doc_id"], index=metadata["chunk_index"],
                 start=metadata["start"], end=metadata["end"], original_text=original_text,
                 metadata=chunk_metadata,
                 text_to_embed=None if text_to_embed == original_text else text_to_embed)


def open_vector_store(directory, model):
    """Open the Chroma collection for `model` in `directory` as a LlamaIndex vector store.

    The database and the collection are created if missing. It compares by
    cosine; Chroma's default is squared L2.
    """
    client = chromadb.PersistentClient(path=str(directory))
    # A collection name is letters, digits, dots, dashes and underscores.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    collection = client.get_or_create_collection(
        name, configuration={"hnsw": {"space": "cosine"}})
    return ChromaVectorStore(chroma_collection=collection)


def embed_nodes(store, nodes, model, client=None, batch_size=BATCH_SIZE, tokenizer=None,
                max_tokens=None, on_batch=None):
    """Embed `nodes` with `model` into `store`, after the ones it already holds.

    Nodes are only ever appended, in order, so the collection's size is the
    number done: a pass interrupted, or one over fewer chunks, is continued.

    `max_tokens` is the model's input limit; a text over it is cut to fit
    before it is sent, counted by `tokenizer` -- a Hugging Face repository
    name, or None for tiktoken (see `embedding.load_tokenizer`). Without
    `max_tokens` nothing is cut, and a text over the limit stops the pass. The
    node keeps its whole text either way. `on_batch(count)` is called once each
    batch of `count` nodes is stored, which is where a caller records what it
    cost. Returns the ids of the nodes whose text was cut, which carry
    `truncated: True` in their metadata.
    """
    collection = store._collection
    done = collection.count()
    # A collection written before nodes carried the whole chunk would be
    # extended with nodes unlike its own.
    if done and "original_text" not in collection.get(limit=1, include=["metadatas"])["metadatas"][0]:
        raise ValueError(f"the {model} collection predates full-chunk node metadata; "
                         f"delete it and embed again")
    if done > len(nodes):
        raise ValueError(f"the {model} collection holds {done} nodes but there are only "
                         f"{len(nodes)} chunks; delete it")
    if done < len(nodes):
        client = client or get_client()
        if max_tokens is not None:
            tokenizer = load_tokenizer(tokenizer)
    for start in tqdm(range(done, len(nodes), batch_size), desc="embedding", unit="batch"):
        batch = nodes[start:start + batch_size]
        vectors, cut = embed_batch([node.text for node in batch], model, client, tokenizer,
                                   max_tokens)
        for position, (node, vector) in enumerate(zip(batch, vectors)):
            node.embedding = vector
            node.metadata["truncated"] = position in cut
        store.add(batch)
        if on_batch:
            on_batch(len(batch))
    return collection.get(where={"truncated": True}, include=[])["ids"]


def build_bm25(nodes, directory, language):
    """Index `nodes` for BM25 and persist the retriever in `directory`.

    Built from all the nodes at once, since BM25's term weights depend on the
    whole corpus; it calls no model, so it is simply redone when the corpus
    grows. `language` is a pysbd code: English is stemmed and loses its stop
    words, and a language in no `BM25_LANGUAGES` entry is indexed as whole words.
    """
    if not nodes:
        return
    stemmer_language = BM25_LANGUAGES.get(language)
    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        # Passed to bm25s as its stop words: a language name, or a list.
        language=stemmer_language or [],
        stemmer=Stemmer.Stemmer(stemmer_language) if stemmer_language else None,
        skip_stemming=stemmer_language is None,
    )
    retriever.persist(str(directory))


def load_bm25(directory, language, top_k=10):
    """Load a BM25Retriever persisted in `directory`, returning `top_k` nodes a query.

    `persist` saves the index but not how it was tokenized, and a loaded
    retriever stems every query as English. The stemming is therefore set
    again from `language`, the pysbd code the index was built with, so a query
    is cut into the same terms its index holds. The query's English stop words
    are still removed whatever the language; the retriever offers no way to say
    otherwise.
    """
    retriever = BM25Retriever.from_persist_dir(str(Path(directory)))
    retriever.similarity_top_k = top_k
    stemmer_language = BM25_LANGUAGES.get(language)
    retriever.skip_stemming = stemmer_language is None
    if stemmer_language:
        retriever.stemmer = Stemmer.Stemmer(stemmer_language)
    return retriever
