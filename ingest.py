"""Ingestion: a dataset's documents chunked, augmented, embedded and written to disk.

One run is one point in the comparison -- a dataset, a rendering, a chunking
method with its parameters and a set of augmentations -- and produces an index
directory retrieval reads without calling a model again:

    indexes/<dataset>/<name>/config.json          what the run is
    indexes/<dataset>/<name>/chunks.jsonl         one chunk per line, in document order
    indexes/<dataset>/<name>/documents.jsonl      one line per finished document
    indexes/<dataset>/<name>/chroma/              Chroma database, one collection per embedding model
    indexes/<dataset>/<name>/bm25/                persisted BM25Retriever

Dense and sparse retrieval read the two stores through LlamaIndex:
`ChromaVectorStore` over a persistent Chroma collection, and `BM25Retriever`
(bm25s underneath) saved with `persist`. Both hold the same nodes -- one per
chunk, with `get_text_to_embed` as its text -- so the two are compared on
equal input. A node's metadata is the whole chunk -- span, `original_text` and
everything the augmentations generated -- so a retrieved node is all retrieval
and evaluation need, and `node_to_chunk` turns it back into a `Chunk`. A node's
id is `<doc_id>:<chunk index>`.

`chunks.jsonl` is not something retrieval reads. It is the build's checkpoint --
where a document's LLM work is kept the moment it is done, before anything is
embedded, and what the stores are filled from -- and the plain-text record of
a run, for checking an experiment without opening either store.

Chroma's index is HNSW, so dense search is approximate: a chunking method's
score includes the index's own recall. At the sizes here (GutenQA, the largest
corpus, is some 45,000 chunks at 256 tokens) that is small, but it is not zero.

The LlamaIndex stores are used for storage only. Vectors still come from
`embed_batch`, which cuts a text over the model's limit to fit before sending
it, and are handed to the store on the nodes.

Everything is resumable, because the expensive runs are long -- contextualising
GutenQA takes days at 128,000 input tokens a minute, the limit of the endpoint
used here (see the README's cost table) -- and `call_llm` raises rather than
skip a chunk. Chunks are written a document at a time and `documents.jsonl`
is appended to only once a
document's chunks are on disk, so a run that stopped anywhere picks up at the
first unfinished document and repeats no LLM call of a finished one. Vectors
are added to the collection a batch at a time, in chunk order, so its size is
where to continue from. The BM25 index calls no model and is rebuilt each run.

Chunks and vectors are separate stages so that the embedding model is its own
axis: running again with another `--embedding-model` adds a second collection
beside the first and leaves the chunks, and the LLM calls behind them, alone.
`config.json` holds what decides a chunk's content; a run directory is refused
if it was started with a different one.
"""

import argparse
import inspect
import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path

import chromadb
import Stemmer
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from augmentation import (add_context, add_keywords, add_questions, add_summary, add_title,
                          use_summary)
from chunking import (Chunk, build_text_to_embed, clustered, detect_language, fixed_char,
                      fixed_token, get_text_to_embed, html, markdown, openai_embedding,
                      semantic, sentence, tiled, update_chunk_metadata, verify)
from chunking.lumberchunker import lumberchunker
from embedding import BATCH_SIZE, embed_batch, load_tokenizer
from llm import get_client

ROOT = Path(__file__).parent

# Where each rendering's files live under datasets/<name>/data/.
RENDERINGS = {"txt": ("documents", ".txt"),
              "md": ("documents_md", ".md"),
              "html": ("documents_html", ".html")}

CHUNKERS = {"fixed_char": fixed_char, "fixed_token": fixed_token, "sentence": sentence,
            "markdown": markdown, "html": html, "semantic": semantic, "tiled": tiled,
            "clustered": clustered, "lumberchunker": lumberchunker}

# The rendering a method reads unless told otherwise. The others default to
# plain text, but any of them can be pointed at another rendering: `sentence`
# over Markdown is a fair baseline for `markdown`.
DEFAULT_RENDERING = {"markdown": "md", "html": "html"}

# In the order they are applied, which is cheapest first. The order does not
# change the result -- `text_to_embed` is rebuilt from the metadata each time.
AUGMENTERS = {
    "title": lambda chunks, document, model, client, workers:
        add_title(chunks, model, client),
    "summary": lambda chunks, document, model, client, workers:
        add_summary(chunks, document, model, client),
    "keywords": lambda chunks, document, model, client, workers:
        add_keywords(chunks, model, client, workers=workers),
    "questions": lambda chunks, document, model, client, workers:
        add_questions(chunks, model, client, workers=workers),
    "chunk_summary": lambda chunks, document, model, client, workers:
        use_summary(chunks, model, client, workers=workers),
    "context": lambda chunks, document, model, client, workers:
        add_context(chunks, document, model, client, workers=workers),
}

# Arguments of a chunking method that the run supplies rather than the user.
SUPPLIED = ("document", "doc_id", "language", "embed_model", "model", "client")

# Snowball stemmer and bm25s stop-word list for a pysbd language code. Polish
# has neither -- PyStemmer ships no Polish algorithm and bm25s no Polish list --
# so PoQuAD is indexed on whole words, with nothing removed.
BM25_LANGUAGES = {"en": "english"}


def read_documents(dataset, rendering="txt", max_documents=None):
    """Yield `(doc_id, text)` for a dataset's documents in one rendering, by name.

    `max_documents` keeps the first that many, in the same order every time, so
    a run started on a few documents can be extended to more.
    """
    folder, extension = RENDERINGS[rendering]
    directory = ROOT / "datasets" / dataset / "data" / folder
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} does not exist: run datasets/{dataset}/load.py first, and check "
            f"that {dataset} has a '{rendering}' rendering")
    for path in sorted(directory.glob(f"*{extension}"))[:max_documents]:
        # newline="" reads the file as it was written. The default would turn
        # a stray "\r\n" into "\n" and shift every offset after it, and the
        # evidence spans are offsets into the file as it is.
        with open(path, encoding="utf-8", newline="") as file:
            yield path.stem, file.read()


def build_config(dataset, chunking="sentence", params=None, rendering=None, augmentations=(),
                 language="en", llm_model=None, embedding_model=None,
                 chunking_embedding_model=None):
    """Describe a run: everything that decides what its chunks contain.

    `params` are the chunking method's keyword arguments; the ones left out are
    filled in from the method's defaults, so a default changed in the code
    later does not pass for the same run. `language` is a pysbd code; `ingest`
    detects it. A model is recorded only where the run calls it, which is what
    lets one set of chunks be embedded by several models -- and it must be
    named where it is called, since no model is assumed.
    `chunking_embedding_model` is the model a semantic chunker embeds
    sentences with, and defaults to `embedding_model`.
    """
    chunking_embedding_model = chunking_embedding_model or embedding_model
    if chunking not in CHUNKERS:
        raise ValueError(f"unknown chunking method {chunking!r}; choose from {sorted(CHUNKERS)}")
    unknown = [name for name in augmentations if name not in AUGMENTERS]
    if unknown:
        raise ValueError(f"unknown augmentation {unknown}; choose from {list(AUGMENTERS)}")
    accepted = inspect.signature(CHUNKERS[chunking]).parameters
    params = dict(params or {})
    refused = [key for key in params if key not in accepted or key in SUPPLIED]
    if refused:
        raise ValueError(f"{chunking} takes no parameter {refused}")
    defaults = {name: parameter.default for name, parameter in accepted.items()
                if name not in SUPPLIED}
    augmentations = [name for name in AUGMENTERS if name in augmentations]
    calls_llm = bool(augmentations) or "client" in accepted
    if calls_llm and not llm_model:
        raise ValueError("this run calls an LLM (augmentation or lumberchunker): name it with "
                         "--llm-model or LLM_MODEL")
    if "embed_model" in accepted and not chunking_embedding_model:
        raise ValueError(f"{chunking} chunking embeds sentences: name the model with "
                         f"--embedding-model or EMBEDDING_MODEL")
    config = {
        "dataset": dataset,
        "rendering": rendering or DEFAULT_RENDERING.get(chunking, "txt"),
        "language": language,
        "chunking": {"method": chunking, "params": {**defaults, **params}},
        "chunking_embedding_model": (chunking_embedding_model if "embed_model" in accepted
                                     else None),
        "augmentations": augmentations,
        "llm_model": llm_model if calls_llm else None,
    }
    # Through JSON and back, so that it compares equal to the copy on disk
    # (a tuple of heading names is read back as a list).
    return json.loads(json.dumps(config))


def run_name(config, params=None):
    """Directory name for a run: rendering, method, the parameters given, augmentations.

    Only the parameters in `params` are spelled out, to keep the name short;
    the full set is in `config.json`.
    """
    name = f"{config['rendering']}-{config['chunking']['method']}"
    name += "".join(f"-{key}={value}" for key, value in sorted((params or {}).items()))
    return name + "".join(f"+{augmentation}" for augmentation in config["augmentations"])


def _read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as file:
        return [json.loads(line) for line in file]


def _append_jsonl(path, records):
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def load_chunks(run_directory):
    """Read an index's chunks, in the order their vectors are stored."""
    return [Chunk(**record) for record in _read_jsonl(Path(run_directory) / "chunks.jsonl")]


# Node metadata keys reserved for the chunk's fields and the stores; chunk
# metadata may not use them, and `node_to_chunk` leaves them out of it.
# `truncated` is written by `embed_corpus` on every node it adds to Chroma --
# True where `embed_batch` had to cut the text to fit the model -- and BM25
# nodes never carry it. `json_encoded_keys` names the values stored as JSON
# (see `to_nodes`).
RESERVED_METADATA_KEYS = ("doc_id", "chunk_index", "start", "end", "original_text", "truncated",
             "json_encoded_keys")


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


def load_vector_store(run_directory, model):
    """Open a run's Chroma collection for `model` as a LlamaIndex vector store.

    The collection is created if the run has none for that model. It compares
    by cosine; Chroma's default is squared L2.
    """
    client = chromadb.PersistentClient(path=str(Path(run_directory) / "chroma"))
    # A collection name is letters, digits, dots, dashes and underscores.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    collection = client.get_or_create_collection(
        name, configuration={"hnsw": {"space": "cosine"}})
    return ChromaVectorStore(chroma_collection=collection)


def load_bm25(run_directory, top_k=10):
    """Load a run's persisted BM25Retriever, returning `top_k` nodes a query.

    `persist` saves the index but not how it was tokenized, and a loaded
    retriever stems every query as English. The stemming is therefore set
    again from the run's language, so a query is cut into the same terms its
    index holds. The query's English stop words are still removed whatever
    the language; the retriever offers no way to say otherwise.
    """
    run_directory = Path(run_directory)
    retriever = BM25Retriever.from_persist_dir(str(run_directory / "bm25"))
    retriever.similarity_top_k = top_k
    language = BM25_LANGUAGES.get(_read_json(run_directory / "config.json")["language"])
    retriever.skip_stemming = language is None
    if language:
        retriever.stemmer = Stemmer.Stemmer(language)
    return retriever


def _drop_unfinished(run_directory):
    """Cut `chunks.jsonl` back to the documents `documents.jsonl` says are finished.

    A run stopped while a document's chunks were being written leaves lines
    that no finished document accounts for; they are removed so the document
    is done again from its start. Returns the finished documents' ids.
    """
    finished = _read_jsonl(run_directory / "documents.jsonl")
    expected = sum(document["chunks"] for document in finished)
    chunks_path = run_directory / "chunks.jsonl"
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8", newline="") as file:
            lines = file.readlines()
        if len(lines) < expected:
            raise ValueError(f"{chunks_path} holds {len(lines)} chunks but documents.jsonl "
                             f"accounts for {expected}")
        if len(lines) > expected:
            with open(chunks_path, "w", encoding="utf-8", newline="\n") as file:
                file.writelines(lines[:expected])
    return {document["doc_id"] for document in finished}


def chunk_corpus(run_directory, config, max_documents=None, workers=4, client=None):
    """Chunk and augment every document of a run that is not finished yet.

    `workers` is how many per-chunk LLM calls are in flight at once. `client`
    is the OpenAI client the augmentations and `lumberchunker` call, and
    defaults to `get_client()`.
    """
    run_directory = Path(run_directory)
    finished = _drop_unfinished(run_directory)

    method = CHUNKERS[config["chunking"]["method"]]
    accepted = inspect.signature(method).parameters
    arguments = dict(config["chunking"]["params"])
    if config["llm_model"]:
        client = client or get_client()
    supplied = {"language": config["language"], "model": config["llm_model"], "client": client}
    if "embed_model" in accepted:
        supplied["embed_model"] = openai_embedding(config["chunking_embedding_model"])
    arguments.update({name: value for name, value in supplied.items() if name in accepted})

    documents = read_documents(config["dataset"], config["rendering"], max_documents)
    for doc_id, document in tqdm(list(documents), desc="chunking", unit="doc"):
        if doc_id in finished:
            continue
        chunks = verify(document, method(document, doc_id=doc_id, **arguments))
        if config["rendering"] == "html":
            chunks = [_as_html(chunk) for chunk in chunks]
        for name in config["augmentations"]:
            chunks = AUGMENTERS[name](chunks, document, config["llm_model"], client, workers)
        _append_jsonl(run_directory / "chunks.jsonl", [asdict(chunk) for chunk in chunks])
        _append_jsonl(run_directory / "documents.jsonl",
                      [{"doc_id": doc_id, "characters": len(document), "chunks": len(chunks)}])


def _as_html(chunk):
    """Mark a chunk cut from the HTML rendering as markup, so it embeds without tags.

    `html` marks its own chunks; any other method can be pointed at the HTML
    rendering too, and its chunks hold markup just the same.
    """
    if chunk.metadata.get("markup") == "html":
        return chunk
    return update_chunk_metadata(chunk, markup="html")


def embed_corpus(run_directory, model, client=None, batch_size=BATCH_SIZE, tokenizer=None,
                 max_tokens=None):
    """Embed the chunks of a run with `model` into its Chroma collection.

    Continues after the nodes the collection already holds, whether they come
    from an interrupted pass or from a finished one over fewer documents:
    chunks are only ever appended and are added in order, so the collection's
    size is the number done.

    `max_tokens` is the model's input limit; a text over it is cut to fit
    before it is sent, counted by `tokenizer` -- a Hugging Face repository
    name, or None for tiktoken (see `embedding.load_tokenizer`). Without
    `max_tokens` nothing is cut, and a text over the limit stops the run. The
    node keeps its whole text either way. Returns the ids of the nodes whose
    text was cut, which carry `truncated: True` in their metadata.
    """
    run_directory = Path(run_directory)
    nodes = to_nodes(load_chunks(run_directory))
    store = load_vector_store(run_directory, model)
    collection = store._collection
    done = collection.count()
    # A collection written before nodes carried the whole chunk would be
    # extended with nodes unlike its own.
    if done and "original_text" not in collection.get(limit=1, include=["metadatas"])["metadatas"][0]:
        raise ValueError(f"the {model} collection predates full-chunk node metadata; "
                         f"delete {run_directory / 'chroma'} and run again")
    if done > len(nodes):
        raise ValueError(f"the {model} collection holds {done} nodes but the run has "
                         f"{len(nodes)} chunks; delete {run_directory / 'chroma'}")
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
    return collection.get(where={"truncated": True}, include=[])["ids"]


def build_bm25(run_directory):
    """Index the chunks of a run for BM25 and persist the retriever under `bm25/`.

    Built from all the chunks at once, since BM25's term weights depend on the
    whole corpus; it calls no model, so it is simply redone when a run grows.
    English is stemmed and loses its stop words. A language in no
    `BM25_LANGUAGES` entry is indexed as whole words.
    """
    run_directory = Path(run_directory)
    config = _read_json(run_directory / "config.json")
    nodes = to_nodes(load_chunks(run_directory))
    if not nodes:
        return
    language = BM25_LANGUAGES.get(config["language"])
    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        # Passed to bm25s as its stop words: a language name, or a list.
        language=language or [],
        stemmer=Stemmer.Stemmer(language) if language else None,
        skip_stemming=language is None,
    )
    retriever.persist(str(run_directory / "bm25"))


def _read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def ingest(dataset, chunking="sentence", params=None, rendering=None, augmentations=(),
           language="auto", llm_model=None, embedding_model=None,
           max_documents=None, name=None, workers=4, embed=True, embedding_tokenizer=None,
           embedding_max_tokens=None, chunking_embedding_model=None):
    """Build or continue the index for one run, and return its directory.

    The arguments up to `embedding_model`, and `chunking_embedding_model`,
    are those of `build_config`.
    `llm_model` is needed by a run that augments or uses `lumberchunker`, and
    `embedding_model` by one that embeds. `name` is the run's directory under
    `indexes/<dataset>/` and defaults to `run_name`. `embed=False` writes the
    chunks and the BM25 index and stops. `embedding_max_tokens` and
    `embedding_tokenizer` are the `max_tokens` and `tokenizer` of
    `embed_corpus`.

    `language="auto"` is settled here, once for the run, from the opening of
    the dataset's first documents: a corpus is in one language, the BM25 index
    needs a single one, and `config.json` should name it rather than say "auto".
    """
    # Checked before any chunking, so a long LLM run does not end in an error.
    if embed and not embedding_model:
        raise ValueError("name the embedding model with --embedding-model or EMBEDDING_MODEL, "
                         "or pass --skip-embedding")
    if embedding_tokenizer and embedding_max_tokens is None:
        raise ValueError("an embedding tokenizer counts towards a limit: set "
                         "--embedding-max-tokens or EMBEDDING_MAX_TOKENS as well")
    if language == "auto":
        openings = "\n\n".join(text[:1000] for _, text in read_documents(dataset, "txt", 5))
        language = detect_language(openings, sample=len(openings))
    config = build_config(dataset, chunking, params, rendering, augmentations, language,
                          llm_model, embedding_model, chunking_embedding_model)
    run_directory = ROOT / "indexes" / dataset / (name or run_name(config, params))
    run_directory.mkdir(parents=True, exist_ok=True)
    config_path = run_directory / "config.json"
    # A directory with no finished document holds nothing the old
    # configuration produced, so a run that failed on its first one can be
    # started again with the model or parameter that made it fail changed.
    started = bool(_read_jsonl(run_directory / "documents.jsonl"))
    if started and _read_json(config_path) != config:
        raise ValueError(
            f"{run_directory} was started with a different configuration; pass another "
            f"--name or delete the directory. On disk: {_read_json(config_path)}")
    _write_json(config_path, config)

    chunk_corpus(run_directory, config, max_documents, workers)
    documents = _read_jsonl(run_directory / "documents.jsonl")
    print(f"{run_directory}: {len(documents)} documents, "
          f"{sum(document['chunks'] for document in documents)} chunks")
    build_bm25(run_directory)
    if embed:
        truncated = embed_corpus(run_directory, embedding_model, tokenizer=embedding_tokenizer,
                                 max_tokens=embedding_max_tokens)
        if truncated:
            print(f"{len(truncated)} chunks were cut to fit {embedding_model}: {truncated}")
    return run_directory


def _parameter(text):
    """Parse one `--param key=value`; the value is JSON where it parses as JSON."""
    key, separator, value = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="a directory of datasets/ whose loader has been run")
    parser.add_argument("--chunking", default="sentence", choices=sorted(CHUNKERS))
    parser.add_argument("--param", action="append", type=_parameter, default=[],
                        metavar="KEY=VALUE",
                        help="a keyword argument of the chunking method, repeatable; "
                             "the value is read as JSON (max_tokens=256, max_tokens=null)")
    parser.add_argument("--rendering", choices=sorted(RENDERINGS),
                        help="defaults to md for markdown, html for html, txt otherwise")
    parser.add_argument("--augment", nargs="*", default=[], choices=list(AUGMENTERS))
    parser.add_argument("--language", default="auto",
                        help="pysbd code; auto, the default, detects it from the documents")
    # Models are named by whoever runs the experiment, never by the code; the
    # environment is the fallback, as it is for the endpoint and its key.
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL"),
                        help="model for augmentation and lumberchunker; defaults to LLM_MODEL")
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL"),
                        help="model for embedding and the semantic chunkers; "
                             "defaults to EMBEDDING_MODEL")
    parser.add_argument("--chunking-embedding-model",
                        help="model the semantic chunkers embed sentences with, if not "
                             "--embedding-model")
    parser.add_argument("--embedding-max-tokens", type=int,
                        default=os.environ.get("EMBEDDING_MAX_TOKENS"),
                        help="the embedding model's input limit; longer texts are cut to fit. "
                             "Defaults to EMBEDDING_MAX_TOKENS; unset, nothing is cut. Counted "
                             "with tiktoken, leave a margin: it undercounts bge-m3 by ~14%%")
    parser.add_argument("--embedding-tokenizer", default=os.environ.get("EMBEDDING_TOKENIZER"),
                        help="Hugging Face repository of the embedding model's tokenizer, "
                             "for an exact count (BAAI/bge-m3); defaults to "
                             "EMBEDDING_TOKENIZER, then to tiktoken's cl100k_base")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--name", help="run directory under indexes/<dataset>/")
    parser.add_argument("--workers", type=int, default=4,
                        help="per-chunk LLM calls in flight at once")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="write the chunks and the BM25 index and stop")
    arguments = parser.parse_args()
    ingest(arguments.dataset, arguments.chunking, dict(arguments.param), arguments.rendering,
           arguments.augment, arguments.language, arguments.llm_model,
           arguments.embedding_model, arguments.max_documents, arguments.name,
           arguments.workers, embed=not arguments.skip_embedding,
           embedding_tokenizer=arguments.embedding_tokenizer,
           embedding_max_tokens=arguments.embedding_max_tokens,
           chunking_embedding_model=arguments.chunking_embedding_model)


if __name__ == "__main__":
    main()
