"""Ingestion: a dataset's documents chunked, augmented, then embedded and indexed.

    .venv\\Scripts\\python.exe -m experiments.ingest experiments/configs/ingest_chunking.json

The configuration file lists the alternatives of each stage, and every
combination of a dataset and one alternative per stage is built:

    {
      "dataset": ["squad", "conditionalqa"],
      "chunking": [{"method": "sentence", "params": {"max_tokens": [256, 512], "overlap": 0}},
                   {"method": "semantic", "params": {"embedding_model": "bge-m3"}}],
      "chunk_augmentation": [{"method": "none"},
                             {"method": "context", "params": {"llm_model": "..."}}],
      "chunk_embedding": [{"model": "bge-m3", "tokenizer": "BAAI/bge-m3",
                           "context_length": 8192}],
      "max_documents": 10,
      "language": "auto",
      "workers": 4
    }

    chunking            `method`, a chunking method; `params`, its parameters,
                        where `llm_model` (lumberchunker) and `embedding_model`
                        (semantic, tiled, clustered) name the models it calls;
                        `rendering` (txt, md, html), to read other than the
                        method's own -- `md` for markdown, `html` for html,
                        `txt` otherwise. An overlap below 1 is a share of the
                        size; a combination whose overlap is not below its size
                        is skipped, and so is a dataset without the rendering.
    chunk_augmentation  `method`, an augmentation, a list of them applied
                        together, or "none"; `params.llm_model` names the model.
                        Left out, no augmentation.
    chunk_embedding     `model`; `tokenizer` and `context_length`, to cut
                        over-long chunks to the model's limit, counted exactly.
                        Left out, only the chunks are built.
    max_documents       the first that many documents of each dataset, by name
    language            a pysbd code, or "auto" (the default): detected once per
                        dataset from the opening of its first documents
    workers             per-chunk LLM calls, and embedding requests, in flight
                        at once (4)

Two stages, two trees (see `common`):

    chunks/<dataset>/<rendering>/<chunking>/<augmentation>/
        config.json       records everything that decides the chunks' content.
        chunks.jsonl      holds one chunk per line, in document order.
        documents.jsonl   holds one line per finished document, with its cost.
        cost.json         adds those costs up.
    indexes/<dataset>/<rendering>/<chunking>/<augmentation>/<embedding model>/
        config.json       names the chunks it was built from and the embedding.
        chroma/           holds the Chroma database (`rag.ingestion.open_vector_store`).
        bm25/             holds the persisted BM25Retriever (`rag.ingestion.load_bm25`).
        cost.json         holds what embedding cost.

The chunks are their own stage so that the embedding model is its own axis:
a second model adds an index beside the first and leaves the chunks, and the
LLM calls behind them, alone. An index reads its chunks from the chunks tree
rather than copying them. `chunks.jsonl` is not something retrieval reads: it
is the build's checkpoint -- where a document's LLM work is kept the moment it
is done -- and the plain-text record of a run.

Everything is resumable, because the expensive runs are long -- contextualising
GutenQA takes days at 128,000 input tokens a minute, the limit of the endpoint
used here (see the README's cost table) -- and `call_llm` raises rather than
skip a chunk. Chunks are written a document at a time and `documents.jsonl`
is appended to only once a document's chunks are on disk, so a run that
stopped anywhere picks up at the first unfinished document and repeats no LLM
call of a finished one. Vectors are added a batch at a time, so a collection's
size is where to continue from. The BM25 index calls no model and is rebuilt
each run.

What a run costs is recorded with what it produced, so that resuming adds to
it rather than starting the count again: each line of `documents.jsonl` holds
the document's `chunking` and `augmentation` -- seconds, and the tokens and
rate-limit waits `llm.usage` counted -- and the chunks' `cost.json` adds them
up; the index's holds a running total brought up to date after every batch.
The `workers` the last run had are recorded too, in both, since they change
the seconds but not the tokens. A document finished before costs were recorded has none,
so the totals say how many documents they cover.

Every combination is checked before the first starts -- unknown keys, methods
and parameters, missing models, two runs that would share a directory but not
its configuration -- so a mistake does not surface hours in.
"""

import inspect
import itertools
import time
from collections import Counter
from dataclasses import asdict

from tqdm import tqdm

from rag.augmentation import (add_context, add_keywords, add_questions, add_summary, add_title,
                              use_summary)
from rag.chunking import (clustered, detect_language, fixed_char, fixed_token, html, markdown,
                          openai_embedding, semantic, sentence, tiled, update_chunk_metadata,
                          verify)
from rag.chunking.lumberchunker import lumberchunker
from rag.ingestion import build_bm25, embed_nodes, open_vector_store, to_nodes
from rag.llm import get_client, usage, usage_since

from experiments.common import (CHUNKS, INDEXES, RENDERINGS, append_jsonl, as_list, check_dataset,
                                check_keys, has_rendering, is_none, load_chunks, load_config,
                                name, normalized, parse_arguments, read_documents, read_json,
                                read_jsonl, relative, run_each, start_run, sweep, write_json)

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

# Arguments of a chunking method that the run supplies rather than the config.
SUPPLIED = ("document", "doc_id", "language", "embed_model", "model", "client")

CONFIG_KEYS = ("dataset", "chunking", "chunk_augmentation", "chunk_embedding", "max_documents",
               "language", "workers")


def chunking_options(entry, where):
    """The chunking alternatives one entry stands for, its sweeps expanded."""
    check_keys(entry, ("method", "params", "rendering"), where)
    method = entry.get("method")
    if method not in CHUNKERS:
        raise ValueError(f"{where}: unknown chunking method {method!r}; choose from "
                         f"{sorted(CHUNKERS)}")
    if entry.get("rendering") not in (None, *RENDERINGS):
        raise ValueError(f"{where}: unknown rendering {entry['rendering']!r}; choose from "
                         f"{sorted(RENDERINGS)}")
    accepted = inspect.signature(CHUNKERS[method]).parameters
    fixed = [key for key, parameter in accepted.items()
             if isinstance(parameter.default, (list, tuple))]
    options = []
    for params in sweep(entry.get("params", {}), fixed):
        params = dict(params)
        llm_model = params.pop("llm_model", None)
        embedding_model = params.pop("embedding_model", None)
        if ("client" in accepted) != bool(llm_model):
            raise ValueError(f"{where}: {method} " + (
                "calls an LLM: name it in params.llm_model" if "client" in accepted
                else "calls no LLM, so takes no llm_model"))
        if ("embed_model" in accepted) != bool(embedding_model):
            raise ValueError(f"{where}: {method} " + (
                "embeds sentences: name the model in params.embedding_model"
                if "embed_model" in accepted else "embeds nothing, so takes no embedding_model"))
        refused = [key for key in params if key not in accepted or key in SUPPLIED]
        if refused:
            raise ValueError(f"{where}: {method} takes no parameter {refused}")
        size = params.get("max_tokens", params.get("size"))
        overlap = params.get("overlap")
        if isinstance(size, int) and isinstance(overlap, int) and overlap >= size:
            print(f"skipped {method} {params}: overlap is not below the chunk size")
            continue
        # The models a method calls name its directory too, so that two of
        # them do not share one.
        named = {**params, **({"llm_model": llm_model} if llm_model else {}),
                 **({"embedding_model": embedding_model} if embedding_model else {})}
        options.append({"method": method, "params": params,
                        "rendering": entry.get("rendering") or DEFAULT_RENDERING.get(method, "txt"),
                        "llm_model": llm_model, "embedding_model": embedding_model,
                        "name": name(method, named)})
    return options


def augmentation_options(entry, where):
    """The augmentation alternatives one entry stands for: `(augmentations, llm_model)` pairs."""
    check_keys(entry, ("method", "params"), where)
    if is_none(entry.get("method")):
        if entry.get("params"):
            raise ValueError(f"{where}: no augmentation takes no params")
        return [((), None)]
    methods = as_list(entry["method"])
    unknown = [method for method in methods if method not in AUGMENTERS]
    if unknown:
        raise ValueError(f"{where}: unknown augmentation {unknown}; choose from "
                         f"{list(AUGMENTERS)} or none")
    check_keys(entry.get("params", {}), ("llm_model",), f"{where}, params")
    options = []
    for params in sweep(entry.get("params", {})):
        if not params.get("llm_model"):
            raise ValueError(f"{where}: augmentation calls an LLM: name it in params.llm_model")
        # Applied, and named, in `AUGMENTERS` order whatever order they are listed in.
        options.append(([method for method in AUGMENTERS if method in methods],
                        params["llm_model"]))
    return options


def embedding_options(entry, where):
    check_keys(entry, ("model", "tokenizer", "context_length"), where)
    if not entry.get("model"):
        raise ValueError(f"{where}: name the embedding model")
    if entry.get("tokenizer") and not entry.get("context_length"):
        raise ValueError(f"{where}: a tokenizer counts towards context_length; set it too")
    return [{"model": entry["model"], "tokenizer": entry.get("tokenizer"),
             "context_length": entry.get("context_length")}]


def chunks_config(dataset, chunking, augmentations, llm_model, language):
    """Describe a chunks run: everything that decides what its chunks contain.

    The chunking method's parameters left out are filled in from its
    defaults, so a default changed in the code later does not pass for the
    same run. A model is recorded only where the run calls it.
    """
    accepted = inspect.signature(CHUNKERS[chunking["method"]]).parameters
    defaults = {key: parameter.default for key, parameter in accepted.items()
                if key not in SUPPLIED}
    return normalized({
        "dataset": dataset,
        "rendering": chunking["rendering"],
        "language": language,
        "chunking": {"method": chunking["method"], "params": {**defaults, **chunking["params"]}},
        "chunking_embedding_model": chunking["embedding_model"],
        "augmentations": list(augmentations),
        "llm_model": llm_model,
    })


def plan(config, where):
    """The chunks runs a configuration asks for, each with the embeddings to index it with.

    Raises on the first thing that could not run, naming where it is in the file.
    """
    datasets = as_list(config.get("dataset"))
    if not datasets:
        raise ValueError(f"{where}: name the dataset")
    for dataset in datasets:
        check_dataset(dataset, where)
    expanders = {"chunking": chunking_options, "chunk_augmentation": augmentation_options,
                 "chunk_embedding": embedding_options}
    defaults = {"chunk_augmentation": [{"method": "none"}], "chunk_embedding": []}
    options = {}
    for stage, expand in expanders.items():
        entries = config.get(stage, defaults.get(stage))
        if entries is None:
            raise ValueError(f"{where}: list at least one {stage} entry")
        options[stage] = [option for number, entry in enumerate(entries, start=1)
                          for option in expand(entry, f"{where}, {stage} {number}")]

    runs, configs = [], {}
    for dataset, chunking, (augmentations, augmentation_llm) in itertools.product(
            datasets, options["chunking"], options["chunk_augmentation"]):
        here = f"{where}, {dataset} / {chunking['name']}"
        if chunking["llm_model"] and augmentation_llm and \
                chunking["llm_model"] != augmentation_llm:
            raise ValueError(f"{here}: a run calls one LLM, but chunking names "
                             f"{chunking['llm_model']} and augmentation {augmentation_llm}")
        # A dataset without the rendering -- SQuAD has no Markdown -- is passed
        # over rather than refused, so one file can list every dataset.
        if not has_rendering(dataset, chunking["rendering"]):
            print(f"skipped {dataset} for {chunking['method']}: no '{chunking['rendering']}' "
                  f"rendering")
            continue
        augmentation = (name("+".join(augmentations), {"llm_model": augmentation_llm})
                        if augmentations else "none")
        directory = CHUNKS / dataset / chunking["rendering"] / chunking["name"] / augmentation
        llm_model = chunking["llm_model"] or augmentation_llm
        # The language does not decide the name, and is detected at run time.
        described = chunks_config(dataset, chunking, augmentations, llm_model, None)
        if configs.setdefault(directory, described) != described:
            raise ValueError(f"{here}: two runs would share {directory} with different "
                             f"configurations")
        runs.append({"dataset": dataset, "directory": directory, "chunking": chunking,
                     "augmentations": augmentations, "llm_model": llm_model,
                     "embeddings": options["chunk_embedding"]})
    if not runs:
        raise ValueError(f"{where}: no run is left to do")
    return runs


def _add_up(records):
    """The sum of `records`, dicts of numbers, name by name."""
    totals = Counter()
    for record in records:
        totals.update(record)
    return {key: round(value, 3) if isinstance(value, float) else value
            for key, value in sorted(totals.items())}


def _cost_since(before, started):
    """The seconds since `started` and what `llm.usage` counted since `before`."""
    return {"seconds": round(time.perf_counter() - started, 3), **usage_since(before)}


def _drop_unfinished(directory):
    """Cut `chunks.jsonl` back to the documents `documents.jsonl` says are finished.

    A run stopped while a document's chunks were being written leaves lines
    that no finished document accounts for; they are removed so the document
    is done again from its start. Returns the finished documents' ids.
    """
    finished = read_jsonl(directory / "documents.jsonl")
    expected = sum(document["chunks"] for document in finished)
    chunks_path = directory / "chunks.jsonl"
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


def _as_html(chunk):
    """Mark a chunk cut from the HTML rendering as markup, so it embeds without tags.

    `html` marks its own chunks; any other method can be pointed at the HTML
    rendering too, and its chunks hold markup just the same.
    """
    if chunk.metadata.get("markup") == "html":
        return chunk
    return update_chunk_metadata(chunk, markup="html")


def chunk_corpus(directory, config, max_documents=None, workers=4, client=None):
    """Chunk and augment every document of a chunks run that is not finished yet.

    `workers` is how many per-chunk LLM calls are in flight at once. `client`
    is the OpenAI client the augmentations and `lumberchunker` call, and
    defaults to `get_client()`.
    """
    finished = _drop_unfinished(directory)
    method = CHUNKERS[config["chunking"]["method"]]
    accepted = inspect.signature(method).parameters
    arguments = dict(config["chunking"]["params"])
    if config["llm_model"]:
        client = client or get_client()
    supplied = {"language": config["language"], "model": config["llm_model"], "client": client}
    if "embed_model" in accepted:
        supplied["embed_model"] = openai_embedding(config["chunking_embedding_model"])
    arguments.update({key: value for key, value in supplied.items() if key in accepted})

    documents = read_documents(config["dataset"], config["rendering"], max_documents)
    for doc_id, document in tqdm(list(documents), desc="chunking", unit="doc"):
        if doc_id in finished:
            continue
        before, started = usage(), time.perf_counter()
        chunks = verify(document, method(document, doc_id=doc_id, **arguments))
        if config["rendering"] == "html":
            chunks = [_as_html(chunk) for chunk in chunks]
        cost = {"chunking": _cost_since(before, started)}
        if config["augmentations"]:
            before, started = usage(), time.perf_counter()
            for augmentation in config["augmentations"]:
                chunks = AUGMENTERS[augmentation](chunks, document, config["llm_model"], client,
                                                  workers)
            cost["augmentation"] = _cost_since(before, started)
        append_jsonl(directory / "chunks.jsonl", [asdict(chunk) for chunk in chunks])
        append_jsonl(directory / "documents.jsonl",
                     [{"doc_id": doc_id, "characters": len(document), "chunks": len(chunks),
                       **cost}])


def record_cost(directory, workers):
    """Add up the chunking and augmentation costs of a run's documents in `cost.json`.

    Recomputed from `documents.jsonl` each time, over the documents that
    recorded a cost.
    """
    cost = {}
    documents = read_jsonl(directory / "documents.jsonl")
    for stage in ("chunking", "augmentation"):
        recorded = [document[stage] for document in documents if stage in document]
        if recorded:
            cost[stage] = {"documents": len(recorded), **_add_up(recorded)}
    cost["workers"] = workers
    write_json(directory / "cost.json", cost)


def build_chunks(run, language, max_documents=None, workers=4):
    """Build or continue one chunks run; return its directory."""
    directory = run["directory"]
    config = chunks_config(run["dataset"], run["chunking"], run["augmentations"],
                           run["llm_model"], language)
    directory.mkdir(parents=True, exist_ok=True)
    start_run(directory, config, started=bool(read_jsonl(directory / "documents.jsonl")))
    chunk_corpus(directory, config, max_documents, workers)
    record_cost(directory, workers)
    documents = read_jsonl(directory / "documents.jsonl")
    print(f"{relative(directory, CHUNKS)}: {len(documents)} documents, "
          f"{sum(document['chunks'] for document in documents)} chunks")
    return directory


def build_index(chunks_directory, embedding, workers=4):
    """Build or continue the index of a chunks run for one embedding; return its directory.

    `embedding` is `{model, tokenizer, context_length}`: a chunk over
    `context_length` tokens, counted by `tokenizer` (tiktoken if None), is
    cut to fit before it is embedded, and without one a chunk over the
    model's limit stops the run. `workers` embedding requests are in flight
    at once.
    """
    model = embedding["model"]
    directory = INDEXES / relative(chunks_directory, CHUNKS) / name(model)
    directory.mkdir(parents=True, exist_ok=True)
    config = {"chunks": relative(chunks_directory, CHUNKS), "embedding": embedding}
    start_run(directory, config, started=(directory / "chroma").exists())
    language = read_json(chunks_directory / "config.json")["language"]
    chunks = load_chunks(chunks_directory)
    build_bm25(to_nodes(chunks), directory / "bm25", language)

    cost_path = directory / "cost.json"
    state = {"before": usage(), "started": time.perf_counter()}

    def record(count):
        """Add one stored batch's seconds and tokens to the running total."""
        cost = read_json(cost_path) if cost_path.exists() else {}
        cost["embedding"] = _add_up([cost.get("embedding", {}),
                                     {"chunks": count,
                                      **_cost_since(state["before"], state["started"])}])
        cost["workers"] = workers
        write_json(cost_path, cost)
        state.update(before=usage(), started=time.perf_counter())

    store = open_vector_store(directory / "chroma", model)
    truncated = embed_nodes(store, to_nodes(chunks), model, tokenizer=embedding["tokenizer"],
                            max_tokens=embedding["context_length"], on_batch=record,
                            workers=workers)
    if truncated:
        print(f"{len(truncated)} chunks were cut to fit {model}: {truncated}")
    return directory


def main():
    arguments = parse_arguments(__doc__.splitlines()[0])
    config = load_config(arguments.config, CONFIG_KEYS)
    languages = {}

    def run(entry):
        language = config.get("language", "auto")
        if language == "auto":
            # Once per dataset, from plain text: a corpus is in one language,
            # the BM25 index needs a single one, and config.json should name
            # it rather than say "auto".
            if entry["dataset"] not in languages:
                openings = "\n\n".join(text[:1000] for _, text in
                                       read_documents(entry["dataset"], "txt", 5))
                languages[entry["dataset"]] = detect_language(openings, sample=len(openings))
            language = languages[entry["dataset"]]
        directory = build_chunks(entry, language, config.get("max_documents"),
                                 config.get("workers", 4))
        for embedding in entry["embeddings"]:
            build_index(directory, embedding, config.get("workers", 4))

    run_each(plan(config, arguments.config),
             lambda entry: f"{relative(entry['directory'], CHUNKS)} ("
                           f"{', '.join(e['model'] for e in entry['embeddings']) or 'no embedding'})",
             run, arguments.dry_run, arguments.keep_going)


if __name__ == "__main__":
    main()
