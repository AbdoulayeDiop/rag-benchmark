"""Experiments: a grid of pipeline configurations, each built by `ingest.py` and scored by `evaluate.py`.

A comparison is many runs that differ in one stage -- the chunking method, say
-- and agree on the others. Written as a file, the whole comparison is one
record that can be read, re-run and extended, instead of a dozen command lines
kept in step by hand:

    .venv\\Scripts\\python.exe experiments.py experiments/chunking_eval_squad_conditionalqa.json

The file is JSON, one key per stage of the pipeline (see `experiments/schema.json`):

    {
      "description": "...",
      "dataset": ["squad", "conditionalqa"],
      "chunking": [{"method": "sentence", "params": {"max_tokens": 256, "overlap": 0}},
                   {"method": "semantic", "params": {"embedding_model": "bge-m3"}}],
      "chunk_augmentation": [{"method": "none"}],
      "chunk_embedding": [{"model": "bge-m3", "tokenizer": "BAAI/bge-m3",
                           "context_length": 8192}],
      "retrieval": [{"method": "hybrid", "params": {"top_k": 50}}],
      "reranking": [{"method": "cross_encoder",
                     "params": {"model": "bge-reranker-v2-m3", "top_k": 10,
                                "max_tokens": 8192, "tokenizer": "BAAI/bge-m3"}}],
      "max_documents": 10,
      "max_items": 200
    }

Each stage lists alternatives, and every combination of a dataset and one
alternative per stage is a run. Runs that differ only in retrieval or
reranking share their index, which is built once; a second embedding model
adds a collection to the same index. A list inside `params` is a sweep -- one
alternative per value, every combination -- except for a parameter whose own
value is a list (`html`'s `headings`).

The stages:

    dataset             a name or a list of names, exactly as under datasets/
    chunking            `method`, a chunking method; `params`, its parameters,
                        where `llm_model` (lumberchunker) and `embedding_model`
                        (semantic, tiled, clustered) name the models it calls;
                        `rendering` (txt, md, html), to read other than the
                        method's own -- `md` for markdown, `html` for html,
                        `txt` otherwise. An overlap below 1 is a share of the
                        size; a combination whose overlap is not below its size
                        is skipped, and so is a dataset without the rendering.
    chunk_augmentation  `method`, one augmentation, or "none"; `params.llm_model`
                        names its model. Left out, no augmentation.
    chunk_embedding     `model`; `tokenizer` and `context_length`, to cut
                        over-long chunks to the model's limit, counted exactly
    retrieval           `method`: dense, sparse or hybrid; `params.top_k`
    reranking           `method`: cross_encoder, or "none"; `params.model`,
                        `top_k` (chunks kept; all retrieved are reranked),
                        and `max_tokens` and
                        `tokenizer`, to cut a chunk to the model's pair limit.
                        Left out, no reranking.

`max_documents` and `max_items` cap the documents indexed and the questions
scored, for a quick run. Any entry, and the file, may carry a `description`,
which stands in for a comment and is otherwise ignored.

Nothing comes from the environment: a run is what its file says, so the file
is the record of what was run. Every run is checked before the first starts
-- unknown keys, methods and parameters, missing models, two runs that would
share an index directory but not its configuration -- so a mistake does not
surface hours in. Both stages resume, so running a file again finishes what an
interrupted run left and redoes nothing.
"""

import argparse
import inspect
import itertools
import json
import os
import re
import traceback

from compare import collect, table
from evaluate import evaluate
from ingest import AUGMENTERS, CHUNKERS, RENDERINGS, ROOT, build_config, ingest, run_name
from retrieval import METHODS

STAGES = ("chunking", "chunk_augmentation", "chunk_embedding", "retrieval", "reranking")

# The reranking methods implemented; a rule-based one would join them here.
RERANKERS = ("cross_encoder",)

# The parameters an entry of each stage may hold, besides those of a chunking
# method, which come from its signature.
AUGMENTATION_PARAMS = ("llm_model",)
RETRIEVAL_PARAMS = ("top_k",)
RERANKING_PARAMS = ("model", "top_k", "max_tokens", "tokenizer")


def _check_keys(entry, allowed, where):
    unknown = [key for key in entry if key not in (*allowed, "description")]
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")


def _sweep(params, fixed=()):
    """Every parameter set the lists in `params` stand for, one per combination.

    A key in `fixed` takes a list as its value and is not swept.
    """
    swept = [key for key, value in params.items() if isinstance(value, list) and key not in fixed]
    return [{**params, **dict(zip(swept, values))}
            for values in itertools.product(*(params[key] for key in swept))]


def _none(method):
    return method in (None, "", "none")


def chunking_options(entry, where):
    """The chunking alternatives one entry stands for, its sweeps expanded."""
    _check_keys(entry, ("method", "params", "rendering"), where)
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
    for params in _sweep(entry.get("params", {}), fixed):
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
        size = params.get("max_tokens", params.get("size"))
        overlap = params.get("overlap")
        if isinstance(size, int) and isinstance(overlap, int) and overlap >= size:
            print(f"skipped {method} {params}: overlap is not below the chunk size")
            continue
        # The models a method calls name its index too, so that two of them
        # do not share one directory.
        named = {**params, **({"llm_model": llm_model} if llm_model else {}),
                 **({"embedding_model": embedding_model} if embedding_model else {})}
        options.append({"method": method, "params": params, "rendering": entry.get("rendering"),
                        "llm_model": llm_model, "embedding_model": embedding_model,
                        "named": named})
    return options


def augmentation_options(entry, where):
    """The augmentation alternatives one entry stands for: `(augmentations, llm_model)` pairs."""
    _check_keys(entry, ("method", "params"), where)
    method = entry.get("method")
    if _none(method):
        if entry.get("params"):
            raise ValueError(f"{where}: no augmentation takes no params")
        return [((), None)]
    if method not in AUGMENTERS:
        raise ValueError(f"{where}: unknown augmentation {method!r}; choose from "
                         f"{list(AUGMENTERS)} or none")
    _check_keys(entry.get("params", {}), AUGMENTATION_PARAMS, f"{where}, params")
    options = []
    for params in _sweep(entry.get("params", {})):
        if not params.get("llm_model"):
            raise ValueError(f"{where}: {method} calls an LLM: name it in params.llm_model")
        options.append(((method,), params["llm_model"]))
    return options


def embedding_options(entry, where):
    _check_keys(entry, ("model", "tokenizer", "context_length"), where)
    if not entry.get("model"):
        raise ValueError(f"{where}: name the embedding model")
    if entry.get("tokenizer") and not entry.get("context_length"):
        raise ValueError(f"{where}: a tokenizer counts towards context_length; set it too")
    return [entry]


def retrieval_options(entry, where):
    _check_keys(entry, ("method", "params"), where)
    if entry.get("method") not in METHODS:
        raise ValueError(f"{where}: unknown retrieval method {entry.get('method')!r}; choose "
                         f"from {METHODS}")
    _check_keys(entry.get("params", {}), RETRIEVAL_PARAMS, f"{where}, params")
    return [{"method": entry["method"], "params": params}
            for params in _sweep(entry.get("params", {}))]


def reranking_options(entry, where):
    _check_keys(entry, ("method", "params"), where)
    method = entry.get("method")
    if _none(method):
        return [None]
    if method not in RERANKERS:
        raise ValueError(f"{where}: unknown reranking method {method!r}; choose from "
                         f"{RERANKERS} or none")
    _check_keys(entry.get("params", {}), RERANKING_PARAMS, f"{where}, params")
    options = []
    for params in _sweep(entry.get("params", {})):
        if not params.get("model"):
            raise ValueError(f"{where}: {method} needs params.model")
        if params.get("max_tokens") and not params.get("tokenizer"):
            raise ValueError(f"{where}: max_tokens is counted with the reranker's tokenizer; "
                             f"set params.tokenizer too")
        options.append({"method": method, "params": params})
    return options


def setup_name(retrieval, embedding_model, reranking):
    """Results directory for one retrieval set-up: the methods, models and parameters given."""
    name = retrieval["method"]
    if retrieval["method"] != "sparse":
        name += f"-{embedding_model}"
    name += "".join(f"-{key}={value}" for key, value in sorted(retrieval["params"].items()))
    if reranking:
        params = reranking["params"]
        name += f"+{params['model']}"
        if "top_k" in params:
            name += f"-top_k={params['top_k']}"
    return re.sub(r"[^A-Za-z0-9._+=-]", "_", name)


def load_experiments(path):
    """Read an experiment file into the index builds it asks for, each with its evaluations.

    A build is one index (a dataset, chunking and augmentation) embedded by
    one model; its evaluations are the retrieval set-ups scored on it. Raises
    on the first thing that could not run, naming where it is in the file.
    """
    with open(path, encoding="utf-8") as file:
        spec = json.load(file)
    _check_keys(spec, ("dataset", *STAGES, "max_documents", "max_items"), path)
    datasets = spec.get("dataset")
    datasets = [datasets] if isinstance(datasets, str) else datasets or []
    if not datasets:
        raise ValueError(f"{path}: name the dataset")
    for dataset in datasets:
        # Exact names only: Windows would open datasets/SQuAD as datasets/squad,
        # but the index and results would be filed under another name.
        if dataset not in os.listdir(ROOT / "datasets") or \
                not (ROOT / "datasets" / dataset / "data").is_dir():
            raise ValueError(f"{path}: datasets/{dataset}/data does not exist (names are "
                             f"case-sensitive); run its loader first")

    expanders = {"chunking": chunking_options, "chunk_augmentation": augmentation_options,
                 "chunk_embedding": embedding_options, "retrieval": retrieval_options,
                 "reranking": reranking_options}
    # Augmentation and reranking are optional stages; the others are not.
    defaults = {"chunk_augmentation": [{"method": "none"}], "reranking": [{"method": "none"}]}
    options = {}
    for stage, expand in expanders.items():
        entries = spec.get(stage) or defaults.get(stage)
        if not entries:
            raise ValueError(f"{path}: list at least one {stage} entry")
        options[stage] = [option for number, entry in enumerate(entries, start=1)
                          for option in expand(entry, f"{path}, {stage} {number}")]

    builds, configs = [], {}
    for dataset, chunking, (augmentations, augmentation_llm), embedding in itertools.product(
            datasets, options["chunking"], options["chunk_augmentation"],
            options["chunk_embedding"]):
        where = f"{path}, {dataset} / {chunking['method']} {chunking['named']}"
        if chunking["llm_model"] and augmentation_llm and \
                chunking["llm_model"] != augmentation_llm:
            raise ValueError(f"{where}: a run calls one LLM, but chunking names "
                             f"{chunking['llm_model']} and augmentation {augmentation_llm}")
        llm_model = chunking["llm_model"] or augmentation_llm
        try:
            config = build_config(dataset, chunking["method"], chunking["params"],
                                  chunking["rendering"], augmentations, "en", llm_model,
                                  embedding["model"], chunking["embedding_model"])
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from error
        # A dataset without the rendering -- SQuAD has no Markdown -- is passed
        # over rather than refused, so one file can list every dataset.
        folder = RENDERINGS[config["rendering"]][0]
        if not (ROOT / "datasets" / dataset / "data" / folder).is_dir():
            print(f"skipped {dataset} for {chunking['method']}: no '{config['rendering']}' "
                  f"rendering")
            continue
        index = run_name(config, chunking["named"])
        # The language does not decide the name, and is detected at run time.
        if configs.setdefault((dataset, index), config) != config:
            raise ValueError(f"{where}: two runs would share indexes/{dataset}/{index} with "
                             f"different configurations (augmentation models differ?)")
        ingest_arguments = {
            "chunking": chunking["method"], "params": chunking["params"],
            "rendering": chunking["rendering"], "augmentations": augmentations,
            "llm_model": llm_model, "embedding_model": embedding["model"],
            "chunking_embedding_model": chunking["embedding_model"],
            "embedding_tokenizer": embedding.get("tokenizer"),
            "embedding_max_tokens": embedding.get("context_length"),
            "max_documents": spec.get("max_documents"), "name": index}
        evaluations = []
        for retrieval, reranking in itertools.product(options["retrieval"], options["reranking"]):
            rerank = reranking["params"] if reranking else {}
            arguments = {"method": retrieval["method"], "embedding_model": embedding["model"],
                         "rerank_model": rerank.get("model"),
                         "rerank_max_tokens": rerank.get("max_tokens"),
                         "rerank_tokenizer": rerank.get("tokenizer"),
                         "max_items": spec.get("max_items")}
            if "top_k" in retrieval["params"]:
                arguments["top_k"] = retrieval["params"]["top_k"]
            if "top_k" in rerank:
                arguments["rerank_top_k"] = rerank["top_k"]
            evaluations.append({"name": setup_name(retrieval, embedding["model"], reranking),
                                "arguments": arguments})
        builds.append({"dataset": dataset, "index": index, "embedding_model": embedding["model"],
                       "ingest": ingest_arguments, "evaluations": evaluations})
    if not builds:
        raise ValueError(f"{path}: no run is left to do")
    return builds


def run(builds, keep_going=False, skip_evaluation=False):
    """Build each index and score each of its set-ups; return what failed, with the error."""
    failed = []
    for number, build in enumerate(builds, start=1):
        print(f"\n=== [{number}/{len(builds)}] {build['dataset']} / {build['index']} "
              f"({build['embedding_model']})")
        try:
            directory = ingest(build["dataset"], **build["ingest"])
            for evaluation in [] if skip_evaluation else build["evaluations"]:
                evaluate(build["dataset"], directory.name, name=evaluation["name"],
                         **evaluation["arguments"])
        except Exception as error:
            if not keep_going:
                raise
            traceback.print_exc()
            failed.append((build, error))
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="a JSON experiment file (see experiments/schema.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the file and list the runs, without running them")
    parser.add_argument("--keep-going", action="store_true",
                        help="go on to the next index when one fails, and list failures at "
                             "the end")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="build the indexes only")
    arguments = parser.parse_args()
    builds = load_experiments(arguments.file)
    for number, build in enumerate(builds, start=1):
        print(f"{number:>3}. {build['dataset']} / {build['index']} ({build['embedding_model']})")
        for evaluation in build["evaluations"]:
            print(f"       {evaluation['name']}")
    print(f"{len(builds)} index builds, "
          f"{sum(len(build['evaluations']) for build in builds)} evaluations")
    if arguments.dry_run:
        return
    failed = run(builds, arguments.keep_going, arguments.skip_evaluation)

    # The table of this file's runs only; compare.py shows every result there is.
    indexes = {(build["dataset"], build["index"]) for build in builds}
    for dataset, summaries in collect().items():
        mine = [summary for summary in summaries if (dataset, summary["index"]) in indexes]
        if mine and not arguments.skip_evaluation:
            print(f"\n### {dataset}\n\n{table(mine)}")
    if failed:
        print(f"\n{len(failed)} of {len(builds)} index builds failed:")
        for build, error in failed:
            print(f"  {build['dataset']} / {build['index']}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
