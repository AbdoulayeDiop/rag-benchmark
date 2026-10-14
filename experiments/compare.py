"""Comparison: the evaluated runs of each dataset side by side, as Markdown tables.

Reads every `results/.../evaluation/summary.json` that `evaluate.py` wrote and
prints one table per dataset, so the chunking methods can be read
off against each other and pasted into the README. By default a method gets
one row, with the parameters that scored best on the first metric shown
(`evidence_recall@5` unless `--columns` says otherwise); `--all` lists every
parameter combination instead. Methods are told apart by rendering and
augmentations too, and by embedding model, retrieval and reranking, so a best
row never mixes `sentence` over Markdown with `sentence` over plain text, or
hybrid retrieval with dense.

The figures are those of the `evidence` group of questions; a dataset with
none (LiteraryQA, TriviaQA) falls back to `no_evidence`, which has only the
document-level metrics.

Two rows are comparable only if they were scored on the same questions, which
means indexes over the same documents: the `docs` and `items` columns say
whether they were, and a dataset whose rows differ is flagged under its table.

`--cost` adds what a row cost, from the `cost.json` of its chunks and of its
index (`ingest.py`) and the summary's `cost` (`evaluate.py`): per document of
the index, the seconds it took to build -- chunking, augmentation, and
embedding the document's share of the chunks -- and the LLM and embedding
tokens it took; per question, the seconds of retrieval, and of reranking for a
reranked run. A run built or scored before costs were recorded shows "–"
where they are missing.

    .venv\\Scripts\\python.exe -m experiments.compare
"""

import argparse
import sys
from pathlib import Path

from experiments.common import CHUNKS, INDEXES, RESULTS, read_json

# Shown by default, and what the best parameters are chosen by: how much of
# the evidence the first five chunks cover.
COLUMNS = ("evidence_recall@5",)

# What `--cost` adds, as `cost` names them.
COST_COLUMNS = ("index s/doc", "LLM tok/doc", "embed tok/doc", "query s")


def describe_method(config):
    """A method as the comparison groups it: rendering, method, augmentations."""
    label = f"{config['rendering']}, {config['chunking']['method']}"
    return label + "".join(f" +{name}" for name in config["augmentations"])


def describe_params(config):
    return ", ".join(f"{key}={value}" for key, value in config["chunking"]["params"].items())


def describe_setup(summary):
    """What follows the chunks: the embedding model, then the retrieval and reranking runs."""
    return f"{summary['embedding']['model']} / {summary['setup']}"


def collect(results=RESULTS):
    """Every evaluation summary, grouped by dataset."""
    datasets = {}
    for path in sorted(Path(results).glob("**/evaluation/summary.json")):
        summary = read_json(path)
        datasets.setdefault(summary["dataset"], []).append(summary)
    return datasets


def _values(summary):
    """A summary's group of questions shown, and that group's metrics."""
    metrics = summary["metrics"]
    group = "evidence" if "evidence" in metrics else "no_evidence"
    return group, metrics.get(group, {})


def best(summaries, metric=COLUMNS[0]):
    """The summary of each method and set-up whose parameters scored highest on `metric`."""
    chosen = {}
    for summary in summaries:
        key = (describe_method(summary["chunking"]), describe_setup(summary))
        value = _values(summary)[1].get(metric, float("-inf"))
        if key not in chosen or value > chosen[key][0]:
            chosen[key] = (value, summary)
    return [summary for _, summary in chosen.values()]


def _format(value):
    return "–" if value is None else f"{value:.3f}"


def cost(summary):
    """What a summary's row cost, by the names in `COST_COLUMNS`; a cost not recorded is left out.

    The index's figures are means over the documents that recorded theirs;
    embedding is counted per chunk and multiplied by the index's chunks per
    document. The index's seconds and embedding tokens are left out when its
    embedding has no cost recorded, since they would leave the embedding out.
    """
    values = {}
    chunks_path = CHUNKS / Path(summary["index"]).parent / "cost.json"
    index_path = INDEXES / summary["index"] / "cost.json"
    built = read_json(chunks_path) if chunks_path.exists() else {}
    stages = [built[name] for name in ("chunking", "augmentation") if name in built]
    if stages:
        def per_document(name):
            return sum(stage.get(name, 0) / stage["documents"] for stage in stages)
        values["LLM tok/doc"] = per_document("llm_input_tokens") + per_document("llm_output_tokens")
        embedded = (read_json(index_path) if index_path.exists() else {}).get("embedding")
        if embedded:
            values["index s/doc"] = per_document("seconds")
            values["embed tok/doc"] = per_document("embedding_tokens")
            chunks_per_document = summary["chunks"] / summary["documents"]
            values["index s/doc"] += embedded["seconds"] / embedded["chunks"] * chunks_per_document
            values["embed tok/doc"] += (embedded.get("embedding_tokens", 0) / embedded["chunks"]
                                        * chunks_per_document)
    queried = summary.get("cost") or {}
    if "retrieval_seconds" in queried:
        values["query s"] = queried["retrieval_seconds"] + queried.get("rerank_seconds", 0)
    return values


def format_cost(name, value):
    """A cost as shown: tokens whole, seconds to three significant figures."""
    if value is None:
        return "–"
    return f"{value:,.0f}" if "tok" in name else f"{value:,.3g}"


def table(summaries, columns=COLUMNS, costs=False):
    """A Markdown table of `summaries`, one dataset's; `costs` adds the `COST_COLUMNS`."""
    shown_costs = COST_COLUMNS if costs else ()
    header = ["method", "params", "set-up", "docs", "items", "chunk chars", *columns,
              *shown_costs]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
    for summary in sorted(summaries, key=lambda s: (describe_method(s["chunking"]),
                                                     describe_setup(s),
                                                     describe_params(s["chunking"]))):
        group, values = _values(summary)
        row = [describe_method(summary["chunking"]), describe_params(summary["chunking"]),
               describe_setup(summary), str(summary["documents"]), str(summary["items"][group]),
               f"{summary['mean_chunk_chars']:,.0f}",
               *(_format(values.get(name)) for name in columns),
               *(format_cost(name, cost(summary).get(name)) for name in shown_costs)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="datasets to show; all by default")
    parser.add_argument("--columns", nargs="+", default=list(COLUMNS),
                        help="metrics to show, as named in summary.json (hit@1, mrr@10, "
                             "evidence_recall@10, ...); the first picks the best parameters")
    parser.add_argument("--all", action="store_true",
                        help="every parameter combination, not only each method's best")
    parser.add_argument("--cost", action="store_true",
                        help="add what each row cost to build and to query")
    arguments = parser.parse_args()
    # The Windows console defaults to cp1252, which has no "–" for a missing value.
    sys.stdout.reconfigure(encoding="utf-8")
    for dataset, summaries in collect().items():
        if arguments.datasets and dataset not in arguments.datasets:
            continue
        if not arguments.all:
            summaries = best(summaries, arguments.columns[0])
        print(f"### {dataset}\n\n{table(summaries, arguments.columns, arguments.cost)}\n")
        if len({summary["documents"] for summary in summaries}) > 1:
            print("Rows were scored on indexes over different documents, so on different "
                  "questions; compare only rows with the same `docs`.\n")


if __name__ == "__main__":
    main()
