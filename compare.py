"""Comparison: the evaluated runs of each dataset side by side, as Markdown tables.

Reads every `results/<dataset>/<index>/<set-up>/summary.json` that `evaluate.py`
wrote and prints one table per dataset, so the chunking methods can be read
off against each other and pasted into the README. By default a method gets
one row, with the parameters that scored best on the first metric shown
(`evidence_recall@5` unless `--columns` says otherwise); `--all` lists every
parameter combination instead. Methods are told apart by rendering and
augmentations too, and by retrieval set-up, so a best row never mixes
`sentence` over Markdown with `sentence` over plain text, or hybrid retrieval
with dense.

The figures are those of the `evidence` group of questions; a dataset with
none (LiteraryQA) falls back to `no_evidence`, which has only the
document-level metrics.

Two rows are comparable only if they were scored on the same questions, which
means indexes over the same documents: the `docs` and `items` columns say
whether they were, and a dataset whose rows differ is flagged under its table.

`--cost` adds what a row cost, from the index's `cost.json` (`ingest.py`) and
the summary's `cost` (`evaluate.py`): per document of the index, the seconds
it took to build -- chunking, augmentation, and embedding the document's share
of the chunks with the set-up's model -- and the LLM and embedding tokens it
took; per question, the seconds of retrieval, and of reranking where the
ranking shown is reranked. A run built or scored before costs were recorded
shows "–" where they are missing.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Shown by default, and what the best parameters are chosen by: how much of
# the evidence the first five chunks cover.
COLUMNS = ("evidence_recall@5",)

# What `--cost` adds, as `cost` names them.
COST_COLUMNS = ("index s/doc", "LLM tok/doc", "embed tok/doc", "query s")


def _read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def describe_method(config):
    """A method as the comparison groups it: rendering, method, augmentations."""
    label = f"{config['rendering']}, {config['chunking']['method']}"
    return label + "".join(f" +{name}" for name in config["augmentations"])


def describe_params(config):
    return ", ".join(f"{key}={value}" for key, value in config["chunking"]["params"].items())


def collect(results=ROOT / "results"):
    """Every evaluation summary, with its configuration, grouped by dataset."""
    datasets = {}
    for path in sorted(Path(results).glob("*/*/*/summary.json")):
        summary = _read_json(path)
        summary["config"] = _read_json(path.parent / "config.json")
        datasets.setdefault(summary["dataset"], []).append(summary)
    return datasets


def _values(summary, stage):
    """The stage shown for a summary, its group of questions, and that group's metrics."""
    metrics = summary["metrics"]
    shown = stage if stage in metrics else "retrieved"
    group = "evidence" if "evidence" in metrics.get(shown, {}) else "no_evidence"
    return shown, group, metrics.get(shown, {}).get(group, {})


def best(summaries, stage="reranked", metric=COLUMNS[0]):
    """The summary of each method and set-up whose parameters scored highest on `metric`."""
    chosen = {}
    for summary in summaries:
        key = (describe_method(summary["config"]["index_config"]), summary["setup"])
        value = _values(summary, stage)[2].get(metric, float("-inf"))
        if key not in chosen or value > chosen[key][0]:
            chosen[key] = (value, summary)
    return [summary for _, summary in chosen.values()]


def _format(value):
    return "–" if value is None else f"{value:.3f}"


def cost(summary, stage="reranked"):
    """What a summary's row cost, by the names in `COST_COLUMNS`; a cost not recorded is left out.

    The index's figures are means over the documents that recorded theirs;
    embedding is counted per chunk and multiplied by the index's chunks per
    document. The index's seconds and embedding tokens are left out when the
    set-up's embedding model has no cost recorded, since they would leave its
    embedding out.
    """
    values = {}
    config = summary["config"]
    path = ROOT / "indexes" / summary["dataset"] / summary["index"] / "cost.json"
    built = _read_json(path) if path.exists() else {}
    stages = [built[name] for name in ("chunking", "augmentation") if name in built]
    if stages:
        def per_document(name):
            return sum(stage.get(name, 0) / stage["documents"] for stage in stages)
        values["LLM tok/doc"] = per_document("llm_input_tokens") + per_document("llm_output_tokens")
        embedded = built.get("embedding", {}).get(config["embedding_model"])
        if embedded or not config["embedding_model"]:
            values["index s/doc"] = per_document("seconds")
            values["embed tok/doc"] = per_document("embedding_tokens")
        if embedded:
            chunks_per_document = config["chunks"] / config["documents"]
            values["index s/doc"] += embedded["seconds"] / embedded["chunks"] * chunks_per_document
            values["embed tok/doc"] += (embedded.get("embedding_tokens", 0) / embedded["chunks"]
                                        * chunks_per_document)
    queried = summary.get("cost") or {}
    if "retrieval_seconds" in queried:
        values["query s"] = queried["retrieval_seconds"]
        if _values(summary, stage)[0] == "reranked":
            values["query s"] += queried.get("rerank_seconds", 0)
    return values


def format_cost(name, value):
    """A cost as shown: tokens whole, seconds to three significant figures."""
    if value is None:
        return "–"
    return f"{value:,.0f}" if "tok" in name else f"{value:,.3g}"


def table(summaries, stage="reranked", columns=COLUMNS, costs=False):
    """A Markdown table of `summaries`, one dataset's, at `stage` (falling back to `retrieved`).

    `costs` adds the `COST_COLUMNS`.
    """
    shown_costs = COST_COLUMNS if costs else ()
    header = ["method", "params", "set-up", "stage", "docs", "items", "chunk chars", *columns,
              *shown_costs]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 4 + ["---:"] * (len(header) - 4)) + "|"]
    for summary in sorted(summaries, key=lambda s: (describe_method(s["config"]["index_config"]),
                                                     s["setup"],
                                                     describe_params(s["config"]["index_config"]))):
        shown, group, values = _values(summary, stage)
        config = summary["config"]
        row = [describe_method(config["index_config"]), describe_params(config["index_config"]),
               summary["setup"], shown, str(config["documents"]), str(summary["items"][group]),
               f"{summary['mean_chunk_chars']:,.0f}",
               *(_format(values.get(name)) for name in columns),
               *(format_cost(name, cost(summary, stage).get(name)) for name in shown_costs)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="datasets to show; all by default")
    parser.add_argument("--stage", default="reranked", choices=("retrieved", "reranked"),
                        help="the ranking scored; a set-up without a reranker shows 'retrieved'")
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
            summaries = best(summaries, arguments.stage, arguments.columns[0])
        print(f"### {dataset}\n\n{table(summaries, arguments.stage, arguments.columns, arguments.cost)}\n")
        if len({summary["config"]["documents"] for summary in summaries}) > 1:
            print("Rows were scored on indexes over different documents, so on different "
                  "questions; compare only rows with the same `docs`.\n")


if __name__ == "__main__":
    main()
