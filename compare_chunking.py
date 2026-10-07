"""Chunking comparison: each chunking method at its best, with the rest of the pipeline fixed.

`compare.py` tabulates every evaluation there is. This script reads only those
of the baseline pipeline the chunking experiments run with (see the README,
"Comparison of methods, one stage at a time") -- no augmentation, bge-m3,
dense retrieval of 50 chunks, no reranking -- so that the chunking method is
the only thing that differs between two rows. Results of other set-ups, such
as the first chunking runs scored with hybrid retrieval, are left out.

    .venv\\Scripts\\python.exe compare_chunking.py                 # every dataset
    .venv\\Scripts\\python.exe compare_chunking.py squad qasper    # some
    .venv\\Scripts\\python.exe compare_chunking.py --all           # every parameter combination

It prints two things:

- per dataset, one row per method (rendering and method: `sentence` over
  Markdown is not `sentence` over plain text) at the parameters that scored
  best on `--metric`, best method first;
- one overview, the methods by the datasets, with each method's mean
  difference to `fixed_token` on the datasets it ran on. `fixed_token` is the
  baseline's method and runs on every dataset, so the difference compares a
  method that ran on five datasets with one that ran on eight, which a mean
  score or rank would not.

The figures are those of the `evidence` group of questions. A dataset with no
evidence spans (LiteraryQA) has none and is left out. Every metric favours
large chunks, so read them against the `chunk chars` column. Two rows of a
dataset are comparable only if they were scored on the same questions: a
dataset whose rows differ in `docs` or `items` is flagged under its table.

`--judge MODEL` adds RAGAS's context precision and recall at `--k`, where
`evaluate_ragas.py` has judged a set-up with that model; a set-up it has not
shows "–". They are means over its sample (200 questions by default, the
`judged` column), not over `items`, and can be picked as `--metric` too.

`--cost` adds what each row cost to build and to query, as in `compare.py`.
"""

import argparse
import json
import sys
from statistics import mean

from compare import (COST_COLUMNS, ROOT, collect, cost, describe_method, describe_params,
                     format_cost)

# The pipeline every stage but chunking is fixed at. A summary's config must
# match it on each key to be read.
BASELINE = {"retrieval": "dense", "embedding_model": "bge-m3", "top_k": 50,
            "rerank_model": None}

# The method every other is measured against in the overview.
REFERENCE = "txt, fixed_token"

# Shown per dataset; `--metric` picks the best parameters and orders the rows.
COLUMNS = ("hit@5", "mrr@10", "ndcg@10", "evidence_recall@5", "evidence_recall@10")


def is_baseline(summary):
    """Whether a summary was scored with the baseline pipeline around its chunking."""
    config = summary["config"]
    return (all(config.get(key) == value for key, value in BASELINE.items())
            and not config["index_config"]["augmentations"])


def metrics(summary):
    """The retrieved ranking's metrics over the questions with evidence; empty if there are none.

    RAGAS's, when `attach_ragas` found them, are among them.
    """
    spans = summary["metrics"].get("retrieved", {}).get("evidence", {})
    return {**spans, **summary.get("ragas", {})} if spans else {}


def attach_ragas(summary, judge_model, k):
    """Add to `summary` the RAGAS metrics `evaluate_ragas.py` wrote for it with `judge_model`."""
    # Imported here: ragas is slow to load, and only this option needs it.
    from evaluate_ragas import judge_directory
    results_directory = ROOT / "results" / summary["dataset"] / summary["index"] / summary["setup"]
    path = judge_directory(results_directory, judge_model, k) / "summary.json"
    if path.exists():
        judged = _read_json(path)
        summary["ragas"] = {name: judged[name] for name in
                            (f"context_precision@{k}", f"context_recall@{k}")}
        summary["judged"] = judged["items"] - judged["failed"]


def best(summaries, metric):
    """Each method's summary whose parameters scored highest on `metric`."""
    chosen = {}
    for summary in summaries:
        method = describe_method(summary["config"]["index_config"])
        value = metrics(summary).get(metric, float("-inf"))
        if method not in chosen or value > chosen[method][0]:
            chosen[method] = (value, summary)
    return {method: summary for method, (_, summary) in chosen.items()}


def _read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _format(value):
    return "–" if value is None else f"{value:.3f}"


def dataset_table(summaries, metric, columns, costs=False):
    """A Markdown table of one dataset's summaries, best on `metric` first.

    `costs` adds the `COST_COLUMNS`.
    """
    columns = [metric, *(name for name in columns if name != metric)]
    shown_costs = COST_COLUMNS if costs else ()
    judged = any("judged" in summary for summary in summaries)
    header = ["method", "params", "docs", "items", *(["judged"] if judged else []), "chunks",
              "chunk chars", *columns, *shown_costs]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|"]
    ordered = sorted(summaries, key=lambda summary: (-metrics(summary).get(metric, float("-inf")),
                                                     describe_method(summary["config"]["index_config"])))
    for summary in ordered:
        config = summary["config"]
        values = metrics(summary)
        row = [describe_method(config["index_config"]), describe_params(config["index_config"]),
               str(config["documents"]), str(summary["items"]["evidence"]),
               *([str(summary.get("judged", "–"))] if judged else []),
               f"{config['chunks']:,}", f"{summary['mean_chunk_chars']:,.0f}",
               *(_format(values.get(name)) for name in columns),
               *(format_cost(name, cost(summary, "retrieved").get(name))
                 for name in shown_costs)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def overview_table(chosen, metric):
    """The methods by the datasets on `metric`, with the mean difference to `REFERENCE`.

    `chosen` maps a dataset to its methods' best summaries.
    """
    datasets = sorted(chosen)
    methods = sorted({method for by_method in chosen.values() for method in by_method})
    header = ["method", *datasets, "datasets", f"mean difference to {REFERENCE}"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|" + "|".join(["---:"] * (len(header) - 1)) + "|"]
    for method in methods:
        values, differences = [], []
        for dataset in datasets:
            summary = chosen[dataset].get(method)
            value = metrics(summary).get(metric) if summary else None
            values.append(value)
            reference = chosen[dataset].get(REFERENCE)
            if value is not None and reference and metrics(reference).get(metric) is not None:
                differences.append(value - metrics(reference)[metric])
        difference = f"{mean(differences):+.3f}" if differences else "–"
        lines.append("| " + " | ".join([method, *map(_format, values),
                                        str(sum(value is not None for value in values)),
                                        difference]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="datasets to show; all by default")
    parser.add_argument("--metric", default="evidence_recall@5",
                        help="the metric that picks each method's best parameters, orders the "
                             "rows and fills the overview, as named in summary.json")
    parser.add_argument("--columns", nargs="+", default=list(COLUMNS),
                        help="metrics to show per dataset, beside --metric")
    parser.add_argument("--judge", help="add RAGAS context precision and recall, as judged by "
                                         "this model with evaluate_ragas.py")
    parser.add_argument("--k", type=int, default=5, help="the cut-off the RAGAS metrics were "
                                                         "judged at")
    parser.add_argument("--all", action="store_true",
                        help="every parameter combination per dataset, not only each "
                             "method's best; the overview still uses the best")
    parser.add_argument("--cost", action="store_true",
                        help="add what each row cost to build and to query")
    arguments = parser.parse_args()
    # The Windows console defaults to cp1252, which has no "–" for a missing value.
    sys.stdout.reconfigure(encoding="utf-8")
    if arguments.judge:
        arguments.columns = [*arguments.columns, f"context_precision@{arguments.k}",
                             f"context_recall@{arguments.k}"]

    chosen = {}
    for dataset, summaries in collect().items():
        if arguments.datasets and dataset not in arguments.datasets:
            continue
        summaries = [summary for summary in summaries
                     if is_baseline(summary) and metrics(summary)]
        if arguments.judge:
            for summary in summaries:
                attach_ragas(summary, arguments.judge, arguments.k)
        if not summaries:
            continue
        chosen[dataset] = best(summaries, arguments.metric)
        shown = summaries if arguments.all else list(chosen[dataset].values())
        print(f"### {dataset}\n\n{dataset_table(shown, arguments.metric, arguments.columns, arguments.cost)}\n")
        if len({(summary["config"]["documents"], summary["items"]["evidence"])
                for summary in shown}) > 1:
            print("Rows were scored on different documents or questions; compare only rows "
                  "with the same `docs` and `items`.\n")
    if not chosen:
        raise SystemExit("No result of the baseline pipeline yet: run an experiment file "
                         "under experiments/ first.")
    print(f"### All datasets: {arguments.metric}, each method at its best\n\n"
          f"{overview_table(chosen, arguments.metric)}")


if __name__ == "__main__":
    main()
