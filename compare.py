"""Comparison: the evaluated runs of each dataset side by side, as Markdown tables.

Reads every `results/<dataset>/<index>/<set-up>/summary.json` that `evaluate.py`
wrote and prints one table per dataset, a row per index and set-up, so the
chunking methods can be read off against each other and pasted into the
README. The figures are those of the `evidence` group of questions; a dataset
with none (LiteraryQA) falls back to `no_evidence`, which has only the
document-level metrics.

Two rows are comparable only if they were scored on the same questions, which
means indexes over the same documents: the `docs` and `items` columns say
whether they were, and a dataset whose rows differ is flagged under its table.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent

# Metrics shown by default: whether the evidence is found and how high, and
# how much of it against how much text that took.
COLUMNS = ("hit@1", "hit@5", "mrr@10", "ndcg@10", "precision@5", "recall@5", "f1@5", "doc_hit@5")


def _read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def describe_index(config):
    """A short label for an index: rendering, method, parameters, augmentations."""
    chunking = config["chunking"]
    params = ", ".join(f"{key}={value}" for key, value in chunking["params"].items())
    label = f"{config['rendering']}, {chunking['method']}({params})"
    return label + "".join(f" +{name}" for name in config["augmentations"])


def collect(results=ROOT / "results"):
    """Every evaluation summary, with its configuration, grouped by dataset."""
    datasets = {}
    for path in sorted(Path(results).glob("*/*/*/summary.json")):
        summary = _read_json(path)
        summary["config"] = _read_json(path.parent / "config.json")
        datasets.setdefault(summary["dataset"], []).append(summary)
    return datasets


def _format(name, value):
    if value is None:
        return "–"
    return f"{value:.3f}"


def table(summaries, stage="reranked", columns=COLUMNS):
    """A Markdown table of `summaries`, one dataset's, at `stage` (falling back to `retrieved`)."""
    header = ["index", "set-up", "stage", "docs", "items", "chunk chars", *columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
    for summary in sorted(summaries, key=lambda s: (describe_index(s["config"]["index_config"]),
                                                     s["setup"])):
        metrics = summary["metrics"]
        shown = stage if stage in metrics else "retrieved"
        group = "evidence" if "evidence" in metrics.get(shown, {}) else "no_evidence"
        values = metrics.get(shown, {}).get(group, {})
        row = [describe_index(summary["config"]["index_config"]), summary["setup"], shown,
               str(summary["config"]["documents"]), str(summary["items"][group]),
               f"{summary['mean_chunk_chars']:,.0f}",
               *(_format(name, values.get(name)) for name in columns)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="datasets to show; all by default")
    parser.add_argument("--stage", default="reranked", choices=("retrieved", "reranked"),
                        help="the ranking scored; a set-up without a reranker shows 'retrieved'")
    parser.add_argument("--columns", nargs="+", default=list(COLUMNS),
                        help="metrics to show, as named in summary.json (hit@3, recall@10, ...)")
    arguments = parser.parse_args()
    for dataset, summaries in collect().items():
        if arguments.datasets and dataset not in arguments.datasets:
            continue
        print(f"### {dataset}\n\n{table(summaries, arguments.stage, arguments.columns)}\n")
        if len({summary["config"]["documents"] for summary in summaries}) > 1:
            print("Rows were scored on indexes over different documents, so on different "
                  "questions; compare only rows with the same `docs`.\n")


if __name__ == "__main__":
    main()
