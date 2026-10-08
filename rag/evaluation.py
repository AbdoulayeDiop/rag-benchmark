"""Evaluation: how well a ranking of chunks finds a question's evidence.

The metrics compare the retrieved chunks' character spans with the evidence
spans a dataset records, so they call no model, cost nothing beyond the
retrieval itself, and come out the same every time:

    hit@k            a chunk among the first k overlaps an evidence span
    mrr@10           1 / rank of the first such chunk, 0 beyond 10
    ndcg@k           binary relevance (the chunk overlaps evidence); the ideal
                     ranking puts every chunk of the index that does first
    evidence_recall@k  share of the evidence characters the first k chunks
                     cover, overlapping spans and chunks counted once
    doc_hit@k        a chunk among the first k is from an evidence document

All of them favour large chunks -- a method that cuts a book into four pieces
hits the evidence at k = 1 by covering a quarter of the book -- so they are
read against the chunks' length.

A ranking is a list of `[doc_id, start, end, ...]` entries, best first; a
span is `(doc_id, start, end)`. A question without spans gets only `doc_hit`.
"""

import math

# The cut-offs every metric is reported at.
KS = (1, 3, 5, 10)


def _merge(intervals):
    """Union of `(start, end)` intervals, as sorted disjoint intervals."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _intersection(first, second):
    """Characters two sets of sorted disjoint intervals have in common."""
    total = i = j = 0
    while i < len(first) and j < len(second):
        total += max(0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return total


def _intervals(spans):
    """`[doc_id, start, end, ...]` spans as each document's merged intervals."""
    by_document = {}
    for span in spans:
        by_document.setdefault(span[0], []).append((span[1], span[2]))
    return {doc_id: _merge(intervals) for doc_id, intervals in by_document.items()}


def _length(intervals_by_document):
    """Characters in a document-to-intervals mapping."""
    return sum(end - start for intervals in intervals_by_document.values()
               for start, end in intervals)


def _overlaps(chunk, spans):
    doc_id, start, end = chunk[:3]
    return any(doc_id == span_doc and start < span_end and span_start < end
               for span_doc, span_start, span_end in spans)


def relevant_chunks(spans, chunks_by_document):
    """How many chunks of the index overlap an evidence span: the ideal ranking's length.

    `chunks_by_document` maps a doc_id to its `Chunk`s.
    """
    return sum(_overlaps((doc_id, chunk.start, chunk.end), spans)
               for doc_id in {span[0] for span in spans}
               for chunk in chunks_by_document.get(doc_id, ()))


def score(ranking, spans, documents, relevant, ks=KS):
    """The metrics of one ranking for one question.

    `spans` are the question's evidence spans, `documents` the ids of the
    documents its answer is in, and `relevant` the chunks of the index that
    overlap the spans (`relevant_chunks`).
    """
    spans = [tuple(span) for span in spans]
    documents = set(documents)
    metrics = {}
    for k in ks:
        top = ranking[:k]
        metrics[f"doc_hit@{k}"] = float(any(chunk[0] in documents for chunk in top))
    if not spans:
        return metrics
    hits = [_overlaps(chunk, spans) for chunk in ranking]
    first = next((rank for rank, hit in enumerate(hits[:10], start=1) if hit), None)
    metrics["mrr@10"] = 1 / first if first else 0.0
    # Evidence spans overlap (SQuAD's annotators often mark the same answer),
    # and so do chunks cut with an overlap: both sides are counted as their
    # union, per document, so no character counts twice.
    evidence = _intervals(spans)
    evidence_chars = _length(evidence)
    for k in ks:
        metrics[f"hit@{k}"] = float(any(hits[:k]))
        dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits[:k], start=1) if hit)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, relevant) + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
        retrieved = _intervals(ranking[:k])
        covered = sum(_intersection(intervals, retrieved.get(doc, []))
                      for doc, intervals in evidence.items())
        metrics[f"evidence_recall@{k}"] = covered / evidence_chars if evidence_chars else 0.0
    return metrics


def mean_scores(scores):
    """The mean of each metric over a list of `score` results with the same keys."""
    return {name: sum(s[name] for s in scores) / len(scores) for name in scores[0]} if scores else {}
