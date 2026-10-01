"""Semantic chunking: cut where the meaning changes, not where the markup does.

Two methods, both starting from embedded sentences and neither bounded in size:
a chunk is as long as its subject lasts.

`semantic` is the breakpoint method. Each sentence is embedded with its
neighbours and a boundary is placed wherever the similarity between consecutive
groups drops far enough -- `breakpoint_percentile` says how far, as a percentile
of the drops seen in that document. It is a local decision: each gap is judged
on its own. LlamaIndex's SemanticSplitterNodeParser implements it.

`clustered` decides globally instead. Agglomerative clustering merges the most
similar pair of groups repeatedly, but under a connectivity constraint that only
lets a group merge with the one beside it, so a cluster is always a contiguous
run of sentences and the document keeps its order. The constraint is what makes
clustering usable for chunking at all: unconstrained, it would gather every
mention of a subject scattered through the document into one cluster, which is
not a passage anyone can retrieve.

The embedding model is injected rather than fixed in both, because it is the
expensive part and the one a result depends on most: the same document chunked
with two encoders is two different experiments.

Embeddings come from an OpenAI-compatible endpoint. `openai_embedding` reads the
usual environment (OPENAI_API_KEY, OPENAI_API_BASE) and takes a `proxy`, since
the endpoint is not always reachable directly.
"""

import os

import httpx
import numpy as np
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.utils import get_tokenizer
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering

from .base import split_spans, to_chunks
from .segment import sentence_pieces, sentence_spans


def openai_embedding(model, api_key=None, api_base=None,
                     proxy=None, batch_size=64, timeout=60.0, **kwargs):
    """Build an embedding client for any OpenAI-compatible endpoint.

    OpenAILikeEmbedding is used rather than OpenAIEmbedding because the latter
    validates the model name against an enum of OpenAI's own models and refuses
    anything else. `model` is the name the endpoint serves the encoder under.
    Choose a multilingual one for these corpora: PoQuAD is Polish, and an
    English-only encoder would put its breakpoints in the wrong places. The
    experiments here use bge-m3.

    `batch_size` is how many sentences go up per request. 64 is the largest the
    endpoint used here accepts -- it answers 413 rather than splitting the batch
    itself -- and a safe value elsewhere; raise it where the endpoint allows.

    `api_key` and `api_base` default to OPENAI_API_KEY and OPENAI_API_BASE.
    `proxy` is an http(s) proxy URL; it is applied by handing the client its own
    httpx transport, which is the only way through to the underlying SDK.
    Anything else -- `dimensions`, `max_retries` -- passes straight through.
    """
    client = httpx.Client(proxy=proxy, timeout=timeout) if proxy else None
    return OpenAILikeEmbedding(
        model_name=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        api_base=api_base or os.environ.get("OPENAI_API_BASE"),
        embed_batch_size=batch_size,
        timeout=timeout,
        http_client=client,
        **kwargs,
    )


def semantic(document, embed_model, breakpoint_percentile=95, buffer_size=1,
             language="auto", doc_id=""):
    """Chunk `document` where consecutive sentences stop resembling each other.

    `breakpoint_percentile` is the cut threshold: 95 breaks at the largest 5% of
    similarity drops, so a lower number means more, smaller chunks.
    `buffer_size` is how many neighbouring sentences are embedded together --
    1 compares single sentences, which is noisier than grouping them.

    `embed_model` is a LlamaIndex embedding model, such as one built by
    `openai_embedding`; every sentence of the document is sent to it. `language` is a pysbd code, used
    for the sentence segmentation the method starts from.
    """
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=breakpoint_percentile,
        buffer_size=buffer_size,
        sentence_splitter=sentence_pieces(language),
    )
    return to_chunks(document, split_spans(document, parser), doc_id)


def _window(document, sentences, position, buffer):
    """Text of the sentence at `position` plus its `buffer` neighbours each side."""
    first_sentence = max(0, position - buffer)
    last_sentence = min(len(sentences) - 1, position + buffer)
    return document[sentences[first_sentence][0]:sentences[last_sentence][1]]


def _extent(node, merge_children, sentence_count, sentences):
    """Character span covered by one node of the merge tree.

    Merges are constrained to neighbours, so a node always covers a contiguous
    run of sentences, and its extent is its first sentence's start to its last
    sentence's end. Finding those means walking down to the node's leaves.
    """
    to_visit, sentences_below = [node], []
    while to_visit:
        current = to_visit.pop()
        if current < sentence_count:
            sentences_below.append(current)
        else:
            to_visit.extend(merge_children[current - sentence_count])
    return (sentences[min(sentences_below)][0],
            sentences[max(sentences_below)][1])


def clustered(document, embed_model, percentile=95, threshold=None, max_tokens=1024,
              buffer=1, language="auto", doc_id=""):
    """Chunk `document` by clustering its sentences, adjacent ones only.

    Each sentence is embedded together with its `buffer` neighbours on either
    side, so a short line is judged in context rather than on its own; the chunk
    boundaries still fall between sentences.

    Clustering merges the closest pair of neighbouring groups over and over,
    building a tree. The chunks are read off that tree from the root down: a
    node becomes a chunk when it holds at most `max_tokens` tokens and was
    formed by a merge closer than the distance cut; otherwise its two halves are
    examined in turn. So a chunk is the largest coherent group that still fits.

    Tokens are counted with the same tokenizer the other methods size by, so a
    budget means the same thing across all of them -- which characters would
    not, since Polish costs about twice as many tokens per character as English.

    The cut is `percentile` of the distances between neighbouring sentences in
    this document -- 95 keeps only the outlying gaps as boundaries, which is
    what marks a real change of subject. `threshold` fixes that distance
    absolutely instead, but it is not portable: bge-m3 puts neighbours 0.47
    apart on average, so the usable band is narrow and moves with the encoder.
    """
    sentences = sentence_spans(document, language)
    sentence_count = len(sentences)
    if sentence_count < 2:
        return to_chunks(document, sentences, doc_id)

    sentence_vectors = np.array(embed_model.get_text_embedding_batch(
        [_window(document, sentences, position, buffer) for position in range(sentence_count)]))

    if threshold is None:
        unit_vectors = sentence_vectors / np.linalg.norm(sentence_vectors, axis=1, keepdims=True)
        neighbour_distances = 1 - np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        threshold = float(np.percentile(neighbour_distances, percentile))

    # The connectivity graph is the chain 0-1-2-...-n: a group may only ever
    # merge with the one beside it, so every node covers a contiguous run and
    # the document keeps its order. Clustering runs all the way to a single
    # root, because the whole tree is what the chunks are read from.
    neighbour_graph = diags([np.ones(sentence_count - 1), np.ones(sentence_count - 1)],
                            offsets=[-1, 1], format="csr")
    tree = AgglomerativeClustering(
        n_clusters=1,
        metric="cosine",
        linkage="average",
        connectivity=neighbour_graph,
        compute_distances=True,
    ).fit(sentence_vectors)

    # sklearn numbers the tree in one flat sequence: 0 .. sentence_count-1 are
    # the sentences, and the node formed by merge i is numbered
    # sentence_count + i. Subtracting sentence_count from a node number is
    # therefore how to look its merge up in children_ and distances_.
    merge_childrens, merge_distances = tree.children_, tree.distances_
    root = sentence_count + len(merge_childrens) - 1
    count_tokens = get_tokenizer()

    chunk_spans, to_visit = [], [root]
    while to_visit:
        node = to_visit.pop()
        if node < sentence_count:
            chunk_spans.append(sentences[node])
            continue

        merge = node - sentence_count
        start, end = _extent(node, merge_childrens, sentence_count, sentences)
        fits_budget = len(count_tokens(document[start:end])) <= max_tokens
        # How far apart the two halves were when they were joined: a merge
        # above the cut reached across a change of subject, so the node is not
        # one coherent passage however well it fits.
        is_coherent = merge_distances[merge] <= threshold

        if fits_budget and is_coherent:
            chunk_spans.append((start, end))
        else:
            to_visit.extend(merge_childrens[merge])

    # The two halves of a merge are not stored in positional order, so the walk
    # above yields nodes in tree order; sorting puts the chunks back in
    # document order.
    return to_chunks(document, sorted(chunk_spans), doc_id)


def _blocks(document, sentences, block):
    """Text of the run of `block` sentences starting at each position.

    One list serves both sides of every comparison: the block starting at
    position s is the *following* block for s, and the *preceding* block for
    s + block. So a document costs one embedding per sentence, not two.
    """
    count = len(sentences)
    return [document[sentences[start][0]:sentences[min(count - 1, start + block - 1)][1]]
            for start in range(count)]


def tiled(document, embed_model, block=3, percentile=90, neighbourhood=2,
          language="auto", doc_id=""):
    """Chunk `document` where a block of sentences stops resembling the next block.

    Rather than comparing one sentence with the next, each candidate boundary is
    judged by what surrounds it: the `block` sentences before it, against the
    `block` sentences from it onwards. A single odd sentence therefore cannot
    open a chunk on its own -- it has to change what follows it -- which is the
    weakness of the consecutive-sentence comparison in `semantic`.

    A position becomes a boundary when its dissimilarity clears the `percentile`
    of all the dissimilarities in the document *and* is the largest within
    `neighbourhood` positions either side. The threshold alone would mark every
    position across a long transition; the local maximum alone would mark the
    best position in every flat stretch, however unremarkable. Set
    `neighbourhood` to 0 to keep only the threshold.

    This is the shape of Hearst's TextTiling, with embeddings in place of its
    word-overlap score.

    A block is a number of sentences, not a number of tokens, although equal
    token lengths would be the fairer comparison: three bullet points are a
    quarter the length of three prose sentences, and the dissimilarity spikes
    where a list begins for that reason alone. Equalising by tokens was tried
    and does remove those spikes -- on gov.uk pages it lifted the share of
    boundaries landing on a real heading from 43% to 60% -- but it reads worse
    on prose (Qasper 22% against 27%), costs two embeddings per position instead
    of one, since neither side's text is reusable, and widens the run of
    positions at each end that cannot be judged at all.
    """
    sentences = sentence_spans(document, language)
    count = len(sentences)
    # Every comparison needs a full block on each side, so a document shorter
    # than two blocks has no position that can be judged at all.
    if count <= 2 * block:
        return to_chunks(document, [(sentences[0][0], sentences[-1][1])] if sentences else [], doc_id)

    block_vectors = np.array(embed_model.get_text_embedding_batch(
        _blocks(document, sentences, block)))
    block_vectors /= np.linalg.norm(block_vectors, axis=1, keepdims=True)

    # A candidate at position s compares sentences [s-block, s-1] with
    # [s, s+block-1]. Positions past count-block are left out: their following
    # block would be truncated, which inflates the dissimilarity for a reason
    # that has nothing to do with the text.
    candidates = list(range(block, count - block + 1))
    dissimilarity = {position: 1 - float(block_vectors[position - block] @ block_vectors[position])
                     for position in candidates}

    threshold = float(np.percentile(list(dissimilarity.values()), percentile))
    boundaries = []
    for position in candidates:
        if dissimilarity[position] < threshold:
            continue
        nearby = [dissimilarity[other] for other in candidates
                  if abs(other - position) <= neighbourhood]
        if dissimilarity[position] >= max(nearby):
            boundaries.append(position)

    spans, start = [], 0
    for boundary in boundaries + [count]:
        if boundary > start:
            spans.append((sentences[start][0], sentences[boundary - 1][1]))
            start = boundary
    return to_chunks(document, spans, doc_id)
