"""Fixed-size chunking: windows of a fixed length, with a fixed overlap.

This is the naive baseline every other method is measured against: nothing
about the document's structure is consulted, so a cut lands mid-sentence. Two
units are offered, characters (`fixed_char`) and tokens (`fixed_token`).

LangChain's CharacterTextSplitter with an empty separator does the same thing,
but it splits a 460k-character book into 460k single-character strings and merges
them back, which costs about 40x more than slicing the windows directly. Slicing
is what this method *is*, so there is no library knowledge to inherit here.

`fixed_token` sizes by tokens, the unit an embedding model's input is measured
in, so it is the fixed-size baseline to set beside `sentence` at the same
`max_tokens`. LlamaIndex's TokenTextSplitter does it: it splits on spaces and
packs words into windows of at most `max_tokens` tokens, so a cut never falls
inside a word, and it reports offsets that `split_spans` checks. Tokens are
tiktoken's cl100k_base, the unit every chunker here budgets in. At 256 tokens
it fills a window to 253 on average (GutenQA, SQuAD; never over) and cuts a
630,000-character book in about a second. LangChain's TokenTextSplitter cuts
on exact token boundaries instead, mid-word, but hands back strings with no
offsets, which would have to be searched for in the document.
"""

from llama_index.core.node_parser import TokenTextSplitter

from .base import resolve_overlap, split_spans, to_chunks
from .segment import trim


def fixed_char(document, size=1000, overlap=200, doc_id=""):
    """Cut `document` every `size` characters, repeating `overlap` characters.

    An `overlap` below 1 is a share of `size`.

    Whitespace at a window's edges is trimmed off, so a chunk never opens on a
    blank line and is at most `size` characters long rather than exactly.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1 character, got {size}")
    overlap = resolve_overlap(overlap, size)
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got {overlap} with size {size}")

    spans = []
    start = 0
    while start < len(document):
        end = min(start + size, len(document))
        span = trim(document, start, end)
        # Trimming can empty a window of pure whitespace, or push its start past
        # the start of the next one; either way the chunk adds nothing.
        if span and (not spans or span[1] > spans[-1][1]):
            spans.append(span)
        if end == len(document):
            break
        start += size - overlap
    return to_chunks(document, spans, doc_id)


def fixed_token(document, max_tokens=1024, overlap=200, doc_id=""):
    """Cut `document` into windows of at most `max_tokens` tokens, on spaces.

    `overlap` is in tokens, and is how much of a window the next one repeats;
    below 1, it is a share of `max_tokens`.
    """
    overlap = resolve_overlap(overlap, max_tokens)
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"overlap must be in [0, max_tokens), got {overlap} with "
                         f"max_tokens {max_tokens}")
    parser = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=overlap)
    return to_chunks(document, split_spans(document, parser), doc_id)
