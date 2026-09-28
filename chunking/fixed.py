"""Fixed-size chunking: blind windows of characters, with a fixed overlap.

This is the naive baseline every other method is measured against. The cut lands
mid-word and mid-sentence because nothing about the document is consulted.

LangChain's CharacterTextSplitter with an empty separator does the same thing,
but it splits a 460k-character book into 460k single-character strings and merges
them back, which costs about 40x more than slicing the windows directly. Slicing
is what this method *is*, so there is no library knowledge to inherit here.

Sizing by tokens rather than characters is the other half of this method; it
waits on the embedding model, whose tokenizer decides what a token is
(LangChain's TokenTextSplitter is the implementation to use for it).
"""

from .base import to_chunks
from .segment import trim


def fixed_char(document, size=1000, overlap=200, doc_id=""):
    """Cut `document` every `size` characters, repeating `overlap` characters.

    Whitespace at a window's edges is trimmed off, so a chunk never opens on a
    blank line and is at most `size` characters long rather than exactly.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1 character, got {size}")
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
