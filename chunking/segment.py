"""Segmenting a document into the units that chunkers group.

Every function here returns character spans into the document it was given,
never substrings, so the offsets survive all the way to evaluation.

Sentences come from pysbd, a rule-based segmenter with no model to download and
explicit support for both languages in the corpora (English, and Polish for
PoQuAD). It is run one paragraph at a time: a sentence never crosses a blank
line, and feeding it a whole 460k-character book at once is markedly slower than
feeding it the book's paragraphs.

One pysbd rule shapes the literary corpora in particular: it does not break
inside a double-quoted passage, so a page of quoted dialogue comes back as a
single "sentence" of several thousand characters. Sentence units are therefore
far more uneven than their median suggests (GutenQA: median 85, maximum 4,255
characters), which is worth remembering when a chunker counts sentences rather
than the characters they hold.
"""

import re
from functools import lru_cache

import pysbd

# A blank line separates paragraphs. Trailing spaces on the "blank" line are
# common in the Wikipedia-derived corpora, so they do not disqualify a break.
PARAGRAPH_BREAK = re.compile(r"\n[ \t\r]*\n\s*")


def trim(text, start, end):
    """Shrink a span past whitespace at either edge, or None if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def paragraph_spans(text):
    """Split `text` on blank lines, returning one span per non-empty paragraph."""
    spans = []
    cursor = 0
    for match in PARAGRAPH_BREAK.finditer(text):
        span = trim(text, cursor, match.start())
        if span:
            spans.append(span)
        cursor = match.end()
    span = trim(text, cursor, len(text))
    if span:
        spans.append(span)
    return spans


@lru_cache(maxsize=None)
def segmenter(language):
    """Return a cached pysbd segmenter; constructing one compiles its rule set."""
    return pysbd.Segmenter(language=language, clean=False, char_span=True)


def sentence_spans(text, language="en"):
    """Split `text` into sentences, returning their spans.

    pysbd's spans absorb the whitespace that follows a sentence; it is trimmed
    off here so that a chunk starting at a sentence starts at its first
    character. Anything the segmenter drops between sentences stays in the
    document and is recovered by any chunk that spans across it.
    """
    engine = segmenter(language)
    spans = []
    for start, end in paragraph_spans(text):
        paragraph = text[start:end]
        for sentence in engine.segment(paragraph):
            span = trim(text, start + sentence.start, start + sentence.end)
            if span:
                spans.append(span)
    return spans


def sentence_pieces(language="en"):
    """Return a `text -> list[str]` splitter for LlamaIndex, backed by pysbd.

    The pieces are contiguous: each one runs to the start of the next sentence,
    so joining them rebuilds the document exactly. That matters because
    LlamaIndex locates a node by matching its text against the source -- feeding
    it the trimmed spans `sentence_spans` returns loses the whitespace between
    sentences and leaves every offset unrecoverable (measured: 0 of 23 nodes).
    """
    def split(text):
        spans = sentence_spans(text, language)
        if not spans:
            return [text]
        pieces, cursor = [], 0
        for end in [start for start, _ in spans[1:]] + [len(text)]:
            pieces.append(text[cursor:end])
            cursor = end
        return pieces

    return split
