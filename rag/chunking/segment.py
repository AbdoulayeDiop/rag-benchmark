"""Segmenting a document into the units that chunkers group.

Every function here returns character spans into the document it was given,
never substrings, so the offsets survive all the way to evaluation.

Sentences come from pysbd, a rule-based segmenter with no model to download and
explicit support for both languages in the corpora (English, and Polish for
PoQuAD). Its rules are per language, so it has to be told which one; by
default (`language="auto"`) lingua identifies it from the document.

It is run one paragraph at a time: a sentence never crosses a blank line, and feeding it a whole 460k-character book at once is markedly slower than
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
from lingua import IsoCode639_1, LanguageDetectorBuilder
from pysbd.languages import LANGUAGE_CODES

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


@lru_cache(maxsize=None)
def _language_detector():
    """Build the lingua detector once; it loads a model per candidate language.

    The candidates are the languages pysbd has rules for and lingua can
    recognise -- all of pysbd's but Amharic and Burmese. Naming a language the
    segmenter cannot use would gain nothing, and fewer candidates means fewer
    ways to be wrong: a Czech document comes back as Slovak or Polish, whose
    rules are the nearest there are.
    """
    codes = [getattr(IsoCode639_1, code.upper()) for code in LANGUAGE_CODES
             if hasattr(IsoCode639_1, code.upper())]
    return LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()


def detect_language(text, sample=3000):
    """Return the pysbd code of the language `text` is written in.

    Only the first `sample` characters are read: a document is in one language
    here, and that much is enough to tell. Falls back to "en" when nothing is
    recognised -- a document of digits, markup or a single name.
    """
    language = _language_detector().detect_language_of(text[:sample])
    return language.iso_code_639_1.name.lower() if language else "en"


def sentence_spans(text, language="auto"):
    """Split `text` into sentences, returning their spans.

    `language` is a pysbd code, or "auto", the default, to have
    `detect_language` choose it from the text.

    pysbd's spans absorb the whitespace that follows a sentence; it is trimmed
    off here so that a chunk starting at a sentence starts at its first
    character. Anything the segmenter drops between sentences stays in the
    document and is recovered by any chunk that spans across it.
    """
    if language == "auto":
        language = detect_language(text)
    engine = segmenter(language)
    spans = []
    for start, end in paragraph_spans(text):
        paragraph = text[start:end]
        for sentence in engine.segment(paragraph):
            span = trim(text, start + sentence.start, start + sentence.end)
            if span:
                spans.append(span)
    return spans


def sentence_pieces(language="auto"):
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
