"""Chunking methods evaluated by the experiments.

Each method takes a document and returns a list of `Chunk`; see `base` for why
chunks are spans rather than strings. The splitting is delegated to LlamaIndex,
whose node parsers report character offsets natively -- except for the two
places where no library will do: slicing fixed-size windows, and reading an HTML
outline without throwing the markup away.
"""

from .base import Chunk, split_spans, to_chunks, verify
from .fixed import fixed_char
from .html import html
from .markdown import markdown
from .semantic import clustered, openai_embedding, semantic, tiled
from .segment import paragraph_spans, sentence_pieces, sentence_spans
from .structure import sentence

__all__ = [
    "Chunk",
    "clustered",
    "fixed_char",
    "html",
    "markdown",
    "paragraph_spans",
    "openai_embedding",
    "semantic",
    "sentence",
    "sentence_pieces",
    "sentence_spans",
    "split_spans",
    "tiled",
    "to_chunks",
    "verify",
]
