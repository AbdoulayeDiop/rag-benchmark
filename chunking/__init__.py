"""Chunking methods evaluated by the experiments.

Each method takes a document and returns a list of `Chunk`; see `base` for why
chunks are spans rather than strings. The splitting is delegated to LlamaIndex,
whose node parsers report character offsets natively -- except for the two
places where no library will do: slicing fixed-size character windows, and reading an HTML
outline without throwing the markup away.
"""

from .base import (Chunk, build_text_to_embed, get_text_to_embed, resolve_overlap,
                   split_spans, to_chunks, update_chunk_metadata, verify)
from .fixed import fixed_char, fixed_token
from .html import html
from .markdown import markdown
from .semantic import clustered, openai_embedding, semantic, tiled
from .segment import detect_language, paragraph_spans, sentence_pieces, sentence_spans
from .structure import sentence

__all__ = [
    "Chunk",
    "build_text_to_embed",
    "clustered",
    "detect_language",
    "get_text_to_embed",
    "fixed_char",
    "fixed_token",
    "html",
    "markdown",
    "paragraph_spans",
    "resolve_overlap",
    "openai_embedding",
    "semantic",
    "sentence",
    "sentence_pieces",
    "sentence_spans",
    "split_spans",
    "tiled",
    "to_chunks",
    "update_chunk_metadata",
    "verify",
]
