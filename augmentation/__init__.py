"""Chunk augmentation methods evaluated by the experiments.

Each method takes the chunks of one document and returns them with
`text_to_embed` set; see `base` for what goes into it and why none of it is
written into the chunk's own text. `chunking.get_text_to_embed` reads it.

All of them generate what they add with an LLM, through the plain OpenAI
client. By how much they read:

    add_title       one call per document, on its opening
    add_summary     one pass over the whole document
    add_keywords    one call per chunk, reading the chunk
    add_questions   one call per chunk, reading the chunk
    use_summary     one call per chunk, reading the chunk
    add_context     one call per chunk, reading the document again each time
"""

from .base import augment_chunk, call_llm, call_llm_for_each
from .contextual import add_context
from .keywords import add_keywords
from .questions import add_questions
from .summary import add_summary, summarize_document, use_summary
from .title import add_title, generate_title

__all__ = [
    "add_context",
    "add_keywords",
    "add_questions",
    "add_summary",
    "add_title",
    "augment_chunk",
    "call_llm",
    "call_llm_for_each",
    "generate_title",
    "summarize_document",
    "use_summary",
]
