"""Structure-preserving chunking: cuts that land on boundaries the text already has.

Where `fixed` cuts blind, this cuts where the document itself says one thing
ends and another begins, under an upper bound on chunk length -- the budget an
embedding model has to fit.

LlamaIndex's SentenceSplitter is a ladder, not just a sentence splitter: it
divides on paragraphs first, then on sentences inside a paragraph that is still
too long, then on clause punctuation, and finally on spaces. That is the same
descent a recursive character splitter makes, with sentences as a rung, so the
one parser covers both methods the README lists.

Sizing is in tokens, counted by LlamaIndex's default tokenizer (tiktoken,
cl100k_base). That is the unit an embedding model's context is actually measured
in; chunk *character* lengths therefore vary with how the text tokenizes, and
Polish costs noticeably more tokens per character than English.
"""

from llama_index.core.node_parser import SentenceSplitter

from .base import resolve_overlap, split_spans, to_chunks
from .segment import detect_language, sentence_pieces

# Our corpora separate paragraphs with a blank line; LlamaIndex defaults to
# three newlines, which never matches here and would skip the paragraph rung.
PARAGRAPH_SEPARATOR = "\n\n"


def sentence(document, max_tokens=1024, overlap=200, language="auto", doc_id=""):
    """Chunk `document` on sentence boundaries, at most `max_tokens` tokens.

    `overlap` is in tokens, or below 1 a share of `max_tokens`. `language` is
    a pysbd language code, or "auto" to detect it from the document -- the
    corpora here are English except PoQuAD, which is Polish ("pl"). pysbd is
    used in place of LlamaIndex's NLTK default, which only knows English: on
    256-token chunks the two end 12% of PoQuAD's chunks differently and 43% of
    GutenQA's, where NLTK cuts inside quoted dialogue.
    """
    # Detected once for the document: the splitter is handed passages a few
    # words long, too little to tell a language from.
    if language == "auto":
        language = detect_language(document)
    parser = SentenceSplitter(
        chunk_size=max_tokens,
        chunk_overlap=resolve_overlap(overlap, max_tokens),
        paragraph_separator=PARAGRAPH_SEPARATOR,
        chunking_tokenizer_fn=sentence_pieces(language),
    )
    return to_chunks(document, split_spans(document, parser), doc_id)
