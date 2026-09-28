"""Markdown chunking: split on the document's own heading structure.

A Markdown document states its own outline, so the sections it declares are the
chunks -- each one a heading and the body under it, kept whole so a retrieved
chunk still says what it is about. Sections over the budget are split by
`sentence`, which descends paragraphs and sentences inside the section rather
than cutting at an arbitrary offset.

LlamaIndex's MarkdownNodeParser does the outline: it returns the source text
verbatim with exact offsets, and records the heading path it was found under.
LangChain's MarkdownHeaderTextSplitter cannot be used for this at all -- it
re-joins lines with "  \n", so its output is not in the document and has no
offsets to recover.
"""

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from .base import split_spans, to_chunks
from .structure import PARAGRAPH_SEPARATOR
from .segment import sentence_pieces


def markdown(document, max_tokens=1024, language="en", doc_id=""):
    """Chunk a Markdown `document` into the sections its headings declare.

    `max_tokens` caps a chunk in tokens; None leaves sections whole however long
    they are -- a chapter heading in LiteraryQA can own 12,000 characters.
    Headings stay in the chunk text as well as in `metadata['header_path']`, so
    a chunk carries its own context even when it is cut out of a section.
    """
    nodes = MarkdownNodeParser().get_nodes_from_documents([_document(document)])
    sections, metadata = [], []
    for node in nodes:
        if node.start_char_idx is None or document[node.start_char_idx:node.end_char_idx] != node.text:
            raise ValueError("MarkdownNodeParser returned a node that is not in the source")
        sections.append((node.start_char_idx, node.end_char_idx))
        metadata.append(dict(node.metadata))

    if max_tokens is None:
        return to_chunks(document, sections, doc_id, metadata)

    parser = SentenceSplitter(
        chunk_size=max_tokens,
        chunk_overlap=0,
        paragraph_separator=PARAGRAPH_SEPARATOR,
        chunking_tokenizer_fn=sentence_pieces(language),
    )
    spans, spans_metadata = [], []
    for (start, end), meta in zip(sections, metadata):
        # Every section goes through the splitter: only it can say whether the
        # section fits a budget counted in tokens. One that fits comes back
        # whole. Splitting a section cuts its heading off the body that
        # follows, so every piece keeps the heading path the whole section had.
        pieces = split_spans(document[start:end], parser, start)
        spans.extend(pieces)
        spans_metadata.extend([meta] * len(pieces))
    return to_chunks(document, spans, doc_id, spans_metadata)


def _document(text):
    """Wrap text for LlamaIndex without importing Document at every call site."""
    from llama_index.core import Document

    return Document(text=text)
