"""What every chunking method returns, and how a library's output is turned into it.

A chunk is a *character span* of a document, not a detached string. The dataset
loaders record evidence as character offsets into the document (see README,
"Dataset processing"), so a retrieved chunk can only be scored by span overlap if
it knows where it came from:

    document[chunk.start:chunk.end] == chunk.original_text

LlamaIndex's node parsers carry those offsets natively, which is why they are
the library used here: `split_spans` reads them off each node and checks them,
instead of searching the source for a string a splitter handed back.
"""

from dataclasses import dataclass, field, replace

from bs4 import BeautifulSoup
from llama_index.core import Document

# What goes between the parts of `text_to_embed`. A blank line is what
# Anthropic's contextual retrieval uses, and it reads as a paragraph break to
# an encoder and to BM25 alike.
SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A span of one document, with the text it covers.

    `metadata` holds what the splitter knew about the chunk but the text does
    not say -- the heading path above a Markdown section, say, which a chunk cut
    out of the middle of that section no longer carries. It is empty for methods
    that derive nothing beyond the span.

    `original_text` is the span exactly as the document has it, and is what a
    chunk is scored on. `text_to_embed` is what is embedded or indexed instead
    when it differs: an HTML chunk without its tags, or a chunk an augmentation
    method has added to. None means `original_text` is embedded as it is.
    """

    doc_id: str
    index: int
    start: int
    end: int
    original_text: str
    text_to_embed: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def text(self):
        """The span's text: `original_text`, under the name older code uses."""
        return self.original_text

    def __len__(self):
        return self.end - self.start


def build_text_to_embed(original_text, metadata):
    """What a chunk with this text and metadata is embedded and indexed as.

    It is put together from the chunk's own text and whatever the augmentation
    methods stored in `metadata`, always in this order, whatever order they
    were added in -- which is what lets the methods be applied in any order:

        metadata['parent_document_title']
        metadata['parent_document_summary']
        metadata['context']                   the note situating the chunk in its document
        metadata['keywords']
        metadata['questions']                 a list, one per line
        the chunk itself                      original_text, without its tags for HTML

    The exception is `metadata['summary']`, a summary of the chunk itself: a
    chunk that has one is embedded as that summary alone, and the other keys
    are left out.

    A chunk whose `metadata['markup']` is "html" keeps its tags in
    `original_text`, because that is what its offsets cover, but is embedded
    and indexed without them -- the tags are noise to an encoder and to BM25
    alike. A chunk of nothing but tags (a lone <hr>) would strip to an empty
    string, which an embedding endpoint refuses, so it keeps its markup.

    It lives with `Chunk` rather than with the augmentation methods because it
    is the one definition of a chunk's `text_to_embed`: the chunkers use it,
    `augmentation.augment_chunk` uses it after every method, and an index that
    stores chunks without the field uses it to rebuild them.
    """
    if metadata.get("summary"):
        return metadata["summary"]
    text = original_text
    if metadata.get("markup") == "html":
        text = BeautifulSoup(original_text, "html.parser").get_text(" ", strip=True) or original_text
    pieces = [metadata.get("parent_document_title"),
              metadata.get("parent_document_summary"),
              metadata.get("context"),
              metadata.get("keywords"),
              "\n".join(metadata.get("questions") or []),
              text]
    return SEPARATOR.join(piece for piece in pieces if piece)


def update_chunk_metadata(chunk, **metadata):
    """Return `chunk` with `metadata` added and `text_to_embed` rebuilt from it.

    `text_to_embed` is left None when it would equal `original_text`, so that
    None keeps meaning "embedded as the span".
    """
    metadata = {**chunk.metadata, **metadata}
    text = build_text_to_embed(chunk.original_text, metadata)
    return replace(chunk, metadata=metadata,
                   text_to_embed=None if text == chunk.original_text else text)


def get_text_to_embed(chunk):
    """The text to embed or index for `chunk`: `text_to_embed`, or the span itself."""
    return chunk.original_text if chunk.text_to_embed is None else chunk.text_to_embed


def to_chunks(document, spans, doc_id="", metadata=None):
    """Number a sequence of (start, end) spans and attach the text they cover.

    `metadata` is one dict per span, in the same order, or None for no metadata.
    `text_to_embed` is set only where it differs from the span, which before
    any augmentation means a chunk of markup (see `build_text_to_embed`).
    """
    return [
        update_chunk_metadata(Chunk(doc_id=doc_id, index=index, start=start, end=end,
                                 original_text=document[start:end]),
                           **(metadata[index] if metadata else {}))
        for index, (start, end) in enumerate(spans)
    ]


def split_spans(text, parser, offset=0):
    """Run a LlamaIndex node parser over `text`, returning spans plus `offset`.

    LlamaIndex reports `start_char_idx`/`end_char_idx` on every node, so unlike
    LangChain -- whose splitters return bare strings that have to be searched
    for -- the offsets come straight from the parser. They are still checked
    against the text they claim to cover: a parser that rewrites what it splits
    breaks the span contract, and should fail here rather than silently
    mis-score a retrieval later.
    """
    spans = []
    for node in parser.get_nodes_from_documents([Document(text=text)]):
        start, end = node.start_char_idx, node.end_char_idx
        if start is None or text[start:end] != node.text:
            raise ValueError(f"parser returned a node that is not at offset {start} of the source")
        spans.append((offset + start, offset + end))
    return spans


def verify(document, chunks):
    """Raise if `chunks` are not well-formed spans of `document`.

    Checks the offset/text invariant every method must hold, plus the ordering
    and bounds that make span-overlap scoring meaningful.
    """
    previous = None
    for chunk in chunks:
        if not 0 <= chunk.start < chunk.end <= len(document):
            raise ValueError(f"chunk {chunk.index} has out-of-range span ({chunk.start}, {chunk.end})")
        if document[chunk.start:chunk.end] != chunk.original_text:
            raise ValueError(f"chunk {chunk.index} text does not match document[{chunk.start}:{chunk.end}]")
        if previous is not None and chunk.start <= previous.start:
            raise ValueError(f"chunk {chunk.index} does not start after chunk {previous.index}")
        previous = chunk
    return chunks
