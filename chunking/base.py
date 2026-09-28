"""What every chunking method returns, and how a library's output is turned into it.

A chunk is a *character span* of a document, not a detached string. The dataset
loaders record evidence as character offsets into the document (see README,
"Dataset processing"), so a retrieved chunk can only be scored by span overlap if
it knows where it came from:

    document[chunk.start:chunk.end] == chunk.text

LlamaIndex's node parsers carry those offsets natively, which is why they are
the library used here: `split_spans` reads them off each node and checks them,
instead of searching the source for a string a splitter handed back.
"""

from dataclasses import dataclass, field

from llama_index.core import Document


@dataclass(frozen=True, slots=True)
class Chunk:
    """A span of one document, with the text it covers.

    `metadata` holds what the splitter knew about the chunk but the text does
    not say -- the heading path above a Markdown section, say, which a chunk cut
    out of the middle of that section no longer carries. It is empty for methods
    that derive nothing beyond the span.
    """

    doc_id: str
    index: int
    start: int
    end: int
    text: str
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return self.end - self.start


def to_chunks(document, spans, doc_id="", metadata=None):
    """Number a sequence of (start, end) spans and attach the text they cover.

    `metadata` is one dict per span, in the same order, or None for no metadata.
    """
    return [
        Chunk(doc_id=doc_id, index=index, start=start, end=end, text=document[start:end],
              metadata=dict(metadata[index]) if metadata else {})
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
        if document[chunk.start:chunk.end] != chunk.text:
            raise ValueError(f"chunk {chunk.index} text does not match document[{chunk.start}:{chunk.end}]")
        if previous is not None and chunk.start <= previous.start:
            raise ValueError(f"chunk {chunk.index} does not start after chunk {previous.index}")
        previous = chunk
    return chunks
