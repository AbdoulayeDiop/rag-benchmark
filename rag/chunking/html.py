"""HTML chunking: split on the document's own heading structure.

The same method as `markdown`, on the rendering that keeps the markup. A chunk
runs from one heading to the next, and a section over the budget is split
further at tag boundaries.

Unlike every other method here, the outline is not taken from a splitter
library, because every HTML splitter in both libraries returns the *extracted
text* of a section with the markup stripped. LangChain's three produce text that
cannot be found in the source file at all; LlamaIndex's HTMLNodeParser gives
offsets for only 25 of the 39 nodes it returns, and for text that no longer
holds the tags. Chunks would then have no offsets into the rendering they came
from -- and ConditionalQA's HTML evidence is raw markup ("<p>You can apply to
become...") recorded as offsets into this exact file, so there would be nothing
to score against. BeautifulSoup's html.parser reports the source position of
every tag it opens, exact on all three HTML corpora, so the outline is read from
those positions instead and the chunks stay spans.

A chunk's `original_text` therefore holds markup, not clean text: changing it
would break `document[start:end] == text` and with it the evaluation. Every
chunk is marked `metadata['markup'] = "html"` instead, and its `text_to_embed`
is the text with the tags stripped (`base.build_text_to_embed`), which augmentation
keeps when it adds to a chunk. What is embedded and indexed is clean text; what
is scored is the span.
"""

from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter

from .base import split_spans, to_chunks

HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _offsets(document):
    """Return a function mapping a tag to its character offset in `document`.

    BeautifulSoup reports a line and a column; the line starts are needed to
    turn that pair into an offset.
    """
    starts = [0]
    for line in document.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return lambda tag: starts[tag.sourceline - 1] + tag.sourcepos


def html(document, max_tokens=1024, headings=HEADINGS, doc_id=""):
    """Chunk an HTML `document` into the sections its headings declare.

    `max_tokens` caps a chunk in tokens -- markup included, since that is what
    the chunk holds, and tags are not cheap to tokenize -- and None leaves
    sections whole. Everything before the first heading (doctype, head, title)
    is a chunk of its own.
    """
    soup = BeautifulSoup(document, "html.parser")
    offset_of = _offsets(document)

    # A heading opens a section that runs to the next heading of any level, and
    # carries the path of headings above it.
    starts, metadata, path = [0], [{"markup": "html"}], {}
    for tag in soup.find_all(list(headings)):
        level = headings.index(tag.name)
        path = {name: value for name, value in path.items() if headings.index(name) < level}
        path[tag.name] = tag.get_text(" ", strip=True)
        starts.append(offset_of(tag))
        metadata.append({"markup": "html", **path})

    bounds = starts[1:] + [len(document)]
    sections = [(start, end) for start, end in zip(starts, bounds) if document[start:end].strip()]
    metadata = [meta for meta, (start, end) in zip(metadata, zip(starts, bounds))
                if document[start:end].strip()]
    if max_tokens is None:
        return to_chunks(document, sections, doc_id, metadata)

    # One element per line is how the loaders write these files, so splitting
    # on newlines cuts between tags rather than inside one.
    splitter = SentenceSplitter(
        chunk_size=max_tokens, chunk_overlap=0, paragraph_separator="\n")
    spans, spans_metadata = [], []
    for (start, end), meta in zip(sections, metadata):
        # Only the splitter can say whether a section fits a token budget; one
        # that fits comes back whole.
        pieces = split_spans(document[start:end], splitter, start)
        spans.extend(pieces)
        spans_metadata.extend([meta] * len(pieces))
    return to_chunks(document, spans, doc_id, spans_metadata)
