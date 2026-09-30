"""What every augmentation method returns, and the LLM the generated ones call.

Augmentation changes what a chunk is *found by* without changing what it *is*.
A chunk cut out of its document has lost which document that was and where in
it the chunk sat; it is also worded as an answer, while it will be searched for
with a question. The methods here add what is missing, or reword what is there.

None of what they produce is in the document at the chunk's offsets, so it
cannot go in `chunk.original_text` -- that would break
`document[chunk.start:chunk.end] == chunk.original_text` and with it the
span-overlap scoring every evaluation here rests on. It is kept in the chunk's
metadata instead, one key per kind, and `text_to_embed` is put together from
whichever are present, always in this order:

    metadata['parent_document_title']
    metadata['parent_document_summary']
    metadata['context']                   the note situating the chunk in its document
    metadata['keywords']
    metadata['questions']
    the chunk itself                      original_text

The exception is `metadata['summary']`, a summary of the chunk itself: a chunk
that has one is embedded as the summary alone, with nothing around it. The
other keys stay in the metadata but are left out of `text_to_embed`.

`text_to_embed` is None on a chunk nothing was added to; `get_text_to_embed` reads
either case. So an augmented chunk retrieves on something other than what it
covers, and is still scored on exactly what it covers. Because the text is
rebuilt from the metadata each time, the methods can be applied in any order.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from llm import DEFAULT_MODEL, call_llm, get_client

# What goes between the parts of `text_to_embed`. A blank line is what
# Anthropic's contextual retrieval uses, and it reads as a paragraph break to
# an encoder and to BM25 alike.
SEPARATOR = "\n\n"


def build_text_to_embed(original_text, metadata):
    """Join a chunk's text and the augmentations in `metadata` into the text to embed.

    A chunk with a summary of itself is embedded as that summary and nothing
    else. Otherwise it is whichever of the keys the module docstring lists are
    present, in the docstring's order, whatever order they were added in.
    """
    if metadata.get("summary"):
        return metadata["summary"]
    pieces = [metadata.get("parent_document_title"),
              metadata.get("parent_document_summary"),
              metadata.get("context"),
              metadata.get("keywords"),
              "\n".join(metadata.get("questions") or []),
              original_text]
    return SEPARATOR.join(piece for piece in pieces if piece)


def augment_chunk(chunk, **augmentations):
    """Return `chunk` with `augmentations` stored in its metadata and `text_to_embed` rebuilt.

    `augmentations` are metadata keys from the module docstring and the text
    generated for each, as in `augment_chunk(chunk, keywords="a, b")`. Every
    method ends by calling this.
    """
    metadata = {**chunk.metadata, **augmentations}
    return replace(chunk, metadata=metadata,
                   text_to_embed=build_text_to_embed(chunk.original_text, metadata))


def get_text_to_embed(chunk):
    """The text to embed or index for `chunk`: `text_to_embed`, or the span itself."""
    return chunk.original_text if chunk.text_to_embed is None else chunk.text_to_embed


def call_llm_for_each(prompts, model=DEFAULT_MODEL, client=None, max_output_tokens=512,
                      workers=4):
    """Send each of `prompts` and return the answers in the same order.

    For the methods that make one call per chunk. `workers` is how many calls
    are in flight at once; the configured endpoint allows 128,000 input tokens
    a minute, and past that more workers only reach the limit sooner.
    """
    client = client or get_client()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda prompt: call_llm(prompt, model, client, max_output_tokens),
                             prompts))
