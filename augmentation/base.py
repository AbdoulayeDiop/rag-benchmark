"""What every augmentation method returns, and the LLM the generated ones call.

Augmentation changes what a chunk is *found by* without changing what it *is*.
A chunk cut out of its document has lost which document that was and where in
it the chunk sat; it is also worded as an answer, while it will be searched for
with a question. The methods here add what is missing, or reword what is there.

None of what they produce is in the document at the chunk's offsets, so it
cannot go in `chunk.original_text` -- that would break
`document[chunk.start:chunk.end] == chunk.original_text` and with it the
span-overlap scoring every evaluation here rests on. It is kept in the chunk's
metadata instead, one key per kind:

    metadata['parent_document_title']     add_title
    metadata['parent_document_summary']   add_summary
    metadata['context']                   add_context
    metadata['keywords']                  add_keywords
    metadata['questions']                 add_questions
    metadata['summary']                   use_summary

and the chunk's `text_to_embed` is rebuilt from them by
`chunking.base.build_text_to_embed`, which owns the order they are joined in
and the rule that a chunk with a `summary` of itself is embedded as that alone.
Every method ends with `augment_chunk`, so the text is rebuilt from the
metadata each time and the methods can be applied in any order. An augmented
chunk retrieves on something other than what it covers, and is still scored on
exactly what it covers.
"""

from concurrent.futures import ThreadPoolExecutor

from chunking.base import SEPARATOR, update_chunk_metadata
from llm import call_llm, get_client


def augment_chunk(chunk, **augmentations):
    """Return `chunk` with `augmentations` stored in its metadata and `text_to_embed` rebuilt.

    `augmentations` are metadata keys from the module docstring and the text
    generated for each, as in `augment_chunk(chunk, keywords="a, b")`. Every
    method ends by calling this.
    """
    return update_chunk_metadata(chunk, **augmentations)


def call_llm_for_each(prompts, model, client=None, max_output_tokens=512,
                      workers=4):
    """Send each of `prompts` and return the answers in the same order.

    For the methods that make one call per chunk. `workers` is how many calls
    are in flight at once. Under a per-minute token limit -- 128,000 input
    tokens on the endpoint used here -- more workers past it only reach the
    limit sooner.
    """
    client = client or get_client()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda prompt: call_llm(prompt, model, client, max_output_tokens),
                             prompts))
