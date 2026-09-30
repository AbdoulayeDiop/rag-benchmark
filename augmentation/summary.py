"""Summary augmentation, two ways: of the document, and of the chunk.

`add_summary` puts a summary of the whole document ahead of each chunk.
`use_summary` summarises each chunk and embeds that *instead of* the chunk --
the only method here that takes text away rather than adding it.

For `add_summary`, one summary is written per document and put ahead of each of its chunks, so
the cost is per document, not per chunk -- which is the point of comparing it
with `add_context`, whose cost is per chunk.

A document the model can read in one call is summarised in one call. A longer
one is cut into pieces that each fit, the pieces are summarised, and those
summaries are summarised in turn. That matters here: a GutenQA book is around
115,000 tokens, and does not fit beside a prompt in a 128,000-token model.

This is written directly rather than taken from LlamaIndex, which has the same
procedure as TreeSummarize: it is a dozen lines, and the library version wants
its own LLM class and wraps the text in a question-answering template.

`use_summary` makes one call per chunk, reading only the chunk. The summary
states what the chunk says in fewer words, so the embedding is of its content
with the asides, examples and phrasing stripped out. The chunk is still what is
retrieved and scored: only what it is found by changes.
"""

from llama_index.core.utils import get_tokenizer

from .base import DEFAULT_MODEL, SEPARATOR, augment_chunk, call_llm, call_llm_for_each

# It asks for the document's own language because the summary is embedded
# beside the chunk, and an English summary on a Polish chunk pulls the two
# apart -- which is what the model wrote until it was told not to.
SUMMARY_PROMPT = (
    "Write a short summary of this document, in one paragraph of at most five "
    "sentences, for the purposes of improving search retrieval of its passages. "
    "Name the document's subject and the main entities it covers, starting with "
    "the subject itself rather than with 'This document'. Write it in the language "
    "the document itself is written in. "
    "Answer only with the summary and nothing else."
)

# For one chunk. It asks for names in place of pronouns because a summary that
# says "he" or "the method" has dropped the one thing a search would match on.
# Its language line is worded differently from the other prompts' on purpose:
# "in the language the chunk itself is written in", which they use, returned
# English summaries for 2 of 4 Polish chunks, and this wording for none.
CHUNK_SUMMARY_PROMPT = """<chunk>
{chunk}
</chunk>
Summarise this chunk in at most {sentences} sentences, for the purposes of improving search retrieval of the chunk. Keep the names, numbers and terms it states, and replace pronouns by what they refer to. Start with the content itself rather than with 'This chunk'. Write in the same language as the chunk: do not translate into English. Answer only with the summary and nothing else."""


def _split_by_tokens(text, max_tokens):
    """Cut `text` into the fewest equal pieces of at most `max_tokens` tokens each.

    Sized in characters, from how this text tokenizes; the cuts fall wherever
    that lands, which a summary of several thousand tokens does not notice.
    """
    tokens = len(get_tokenizer()(text))
    count = -(-tokens // max_tokens)  # ceiling division
    size = -(-len(text) // count)
    return [text[start:start + size] for start in range(0, len(text), size)]


def summarize_document(document, model=DEFAULT_MODEL, client=None, prompt=SUMMARY_PROMPT,
                       max_input_tokens=64000):
    """Summarise a whole `document`, however long, and return the summary.

    `max_input_tokens` is how much text goes into one call. The default is half
    the default model's context, because the configured endpoint allows 128,000
    input tokens a minute: a call that size would be the whole minute's budget.
    """
    text = document
    while True:
        pieces = _split_by_tokens(text, max_input_tokens)
        summaries = [call_llm(f"<document>\n{piece}\n</document>\n{prompt}", model, client)
                     for piece in pieces]
        if len(summaries) == 1:
            return summaries[0]
        # The summaries of the pieces, in order, stand in for the document.
        text = SEPARATOR.join(summaries)


def add_summary(chunks, document, model=DEFAULT_MODEL, client=None, prompt=SUMMARY_PROMPT,
                max_input_tokens=64000):
    """Put a summary of `document` ahead of every one of its `chunks`.

    The summary is generated once per call, so pass all of a document's chunks
    together. To reuse one summary across several chunkings of the same
    document, call `summarize_document` once and attach the result with
    `augment_chunk(chunk, parent_document_summary=...)`.
    """
    if not chunks:
        return []
    document_summary = summarize_document(document, model, client, prompt, max_input_tokens)
    return [augment_chunk(chunk, parent_document_summary=document_summary) for chunk in chunks]


def use_summary(chunks, model=DEFAULT_MODEL, client=None, sentences=3,
                prompt=CHUNK_SUMMARY_PROMPT, workers=4):
    """Embed each of `chunks` as a generated summary of itself, not as its text.

    The summary is kept in `metadata['summary']` and becomes the whole of
    `text_to_embed`: a title, context, keywords or questions added by the
    other methods stay in the metadata but are not embedded with it.
    `original_text` and the offsets are untouched.
    `prompt` is a template with `{chunk}` and `{sentences}` fields.
    """
    prompts = [prompt.format(chunk=chunk.original_text, sentences=sentences) for chunk in chunks]
    summaries = call_llm_for_each(prompts, model, client, max_output_tokens=256, workers=workers)
    return [augment_chunk(chunk, summary=text) for chunk, text in zip(chunks, summaries)]
