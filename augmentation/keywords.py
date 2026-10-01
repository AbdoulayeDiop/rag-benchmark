"""Keyword augmentation: every chunk is followed by the terms it is about.

One call per chunk, reading only the chunk. The model lists the chunk's main
topics and entities, which adds the words a question is likely to use when the
chunk itself says them only once, by pronoun, or not at all -- "the mission"
for Cassini-Huygens. It helps lexical search most, since BM25 matches on terms
and nothing else.
"""

from .base import augment_chunk, call_llm_for_each

# Same-language line for the reason given in `summary`: keywords in English
# after a Polish chunk match no Polish question.
KEYWORDS_PROMPT = """<chunk>
{chunk}
</chunk>
Give up to {count} keywords or short key phrases naming the main topics and entities of this chunk, for the purposes of improving search retrieval of the chunk. Replace pronouns by what they refer to. Write them in the language the chunk itself is written in. Answer only with the keywords, separated by commas, on one line."""


def add_keywords(chunks, model, client=None, count=8, prompt=KEYWORDS_PROMPT,
                 workers=4):
    """Put up to `count` generated keywords after each of `chunks`.

    The keywords are kept as one comma-separated line in
    `metadata['keywords']`. `prompt` is a template with `{chunk}` and `{count}`
    fields.
    """
    prompts = [prompt.format(chunk=chunk.original_text, count=count) for chunk in chunks]
    answers = call_llm_for_each(prompts, model, client, max_output_tokens=128, workers=workers)
    # One line, whatever the model did with newlines or a trailing comma, and
    # no more than was asked for: "up to 8" came back as 14.
    keywords = [", ".join([word.strip() for word in answer.replace("\n", ",").split(",")
                           if word.strip()][:count])
                for answer in answers]
    return [augment_chunk(chunk, keywords=line) for chunk, line in zip(chunks, keywords)]
