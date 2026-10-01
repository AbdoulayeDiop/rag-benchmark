"""Title augmentation: every chunk says which document it was cut from.

The cheapest of the generated methods: one short call per document, reading
only its opening. A chunk from the middle of a paper or a novel rarely names
the paper or the novel, while the question usually does.

The title is generated rather than looked up because the corpora do not agree
on where one lives -- five datasets record it with their questions, three only
in a file name, and TechQA's file names are IBM document numbers. A model
reading the first chunks is the one source that works for all of them, and the
opening of a document is where it states its title when it has one.
"""

from .base import SEPARATOR, augment_chunk, call_llm

# Same-language line for the reason given in `summary`: the title is embedded
# beside the chunk, and an English title on a Polish chunk pulls the two apart.
TITLE_PROMPT = """<document_start>
{text}
</document_start>
Above is the beginning of a document. Give the title of the whole document, not the heading of one of its sections: the one it states if it states one, otherwise a short title naming its subject. Write it in the language the document itself is written in. Answer only with the title and nothing else."""


def generate_title(chunks, model, client=None, k=5, prompt=TITLE_PROMPT):
    """Title the document `chunks` were cut from, reading its first `k` chunks.

    `chunks` must be in document order, as every chunking method returns them.
    `prompt` is a template with a `{text}` field.
    """
    opening = SEPARATOR.join(chunk.original_text for chunk in chunks[:k])
    return call_llm(prompt.format(text=opening), model, client, max_output_tokens=64)


def add_title(chunks, model, client=None, k=5, prompt=TITLE_PROMPT):
    """Put a generated title of the document ahead of every one of its `chunks`.

    The title is generated once per call, from the first `k` chunks, so pass
    all of a document's chunks together. How much the model reads therefore
    depends on the chunk size as well as on `k`. To attach a title that is
    already known, use `augment_chunk(chunk, parent_document_title=...)` instead.
    """
    if not chunks:
        return []
    document_title = generate_title(chunks, model, client, k, prompt)
    return [augment_chunk(chunk, parent_document_title=document_title) for chunk in chunks]
