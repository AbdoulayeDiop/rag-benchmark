"""Contextual augmentation: every chunk carries a note situating it in its document.

This is Anthropic's Contextual Retrieval. The model is shown the document and
one chunk of it, and writes a sentence or two saying where the chunk sits and
what it is about -- naming the things the chunk only refers to. One call per
chunk, each reading the document again, so it is the expensive method: the cost
is chunks x document length, against one pass over the document for `add_summary`.

The prompt is the one from Anthropic's post, sent with the plain OpenAI client.
LlamaIndex has the method as DocumentContextExtractor, but it works on nodes
that find their document through a docstore, and returns a chunk without a
context when a call fails; going through it meant converting every chunk to a
node and back for the sake of one prompt.

The published method sends the whole document with every chunk and assumes it
fits. A GutenQA book is around 115,000 tokens: it does not fit some models at
all, and at some 225 chunks a book would cost 26M tokens to contextualise. So a
document over `max_context_tokens` is not sent whole; each chunk is shown the
window of that size centred on it instead. That is a departure from the
published method, and it is the documents it applies to that need their results
read with that in mind.
"""

from llama_index.core.utils import get_tokenizer

from .base import DEFAULT_MODEL, augment_chunk, call_llm_for_each

# Anthropic's prompt, plus the last line. The prompt is in English and says
# nothing about language, and on a Polish document the notes came back in
# English for some chunks and Polish for others -- so which chunks a Polish
# question could reach depended on a coin toss. With the line added, all six of
# six were Polish.
CONTEXT_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
Write it in the language the chunk itself is written in."""


def _window_around_chunk(document, chunk, size):
    """Span of `size` characters of `document` centred on `chunk`.

    A window that would run past either end of the document is shifted back
    inside it rather than cut short, so every chunk is shown the same amount.
    """
    start = max(0, (chunk.start + chunk.end - size) // 2)
    end = min(len(document), start + size)
    return max(0, end - size), end


def add_context(chunks, document, model=DEFAULT_MODEL, client=None, max_context_tokens=8000,
               max_output_tokens=512, prompt=CONTEXT_PROMPT, workers=4):
    """Put a generated note ahead of each of `chunks`, situating it in `document`.

    `max_context_tokens` is how much of the document the model is shown beside
    a chunk. A document within it is shown whole, as in the published method;
    a longer one is shown as a window around the chunk. Raising it buys wider
    context at a cost that grows with it for every chunk.

    `workers` is how many chunks are in flight at once: at the default window
    the endpoint's limit is 16 chunks a minute. `max_output_tokens` bounds the
    note; a reasoning model spends it on reasoning first and needs
    more than the note itself takes. `prompt` is a template with `{document}`
    and `{chunk}` fields.
    """
    if not chunks:
        return []
    # The window is sized in characters, from how this document tokenizes --
    # Polish costs about twice the tokens per character that English does.
    document_tokens = len(get_tokenizer()(document))
    fits = document_tokens <= max_context_tokens
    window_size = len(document) if fits else len(document) * max_context_tokens // document_tokens

    prompts = []
    for chunk in chunks:
        start, end = _window_around_chunk(document, chunk, window_size)
        prompts.append(prompt.format(document=document[start:end], chunk=chunk.original_text))
    contexts = call_llm_for_each(prompts, model, client, max_output_tokens, workers)
    return [augment_chunk(chunk, context=context) for chunk, context in zip(chunks, contexts)]
