"""Question augmentation: every chunk is followed by questions it answers.

One call per chunk, reading only the chunk. A chunk is worded as an answer and
is searched for with a question, and the two are not phrased alike; writing out
the questions the chunk answers puts text shaped like the query beside it. This
is the idea of doc2query, and of the "hypothetical questions" index.

The questions are embedded together with the chunk, as one vector. Indexing
each question as a vector of its own, all pointing back at the chunk, is a
different method: it changes what the index holds, not what a chunk embeds as.
"""

import re

from .base import augment_chunk, call_llm_for_each

# The language line is there because the questions stand in for the ones that
# will be asked, and PoQuAD's are asked in Polish. Its wording matters: "do not
# translate into English" made the model write German questions for an English
# paper, so every prompt here names the chunk's own language instead.
QUESTIONS_PROMPT = """<chunk>
{chunk}
</chunk>
Write {count} questions that this chunk answers, for the purposes of improving search retrieval of the chunk. Each question must be answerable from the chunk alone and must name what it is about rather than refer to "the chunk" or "the text". Write them in the language the chunk itself is written in. Answer only with the questions, one per line."""

# Models number or bullet a list whether asked to or not.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def add_questions(chunks, model, client=None, count=3, prompt=QUESTIONS_PROMPT,
                  workers=4):
    """Put `count` generated questions after each of `chunks`.

    The questions are kept as a list in `metadata['questions']`, and embedded
    one per line. `prompt` is a template with `{chunk}` and `{count}` fields.
    """
    prompts = [prompt.format(chunk=chunk.original_text, count=count) for chunk in chunks]
    answers = call_llm_for_each(prompts, model, client, max_output_tokens=256, workers=workers)
    augmented = []
    for chunk, answer in zip(chunks, answers):
        questions = [_LIST_MARKER.sub("", line).strip() for line in answer.splitlines()]
        augmented.append(augment_chunk(chunk, questions=[q for q in questions if q]))
    return augmented
