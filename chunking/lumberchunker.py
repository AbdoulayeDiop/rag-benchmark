import json
from typing import List

from llm import DEFAULT_MODEL, call_llm

from .base import Chunk, to_chunks, verify
from .segment import paragraph_spans

# System prompt used by the original LumberChunker segmentation script.
_system_prompt = """You will receive as input an English document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Respond with a JSON object containing the field `answer_id` set to the ID of the paragraph with the content shift, e.g., `{\"answer_id\": 123}`.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


def _count_words(text: str) -> int:
    """Approximate token count: 1 word ≈ 1.2 tokens (mirrors original code)."""
    return round(1.2 * len(text.split()))


def lumberchunker(document: str, model: str = None, client=None, doc_id: str = "") -> List[Chunk]:
    """Chunk a plain‑text *document* using the LumberChunker LLM strategy.

    The algorithm follows the reference implementation in
    ``LumberChunker‑Segmentation.py``:

    1. Split the document into paragraphs (blank‑line delimiters).
    2. Prefix each paragraph with ``ID <n>: `` where ``n`` is a running integer.
    3. Repeatedly feed a growing window of paragraphs to the LLM until it
       signals a *content shift*.
    4. Record the paragraph index returned by the model as a chunk boundary and
       continue from there.
    5. Return a list of :class:`~chunking.base.Chunk` objects spanning the raw
       document text.
    """
    # 1. Paragraph split. Spans rather than strings, so the chunk offsets are
    # read off the document instead of being reconstructed from paragraph lengths.
    char_offsets = paragraph_spans(document)
    if not char_offsets:
        return []

    # 2. Prefix each paragraph with an ID.
    paragraphs = [f"ID {idx}: {document[start:end]}"
                  for idx, (start, end) in enumerate(char_offsets)]

    # 3. Main loop – follows original script logic.
    chunk_boundaries: List[int] = []
    i = 0
    while i < len(paragraphs) - 5:
        # Build a window up to ~550 token‑equivalent words.
        word_count = 0
        window = 0
        while word_count < 550 and i + window < len(paragraphs) - 1:
            window += 1
            window_text = "\n".join(paragraphs[i : i + window])
            word_count = _count_words(window_text)
        # Determine the document sent to the model (mirrors script nuance).
        sent = 1 if window == 1 else window - 1
        final_doc = "\n".join(paragraphs[i : i + sent])
        prompt = _system_prompt + "\nDocument:\n" + final_doc
        # Rate limits are retried by `call_llm` and any other error raises. An
        # empty answer, or one without a usable `answer_id`, is a refusal: -1,
        # the original's `content_flag_increment`.
        try:
            raw = call_llm(
                prompt,
                model or DEFAULT_MODEL,
                client,
                system=_system_prompt,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            # Expected format: {"answer_id": 123}
            answer_id = int(json.loads(raw)["answer_id"])
        except (ValueError, KeyError, TypeError):
            answer_id = -1
        # A boundary must be a paragraph the model was shown, past the first:
        # anything else is treated as a refusal, so boundaries only move forward
        # and stay inside the document.
        if not i < answer_id < i + sent:
            i += 1
            continue
        chunk_boundaries.append(answer_id)
        # The next window opens on the boundary paragraph itself, so the model
        # sees how the chunk begins.
        i = answer_id

    # Add final boundary (end of document).
    chunk_boundaries.append(len(paragraphs))

    # Convert paragraph‑level boundaries to character spans.
    char_spans: List[tuple[int, int]] = []
    prev = 0
    for boundary in chunk_boundaries:
        char_spans.append((char_offsets[prev][0], char_offsets[boundary - 1][1]))
        prev = boundary

    return verify(document, to_chunks(document, char_spans, doc_id=doc_id, metadata=None))
