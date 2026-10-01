# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Experiments comparing RAG methods stage by stage: dataset preparation, chunking,
chunk augmentation, embedding, retrieval, reranking and retrieval evaluation
(generation and the RAGAS metrics are not written yet). [README.md](README.md) is the methodology record: dataset statistics,
per-dataset processing decisions and cost tables live there. Keep it in step
with the code when a method or a loader changes.

There is no test suite, linter config or package metadata yet. The repo is a
set of importable modules, one script per dataset, `ingest.py`, which runs
chunking, augmentation and embedding for one dataset and one configuration,
`evaluate.py`, which scores one index's retrieval against the evidence spans,
and `compare.py`, which tabulates the evaluations.

## Commands

Windows, Python 3.13, virtualenv in `.venv`. Run everything from the repository root.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe datasets\squad\load.py            # build one dataset
.venv\Scripts\python.exe -m streamlit run app.py           # chunk visualizer
.venv\Scripts\python.exe ingest.py squad --param max_tokens=256 --max-documents 3   # build an index
.venv\Scripts\python.exe evaluate.py squad txt-sentence-max_tokens=256 --embedding-model bge-m3 --rerank-model bge-reranker-v2-m3   # score it
.venv\Scripts\python.exe compare.py                       # Markdown table per dataset
```

`ingest.py` flags: `--chunking`, `--param KEY=VALUE` (repeatable, JSON values),
`--rendering`, `--augment`, `--llm-model`, `--embedding-model`,
`--embedding-max-tokens` and `--embedding-tokenizer` (truncate over-long texts
before embedding; for bge-m3, `8192` and `BAAI/bge-m3`), `--max-documents`,
`--name`, `--skip-embedding`. A small `--max-documents` run is the quickest
end-to-end check; re-running the same command resumes rather than repeats.

`evaluate.py <dataset> <index>` flags: `--retrieval dense|sparse|hybrid`,
`--top-k`, `--embedding-model`, `--rerank-model` or `--no-rerank`,
`--rerank-depth`, `--rerank-max-tokens` and `--rerank-tokenizer` (cut chunks to
the reranker's pair limit; `8192` and `BAAI/bge-m3`), `--max-items` (a fixed
sample order), `--name`. It resumes too, and refuses to extend results after
the index has grown.

Loader flags vary per dataset: `--splits` (conditionalqa, literaryqa, poquad,
qasper, squad), `--split` (triviaqa), `--max-documents` (naturalquestions, squad,
techqa, triviaqa), `--delay` (literaryqa, poquad, which fetch from the web).

LLM and embedding calls read `OPENAI_API_KEY` and `OPENAI_API_BASE` and target an
OpenAI-compatible endpoint, not OpenAI itself. No model name is hard-coded:
every library function that calls a model takes `model` (or `embed_model`)
explicitly, and only the entry points fall back to the environment --
`ingest.py --llm-model/--embedding-model` and `app.py` read `LLM_MODEL` and
`EMBEDDING_MODEL`, and `evaluate.py` reads `EMBEDDING_MODEL` and `RERANK_MODEL`.
Do not add a default model back. The experiments here use
`bge-m3` for embeddings and `mistral-small-3-2-24b-instruct-2506` for generation. The
endpoint allows 128,000 input tokens per minute and at most 64 texts per
embedding request; both limits shape the code. Its `/rerank` route
(`bge-reranker-v2-m3`) takes at most 64 documents a request and refuses a
query–document pair over 8,192 tokens.

## Layout

| Path | Contents |
|---|---|
| `datasets/<name>/load.py` | One standalone loader per dataset (9 built; NovelQA and MS MARCO deliberately skipped, see README) |
| `datasets/<name>/data/`, `.cache/` | Loader output and download cache. Gitignored, rebuilt rather than committed |
| `datasets/literaryqa/vendor/` | Upstream LiteraryQA cleaning code pinned at the commit in `vendor/COMMIT`. Do not edit; `structure.py` is the adapted variant |
| `chunking/` | Chunking methods and the `Chunk` type |
| `augmentation/` | Chunk augmentation methods |
| `llm.py` | `get_client()` and `call_llm()`, shared by `chunking/lumberchunker.py` and `augmentation/` |
| `embedding.py` | `embed_batch()`, on the plain OpenAI client; handles over-long inputs |
| `ingest.py` | Pipeline runner: documents → chunks → augmented chunks → Chroma collection and BM25 index, resumable per document and per batch |
| `indexes/<dataset>/<run>/` | Runner output: `config.json`, `chunks.jsonl`, `documents.jsonl`, `chroma/` (LlamaIndex `ChromaVectorStore`), `bm25/` (LlamaIndex `BM25Retriever`; load it with `load_bm25`, which restores the stemming `persist` drops). Node metadata is the whole chunk; `node_to_chunk` rebuilds it. `chunks.jsonl` is the checkpoint and verification record, not read by retrieval. Gitignored |
| `retrieval.py` | `dense`, `sparse`, `hybrid` (reciprocal rank fusion) over a run's stores; returns `(Chunk, score)` pairs |
| `rerank.py` | `rerank()` through the endpoint's `/rerank`, on the plain OpenAI client's `post`; cuts chunks to the pair limit |
| `evaluate.py` | Retrieval evaluation against evidence spans (hit, MRR, nDCG, character precision/recall/F1, doc hit), before and after reranking |
| `results/<dataset>/<index>/<set-up>/` | `evaluate.py` output: `config.json`, `items.jsonl` (rankings, the checkpoint), `summary.json`. Gitignored |
| `compare.py` | Every `summary.json` as one Markdown table per dataset |
| `app.py` | Streamlit chunk visualizer |
| `papers.md`, `archive/datasets.md` | Reading list and dataset survey notes |

`datasets/` is a plain directory of scripts, not a package, and shares its name
with the Hugging Face `datasets` library. Loaders use `huggingface_hub` and
pandas directly.

## The span contract

Everything rests on one invariant, defined in [chunking/base.py](chunking/base.py):

```python
document[chunk.start:chunk.end] == chunk.original_text
```

Loaders record evidence as character offsets into the document, so retrieval is
scored by span overlap however a document was cut. Consequences:

- A chunking method returns `list[Chunk]` built with `to_chunks(document, spans, doc_id, metadata)`. Never return text that was rewritten, stripped or re-joined.
- Library splitters go through `split_spans`, which reads LlamaIndex's native offsets and raises if a node is not at the offset it claims. `verify(document, chunks)` checks bounds, text and ordering.
- HTML chunks hold raw markup in `original_text` for the same reason. They carry `metadata['markup'] = "html"`, and their `text_to_embed` has the tags stripped, from the chunker on and through every augmentation. `ingest.py` marks chunks of the HTML rendering whatever method cut them.
- Each rendering (`documents/`, `documents_md/`, `documents_html/`) has its own offsets and its own evidence text. One span never applies to two renderings.
- Every loader asserts `document[start:end] == text` for each evidence span before finishing, and records how the span was found in `match`.

## Dataset loaders

All loaders write the same layout (`documents/<doc_id>.txt`, optional
`documents_md/` and `documents_html/`, and `items.jsonl`); the item schema is in
the README under "Dataset processing". Each loader's module docstring explains
why the source was reshaped. Loaders default to the smallest representative
split because corpus size is the binding constraint on local compute.

## Chunking

Every method has the signature `method(document, ..., doc_id="") -> list[Chunk]`.

| Function | File | Notes |
|---|---|---|
| `fixed_char` | `fixed.py` | Character windows, hand-sliced |
| `sentence` | `structure.py` | LlamaIndex `SentenceSplitter`; covers both the sentence and recursive methods |
| `markdown` | `markdown.py` | `MarkdownNodeParser`, then `SentenceSplitter` for sections over budget; heading path in metadata |
| `html` | `html.py` | Outline read from BeautifulSoup source positions; no library keeps the markup |
| `semantic`, `tiled`, `clustered` | `semantic.py` | Embedding-based; `embed_model` is required, built with `openai_embedding(model)` |
| `lumberchunker` | `lumberchunker.py` | LLM-based, through `call_llm`; an unusable model answer counts as a refusal |

Conventions:

- Prefer a LlamaIndex node parser. Write plain code only where no library fits, and say why in the module docstring, as the existing modules do.
- Budgets are in tokens (tiktoken `cl100k_base` via LlamaIndex's `get_tokenizer`), not characters. Polish costs about twice the tokens per character that English does.
- Sentence segmentation is pysbd, run per paragraph (`segment.py`). `language` defaults to `"auto"`: `detect_language` (lingua, restricted to the languages pysbd supports) reads it from the document. A pysbd code (`"pl"`, `"en"`) overrides it. `ingest.py` detects once per run and records the result in `config.json`.
- Paragraphs are separated by a blank line (`PARAGRAPH_SEPARATOR = "\n\n"`), not LlamaIndex's default of three newlines.

## Augmentation

A method takes the chunks of one document and returns new chunks (`Chunk` is
frozen) with `text_to_embed` set. `original_text` and the offsets are never
touched. Generated text is stored under one metadata key per method, and every
method ends with `augment_chunk(chunk, key=value)`, which rebuilds
`text_to_embed` from the metadata via `build_text_to_embed`. That is what makes
the methods order-independent. Read the result with `get_text_to_embed(chunk)`.
Both live in `chunking/base.py`, next to `Chunk`: `build_text_to_embed` is the one
definition of what a chunk is embedded as (augmentations in a fixed order, HTML
tags stripped), used by the chunkers, by `augment_chunk` and by `node_to_chunk`.

| Function | Metadata key | LLM calls |
|---|---|---|
| `add_title` | `parent_document_title` | One per document |
| `add_summary` | `parent_document_summary` | One pass per document, recursive for long ones |
| `add_context` | `context` | One per chunk, with a document window of `max_context_tokens` |
| `add_keywords` | `keywords` | One per chunk |
| `add_questions` | `questions` (a list) | One per chunk |
| `use_summary` | `summary` | One per chunk; replaces the chunk text in `text_to_embed` |

Conventions:

- Call the plain OpenAI client through `call_llm` (in `llm.py`) / `call_llm_for_each`. No LlamaIndex nodes, docstores or LLM classes.
- `call_llm` retries only on rate limits (60 s wait) and raises on an empty answer, so a run never silently mixes augmented and bare chunks.
- Every prompt tells the model to answer in the chunk's own language. The exact wording was tuned against Polish output and is explained in comments above each prompt; do not reword it without re-testing on PoQuAD.

## Code style

- Module docstrings carry the rationale: what the method is, which library was chosen or rejected, and measured figures behind the decision. Follow that pattern in new modules.
- Comments explain why, often with a measurement. Function docstrings describe parameters in prose.
- Plain functions and one dataclass; no class hierarchies or framework abstractions.
- Files are written with `encoding="utf-8", newline="\n"`.

## Known gaps

- `chunking/lumberchunker.py` is not exported from `chunking/__init__.py`.
- `requirements.txt` lists `llama-index-embeddings-openai`, but the code imports `llama_index.embeddings.openai_like` (package `llama-index-embeddings-openai-like`). `scikit-learn`, `scipy`, `numpy`, `httpx` and `tiktoken` are imported but not listed.
- `app.py` and `lumberchunker.py` were written in a different style from the rest (type hints, no rationale docstrings).
