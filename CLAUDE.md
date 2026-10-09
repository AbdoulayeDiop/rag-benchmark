# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Experiments comparing RAG methods stage by stage: dataset preparation, chunking,
chunk augmentation, embedding, retrieval, reranking and retrieval evaluation
(generation, and the RAGAS metrics that need it, are not written yet; RAGAS context precision and recall are). [README.md](README.md) is the methodology record: dataset statistics,
per-dataset processing decisions and cost tables live there. Keep it in step
with the code when a method or a loader changes.

There is no test suite, linter config or package metadata yet. The code is
split in two:

- `rag/` is the library, one module per pipeline stage. Every function takes
  its inputs, models and directories as arguments; nothing in it reads a
  dataset, a configuration file or a model name from the environment.
- `experiments/` holds one script per stage (`ingest`, `retrieve`, `rerank`,
  `evaluate`) plus `compare` and `compare_chunking`. Each takes one JSON
  configuration file, decides where runs live on disk, and resumes what a
  stopped run left. They are run as modules (`-m experiments.ingest`), so that
  `rag` is importable.

## Commands

Windows, Python 3.13, virtualenv in `.venv`. Run everything from the repository root.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe datasets\squad\load.py                                            # build one dataset
.venv\Scripts\python.exe -m streamlit run app.py                                           # chunk visualizer
.venv\Scripts\python.exe -m experiments.ingest experiments/configs/ingest_chunking.json --dry-run   # list, then drop --dry-run to build
.venv\Scripts\python.exe -m experiments.retrieve experiments/configs/retrieve_dense.json
.venv\Scripts\python.exe -m experiments.rerank <config>.json
.venv\Scripts\python.exe -m experiments.evaluate experiments/configs/evaluate_spans.json    # span metrics
.venv\Scripts\python.exe -m experiments.evaluate experiments/configs/evaluate_ragas.json    # and RAGAS
.venv\Scripts\python.exe -m experiments.compare                                            # Markdown table per dataset
.venv\Scripts\python.exe -m experiments.compare_chunking                                   # chunking methods, baseline pipeline only
```

Every experiment script takes `--dry-run` (list the runs) and `--keep-going`
(go past a failed run). The configuration keys are in each script's module
docstring:

- `ingest`: `dataset`, `chunking`, `chunk_augmentation`, `chunk_embedding`
  (`model`, `tokenizer`, `context_length`; for bge-m3, `BAAI/bge-m3` and
  `8192`), `max_documents`, `language`, `workers`.
- `retrieve`: `index` (a path under `experiments/indexes/`, a glob pattern,
  `"all"`, or a list), `retrieval` (`dense|sparse|hybrid`, `top_k`),
  `max_items` (a fixed sample order).
- `rerank`: `retrieval` (paths under `experiments/results/`), `reranking`
  (`cross_encoder` with `model`, `top_k`, `max_tokens`, `tokenizer`; `8192`
  and `BAAI/bge-m3` cut chunks to the reranker's pair limit), `max_items`,
  `workers`.
- `evaluate`: `results` (retrieval or reranking runs), `ragas` (`judge_model`,
  `k`, `max_items`, `requests_per_minute`, `workers`).

A list inside `params` is a sweep. A small `max_documents` run is the quickest
end-to-end check; re-running the same file resumes rather than repeats, and a
directory started with another configuration is refused.

Loader flags vary per dataset: `--splits` (conditionalqa, literaryqa, poquad,
qasper, squad), `--split` (triviaqa), `--max-documents` (naturalquestions, squad,
techqa, triviaqa), `--delay` (literaryqa, poquad, squad, which fetch from the web).

LLM and embedding calls read `OPENAI_API_KEY` and `OPENAI_API_BASE` and target an
OpenAI-compatible endpoint, not OpenAI itself. No model name is hard-coded:
every library function that calls a model takes `model` (or `embed_model`)
explicitly, the experiment scripts take every model from their configuration
file, and only `app.py` falls back to the environment (`LLM_MODEL`,
`EMBEDDING_MODEL`). Do not add a default model back. The experiments here use
`bge-m3` for embeddings and `mistral-small-3-2-24b-instruct-2506` for generation. The
endpoint allows LLM calls 128,000 input tokens per minute, which makes
augmentation and LumberChunker the slow stages. Embedding is not under that
limit: one request at a time runs at about 740,000 tokens a minute, and 8
concurrent requests at about 2.8M (measured 2026-10-02). An embedding request
takes at most 64 texts. Its `/rerank` route
(`bge-reranker-v2-m3`) takes at most 64 documents a request and refuses a
query–document pair over 8,192 tokens.

## Layout

| Path | Contents |
|---|---|
| `datasets/<name>/load.py` | One standalone loader per dataset (9 built; NovelQA and MS MARCO deliberately skipped, see README) |
| `datasets/<name>/data/`, `.cache/` | Loader output and download cache. Gitignored, rebuilt rather than committed |
| `datasets/literaryqa/vendor/` | Upstream LiteraryQA cleaning code pinned at the commit in `vendor/COMMIT`. Do not edit; `structure.py` is the adapted variant |
| `rag/chunking/` | Chunking methods and the `Chunk` type |
| `rag/augmentation/` | Chunk augmentation methods |
| `rag/llm.py` | `get_client()` and `call_llm()`, shared by `chunking/lumberchunker.py` and `augmentation/`; `usage()`/`usage_since()`: tokens the endpoint reports, counted by an httpx hook (`count_usage`) on every client, and rate-limit waits |
| `rag/embedding.py` | `embed_batch()`, on the plain OpenAI client; handles over-long inputs |
| `rag/ingestion.py` | `to_nodes`/`node_to_chunk` (node metadata is the whole chunk), `open_vector_store` (LlamaIndex `ChromaVectorStore`), `embed_nodes` (resumable by collection size), `build_bm25`/`load_bm25` (`load_bm25` restores the stemming `persist` drops) |
| `rag/retrieval.py` | `dense`, `sparse`, `hybrid` (reciprocal rank fusion) over those stores; returns `(Chunk, score)` pairs |
| `rag/rerank.py` | `rerank()` through the endpoint's `/rerank`, on the plain OpenAI client's `post`; cuts chunks to the pair limit |
| `rag/evaluation.py` | Span metrics of one ranking (hit, MRR, nDCG, evidence recall, doc hit) |
| `rag/judge.py` | RAGAS `ContextPrecision` and `ContextRecall`, paced to the judge's requests per minute |
| `experiments/common.py` | The layout on disk, JSON I/O, sweeps, target resolution (`resolve`), dataset reading (`read_documents`, `load_items`), the shared run loop |
| `experiments/ingest.py` | Documents → chunks → augmented chunks (`experiments/chunks/`), then Chroma collection and BM25 index per embedding model (`experiments/indexes/`), resumable per document and per batch |
| `experiments/retrieve.py`, `rerank.py` | Rankings per question into `experiments/results/<index>/<retrieval>/[<rerank>/]ranking.jsonl` |
| `experiments/evaluate.py` | Span metrics, and RAGAS on a sample (200) with chunks shown via `build_text_for_llm`, into the run's `evaluation/summary.json`; RAGAS verdicts in `evaluation/ragas-<judge>-k=<k>.jsonl` |
| `experiments/chunks/<dataset>/<rendering>/<chunking>/<augmentation>/` | `config.json`, `chunks.jsonl` (the checkpoint and verification record, not read by retrieval), `documents.jsonl` (each document's chunking and augmentation cost), `cost.json`. Gitignored |
| `experiments/indexes/<chunks path>/<embedding model>/` | `config.json` (names its chunks directory), `chroma/`, `bm25/`, `cost.json` (embedding). Gitignored |
| `experiments/results/<index path>/<retrieval>/[<rerank>/]` | `config.json`, `ranking.jsonl` (`[doc_id, start, end, chunk index, score]` per chunk, and the question's cost; the checkpoint), `evaluation/`. Gitignored |
| `experiments/compare.py` | Every `evaluation/summary.json` as one Markdown table per dataset: each method's best parameters on the first metric (default `evidence_recall@5`), or every combination with `--all`; `--cost` adds build cost per document and query seconds |
| `experiments/compare_chunking.py` | The chunking comparison: only results of the baseline pipeline (no augmentation, bge-m3, dense, no reranking), each method at its best per dataset, and an overview of methods by datasets with the mean difference to `fixed_token`; `--judge` adds RAGAS |
| `experiments/configs/*.json` | Configuration files of the experiment scripts |
| `experiments/*_eval*.json` | Whole comparison grids in the former runner's format (one key per stage); kept as the record of planned comparisons, run stage by stage until a runner reads them again |
| `app.py` | Streamlit chunk visualizer |
| `papers.md`, `archive/datasets.md` | Reading list and dataset survey notes |

`datasets/` is a plain directory of scripts, not a package, and shares its name
with the Hugging Face `datasets` library. Loaders use `huggingface_hub` and
pandas directly.

## The span contract

Everything rests on one invariant, defined in [rag/chunking/base.py](rag/chunking/base.py):

```python
document[chunk.start:chunk.end] == chunk.original_text
```

Loaders record evidence as character offsets into the document, so retrieval is
scored by span overlap however a document was cut. Consequences:

- A chunking method returns `list[Chunk]` built with `to_chunks(document, spans, doc_id, metadata)`. Never return text that was rewritten, stripped or re-joined.
- Library splitters go through `split_spans`, which reads LlamaIndex's native offsets and raises if a node is not at the offset it claims. `verify(document, chunks)` checks bounds, text and ordering.
- HTML chunks hold raw markup in `original_text` for the same reason. They carry `metadata['markup'] = "html"`, and their `text_to_embed` has the tags stripped, from the chunker on and through every augmentation. `experiments/ingest.py` marks chunks of the HTML rendering whatever method cut them.
- Each rendering (`documents/`, `documents_md/`, `documents_html/`) has its own offsets and its own evidence text. One span never applies to two renderings.
- Every loader asserts `document[start:end] == text` for each evidence span before finishing, and records how the span was found in `match`.

## Dataset loaders

All loaders write the same layout (`documents/<doc_id>.txt`, optional
`documents_md/` and `documents_html/`, and `items.jsonl`); the item schema is in
the README under "Dataset processing". Each loader's module docstring explains
why the source was reshaped. Loaders default to the smallest representative
split because corpus size is the binding constraint on local compute.

## Chunking

Every method has the signature `method(document, ..., doc_id="") -> list[Chunk]`. Files are in `rag/chunking/`.

| Function | File | Notes |
|---|---|---|
| `fixed_char` | `fixed.py` | Character windows, hand-sliced |
| `fixed_token` | `fixed.py` | LlamaIndex `TokenTextSplitter`: token windows cut on spaces |
| `sentence` | `structure.py` | LlamaIndex `SentenceSplitter`; covers both the sentence and recursive methods |
| `markdown` | `markdown.py` | `MarkdownNodeParser`, then `SentenceSplitter` for sections over budget; heading path in metadata |
| `html` | `html.py` | Outline read from BeautifulSoup source positions; no library keeps the markup |
| `semantic`, `tiled`, `clustered` | `semantic.py` | Embedding-based; `embed_model` is required, built with `openai_embedding(model)` |
| `lumberchunker` | `lumberchunker.py` | LLM-based, through `call_llm`; an unusable model answer counts as a refusal |

Conventions:

- Prefer a LlamaIndex node parser. Write plain code only where no library fits, and say why in the module docstring, as the existing modules do.
- Budgets are in tokens (tiktoken `cl100k_base` via LlamaIndex's `get_tokenizer`), not characters. Polish costs about twice the tokens per character that English does.
- Sentence segmentation is pysbd, run per paragraph (`segment.py`). `language` defaults to `"auto"`: `detect_language` (lingua, restricted to the languages pysbd supports) reads it from the document. A pysbd code (`"pl"`, `"en"`) overrides it. `experiments/ingest.py` detects once per dataset and records the result in the chunks' `config.json`.
- `overlap` (`sentence`, `fixed_token`, `fixed_char`) below 1 is a share of the chunk size (`resolve_overlap` in `base.py`): 0.25 at `max_tokens=256` is 64 tokens.
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
The files are in `rag/augmentation/`.

| Function | Metadata key | LLM calls |
|---|---|---|
| `add_title` | `parent_document_title` | One per document |
| `add_summary` | `parent_document_summary` | One pass per document, recursive for long ones |
| `add_context` | `context` | One per chunk, with a document window of `max_context_tokens` |
| `add_keywords` | `keywords` | One per chunk |
| `add_questions` | `questions` (a list) | One per chunk |
| `use_summary` | `summary` | One per chunk; replaces the chunk text in `text_to_embed` |

Conventions:

- Call the plain OpenAI client through `call_llm` (in `rag/llm.py`) / `call_llm_for_each`. No LlamaIndex nodes, docstores or LLM classes.
- `call_llm` retries only on rate limits (60 s wait) and raises on an empty answer, so a run never silently mixes augmented and bare chunks.
- Every prompt tells the model to answer in the chunk's own language. The exact wording was tuned against Polish output and is explained in comments above each prompt; do not reword it without re-testing on PoQuAD.

## Code style

- Module docstrings carry the rationale: what the method is, which library was chosen or rejected, and measured figures behind the decision. Follow that pattern in new modules.
- Comments explain why, often with a measurement. Function docstrings describe parameters in prose.
- Plain functions and one dataclass; no class hierarchies or framework abstractions.
- Files are written with `encoding="utf-8", newline="\n"`.

## Known gaps

- `rag/chunking/lumberchunker.py` is not exported from `rag/chunking/__init__.py`.
- `requirements.txt` lists `llama-index-embeddings-openai`, but the code imports `llama_index.embeddings.openai_like` (package `llama-index-embeddings-openai-like`). `scikit-learn`, `scipy`, `numpy`, `httpx` and `tiktoken` are imported but not listed.
- `app.py` and `lumberchunker.py` were written in a different style from the rest (type hints, no rationale docstrings).
