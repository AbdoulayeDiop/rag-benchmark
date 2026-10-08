# RAG experiments

This is a repository of experiments on retrieval augmented generation (RAG) that evaluate different RAG methods and configurations.
You will find the implementation of various methods of the different RAG steps/modules :
1. Indexation
   1. Chunking
   2. Chunk augmentation
   3. Embedding
2. Retrieval
3. Reranking
4. Evaluation

## Methodology

### Datasets
We consider datasets that are representative of real-word use cases and allow to evaluate the different steps of the RAG pipeline.
A particular focus is put on datasets documents size and domain diversity. The datasets should include enough long documents for the chunking step to be pertinent. If documents are too short, there is no need for chunking.

Here are the considered datasets :

1. **GutenQA**: a benchmark of 100 carefully cleaned public-domain books paired with 3,000 needle-in-a-haystack type of questions. [github](https://github.com/joaodsmarques/LumberChunker) | [hf](https://huggingface.co/datasets/LumberChunker/GutenQA) | [paper](https://arxiv.org/html/2406.17526v1)

2. **LiteraryQA**: a long-context question-answering benchmark focusing on literary works. [github](https://github.com/sapienzaNLP/literaryQA) | [hf](https://huggingface.co/datasets/sapienzanlp/LiteraryQA) | [paper](https://arxiv.org/html/2510.13494v1)

3. **NovelQA**: a benchmark to evaluate the long-text understanding and retrieval ability of LLMs. The dataset is constructed by manually collecting questions and answers about English novels that are above 50,000 words. [github](https://github.com/NovelQA/novelqa.github.io) | [hf](https://huggingface.co/datasets/NovelQA/NovelQA) | [paper](https://arxiv.org/html/2403.12766v3)
   
4. **Qasper**: a dataset for question answering on scientific research papers. It consists of 5,049 questions over 1,585 Natural Language Processing papers. [hf](https://huggingface.co/datasets/allenai/qasper) | [paper](https://arxiv.org/html/2105.03011v1)

5. **TriviaQA**: a reading comprehension dataset of question-answer-evidence triples. It consists of over 650K triples: 95K question-answer pairs authored by trivia enthusiasts, together with independently gathered evidence documents (six per question on average) from Wikipedia and the web.
[hf](https://huggingface.co/datasets/mandarjoshi/trivia_qa) |
[paper](https://arxiv.org/abs/1705.03551)

6. **PoQuAD**: the Polish Question Answering Dataset. It is modeled on the SQuAD 2.0, including the impossible questions.
[github](https://github.com/weed478/poquad) |
[paper](https://dl.acm.org/doi/10.1145/3587259.3627548)

7. **ConditionalQA**: a Question Answering (QA) dataset that contains complex questions with conditional answers, i.e. the answers are only applicable when certain conditions apply.
[github](https://github.com/haitian-sun/ConditionalQA) |
[paper](https://arxiv.org/html/2110.06884v1)

8. **TechQA**: a domain-adaptation question answering dataset for the technical support domain.
[hf](https://huggingface.co/datasets/rojagtap/tech-qa) |
[paper](https://arxiv.org/abs/1911.02984)

9. **MS MARCO**: a large scale MAchine Reading COmprehension dataset, which comprises of 1,010,916 anonymized questions, sampled from Bing's search query logs, each with a human generated answer and 182.669 completely human rewritten generated answers.
[hf](https://huggingface.co/datasets/microsoft/ms_marco) |
[paper](https://arxiv.org/html/1611.09268v3)

10. **SQuAD**: the Stanford Question Answering Dataset. 100,000+ questions posed by crowdworkers on Wikipedia articles, where the answer to every question is a span of text from the corresponding passage.
[site](https://rajpurkar.github.io/SQuAD-explorer/) |
[hf](https://huggingface.co/datasets/rajpurkar/squad) |
[paper](https://arxiv.org/abs/1606.05250)

11.  **Natural Questions**:
[github](https://github.com/google-research-datasets/natural-questions)
[paper](https://aclanthology.org/Q19-1026/)

### Dataset processing
[Smigielski et al.](https://arxiv.org/abs/2606.00881v1) share a [github page](https://github.com/ApriiM/Chunking-Research) where they propose an extraction of the different datasets except TechQA and ConditionnalQA. However, for several datasets (including GutenQA, PoQuAd, and Qasper) they do not consider the complete source documents and rely on passages, paragraphs, or abstracts only. Further more they convert all document to raw text without conserving the original document format which may be exploited by retrievers.

Every dataset is loaded by `datasets/<name>/load.py`, run from the repository root:

```bash
python datasets/gutenqa/load.py
```

Each loader downloads its source, reconstructs the documents and writes the same
layout, so that any dataset can feed the pipeline unchanged:

```
datasets/<name>/data/documents/<doc_id>.txt        always
datasets/<name>/data/documents_md/<doc_id>.md      when the source has structure
datasets/<name>/data/documents_html/<doc_id>.html  when the source is HTML
datasets/<name>/data/items.jsonl                   one record per question
```

Downloads are cached under `datasets/<name>/.cache/`, so re-running a loader is
cheap. `data/` and `.cache/` are gitignored; the corpora are rebuilt, not committed.

An item looks like this:

```json
{
  "id": "qasper-b584739622d0",
  "question": "What are the evaluation metrics used?",
  "answer": "Perplexity; Response-Intent Prediction (RIP)",
  "evidences": [
    {
      "doc_id": "1911.09920",
      "text": "Experiments ::: Automatic Evaluation Metrics",
      "start": 22928,
      "end": 22972, "match": "exact",
      "kind": "paragraph",
      "markdown": {
         "text": "### Automatic Evaluation Metrics", 
         "start": 23180, 
         "end": 23212
      }
    }
  ],
  "metadata": {
      "doc_id": "1911.09920",
      "answer_type": "extractive",
      "...": "..."
   }
}
```

`start`/`end` are character offsets into the document, so a retrieved chunk can be
scored by span overlap however it was cut. Because headings render differently in
each version, an evidence carries its own offsets **and its own text** per
rendering — never assume one span works for all of them. `match` records how the
span was found (`exact`, `folded`, `element`, `document`, …), so weak spans can be
filtered out. Every loader verifies `document[start:end] == text` for every span
before finishing.

#### Status

| Dataset | Domain | Docs | Median chars | Items | Evidence | Renderings |
|---|---|---:|---:|---:|---|---|
| GutenQA | Literature, novels | 100 | 459,613 | 3,000 | 3,000 spans | txt |
| LiteraryQA | Literature, novels | 138 | 351,107 | 3,785 | none | txt, md, html |
| NovelQA | Literature, novels | — | — | — | — | **skipped**, see below |
| Qasper | Research papers | 416 | 23,187 | 1,451 | 3,726 spans | txt, md |
| TriviaQA | Open-domain, Wikipedia (trivia) | 1,000 | 21,597 | 947 | document-level | txt, md |
| PoQuAD | Open-domain, Wikipedia (**Polish**) | 942 | 12,477 | 2,805 | 2,336 spans | txt, md |
| ConditionalQA | Government policy, UK gov.uk | 652 | 5,977 | 2,623 | 10,284 spans | txt, md, html |
| TechQA | Technical support, IBM technotes | 2,000 | 3,249 | 910 | 601 spans | txt, md |
| MS MARCO | Web pages, Bing search results | — | — | — | — | **skipped**, see below |
| SQuAD 1.1 | Open-domain, Wikipedia | 100 | 31,579 | 23,557 | 31,203 spans | txt |
| Natural Questions | Open-domain, Wikipedia (search queries) | 300 | 22,263 | 315 | 640 spans | txt, md, html |

Four of the nine built datasets are Wikipedia, so they differ mainly in question
style — trivia, translated search queries, crowdworker questions and Polish —
rather than in the writing itself. The genuinely distinct registers are the
literary ones, scientific papers, legal-administrative prose and technical support
notes. Worth keeping in mind when reading per-dataset results: agreement between
two Wikipedia sets is weaker evidence than agreement between gov.uk and IBM notes.

#### Processing decisions

Choices that are not recoverable from the loader output alone:

- **GutenQA** ships pre-segmented into 36,917 LumberChunker chunks, not books. The
  loader concatenates each book's chunks back in order, so our own chunking is
  measured instead of LumberChunker's. 349 of 3,000 evidence excerpts are LLM
  paraphrases that appear nowhere in the book; those fall back to the whole gold
  chunk and are marked `match: "chunk"`.
- **LiteraryQA** cannot be loaded with `load_dataset`: the Hub repository is a
  script (so it needs `trust_remote_code`) and its published copy declares an
  empty `_LOCAL_FILES`, which raises `KeyError` before downloading anything. The
  loader instead drives the authors' own cleaning code, vendored under
  `datasets/literaryqa/vendor/` at a pinned commit. Answers are abstractive and
  written from plot summaries, so **`evidences` is empty for every item** — usable
  for answer quality, not for retrieval precision.
- **NovelQA is skipped: its ground truth is held out.** The dataset is gated on
  Hugging Face (auto-approved) and 60 of its 89 novels include text — the other 28
  are copyright-protected and ship questions without a book. But all 1,526
  questions over those 60 novels carry only `Aspect`, `Complexity`, `Question` and
  `Options`: no answer, no gold option, no evidence. Those are reserved for the
  [Codabench leaderboard](https://www.codabench.org/competitions/2727/). Only the
  demonstration book (B30) ships the full schema, and even there the evidence
  strings are reformatted quotations — 8 of 48 are locatable in the book text. The
  corpus would be the longest here (median 219,711 tokens per novel), but GutenQA
  and LiteraryQA already cover long literary documents and GutenQA has real spans.
  **Revisit when the authors provide the gold answers and evidence** — the loader
  only needs the corpus, which is already accessible.
- **Qasper** resolves a conflict in the literature: full papers average 24,471
  characters, so the figure of ~1,014 characters reported elsewhere describes
  abstracts, not papers. 459 evidence annotations point at a figure or table
  caption, which the source stores positionless, so captions are appended under a
  trailing "Figures and Tables" heading to keep them locatable.
- **TriviaQA** is capped at 1,000 articles; the full validation split is 10,033
  pages and 234M characters. Its annotation is **document-level** — a question is
  linked to whole articles, never to a passage — so `evidences` spans the entire
  document with `text: null`. `metadata.answer_occurrences` adds where the answer
  string literally appears (99.7% of items), which is distant supervision inferred
  by the loader, not annotation, and is deliberately kept out of `evidences`.
  Wikipedia section headings are detected heuristically (short trailing-space
  lines) and the original nesting is unrecoverable, so all render at one level.
- **PoQuAD** ships one ~930-character paragraph per record, which gives chunking
  nothing to do. The loader refetches each record's full Wikipedia article and
  relocates the answer spans into it, turning ~930-character passages into
  documents with a median of 12,477 characters. The paragraphs date from 2022, so
  379 of 959 no longer appear verbatim; those 1,837 questions are dropped and the
  loader reports the loss. Every one of the 2,336 surviving answer spans was
  located. The 372 articles whose passage drifted are kept as distractors.
  Wikipedia refuses requests without a descriptive `User-Agent`, rejects batching
  for whole-article extracts, and answers 429 with a `Retry-After` header that
  must be honoured, hence roughly 4.5s per article.
- **TechQA** is loaded from NVIDIA's `TechQA-RAG-Eval` repackaging, not the link
  in the list above. `rojagtap/tech-qa` keeps only the gold document per question,
  leaving no corpus to retrieve from; the original `PrimeQA/TechQA` archive is
  2.96GB, nearly all of it a 800k-technote corpus that would be subsampled to a
  fraction of a percent. NVIDIA's version dropped the original offset fields, but
  TechQA answers *are* the annotator-marked excerpt (median 255 characters), so
  locating that text recovers the human annotation rather than inventing one —
  unlike TriviaQA, where a short entity string can occur incidentally. The corpus
  is capped at 2,000 technotes: all 496 gold documents plus distractors. Two
  questions flagged answerable ship an empty answer and are recorded as
  unanswerable. Technotes quote shell snippets, so some body lines begin with `#`
  and read as headings to a Markdown parser although the loader did not mark them.
- **ConditionalQA** is the cleanest source here: all 10,284 evidence entries match
  a page element verbatim. A question is meaningless without its scenario
  ("How long will it be before I hear back from the court?"), so **`question` holds
  the scenario and the question joined**, with the two parts kept separately in
  `metadata.scenario` and `metadata.original_question`. Markdown comes from
  `markdownify` with escaping disabled; table rows are the exception, since the
  dataset stores cells as pipe-separated text in a bare `<tr>` with no `<td>`
  markup to convert.

- **MS MARCO is skipped.** Its passages average 305 characters, which leaves
  chunking nothing to do. Grouping passages by source URL does not rescue them:
  73,884 of 114,373 URLs contribute a single passage. The document-ranking variant
  does have real documents, but the uncompressed collection is not served and gzip
  is not seekable, so the published byte-offset lookup cannot be used and
  extracting any subset means streaming all 8.4GB; its qrels are document-level
  only, the same granularity TriviaQA already provides, and they live in a
  different id space from the `is_selected` passage labels. The cost was judged
  not to buy anything the other datasets lack. Reconsider if web-search query
  distribution becomes important — that is the one thing no other dataset here
  covers.

- **Natural Questions** is the best-annotated source here: spans arrive as token
  indices, so the loader builds the text from the token stream and keeps a
  token-to-character map, making every span exact by construction with no text
  matching anywhere. It carries evidence at two granularities — the *long answer*
  (the paragraph, table or list that answers) and the *short answer* inside it —
  from five annotators per question. The raw Wikipedia HTML is shipped but is five
  to seven times the body text (median 168k vs 22k characters), nearly all of it
  navigation, scripts and footers; dumping it would leave the HTML rendering
  holding different content from the other two and destroy the comparison they
  exist for, so it is rebuilt from the same blocks. Two offset traps are worth
  knowing: `html.escape` defaults to escaping apostrophes, which appear in almost
  every paragraph, and a long answer covering a whole table spans several blocks —
  both break any assumption of a single constant offset shift. 8 annotated ranges
  cover table regions whose cells the tokeniser strips and hold no body text.

- **SQuAD 1.1** is distributed as ~700-character paragraphs, but every paragraph
  carries the title of its source article and the paragraphs under one title are
  consecutive sections of it. Joining them in order rebuilds the article — a
  median of 44 paragraphs and 31,579 characters — with no refetching and nothing
  invented, which is why SQuAD grouped cleanly where PoQuAD (one paragraph per
  record) did not. Answer offsets are paragraph-relative and shift by the
  paragraph's position, staying exact: all 31,203 spans verify.
  **Version 1.1 is used rather than 2.0 on purpose.** 2.0's unanswerable questions
  were written to be unanswerable from the *single paragraph* shown, a label that
  grouping invalidates since the rest of the article may answer them; 27% had a
  near-duplicate answerable question elsewhere in the same article, and with no
  gold answer the label can only ever be contradicted, never confirmed. 1.1 has no
  such questions, so nothing is discarded. For abstention use TechQA, Qasper or
  Natural Questions, where the judgement covers the whole document.
  The corpus is capped at 100 articles, all 48 validation ones plus 52 from train,
  so retrieval is not trivial. Validation questions carry up to six annotator
  answers, giving a tolerance band on span boundaries rather than one arbitrary
  span. `metadata.context_span` marks the paragraph the question was written
  against — the gold passage a retriever should return — and every answer was
  checked to fall inside it. Only plain text is written: SQuAD keeps no section
  headings, so there is no structure to mark.

#### Splits

Loaders default to the smallest representative split, because corpus size is the
binding constraint on local indexing and retrieval. Where a dataset's test split
is held out (TriviaQA answers are `<unk>`, PoQuAD and ConditionalQA test ship no
answers), the loader uses validation or dev instead and says so. Use `--splits`
to build more.

### Chunking methods

1. Fixed size: tokens (`fixed_token`, LlamaIndex `TokenTextSplitter`, cut on spaces) or characters (`fixed_char`)
2. Structure preserving (with upper bound)
   1. Sentence
   2. Markdown (used on md documents only)
   3. HTML (used on html documents only)
3. Semantic
   1. Breakpoint/consecutive sentences similarity
   2. Clustering-based
   3. LLM-based ([LumberChunker](https://github.com/joaodsmarques/LumberChunker/blob/main/Code/LumberChunker-Segmentation.py))

### CHunk augmentation methods
1. Parent document title
2. Parent document summary
3. Contextual chunking (by anthropic)
4. Keywords
5. Hypothetical questions
6. Chunk summary (embedded in place of the chunk)

Implemented in `rag/augmentation/`. A method takes the chunks of one document and
returns them with `text_to_embed` set. `original_text` and the offsets are left
alone, so an augmented chunk is still scored by span overlap. `text_to_embed`
is None on a chunk nothing was added to; `get_text_to_embed(chunk)` returns the
right one either way. Every method calls an OpenAI-compatible endpoint
directly, with no LlamaIndex nodes or docstores in between.

```python
from augmentation import (add_context, add_keywords, add_questions, add_summary,
                          add_title, use_summary)

chunks = add_title(chunks)                   # one LLM call per document, on its first k chunks
chunks = add_summary(chunks, document)       # one LLM pass over the whole document
chunks = add_context(chunks, document)       # one LLM call per chunk, reading the document
chunks = add_keywords(chunks)                # one LLM call per chunk, reading the chunk
chunks = add_questions(chunks)               # one LLM call per chunk, reading the chunk
chunks = use_summary(chunks)                 # one LLM call per chunk, reading the chunk
```

What each method generated is kept under its own metadata key, and
`text_to_embed` is rebuilt from them in a fixed order, so methods combine in
any order:

| Order | Metadata key | Set by |
|---|---|---|
| 1 | `parent_document_title` | `add_title` |
| 2 | `parent_document_summary` | `add_summary` |
| 3 | `context` | `add_context` |
| 4 | the chunk's `original_text` | — |
| 5 | `keywords` | `add_keywords` |
| 6 | `questions` | `add_questions` |

`use_summary` is the exception: a chunk with a `summary` of itself is embedded
as that summary alone, and the other keys are left out of `text_to_embed`.

Cost is what separates them. `add_context` shows the model the document again
for every chunk; documents over `max_context_tokens` (8,000) are shown as a
window around the chunk instead of whole. At the endpoint's limit of 128,000
input tokens a minute, and estimating 4 characters per token, contextualising a
whole corpus takes:

| Dataset | 256-token chunks | 1,024-token chunks |
|---|---:|---:|
| ConditionalQA | 1.8 h | 0.5 h |
| TechQA | 2.4 h | 0.6 h |
| SQuAD | 3.1 h | 0.8 h |
| Qasper | 8.3 h | 2.1 h |
| Natural Questions | 9.5 h | 2.4 h |
| PoQuAD | 15.5 h | 3.9 h |
| TriviaQA | 26.0 h | 6.5 h |
| LiteraryQA | 57.9 h | 14.5 h |
| GutenQA | 69.5 h | 17.4 h |

### Embedding models

`bge-m3` through the OpenAI-compatible endpoint, in `rag/embedding.py`. It is
multilingual, which PoQuAD needs. Measured on the endpoint: 1,024 dimensions,
unit-length vectors (a dot product is a cosine), at most 64 texts a request,
and an input over 8,192 of the model's own tokens is refused rather than
truncated. With `context_length` in the `chunk_embedding` entry, a chunk over the limit is cut to fit
before it is sent, and its node is marked `truncated`, so results on it can be
read with that in mind; only the unbounded chunkers (`semantic`, `tiled`,
`lumberchunker`) or a long generated context produce one. The node keeps its
whole text; only what is sent is cut.

The cut is counted in tokens. `"tokenizer": "BAAI/bge-m3"` counts in the
model's own tokenizer, which matches the server exactly: a text cut to 8,192
tokens is accepted. Without it, tiktoken's cl100k_base stands in, and it
undercounts English for bge-m3 by 12-14% (8,192 tiktoken tokens of a GutenQA
book came to 9,161-9,372 bge-m3 tokens), so the limit then needs a margin --
7,000 rather than 8,192. Without `context_length` nothing is cut and an
over-long chunk stops the run.

What is embedded is the chunk's `text_to_embed` (`get_text_to_embed(chunk)`),
and sparse retrieval indexes the same text. For an HTML chunk it is already
free of tags: the chunk keeps its markup in `original_text`, which its offsets
cover, and is marked `markup: "html"`, and `text_to_embed` is the stripped text
-- set by the chunker and kept by every augmentation.

### Ingestion

The pipeline is split in two. `rag/` is the library, one module per stage,
which takes every input, model and directory as an argument and can be used
outside these experiments. `experiments/` holds one script per stage, each
driven by a JSON configuration file, which decide where runs live on disk:

```powershell
.venv\Scripts\python.exe -m experiments.ingest experiments/configs/ingest_chunking.json --dry-run   # list the runs
.venv\Scripts\python.exe -m experiments.ingest experiments/configs/ingest_chunking.json             # run; re-run to resume
```

`experiments.ingest` chunks, augments, embeds and indexes every combination of
the file's `dataset`, `chunking`, `chunk_augmentation` and `chunk_embedding`
entries (the keys are described in its module docstring). No model is assumed
anywhere in the code, and none is read from the environment: the file names
every model a run calls, and a run that needs one and was given none stops
before doing anything. The model names a run used are recorded in its
`config.json`.

Chunks and indexes are two trees, one directory level per choice, each named
`<method>-<param>=<value>-...` with the parameters given:

```
experiments/chunks/<dataset>/<rendering>/<chunking>/<augmentation>/
    config.json        records everything that decides the chunks' content.
    chunks.jsonl       holds one chunk per line, in document order.
    documents.jsonl    holds one line per finished document, with its cost.
    cost.json          adds those costs up.
experiments/indexes/<dataset>/<rendering>/<chunking>/<augmentation>/<embedding model>/
    config.json        names the chunks it was built from and the embedding.
    chroma/            holds the Chroma database.
    bm25/              holds the persisted BM25Retriever.
    cost.json          holds what embedding cost.
```

For example `squad/txt/sentence-max_tokens=256-overlap=0/none/bge-m3`, or
`qasper/md/markdown-max_tokens=512/context-llm_model=.../bge-m3`. An index
reads its chunks from the chunks tree rather than copying them. It is read
back with `rag.ingestion.open_vector_store(chroma, model)` and
`load_bm25(bm25, language, top_k)`; `node_to_chunk` turns a retrieved node
into its `Chunk`, and `experiments.common.load_chunks(directory)` reads a
whole chunks run.

- **Dense: Chroma through LlamaIndex.** Vectors go into a persistent Chroma
  collection wrapped in `ChromaVectorStore`, compared by cosine. Chroma's index
  is HNSW, so dense search is approximate and a method's score includes the
  index's own recall -- small at these sizes (at most some 45,000 chunks), not
  zero. Vectors are still computed by `embed_batch` and handed to the store on
  the nodes, because it handles the endpoint's length refusals.
- **Sparse: LlamaIndex `BM25Retriever`**, persisted with `persist` and rebuilt
  on each run, since it calls no model. English is stemmed and loses its stop
  words. Polish has neither a Snowball stemmer in PyStemmer nor a stop-word
  list in bm25s, so PoQuAD is indexed on whole words. `persist` does not save
  the stemming settings; load with `load_bm25`, which restores them, rather
  than `BM25Retriever.from_persist_dir` directly.
- **Both stores hold the same nodes, and a node is the whole chunk**: one per
  chunk, its text being the chunk's `text_to_embed`, its id `<doc_id>:<chunk index>`.
  Its metadata holds `doc_id`, `chunk_index`, `start`, `end`, `original_text`,
  `truncated`, and every key of the chunk's own metadata (heading path, title,
  summary, context, keywords, questions). `text_to_embed` is the node's text,
  not a field. `node_to_chunk(node)` rebuilds the `Chunk` from a node either
  store returns, recomputing `text_to_embed` from the augmentations. Chroma takes only flat values, so a list such as the questions
  is stored as JSON and `json_encoded_keys` names it. None of it is in the text BM25
  indexes.
- **`chunks.jsonl` stays** as the build's checkpoint and the plain-text record
  of a run, for verifying an experiment. Retrieval does not need it.
- **Resumable.** A document's chunks are written together and the document is
  logged only once they are on disk; vectors are added a batch at a time, in
  chunk order. A stopped run continues at the first unfinished document and
  repeats no LLM call of a finished one, which matters for the runs in the cost
  table above. `max_documents` can be raised later: the run is extended, not
  redone.
- **Costs are recorded with the work.** Each line of `documents.jsonl` holds
  the document's `chunking` and `augmentation` cost: seconds, LLM input and
  output tokens, embedding tokens (the semantic chunkers'), and
  `rate_limit_seconds`, the minutes waited out after a refusal. the chunks'
  `cost.json` adds them up, with the `workers` of the last run, and the
  index's `cost.json` keeps a running total for its embedding, updated after
  each batch. Tokens are what the endpoint
  reports, counted by a hook on the HTTP client (`llm.usage`). For the LLM
  methods they, not the seconds, are the cost to compare: the seconds are
  mostly the endpoint's 128,000 tokens a minute. Documents finished before
  costs were recorded have none.
- **The embedding model is its own axis.** Another `chunk_embedding` entry
  adds an index beside the first and leaves the chunks, and the LLM calls
  behind them, alone.
- **One directory, one configuration.** `config.json` records everything that
  decides a chunk's content, with the chunking method's defaults filled in; a
  directory that already holds documents refuses a different one.
- The rendering follows the method (`md` for `markdown`, `html` for `html`,
  `txt` otherwise) and can be overridden with a chunking entry's `rendering`.
  The language is detected, once per dataset, from the opening of the first
  documents, unless the file's `language` names one; the chunking functions detect it per document by
  default (`language="auto"`). The detector is
  lingua, restricted to the languages pysbd has rules for; it named the right
  language for every document tried (up to 150 per dataset, 1,238 in all) at
  about 3 ms a document.


### Retrieval methods
1. Semantic (`dense` in `rag/retrieval.py`): the query embedded with `embed_batch`, as
   the chunks were, and searched in the run's Chroma collection by cosine
2. Sparse (BM25, `sparse`): the run's persisted `BM25Retriever`, stemmed as the
   index was (see `load_bm25`)
3. Hybrid (`hybrid`): the top-k of each, merged by reciprocal rank fusion with
   k = 60. Fusion uses ranks only, so BM25 scores and cosine similarities need
   no normalising against each other
4. Hierarchical: not implemented

The query is not embedded through a LlamaIndex embedding class: those send a
text with its newlines replaced, so it would not be embedded as the chunks were.
For the same reason the fusion is written out rather than taken from
`QueryFusionRetriever`.

```powershell
.venv\Scripts\python.exe -m experiments.retrieve experiments/configs/retrieve_dense.json
```

`experiments.retrieve` runs every retrieval entry of its file on every index
its `index` selects -- a path under `experiments/indexes/`, a glob pattern
(`squad/*/*/none/bge-m3`), `"all"`, or a list -- over the first `max_items`
questions of the sample order. Each run writes
`experiments/results/<index>/<method>-top_k=<k>/`: `config.json`, and
`ranking.jsonl`, one line per question with its ranking (`[doc_id, start,
end, chunk index, score]` per chunk, no text) and its `cost`:
`retrieval_seconds`, with its share of the batched query embedding, and
`embedding_tokens`.

### Re-ranking
`rag/rerank.py` sends the query and the retrieved chunks to the endpoint's `/rerank`
route (bge-reranker-v2-m3) and reorders them by its score. A chunk is scored as
its `text_to_embed`, so augmentations reach the reranker too. The endpoint
accepts at most 64 documents a request and refuses a query–chunk pair over
8,192 tokens rather than truncating it. `"max_tokens": 8192, "tokenizer":
"BAAI/bge-m3"` cuts a chunk to fit, counted with the reranker's own tokenizer,
which matches the server's count exactly. Only GutenQA's 40,000-character
chunks (9,700–10,800 tokens) need it; the reranker then reads only their first
~80%.

`experiments.rerank` reranks every retrieval run its `retrieval` selects (a
path under `experiments/results/`, a pattern, `"all"`, or a list) with every
reranking entry, and writes the reordered rankings under the retrieval run:
`<retrieval>/<model>-max_tokens=<n>-top_k=<k>/`, in the same shape as a
retrieval run, the question's cost adding `rerank_seconds` and
`rerank_tokens` to the retrieval's. Reranking seconds depend on the
concurrency (`workers`, 16) and on the shared endpoint's load, so compare
them within one sitting.

### Generation
Not run yet: see Evaluation.

### Evaluation
Chunking is compared on **retrieval** alone for now. An answer generated from
the retrieved chunks measures the generator and its judge as much as the chunks.

`experiments.evaluate` scores the rankings of retrieval and reranking runs
against the evidence spans the loaders record (`rag/evaluation.py`).
It calls no model beyond retrieval itself and gives the same result every time:

| Metric | Meaning |
|---|---|
| `hit@k` | a chunk among the first k overlaps an evidence span |
| `mrr@10` | 1 / rank of the first such chunk (0 beyond 10) |
| `ndcg@k` | binary relevance; the ideal ranking puts first every chunk of the index that overlaps evidence |
| `evidence_recall@k` | share of the evidence characters (their union) the first k chunks cover |
| `doc_hit@k` | a chunk among the first k comes from an evidence document |

`evidence_recall` is counted in characters, overlapping evidence spans and
overlapping chunks once. All the metrics favour large chunks, so read them
against the chunk length `experiments.compare` prints beside them. Each
metric is computed at k = 1, 3, 5, 10, over two groups of questions kept
apart:
- `evidence`: questions with spans in the run's rendering. All metrics apply.
- `no_evidence`: questions tied to a document but with no span. These are
  unanswerable questions, and all of LiteraryQA. Only `doc_hit` applies.

Questions whose documents are not all in the index are left out, so an index
over `max_documents` is scored on its own documents' questions.

Evidence differs by dataset:
- **SQuAD:** several spans are alternative annotators' answers.
- **Qasper:** spans merge every annotator's evidence, so `evidence_recall` undercounts.
- **ConditionalQA:** spans are all needed together.
- **TriviaQA:** the evidence is a whole document, so there `hit` = `doc_hit`, and `evidence_recall` means little.
- **GutenQA:** 12% of spans are the source dataset's gold chunk, not the answer itself.

```powershell
.venv\Scripts\python.exe -m experiments.evaluate experiments/configs/evaluate_spans.json
.venv\Scripts\python.exe -m experiments.compare             # per dataset, each method's best parameters
.venv\Scripts\python.exe -m experiments.compare --all       # every parameter combination
```

`experiments.evaluate` takes `results`, a path under `experiments/results/`
(a retrieval or a reranking run), a pattern, `"all"`, or a list, and writes
into each run's own directory `evaluation/summary.json`: the metrics, and
under `cost` the mean cost of a question and the median seconds, from
`ranking.jsonl`. The span metrics are recomputed from the rankings each time.

`experiments.compare --cost` and `experiments.compare_chunking --cost` add the
cost columns: `index s/doc` (chunking, augmentation, and embedding the
document's chunks), `LLM tok/doc`, `embed tok/doc` and `query s` (retrieval,
plus reranking for a reranked run).

`max_items` takes a sample in a fixed order (by a hash of the question id),
so a larger sample contains a smaller one.

#### RAGAS context precision and recall

With a `ragas` object in its file, `experiments.evaluate` also judges each run with two of the
[RAGAS library](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)'s
metrics, which read the retrieved text rather than its offsets, so they check
the span metrics where evidence is loose or another chunk says the same thing:

| Metric | Meaning |
|---|---|
| [`context_precision@k`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) | each of the first k chunks judged useful or not for reaching the reference answer (one call per chunk); the average precision of the verdicts |
| [`context_recall@k`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) | the reference answer split into statements, each judged attributable to the first k chunks or not (one call); the share attributed |

```powershell
.venv\Scripts\python.exe -m experiments.evaluate experiments/configs/evaluate_ragas.json
.venv\Scripts\python.exe -m experiments.compare_chunking --judge openweight-large   # adds both as columns
```

- **Library, not rewritten:** `ContextPrecision` (with reference) and
  `ContextRecall` from `ragas.metrics.collections`, so the figures are RAGAS's
  own. The precision without reference judges against a generated answer,
  which there is none of yet. Only questions with evidence spans and a
  reference answer are judged.
- **What the judge reads:** each chunk as `build_text_for_llm` gives it
  (`rag/chunking/base.py`): its augmentations, each headed by its metadata key,
  then `original_text` (tags stripped for HTML). `text_to_embed` joins the
  augmentations bare, which an encoder can take but a judge could mistake for
  the document's words; and a chunk summary is shown above the text rather
  than in its place.
- **Cost:** k + 1 calls a question. The endpoint allows gpt-oss-120b 10
  requests a minute, and `requests_per_minute` spaces the calls to that; a
  refused call waits a minute and is sent again. Measured at k = 5: 4 SQuAD
  questions in 3 min 26 s, about 50 s a question, so the default sample of
  200 (`max_items`, the first of the sample order) takes close
  to 3 hours a set-up. RAGAS is for comparing a few set-ups, not a grid.
- **Lenient judge.** "What does the original manuscript show?" (SQuAD, on
  Luther's hymn; reference "multiple revisions") got precision and recall of
  1.0 from both gpt-oss-120b and mistral-small-3-2-24b-instruct-2506 for five
  chunks of other articles, the first on the manuscript of *Go Set a
  Watchman*, none mentioning a revision; the span metrics scored it 0. SQuAD's
  questions take their passage for granted, and ConditionalQA's answers are
  often a bare "yes", one statement to attribute. Read the two kinds of
  metric together.
- **Output and resuming:** the run's `evaluation/summary.json` holds one
  entry per judge and `k` under `ragas`, beside the span metrics, and
  `evaluation/ragas-<judge>-k=<k>.jsonl` the verdict on each question, which a
  stopped run continues from. A question the judge's answer cannot be parsed
  for is recorded with its error and counted in `failed`, not averaged.
- **Configuration:** `ragas` takes `judge_model`, and optionally `k` (5),
  `max_items` (200), `requests_per_minute`, `workers` (8) and
  `max_output_tokens` (2,048).
- **Dependency pin:** ragas 0.4.3 imports
  `langchain_community.chat_models.vertexai`, which langchain-community 0.4
  removed, so `requirements.txt` pins it below 0.4.

Faithfulness and answer relevancy need the generation step and come with it.
- [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness)
- [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)

## Comparison of methods, one stage at a time
The methods of each stage -- chunking, augmentation, embedding, retrieval,
reranking -- are compared with each other, one stage at a time. While one
stage's methods vary, every other stage stays fixed at the baseline pipeline:

| Stage | Baseline |
|---|---|
| Chunking | `fixed_token`, `max_tokens` 256, 512, 1,024 and `overlap` 0, 0.25 swept |
| Chunk augmentation | none |
| Embedding | bge-m3 |
| Retrieval | dense, `top_k` 50 |
| Reranking | none |

Metrics are the evidence-span retrieval metrics of `experiments.evaluate` (see
Evaluation), on the same 2,000-question sample per dataset (`max_items`).
Generation (gpt-oss-120b) and the RAGAS metrics come later.

Only the chunking method is fixed, not its parameters. The best size and
overlap differ by dataset, and they may differ by alternative too: an
augmentation that adds a document's title or context may suit smaller chunks
than bare text does. So every comparison sweeps the six `fixed_token`
combinations under each alternative, and `experiments.compare` reports each
alternative at its best combination (on `evidence_recall@5` by default;
`--all` shows them all). Alternatives are then compared each at its best,
not at one size chosen for the baseline. The baseline alone, on dense
retrieval, scores at best:

| Dataset | `max_tokens` | `overlap` | `evidence_recall@5` | Mean chunk chars |
|---|---:|---:|---:|---:|
| SQuAD | 256 | 0.25 | 0.859 | 1,199 |
| ConditionalQA | 1,024 | 0.25 | 0.877 | 4,003 |

The margins are small: the runner-up is 0.015 behind on SQuAD (0.844, at
1,024 tokens with no overlap) and 0.009 on ConditionalQA (0.868, at 1,024
tokens with no overlap).

| Stage compared | Experiment file | Alternatives | Index builds | Evaluations |
|---|---|---|---:|---:|
| Chunking | `chunking_eval.json` | `fixed_token`, `sentence` (256/512/1,024 tokens, overlap 0/0.25), `markdown`, `html` (256/512/1,024) | 120 | 120 |
| Chunking, semantic | `semantic_chunking_eval.json` | `semantic`, `clustered` | 10 | 10 |
| Chunking, LLM-based | `lumberchunker_eval.json` | `lumberchunker`, by mistral-small-3-2-24b-instruct-2506 | 2 | 2 |
| Chunk augmentation | `augmentation_eval_squad_conditionalqa.json` | none, title, summary, context, keywords, questions, chunk summary, each alone, by mistral-small-3-2-24b-instruct-2506 | 84 | 84 |
| Embedding | `embedding_eval_squad_conditionalqa.json` | bge-m3, qwen3-vl-embedding-8b | 24 | 24 |
| Retrieval | `retrieval_eval_squad_conditionalqa.json` | dense, sparse, hybrid | 12 | 36 |
| Reranking | `reranking_eval_squad_conditionalqa.json` | none, bge-reranker-v2-m3 over the 50 retrieved, keeping 10 | 12 | 24 |

The chunking methods are split over three files by what they cost. The
library splitters run on every built dataset but LiteraryQA, whose questions
have no evidence spans to score. The semantic methods embed every sentence,
so they run on the datasets under 5M tokens (cl100k): ConditionalQA (1.05M),
SQuAD (0.71M), Qasper (2.12M), TechQA (2.22M) and Natural Questions (2.60M),
leaving out TriviaQA (6.39M), PoQuAD (7.06M) and GutenQA (15.66M).
LumberChunker calls the LLM for every boundary, under the endpoint's 128,000
input tokens a minute, so it runs on SQuAD and ConditionalQA only. The
semantic methods and LumberChunker have no size parameter to sweep, and no
upper bound on a chunk, so those files cut over-long chunks to bge-m3's
8,192 tokens.

The other stages' files cover SQuAD and ConditionalQA. On those two, the
baseline's twelve indexes (six combinations, two datasets) have the same names
in every file, so they are built and scored once and shared; the counts above
include them.

- **Chunking:** `markdown` and `html` run only on datasets with that
  rendering (`markdown`: ConditionalQA, Natural Questions, PoQuAD, Qasper,
  TechQA, TriviaQA; `html`: ConditionalQA, Natural Questions), and the other
  methods on the plain text.
- **Augmentation** is the slow comparison. It is limited by the endpoint's
  128,000 LLM input tokens a minute, and the sweep multiplies it: each
  method runs at all six combinations, and the per-chunk methods cost most at
  256 tokens (see the cost table under Chunk augmentation methods).
- **Embedding:** the endpoint serves two embedding models. qwen3-vl-embedding-8b
  returns 4,096 dimensions, unit length, and takes 32,768 tokens, so no
  chunk here needs cutting. A second model adds an index beside the first,
  so it re-embeds but does not re-chunk. Its queries are embedded as
  bge-m3's are, without the instruction prefix Qwen recommends for queries,
  so its scores may be slightly low.
- **Reranking:** bge-reranker-v2-m3 is the only reranker the endpoint serves,
  so the comparison is reranking against none.

The experiment files in the table describe each comparison as a whole grid,
one key per stage, in the format of the former runner, which was removed when
the pipeline was split into one script per stage. Until a runner reads them
again, a comparison is run stage by stage, each script with its own file
under `experiments/configs/` (the chunking comparison's are there):

```powershell
.venv\Scripts\python.exe -m experiments.ingest experiments/configs/ingest_chunking.json
.venv\Scripts\python.exe -m experiments.retrieve experiments/configs/retrieve_dense.json
.venv\Scripts\python.exe -m experiments.evaluate experiments/configs/evaluate_spans.json
.venv\Scripts\python.exe -m experiments.compare squad                    # every result for a dataset
.venv\Scripts\python.exe -m experiments.compare_chunking                 # the chunking methods, baseline pipeline only
```

`experiments.compare_chunking` reads only the results scored with the baseline pipeline
(no augmentation, bge-m3, dense retrieval of 50, no reranking), so the
chunking method is the only difference between two rows. Per dataset it
lists each method at its best parameters on `--metric` (`evidence_recall@5`
by default), best first, with the chunk count and mean chunk length beside
the metrics; `--all` lists every combination. An overview then sets the
methods against the datasets, with each method's mean difference to
`fixed_token` over the datasets it ran on. `fixed_token` runs on every
dataset, so that difference compares methods run on different datasets
(the semantic methods on five, LumberChunker on two) where a mean score
would not.

Every configuration file lists a stage's alternatives as `{"method": ...,
"params": {...}}`, and every combination with the runs it selects is a run.
Models a chunker calls are parameters too: `llm_model` for `lumberchunker`,
`embedding_model` for the semantic chunkers. Any entry may carry a
`description` in place of a comment.

A list inside `params` is a sweep: `"max_tokens": [256, 512], "overlap": [0, 0.25]`
runs every combination, each as its own run. A parameter whose value is
itself a list (`html`'s `headings`) is not swept.

- **Overlap:** an `overlap` below 1 is a share of the chunk size (0.25 at 256
  tokens is 64). Combinations whose overlap is not below the size are skipped.
- **Renderings:** a dataset without the rendering a method reads (SQuAD has no
  `md` or `html`) is skipped for that method, so one file can list every
  dataset.
- **Dataset names** must match their directory exactly, case included.
- **No environment variables:** nothing is read from the environment, so the
  file is the record of what was run.
- **Checked up front:** every run is validated before the first starts, and
  `--dry-run` lists them without running. `--keep-going` moves on past a
  failed run and lists the failures at the end.
- **Resumable, one configuration per directory:** every script continues what
  a stopped run left, and refuses a directory started with another
  configuration.

Rows are comparable only when their indexes cover the same documents (the
`docs` column), so every method of a dataset should be built with the same
`max_documents`.