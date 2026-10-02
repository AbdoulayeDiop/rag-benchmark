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

Implemented in `augmentation/`. A method takes the chunks of one document and
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

`bge-m3` through the OpenAI-compatible endpoint, in `embedding.py`. It is
multilingual, which PoQuAD needs. Measured on the endpoint: 1,024 dimensions,
unit-length vectors (a dot product is a cosine), at most 64 texts a request,
and an input over 8,192 of the model's own tokens is refused rather than
truncated. With `--embedding-max-tokens`, a chunk over the limit is cut to fit
before it is sent, and its node is marked `truncated`, so results on it can be
read with that in mind; only the unbounded chunkers (`semantic`, `tiled`,
`lumberchunker`) or a long generated context produce one. The node keeps its
whole text; only what is sent is cut.

The cut is counted in tokens. `--embedding-tokenizer BAAI/bge-m3` counts in the
model's own tokenizer, which matches the server exactly: a text cut to 8,192
tokens is accepted. Without it, tiktoken's cl100k_base stands in, and it
undercounts English for bge-m3 by 12-14% (8,192 tiktoken tokens of a GutenQA
book came to 9,161-9,372 bge-m3 tokens), so the limit then needs a margin --
7,000 rather than 8,192. Without `--embedding-max-tokens` nothing is cut and an
over-long chunk stops the run.

What is embedded is the chunk's `text_to_embed` (`get_text_to_embed(chunk)`),
and sparse retrieval indexes the same text. For an HTML chunk it is already
free of tags: the chunk keeps its markup in `original_text`, which its offsets
cover, and is marked `markup: "html"`, and `text_to_embed` is the stripped text
-- set by the chunker and kept by every augmentation.

### Ingestion

`ingest.py` runs the indexation steps end to end for one dataset and one
configuration, and writes an index that retrieval reads without calling a model:

```bash
export EMBEDDING_MODEL=bge-m3                             # or --embedding-model
export LLM_MODEL=mistral-small-3-2-24b-instruct-2506      # or --llm-model
python ingest.py squad --param max_tokens=256 --param overlap=0
python ingest.py conditionalqa --chunking html --param max_tokens=256
python ingest.py qasper --chunking markdown --augment title context
```

No model is assumed anywhere in the code: the library functions take the model
as an argument, and the runner takes it from its flags or the environment, as
it takes the endpoint from `OPENAI_API_BASE`. A run that needs a model and was
given none stops before doing anything. The model names a run used are
recorded in its `config.json`.

```
indexes/<dataset>/<name>/config.json              what the run is
indexes/<dataset>/<name>/chunks.jsonl             one chunk per line, in document order
indexes/<dataset>/<name>/documents.jsonl          one line per finished document
indexes/<dataset>/<name>/chroma/                  Chroma database, one collection per embedding model
indexes/<dataset>/<name>/bm25/                    persisted BM25Retriever
```

`<name>` defaults to the rendering, the method, the parameters given and the
augmentations (`txt-sentence-max_tokens=256-overlap=0`, `md-markdown+title+context`).
An index is read back with `load_vector_store(run, model)` and
`load_bm25(run, top_k)`; `node_to_chunk` turns a retrieved node into its
`Chunk`, and `load_chunks(run)` reads the whole run from `chunks.jsonl`.

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
  table above. `--max-documents` can be raised later: the run is extended, not
  redone.
- **The embedding model is its own axis.** Running again with another
  `--embedding-model` adds a collection beside the first and leaves the chunks
  alone.
- **One directory, one configuration.** `config.json` records everything that
  decides a chunk's content, with the chunking method's defaults filled in; a
  directory that already holds documents refuses a different one.
- The rendering follows the method (`md` for `markdown`, `html` for `html`,
  `txt` otherwise) and can be overridden with `--rendering`. The language is
  detected, once per run, from the opening of the first documents, unless
  `--language` names one; the chunking functions detect it per document by
  default (`language="auto"`). The detector is
  lingua, restricted to the languages pysbd has rules for; it named the right
  language for every document tried (up to 150 per dataset, 1,238 in all) at
  about 3 ms a document.


### Retrieval methods
1. Semantic (`dense` in `retrieval.py`): the query embedded with `embed_batch`, as
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

### Re-ranking
`rerank.py` sends the query and the retrieved chunks to the endpoint's `/rerank`
route (bge-reranker-v2-m3) and reorders them by its score. A chunk is scored as
its `text_to_embed`, so augmentations reach the reranker too. The endpoint
accepts at most 64 documents a request and refuses a query–chunk pair over
8,192 tokens rather than truncating it. `--rerank-max-tokens 8192
--rerank-tokenizer BAAI/bge-m3` cuts a chunk to fit, counted with the
reranker's own tokenizer, which matches the server's count exactly. Only
GutenQA's 40,000-character chunks (9,700–10,800 tokens) need it; the reranker
then reads only their first ~80%.

### Generation
Not run yet: see Evaluation.

### Evaluation
Chunking is compared on **retrieval** alone for now. An answer generated from
the retrieved chunks measures the generator and its judge as much as the chunks.

`evaluate.py` scores retrieval against the evidence spans the loaders record.
It calls no model beyond retrieval itself and gives the same result every time:

| Metric | Meaning |
|---|---|
| `hit@k` | a chunk among the first k overlaps an evidence span |
| `mrr@10` | 1 / rank of the first such chunk (0 beyond 10) |
| `ndcg@k` | binary relevance; the ideal ranking puts first every chunk of the index that overlaps evidence |
| `precision@k` | share of the characters of the first k chunks that are evidence |
| `recall@k` | share of the evidence characters the first k chunks cover |
| `f1@k` | harmonic mean of precision@k and recall@k |
| `doc_hit@k` | a chunk among the first k comes from an evidence document |

Precision and recall are counted in characters. Overlapping evidence spans and
overlapping chunks are counted once. Chroma's chunking evaluation (Smith and
Troynikov, 2024) counts the same in tokens. Precision keeps the comparison fair:
hit, MRR, nDCG and recall favour large chunks, and precision pays for the text
they bring along. With short answer spans (SQuAD), precision is small for any
chunk size, so read it across methods, not on its own. Each metric is computed before (`retrieved`) and after
(`reranked`) reranking, at k = 1, 3, 5, 10, and over two groups of questions
kept apart:
- `evidence`: questions with spans in the run's rendering. All metrics apply.
- `no_evidence`: questions tied to a document but with no span. These are
  unanswerable questions, and all of LiteraryQA. Only `doc_hit` applies.

Questions whose documents are not all in the index are left out, so an index
over `--max-documents` is scored on its own documents' questions.

Evidence differs by dataset:
- **SQuAD:** several spans are alternative annotators' answers.
- **Qasper:** spans merge every annotator's evidence, so `recall` undercounts.
- **ConditionalQA:** spans are all needed together.
- **TriviaQA:** the evidence is a whole document, so there `hit` = `doc_hit`, and precision and recall mean little.
- **GutenQA:** 12% of spans are the source dataset's gold chunk, not the answer itself.

```powershell
.venv\Scripts\python.exe evaluate.py squad txt-sentence-max_tokens=256-overlap=0 --embedding-model bge-m3 --rerank-model bge-reranker-v2-m3 --rerank-max-tokens 8192 --rerank-tokenizer BAAI/bge-m3
.venv\Scripts\python.exe compare.py                   # one Markdown table per dataset
```

Results go to `results/<dataset>/<index>/<set-up>/`:
- `config.json`
- `items.jsonl`: rankings per question, resumable
- `summary.json`

`--max-items` takes a sample in a fixed order (by a hash of the question id),
so a larger sample contains a smaller one. The models default to
`EMBEDDING_MODEL` and `RERANK_MODEL`. `--no-rerank` and
`--retrieval dense|sparse|hybrid` give the other set-ups.

Later, the following RAG metrics implemented in the [RAGAS library](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/).
Context precision and recall need no generation and could check the span
metrics where evidence is loose; faithfulness and answer relevancy need the
generation step. The judge would be gpt-oss-120b.
- [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#:~:text=Examples-,Context%20Precision,-The%20ContextPrecision%20metric)
- [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness)
- [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)

## Comparison of chunking methods
We evaluate each chunking method on the different datasets with the following fixed pipeline:
- No chunk augmentation
- Embedding model : BGE-M3
- Retrieval strategy : Hybrid
- No reranking
- Metrics : the evidence-span retrieval metrics of `evaluate.py` (see Evaluation); generation (gpt-oss-120b) and the RAGAS metrics come later

markdown and html based chunking methods are evaluated on only markdown and html datasets respectively, while other methods are evaluated on txt datasets.

Each method is one `ingest.py` index and one `evaluate.py` set-up. A
comparison is written as a JSON file under `experiments/`, in the shape of
`experiments/schema.json`. `experiments.py` builds and scores every run it
describes, then prints the table:

```powershell
.venv\Scripts\python.exe experiments.py experiments/chunking_eval_squad_conditionalqa.json --dry-run   # list the runs
.venv\Scripts\python.exe experiments.py experiments/chunking_eval_squad_conditionalqa.json             # run; re-run to resume
.venv\Scripts\python.exe compare.py squad                                                               # every result for a dataset
```

The file has one key per stage of the pipeline: `chunking`,
`chunk_augmentation`, `chunk_embedding`, `retrieval` and `reranking`. Each
lists alternatives as `{"method": ..., "params": {...}}`, and `dataset` is one
name or a list. Every combination of a dataset and one alternative per stage
is a run. Runs that differ only in retrieval or reranking share one index,
built once.

- **chunking:** a method and its parameters. Models it calls are parameters
  too: `llm_model` for `lumberchunker`, `embedding_model` for the semantic
  chunkers. `rendering` (`txt`, `md`, `html`) overrides the method's own.
- **chunk_augmentation:** one method per entry, or `"none"`; `params.llm_model`
  names its model. If left out, chunks are not augmented.
- **chunk_embedding:** `model`, plus `tokenizer` and `context_length` to cut
  chunks over the model's limit. Used only to embed chunks.
- **retrieval:** `dense`, `sparse` or `hybrid`, with `params.top_k`.
- **reranking:** `cross_encoder` (the `/rerank` model) or `"none"`. Its
  `params` are `model`, `top_k` (chunks kept; every retrieved chunk is
  reranked), and `max_tokens` with `tokenizer`. If left
  out, results are not reranked.

Optional `max_documents` and `max_items` cap a run for a quick check, and any
entry may carry a `description` in place of a comment.

A list inside `params` is a sweep: `"max_tokens": [256, 512], "overlap": [0, 0.25]`
runs every combination, each as its own index. A parameter whose value is
itself a list (`html`'s `headings`) is not swept.

- **Overlap:** an `overlap` below 1 is a share of the chunk size (0.25 at 256
  tokens is 64). Combinations whose overlap is not below the size are skipped.
- **Renderings:** a dataset without the rendering a method reads (SQuAD has no
  `md` or `html`) is skipped for that method, so one file can list every
  dataset.
- **Dataset names** must match their directory exactly, case included.
- **No environment variables:** nothing is read from the environment, so the
  file is the record of what was run.
- **Checked up front:** every run is validated before the first starts.
  `--keep-going` moves on past a failed index; `--skip-evaluation` only builds
  the indexes.

Rows are comparable only when their indexes cover the same documents (the
`docs` column), so every method of a dataset should be built with the same
`--max-documents`.