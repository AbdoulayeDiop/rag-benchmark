# Datasets referenced in papers.md

Working file. Companion to [papers.md](papers.md).

**Last updated:** 2026-09-18

## Method & provenance

Papers were downloaded as raw HTML (`curl https://arxiv.org/html/<id>`) and read directly.
An earlier pass used a fetch tool that returns a small model's *summary* rather than page
text; that pass silently dropped rows (see [Extraction audit](#extraction-audit)). Treat any
figure without a ✓ as unverified.

| Mark | Meaning |
|---|---|
| ✓ | Read directly from the paper's HTML source |
| † | From general knowledge, **not verified** — check before relying on it |
| — | No data; needs lookup |

**Paper codes**

| Code | Paper |
|---|---|
| S1 | [RAG for LLMs: A Survey](https://arxiv.org/abs/2312.10997) (v5 = latest, confirmed) |
| S2 | [Evaluation of RAG: A Survey](https://arxiv.org/abs/2405.07437) |
| S3 | [Can LLMs Be Trusted for Evaluating RAG Systems?](https://arxiv.org/abs/2504.20119) |
| RE | [RAGEval](https://arxiv.org/html/2408.01262) |
| RA | [Ragas](https://arxiv.org/html/2309.15217) |
| KY | [Know Your RAG](https://arxiv.org/html/2411.19710v1) |
| C1 | [Chunking Methods on RAG — Effectiveness vs Cost](https://arxiv.org/html/2606.00881) |
| C2 | [When Is Complex Chunking Worth It?](https://arxiv.org/html/2608.16586) |
| C3 | [Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/html/2410.13070v1) |
| C4 | [Beyond Chunk-Then-Embed](https://arxiv.org/html/2602.16974) |

---

## Verified document statistics

These are the only per-document size figures actually reported in any of the nine papers.

### C1 Table 5 — document length in **characters** ✓

Unit confirmed internally: GutenQA merged 67,058,241 ÷ 36,917 docs = 1,816 ≈ stated avg 1,814.
Same check on TriviaQA: 14,241,172 ÷ 1,000 = 14,241 ≈ stated avg 14,239.

| Dataset | Documents | Min | Max | Avg |
|---|---:|---:|---:|---:|
| GutenQA | 36,917 | 43 | 2,767 | 1,814 |
| GutenQA (merged) | 1 | 67,058,241 | 67,058,241 | 67,058,241 |
| LiteraryQA | 138 | 1,459 | 1,833,987 | 411,471 |
| NaturalQuestions | 300 | 822 | 215,119 | 45,634 |
| NovelQA | 60 | 322,441 | 6,839,619 | 1,007,786 |
| PoQuAD | 1,449 | 50 | 13,860 | 922 |
| Qasper | 416 | 314 | 2,003 | 1,014 |
| SQuAD | 100 | 3,715 | 92,637 | 29,971 |
| TriviaQA | 1,000 | 113 | 430,874 | 14,239 |
| TriviaQA (merged) | 1 | 14,241,172 | 14,241,172 | 14,241,172 |

**Two things to note.**

1. **GutenQA is distributed as chunks, not books.** 36,917 "documents" averaging 1,814
   characters. The books were pre-segmented by LumberChunker. This is exactly the
   "documents déjà en chunks" problem from papers.md:53 — and C1 uses it anyway.
2. **Qasper at 1,014 chars avg / 2,003 max contradicts C3**, which reports Qasper at
   130 sentences per document (see below). Two papers using "Qasper" at completely
   different granularity. One of them is not using full papers. Needs resolving.

### C3 Table 4 — document retrieval datasets ✓

`#D` = number of selected long documents · `S/D(*)` = avg sentences per long document
*after* stitching · `S/D` = avg sentences per document *before* stitching ·
`D/Q` = avg relevant long documents per query

| Dataset | Type | Split | #D | S/D(*) | S/D | D/Q |
|---|---|---|---:|---:|---:|---:|
| MIRACL | Stitched | train | 1,184 | 102 | 4 | 3 |
| NQ | Stitched | test | 488 | 88 | 5 | 1 |
| SciDocs | Stitched | test | 1,692 | 88 | 8 | 5 |
| SciFact | Stitched | test | 420 | 99 | 8 | 1 |
| BioASQ | Stitched | train | 2,368 | 93 | 9 | 6 |
| NFCorpus | Stitched | test | 364 | 118 | 12 | 37 |
| HotpotQA | Original | test | 800 | 20 | 20 | 2 |
| MSMARCO | Original | — | — | — | — | — |

**The "long documents" here are synthetic.** Original documents average 4–12 sentences;
they were concatenated into ~100-sentence pseudo-documents. C3 admits the consequence
directly: *"in real life, the topics in a document may not be as diverse as in our
artificially noisy, stitched data, and hence semantic chunkers may not have an edge over
fixed-size chunker there."* ✓

### C3 Table 5 — evidence retrieval datasets ✓

Not stitched. `E/Q` = avg evidence **sentences** per query.

| Dataset | Split | #D | S/D | E/Q |
|---|---|---:|---:|---:|
| ExpertQA | test | 777 | 20 | 12 |
| DelucionQA | test | 235 | 23 | 9 |
| TechQA | test | 648 | 49 | 15 |
| ConditionalQA | dev | 652 | 120 | 5 |
| Qasper | test | 416 | 130 | 4 |

**This is the closest thing in any of the nine papers to what papers.md:56 asks for** —
real documents with sentence-level evidence annotations. ConditionalQA (120 sent/doc) and
Qasper (130 sent/doc) are the only two long enough for chunking to plausibly matter.

### C2 — corpus scale (document *counts*, not lengths) ✓

| Dataset | Scales | Derived from |
|---|---|---|
| CoRE | 10K / 100K / 1M / 10M docs | MS MARCO v2 (Caspari et al. 2025) |
| KILT-NQ | 10K / 100K / 1M / 6M docs | KILT (Petroni et al. 2021) + NQ queries |

C2 notes it capped at 1M on KILT due to computational cost. No per-document length given.

---

## Master dataset list

Document-bearing datasets only. Query/answer-only sets excluded (see [Exclusions](#exclusions)).

| Dataset | Link | Used by | Verified facts |
|---|---|---|---|
| 2WikiMultiHopQA | [gh](https://github.com/Alab-NII/2wikimultihop) † | S1 | Multi-hop sub-task ✓ |
| ArguAna | [hf](https://huggingface.co/datasets/BeIR/arguana) † | C4 | BEIR subset ✓ |
| ASQA | — | S1 | Long-form QA sub-task ✓ |
| BioASQ | [site](http://bioasq.org/) † | C3 | Stitched: 2,368 docs, 93 sent/doc after, 9 before, 6 D/Q ✓ |
| Biography | — | S1 | Text generation ✓ |
| CamRest | — | S1 | Task-oriented dialog ✓ |
| CDQA | — | S2, S3 | "Generation (Source: News), Labeller" ✓ |
| ClimRetrieve | [arxiv](https://arxiv.org/abs/2406.09818) | S3 | 30 sustainability reports, 16 questions, >8.5K Q-source-answer pairs, graded relevance levels |
| CMB | — | S1 | Domain QA, medical ✓ |
| CodeSearchNet | [gh](https://github.com/github/CodeSearchNet) † | S1 | Code search ✓ |
| ConditionalQA | [gh](https://github.com/haitian-sun/ConditionalQA) † | C3 | 652 docs, **120 sent/doc**, 5 evidence sent/query ✓ |
| CORAL | — | S3 | Multi-turn conversational RAG |
| CoRE | [hf](https://huggingface.co/datasets/PaDaS-Lab/CoRE) ✓ | C2 | 10K/100K/1M/10M docs from MS MARCO v2 ✓ |
| COVID-QA | — | S1 | Domain QA ✓ |
| CoT Reasoning | — | S1 | Own task row ✓ |
| CRAG | — | S3, RE | "Comprehensive RAG benchmark", mock APIs |
| CRUD-RAG | [gh](https://github.com/IAAR-Shanghai/CRUD_RAG) † | S1, S2, S3, RE | Chinese; "Generated (Source: News)" ✓ |
| DelucionQA | [hf](https://huggingface.co/datasets/rungalileo/ragbench) | C3 | 235 docs, 23 sent/doc, 9 evidence sent/query ✓ · Jeep 2023 Gladiator manual |
| DomainRAG | — | S2, S3 | "Generation (Source: College Admission Information)" ✓ |
| DragonBall | [gh](https://github.com/OpenBMB/RAGEval) | RE | 6,711 questions; finance/law/medical; ZH+EN; CC-BY-NC |
| ELI5 | — | S1 | Long-form QA ✓ |
| EventKG | — | S2 | Used by RECALL ✓ |
| ExpertQA | [gh](https://github.com/chaitanyamalaviya/ExpertQA) † | C3 | 777 docs, 20 sent/doc, 12 evidence sent/query ✓ |
| Face4RAG | — | S3 | Chinese factual consistency |
| FEVER | [site](https://fever.ai/) † | S1, S2 | Fact checking; used by ARES ✓ |
| FiQA | [hf](https://huggingface.co/datasets/BeIR/fiqa) † | C4 | BEIR subset ✓ |
| FRAMES | — | S3 | "Fact, Fetch, and Reason" |
| GraphQA | — | S1 | Graph QA ✓ |
| GutenQA | [hf](https://huggingface.co/datasets/LumberChunker/GutenQA) | C1, C4 | **36,917 docs, avg 1,814 chars** ✓ · 100 books, 30 QA each · pre-segmented by LumberChunker |
| HotpotQA | [site](https://hotpotqa.github.io/) † | S1, S2, S3, RE, KY, C3 | Original (unstitched): 800 docs, 20 sent/doc, 2 D/Q ✓ |
| JRC-Acquis | — | S1 | Machine translation ✓ |
| KILT-NQ | [hf](https://huggingface.co/datasets/PaDaS-Lab/kilt-nq) ✓ | C2 | 10K/100K/1M/6M doc subcorpora ✓ |
| LegalBench-RAG | [arxiv](https://arxiv.org/abs/2408.10343) | S3 | 6,858 QA over >79M chars; **human-annotated spans by filename + character index**; NDAs, M&A, contracts, privacy policies |
| LiteraryQA | [hf](https://huggingface.co/datasets/sapienzanlp/LiteraryQA) | C1 | **138 docs, avg 411,471 chars** (max 1.83M) ✓ · cleaned NarrativeQA subset |
| Long²RAG | [arxiv](https://arxiv.org/abs/2410.23000) | S3 | 280 questions, 5 docs each, avg 2,444 words/doc |
| MIRACL | [hf](https://huggingface.co/datasets/miracl/miracl) † | C3 | Stitched: 1,184 docs, 102 sent/doc after, **4 before**, 3 D/Q ✓ |
| MIRAGE (MedRAG) | — | S2 | Used by MedRAG ✓ |
| MIRAGE-bench | — | S3 | Multilingual RAG arena |
| MMCU_Medical | — | S1 | Domain QA, medical ✓ |
| MS MARCO | [site](https://microsoft.github.io/msmarco/) † | S1, RE, KY, C3 | KY used v2.1 ✓ · CoRE derived from v2 ✓ · RAGBench component |
| MultiHop-RAG | [gh](https://github.com/yixuantt/MultiHop-RAG) † | S2, S3, RE | "Generated (Source: News)" ✓ |
| MultiRC | — | S2 | Used by ARES ✓ |
| MuSiQue | [gh](https://github.com/StonyBrookNLP/musique) † | S1 | Multi-hop ✓ |
| NarrativeQA | [hf](https://huggingface.co/datasets/deepmind/narrativeqa) | S1 | Long-form QA ✓ · parent of LiteraryQA |
| Natural Questions | [site](https://ai.google.com/research/NaturalQuestions) † | S1, S2, RE, KY, C1, C3 | C1: **300 docs, avg 45,634 chars** ✓ · C3 stitched: 488 docs, 88 sent after, 5 before ✓ |
| NewsQA | [gh](https://github.com/Maluuba/newsqa) † | KY | News stories |
| NFCorpus | [hf](https://huggingface.co/datasets/BeIR/nfcorpus) † | C3, C4 | Stitched: 364 docs, 118 sent after, 12 before, **37 D/Q** ✓ |
| **NoMIRACL** | — | S1 | Task: **Robustness Evaluation** — only dataset S1 lists for it ✓ |
| NovelQA | [hf](https://huggingface.co/datasets/NovelQA/NovelQA) | C1 | **60 docs, avg 1,007,786 chars** (max 6.8M) ✓ · 89 novels / 2,305 QA (28 not public) |
| PoQuAD | [gh](https://github.com/ipipan/poquad) | C1 | **1,449 docs, avg 922 chars** ✓ · 70k QA, Polish Wikipedia, SQuAD 2.0 style + abstractive layer |
| PubHealth | — | S1 | Fact checking ✓ |
| PubMedQA | [site](https://pubmedqa.github.io/) † | KY | Biomedical; RAGBench component |
| Qasper | [data](https://allenai.org/data/qasper) † | S1, C1, C3 | C1: 416 docs, avg 1,014 chars ✓ · C3: 416 docs, **130 sent/doc**, 4 evidence sent/query ✓ — **figures conflict** |
| QMSum | [gh](https://github.com/Yale-LILY/QMSum) † | S1 | Long-form QA ✓ |
| QuALITY | [gh](https://github.com/nyu-mll/quality) † | S1 | Multi-choice QA ✓ |
| RAD-bench | — | S3 | Retrieval augmented dialogues |
| RAGBench | [hf](https://huggingface.co/datasets/rungalileo/ragbench) · [gh](https://github.com/rungalileo/ragbench) | S3, C3 | 100k examples, 5 domains, 12 components: CovidQA, PubmedQA, HotpotQA, MS Marco, CUAD, EManual, TechQA, FinQA, TAT-QA, ExpertQA, HAGRID, DelucionQA |
| RAG-ConfusionQA | — | S3 | Confusing questions |
| RAG-QA Arena | — | S3 | Domain robustness, long-form |
| RAGTruth | [gh](https://github.com/ParticleMedia/RAGTruth) † | RE | Hallucination corpus |
| RAMS | [site](https://nlp.jhu.edu/rams/) † | S1 | Event argument extraction ✓ |
| RealTimeQA | [gh](https://github.com/realtimeqa/realtimeqa_public) † | S2 | Used by ReEval, with NQ ✓ |
| ReCoRD | — | S2 | Used by ARES; SuperGLUE ✓ |
| RGB | [gh](https://github.com/chen700564/RGB) † | S1, S2, RE | "Generated (Source: News)" ✓ |
| SciDocs | [hf](https://huggingface.co/datasets/BeIR/scidocs) † | C3, C4 | Stitched: 1,692 docs, 88 sent after, 8 before, 5 D/Q ✓ |
| SciFact | [hf](https://huggingface.co/datasets/BeIR/scifact) † | C3, C4 | Stitched: 420 docs, 99 sent after, 8 before, 1 D/Q ✓ |
| SQuAD / SQuAD 2.0 | [site](https://rajpurkar.github.io/SQuAD-explorer/) † | S1, KY, C1 | C1: **100 docs, avg 29,971 chars** ✓ · KY used SQuAD2 training set ✓ |
| SST-2 | — | S1 | Sentiment ✓ |
| StrategyQA | — | S1 | Listed under Others ✓ |
| T-REx | [site](https://hadyelsahar.github.io/t-rex/) † | S1 | Relation extraction ✓ |
| TechQA | [hf](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval) | C3 | 648 docs, 49 sent/doc, **15 evidence sent/query** ✓ · IBM developer forums |
| TREC | — | S1 | Text classification ✓ |
| TREC-COVID | [hf](https://huggingface.co/datasets/BeIR/trec-covid) † | C4 | BEIR subset ✓ |
| TriviaQA | [site](https://nlp.cs.washington.edu/triviaqa/) † | S1, RE, C1 | C1: **1,000 docs, avg 14,239 chars** (max 430,874) ✓ |
| UDA | [gh](https://github.com/qinchuanhui/UDA-Benchmark) · [hf](https://huggingface.co/datasets/qinchuanhui/UDA-QA) | S3 | 2,965 real documents, 29,590 expert-annotated QA, 6 sub-datasets; **documents kept in original PDF/HTML, unparsed and unsegmented**; CC-BY-SA 4.0 |
| **UHGEval** | — | S2 | In Table 2 alongside CRUD-RAG ✓ |
| UJ | — | S2 | Used by RECALL ✓ |
| VioLens | — | S1 | Text classification ✓ |
| WeQA | — | S3 | Wind energy domain |
| WikiASP | — | S1 | Text summarization ✓ |
| WikiEval | [hf](https://huggingface.co/datasets/explodinggradients/WikiEval) | S2, RA | 50 Wikipedia pages, events post-2022; questions by ChatGPT; 2 human annotators on faithfulness / answer relevance / context relevance |
| WikiEvent | — | S1 | Event argument extraction ✓ |
| WikiText-103 | [hf](https://huggingface.co/datasets/Salesforce/wikitext) † | S1 | Language modeling ✓ |
| Wizard of Wikipedia | [site](https://parl.ai/projects/wizard_of_wikipedia/) † | S1, S2 | Dialog generation; used by ARES ✓ |
| XSum | [hf](https://huggingface.co/datasets/EdinburghNLP/xsum) † | S1 | Text summarization ✓ |
| zsRE | — | S1 | Relation extraction ✓ |

**Aggregates, not single datasets:** BEIR ([gh](https://github.com/beir-cellar/beir)) — C3 and C4
both draw subsets. PIRB (Polish Information Retrieval Benchmark) — C1's retrieval evaluation
framework, uses bge-m3 retriever + bge-reranker-v2-m3 ✓.

### Exclusions

**Query/answer or classification only, no corpus:** WebQuestions, PopQA, ARC,
CommonsenseQA/CSQA, MMLU, HellaSwag, GSM8K, GraphQA, DuLeMon, KBP, Amazon (Toys/Sport/Beauty).

**Frameworks, not data:** RAGAS, ARES, TruLens, eRAG, RAGChecker, CoFE-RAG, VERA, Lynx,
RAGElo, RAGProbe, BERGEN, RAGQuestEval, RECALL, ReEval, LangChain Benchmarks,
TruEra RAG Triad, Databricks Eval, IRSC.

---

## Analysis: the gap

Cross-tabulating verified document length against ground-truth granularity:

|  | Doc-level GT only | Sub-document GT |
|---|---|---|
| **Long (>100K chars)** | LiteraryQA (411K), NovelQA (1.0M) | — |
| **Medium (10K–100K)** | — | NaturalQuestions (45.6K), SQuAD (30K), TriviaQA (14.2K) |
| **Short (<10K)** | BEIR ×7, MS MARCO, MIRACL | Qasper, ConditionalQA, TechQA, ExpertQA, DelucionQA, GutenQA (1.8K), PoQuAD (922) |

**Nothing verified sits in long + sub-document GT.** The two genuinely long datasets
(LiteraryQA, NovelQA) have abstractive answers only. Everything with sentence-level evidence
is short.

Two candidates that may fill it, both **unverified**:

- **LegalBench-RAG** — character-index spans over a 79M-character legal corpus. If document
  lengths are what the corpus size implies, this is the only dataset found so far that is both
  long and annotated below document level.
- **UDA** — 2,965 documents deliberately shipped unsegmented, in original PDF/HTML. Directly
  addresses papers.md:53. GT granularity per sub-dataset unknown.

C1 independently reaches the same conclusion as papers.md:51-56 ✓:

> *"Chunking is inherently irrelevant for short texts... many widely used QA datasets were
> excluded or deemphasized because their documents are too short... most existing datasets do
> not provide explicit annotations of golden chunks... we selected datasets that mostly include
> answer annotations grounded in the source documents (e.g., answer spans, supporting passages,
> or evidence labels). These annotations allow us to approximate gold chunks indirectly."*

Its workaround — merging entire datasets into one 67M-character document to create
"stress-test conditions" — is the mirror image of C3's stitching. Both papers manufacture long
documents because none were available.

---

## Extraction audit

Re-read from raw HTML, compared against the earlier summarizer-based pass.

| Source | Ground truth | Summarizer returned | Drops |
|---|---|---|---|
| S1 Table II | 48 datasets (paper says "26 tasks, nearly 50 datasets") | 47 | **NoMIRACL** |
| S2 Table 2 | 11 benchmark→dataset rows | 11 rows, cells incomplete | **UHGEval**; NQ missing from ReEval row |
| S3 | **No dataset table exists** — Tables I–V are all metrics tables | ~38 names from prose | none |
| C1 | Table 5 full statistics | no statistics at all | entire table |
| C3 | Tables 4 & 5 full statistics | no statistics at all | both tables |

S3's "87 QA datasets" figure appears once in prose and is **never enumerated**. Recovering
those names requires going through its 63 references.

---

## Open items

1. **Resolve the Qasper conflict** — C1 reports avg 1,014 chars / max 2,003; C3 reports 130
   sentences per document. These cannot both describe full papers.
2. **MS MARCO row in C3 Table 4** — not extracted, HTML cell boundaries lost.
3. **Verify LegalBench-RAG and UDA** from their own papers (2408.10343, 2406.15187) —
   per-document length and GT format. These are the two strongest candidates.
4. **UDA's six sub-dataset names** — not stated in anything read so far.
5. **LegalBench-RAG's source corpora** — "retrofits human-annotated queries from established
   legal datasets", unnamed.
6. **RAGBench components without rows:** CUAD, EManual, FinQA, TAT-QA, HAGRID, CovidQA.
   CUAD matters most — legal contracts with clause-level spans.
7. **S3's seven methods for determining document relevance** — S3 frames retriever evaluation
   explicitly as *"whether the retrieved chunks are relevant to the given query"* across 24
   papers. Not yet extracted; directly on topic.
8. **Text segmentation literature is absent from all nine papers** — Wiki-727K, WikiSection,
   Choi, Cities/Elements, RST-DT carry gold *section boundary* annotations, a different
   ground truth from evidence retrieval. Unverified, worth a look.
9. **Papers missing from papers.md:** LitSeg (arXiv 2605.27156, narrative-aware segmentation
   for literary RAG); MLEB (arXiv 2510.19365, legal embedding benchmark). Neither read.
10. **FanoutQA** appears in C3's references — multi-hop multi-document QA. Not investigated.
