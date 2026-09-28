# RAG papers

# Surveys

[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)

[Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2405.07437)

# Dataset, eval et RAGAS

[RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](https://arxiv.org/html/2408.01262)

[Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/html/2309.15217)

[Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of...](https://arxiv.org/abs/2504.20119)

[Know Your RAG:Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems](https://arxiv.org/html/2411.19710v1#bib.bib45)

# Chunking

[Chunking Methods on Retrieval-Augmented Generation – Effectiveness Evaluation Against Computational Cost and Limitations](https://arxiv.org/html/2606.00881)

Pros :

* Adapted datasets with long documents and large domain coverage
* chunk-level eval metrics (based on evidence retrieval) + answer generation metric

[When Is Complex Chunking Worth It? A Multi-Objective Evaluation of Chunking Methods at Scale](https://arxiv.org/html/2608.16586)

Limit :

* Document-level eval metrics instead of evaluation chunk-level

[Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/html/2410.13070v1)

Pros :

* Datasets OK
* Metrics OK

Limits :

* Only 3 chunking methods compared

[Beyond Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document Chunking Strategies for Information Retrieval](https://arxiv.org/html/2602.16974)

Limits :

* Datasets : too small - only one is OK (gutenQA)

Limitations côté chunking :

* Datasets non adaptés (documents trop court ou déjà en chunks). Idéalement il faut des datasets avec de long documents avec des query et annotation des sections/phrases permettant de répondre à ces query.
* Mesures d’évaluation non adapté dans la plupart des papiers (souvent ramené au document étant donnée qu’on ne peut pas avoir de vérité terrain sur les chunks).
  * Sachant que la phase de génération s’appuie uniquement sur les chunks et non tout le document d’origine, les mesures d’évaluation devraient permettre d’évaluer la pertinence et la complétude des chunks en eux même.
  * Une mesure intéressante qui prend cela en compte est “Evidence Retrieval” qui mesure compare les chunks aux “ground truth evidences” (phrases annoté comme permettant de répondre à la query)
