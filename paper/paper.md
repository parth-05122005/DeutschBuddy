# Does Retrieval Make LLM Grammar Explanations Trustworthy? A Benchmark Study for German as a Foreign Language

@author Parth Khandelwal | [Department] | [Institution] | [City], India | [college email]
@author Dhananjay Kumar | [Department] | [Institution] | [City], India | [email]

## Abstract

Large language models (LLMs) can explain grammar on demand, but their explanations are often incomplete or wrong, and learners cannot check them. We study whether retrieval-augmented generation (RAG) over a small, teacher-curated grammar knowledge base makes LLM explanations more trustworthy for learners of German as a foreign language. We introduce DaF-GQA, a benchmark of 180 learner questions at CEFR levels A1–B1, each linked to the knowledge-base section that answers it, including 20 questions deliberately outside the knowledge base. Using DeutschBuddy, an open-source RAG tutor built on Gemini 3.5 Flash-Lite, multilingual Gemini embeddings and ChromaDB, we compare four configurations: closed-book generation, RAG, RAG with CEFR level control, and RAG with level control and an abstention rule. Two German instructors rate correctness and level fit; we also measure retrieval hit rate, faithfulness and citation precision. Retrieval raises fully correct explanations from [XX]% to [XX]%, level control changes faithfulness by [XX] points, and the abstention rule declines [XX]% of out-of-scope questions. We release the benchmark, code and annotations.

## Index Terms

retrieval-augmented generation, large language models, grammar explanation, German as a foreign language, CEFR, educational NLP

## I. Introduction

Learners of a foreign language increasingly turn to LLM chatbots for grammar help, but an explanation that sounds right is not necessarily right. LLMs offer personalised, always-available explanations [1], yet they also produce fluent statements that are unsupported or false, a behaviour known as hallucination [2]. For a beginner, a confident but wrong rule about case endings is worse than no answer, because the learner has no means to detect the error.

Recent work confirms this risk for grammar. With one-shot prompting, GPT-4 explained only about 40% of learner errors in German and Chinese; a multi-step pipeline was needed to reach high accuracy [3]. A 2026 evaluation of several LLMs on English learner errors found strong surface-level correction but explanations lacking "the terminological and domain knowledge that effective language teaching requires" [4]. Neither study supplies that knowledge to the model at generation time.

Retrieval-augmented generation [5] is the standard way to give a model such knowledge: before answering, the system retrieves passages from a trusted collection and asks the model to answer from them. For language teaching, the trusted collection can be a set of grammar notes written or checked by a teacher, and the retrieved passages can be shown to the learner as sources. Whether this actually improves grammar explanations has not been measured.

Two further questions arise in a learner-facing setting. First, learners need explanations at their proficiency level, defined by the Common European Framework of Reference (CEFR) [6]; controlling the level of LLM output is an open problem [7], and LLMs often write above beginner level [8], and simplifying an answer may pull it away from its sources. Second, a tutor should say when a question falls outside its material rather than answer from memory.

We address these gaps with three research questions:

- **RQ1:** Does retrieval from a curated grammar knowledge base improve the correctness of LLM grammar explanations compared with the same model without retrieval?
- **RQ2:** Does instructing the model to answer at the learner's CEFR level reduce faithfulness to the retrieved sources or the precision of its citations?
- **RQ3:** Does an explicit abstention rule make the system flag out-of-scope questions without harming in-scope answers?

Our contributions are:

1. **DaF-GQA**, a benchmark of 180 German grammar questions at A1–B1 with gold knowledge-base sections and teacher-checked reference answers, including 20 out-of-scope questions.
2. **DeutschBuddy**, an open-source RAG tutor for German with heading-aware chunking, cross-lingual retrieval and cited, level-adapted answers.
3. A **controlled comparison of four configurations** with expert ratings of correctness and level fit, plus automatic measures of retrieval, faithfulness and citation precision.

## II. Related Work

Prior work has studied grammar explanation, retrieval grounding and level control separately; no study combines them for a foreign-language learner.

### A. LLMs for Grammar Explanation

Song et al. [3] define grammar error explanation (GEE) as producing one natural-language explanation per learner error. One-shot GPT-4 explained only about 40% of errors and skipped 39.8%; a two-step pipeline that first extracts atomic edits and then explains each one reached 93.9% correct explanations for German, as judged by two German teachers. Šekrst and Kovačić [4] evaluated several LLMs on English learner errors with automatic metrics and expert judgments and concluded that explanations lack the domain knowledge of a teacher. Both studies rely on the model's internal knowledge; neither retrieves reference material.

### B. Retrieval-Augmented Generation and Its Evaluation

RAG conditions a generator on passages returned by a retriever [5]. Dense retrievers based on learned embeddings outperform keyword search for open-domain question answering [9], and later systems add query rewriting, re-ranking and modular pipelines [10]. RAGAS [11] evaluates RAG without reference answers: faithfulness checks whether each claim in an answer can be inferred from the retrieved context, and context precision and recall assess the retriever. Faithfulness does not verify that a specific inline citation points to the passage that supports the cited sentence, which matters when sources are shown to learners.

### C. CEFR-Controlled Generation

The CEFR describes six proficiency levels from A1 to C2 [6]. Malik et al. [7] compare prompting, fine-tuning and reinforcement learning for controlling the proficiency level of generated content, and Jin et al. [8] report that most LLMs write at near-native complexity unsuitable for A1–A2 learners, with controllable generation raising comprehensibility for beginners from 39.4% to 83.3%. Rabih et al. [12] show that CEFR-guided prompting with lexical constraints aligns Arabic generation with target levels far better than unconstrained prompting. These studies control level in free generation; they do not ask whether simplifying a grounded answer weakens its link to its sources.

### D. Chatbots in Second-Language Learning

A meta-analysis of 41 experimental studies found a moderate-to-large positive effect of generative-AI chatbots on second-language acquisition (effect size 0.576) [13], while warnings about unverified output persist [1]. None of the included studies evaluates a retrieval-grounded tutor.

### E. Positioning

Table: TABLE I | Comparison with Prior Work
| Work | Language | Retrieval | Level control | Sources shown | Expert rating |
| --- | --- | --- | --- | --- | --- |
| Song et al. [3] | German, Chinese | No | No | No | Yes |
| Šekrst and Kovačić [4] | English | No | No | No | Yes |
| Malik et al. [7], Jin et al. [8] | English, Japanese | No | Yes | No | Varies |
| RAGAS [11] | General | Yes | No | No | No |
| **This work** | **German** | **Yes** | **Yes** | **Yes** | **Yes** |

## III. System: DeutschBuddy

DeutschBuddy is a RAG tutor with an indexing pipeline and an answering pipeline that share one vector index (Fig. 1).

![Fig. 1. DeutschBuddy architecture: indexing (top) and answering (bottom) share one ChromaDB index.](architecture.png)

### A. Knowledge Base

The knowledge base consists of 16 Markdown documents written for German learners at A1–B1: six on grammar (articles and gender, the four cases, present-tense conjugation, word order, tenses, adjective endings), five on themed vocabulary and five on German culture. Explanations are in English, examples in German with translations, nouns always carry their article and plural, and paradigms such as article declension are given as tables. Only the grammar documents are used in this study.

### B. Indexing

Documents are split by Markdown heading, then into chunks of at most about 900 characters on paragraph and sentence boundaries, with a 150-character overlap. The full heading path is prepended to each chunk (e.g. *The Four German Cases (Kasus) > Prepositions and their cases > Always Dativ*) so that short chunks such as a preposition list remain interpretable. With these settings the knowledge base yields 123 chunks. Chunks are embedded with the multilingual gemini-embedding-001 model [14] using the RETRIEVAL_DOCUMENT task type, reduced to 768 dimensions, L2-normalised and stored in ChromaDB with an HNSW cosine index [15]. Chunk identifiers are content hashes, so re-indexing is idempotent.

### C. Answering

A question is embedded with the RETRIEVAL_QUERY task type. Because the embedding space is shared across languages, English, German and mixed questions retrieve the same German examples. The top-k chunks (k = 5) with cosine similarity of at least 0.45 are numbered and inserted into the prompt. Short follow-up questions are concatenated with the previous question before retrieval. The system prompt instructs Gemini 3.5 Flash-Lite to cite passages as [1], [2], to give nouns with their articles, to show German examples with translations, and, depending on the configuration, to answer at a target CEFR level and to state when the passages do not contain the answer.

The system is implemented in Python with Streamlit and is released as open source.

## IV. Benchmark and Experimental Setup

### A. The DaF-GQA Benchmark

DaF-GQA (German as a Foreign Language Grammar Question Answering) contains 180 questions written in the form learners ask them. 160 in-scope questions are answerable from the grammar knowledge base; each is labelled with its CEFR level, its topic, the gold knowledge-base section(s) that answer it, and a reference answer checked by the second author, a German instructor. 20 out-of-scope questions cover topics deliberately absent from the knowledge base (Konjunktiv II, passive voice, relative clauses, reflexive verbs, n-declension).

Table: TABLE II | Composition of DaF-GQA by Topic and CEFR Level
| Topic | A1 | A2 | B1 | Total |
| --- | --- | --- | --- | --- |
| Articles and gender | 15 | 10 | 0 | 25 |
| Cases and prepositions | 10 | 10 | 10 | 30 |
| Verb conjugation | 15 | 10 | 5 | 30 |
| Word order | 10 | 10 | 5 | 25 |
| Tenses | 10 | 10 | 10 | 30 |
| Adjective endings | 0 | 10 | 10 | 20 |
| **In-scope total** | **60** | **60** | **40** | **160** |
| Out-of-scope | – | – | – | 20 |

To reflect real learner input, 106 questions are written in English ("Why is it *das Mädchen* and not *die Mädchen*?"), 43 in German ("Welcher Fall kommt nach *ohne*?") and 31 mix both ("Ich fahre mit ___ Bus, *dem* or *den*?"). The language of each question is recorded for analysis.

### B. Configurations

All configurations use Gemini 3.5 Flash-Lite at temperature 0.2 with the same base tutor prompt; only the listed components change. Retrieval is restricted to the grammar documents, with k = 5 and a minimum cosine similarity of 0.45.

Table: TABLE III | Experimental Configurations
| ID | Configuration | Retrieved passages | CEFR instruction | Abstention rule |
| --- | --- | --- | --- | --- |
| C1 | Closed-book | No | No | No |
| C2 | RAG | Yes | No | No |
| C3 | RAG + level | Yes | Yes | No |
| C4 | RAG + level + abstain | Yes | Yes | Yes |

C1 vs C2 answers RQ1, C2 vs C3 answers RQ2, and C3 vs C4 answers RQ3.

### C. Metrics

- **Correctness** (RQ1): each answer is rated *correct*, *partly correct* (true but incomplete or with a minor error) or *incorrect* against the reference answer.
- **Retrieval Hit@5 and MRR**: whether a gold section appears among the five retrieved chunks, and the reciprocal rank of the first gold chunk.
- **Faithfulness** (RQ2): the share of answer claims supported by the retrieved context, following the RAGAS definition [11]: an LLM judge at temperature 0 splits the answer into factual claims and checks each against the retrieved passages.
- **Citation precision** (RQ2): the share of inline citations whose passage supports the sentence it is attached to, judged by an LLM and verified by a human on a 20% sample. The judge is Gemma 4 31B, a different model from the one being evaluated.
- **Level fit** (RQ2): raters judge whether an explanation is *suitable*, *too difficult* or *too simple* for the question's CEFR level.
- **Abstention** (RQ3): precision and recall of flagging out-of-scope questions, and the false-abstention rate on in-scope questions.

### D. Human Evaluation

Two German instructors rate correctness and level fit for all 720 answers (180 questions × 4 configurations). Answers are shuffled and shown without their configuration. Inter-rater agreement is reported as Cohen's κ [16]; disagreements are resolved by discussion. Differences in the proportion of correct answers between paired configurations are tested with McNemar's test [17] at α = 0.05.

## V. Results

### A. Correctness (RQ1)

Table: TABLE IV | Results on the 160 In-Scope Questions
| Config. | Correct (%) | Partly (%) | Incorrect (%) | Faithful. | Cit. prec. (%) | Level fit (%) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 Closed-book | [XX] | [XX] | [XX] | – | – | [XX] |
| C2 RAG | [XX] | [XX] | [XX] | [XX] | [XX] | [XX] |
| C3 RAG + level | [XX] | [XX] | [XX] | [XX] | [XX] | [XX] |
| C4 RAG + level + abstain | [XX] | [XX] | [XX] | [XX] | [XX] | [XX] |

Inter-rater agreement: correctness κ = [XX], level fit κ = [XX]. Retrieval changed the share of fully correct explanations from [XX]% (C1) to [XX]% (C2) (McNemar p = [XX]). [Describe the most frequent error types in C1 that C2 fixed, and any errors retrieval introduced.]

### B. Retrieval Quality

Table: TABLE V | Retrieval Quality by Question Language
| Question language | n | Hit@5 | MRR |
| --- | --- | --- | --- |
| English | [XX] | [XX] | [XX] |
| German | [XX] | [XX] | [XX] |
| Mixed | [XX] | [XX] | [XX] |
| **All in-scope** | **160** | **[XX]** | **[XX]** |

[Report whether incorrect answers in C2 coincide with retrieval misses, and whether mixed-language questions retrieve worse than single-language ones.]

### C. Level Control vs Grounding (RQ2)

Adding the CEFR instruction changed level fit from [XX]% to [XX]%, faithfulness from [XX] to [XX], and citation precision from [XX]% to [XX]%. [Report results separately for A1, A2 and B1, and whether answers overshoot the target level.]

### D. Abstention (RQ3)

Table: TABLE VI | Abstention on 20 Out-of-Scope and 160 In-Scope Questions
| Config. | Out-of-scope flagged (%) | Precision | False abstentions (%) |
| --- | --- | --- | --- |
| C3 RAG + level | [XX] | [XX] | [XX] |
| C4 RAG + level + abstain | [XX] | [XX] | [XX] |

## VI. Discussion and Limitations

[Write after the results: what retrieval fixes and what it does not, whether level control and grounding trade off, and what the abstention rule costs. Keep claims to what Tables IV–VI show.]

**Implications for teaching.** Showing learners the passage behind each explanation turns the tutor into something a teacher can audit. A small knowledge base written by the teacher defines what the tutor knows, so the curriculum, not the model's pretraining, sets the content.

**Limitations.** The knowledge base covers six grammar topics at A1–B1 and the benchmark has 180 questions, so results may not hold for advanced grammar or larger collections. All configurations use one model, Gemini 3.5 Flash-Lite; other LLMs may benefit more or less from retrieval. Questions are written by the authors in learner style rather than collected from real learners. Two raters from one institution judge correctness and level fit, and level fit in particular is subjective. Faithfulness and citation precision rely partly on an LLM judge, which we check against human judgment only on a sample.

## VII. Conclusion and Future Work

We asked whether retrieving a teacher-curated grammar knowledge base makes LLM grammar explanations trustworthy for learners of German. We introduced the DaF-GQA benchmark and the open-source DeutschBuddy tutor, and compared closed-book generation with three retrieval-based configurations. Retrieval [improved / did not improve] correctness from [XX]% to [XX]%, CEFR level control [did / did not] weaken grounding, and an abstention rule flagged [XX]% of out-of-scope questions.

Future work will (i) collect real learner questions from German courses, (ii) extend the approach to explaining errors in learner writing using the MERLIN corpus of CEFR-rated German exam texts [18], (iii) test hybrid keyword-and-vector retrieval for mixed-language questions, and (iv) measure learning gains in a classroom study.

## References

1. E. Kasneci *et al.*, "ChatGPT for good? On opportunities and challenges of large language models for education," *Learning and Individual Differences*, vol. 103, 2023.
2. Z. Ji *et al.*, "Survey of hallucination in natural language generation," *ACM Computing Surveys*, vol. 55, no. 12, 2023.
3. Y. Song, K. Krishna, R. Bhatt, K. Gimpel, and M. Iyyer, "GEE! Grammar error explanation with large language models," in *Findings of the Association for Computational Linguistics: NAACL 2024*, 2024.
4. K. Šekrst and A. Kovačić, "Clause encounters of the third kind: Can LLMs replace language teachers?" arXiv:2608.16286, 2026.
5. P. Lewis *et al.*, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
6. Council of Europe, *Common European Framework of Reference for Languages: Learning, Teaching, Assessment – Companion Volume*. Strasbourg: Council of Europe Publishing, 2020.
7. A. Malik, S. Mayhew, C. Piech, and K. Bicknell, "From Tarzan to Tolkien: Controlling the language proficiency level of LLMs for content generation," in *Findings of the Association for Computational Linguistics: ACL 2024*, 2024.
8. M. Jin, L. Dugan, and C. Callison-Burch, "Toward beginner-friendly LLMs for language learning: Controlling difficulty in conversation," in *Proc. EACL*, 2026.
9. V. Karpukhin *et al.*, "Dense passage retrieval for open-domain question answering," in *Proc. EMNLP*, 2020.
10. Y. Gao *et al.*, "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023.
11. S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "Ragas: Automated evaluation of retrieval augmented generation," arXiv:2309.15217, 2023.
12. N. Rabih, C. Qwaider, and T. Briscoe, "Can LLMs control readability? A multi-dimensional evaluation framework for CEFR-controlled Arabic generation," in *Proc. READIxTSAR Workshop, LREC*, 2026.
13. M. Li, Y. Wang, and X. Yang, "Can generative AI chatbots promote second language acquisition? A meta-analysis," *Journal of Computer Assisted Learning*, 2025, doi: 10.1111/jcal.70060.
14. J. Lee *et al.*, "Gemini Embedding: Generalizable embeddings from Gemini," arXiv:2503.07891, 2025.
15. Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 4, 2020.
16. J. Cohen, "A coefficient of agreement for nominal scales," *Educational and Psychological Measurement*, vol. 20, no. 1, 1960.
17. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, 1947.
18. A. Boyd *et al.*, "The MERLIN corpus: Learner language and the CEFR," in *Proc. LREC*, 2014.
