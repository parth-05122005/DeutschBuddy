## Table III. Results on in-scope questions

| Config. | n | Correct (%) | Partly (%) | Incorrect (%) | Faithfulness | Citation precision | Level fit (%) | First token (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 Closed-book | 160 | 95.0 | 5.0 | 0.0 | – | – | 100.0 | 3.018 |
| C2 RAG | 160 | 98.1 | 1.9 | 0.0 | 0.987 | 0.995 | 100.0 | 4.010 |
| C3 RAG + level | 160 | 99.4 | 0.6 | 0.0 | 0.987 | 0.998 | 100.0 | 2.871 |
| C4 RAG + level + abstain | 160 | 96.9 | 3.1 | 0.0 | 0.988 | 0.993 | 100.0 | 2.657 |

Correctness and level fit rated by the LLM judge. Teacher spot-check not filled in yet.

### Paired tests (fully correct, McNemar exact)

| Comparison | b | c | p |
| --- | --- | --- | --- |
| C1 vs C2 | 3 | 8 | 0.2266 |
| C2 vs C3 | 1 | 3 | 0.6250 |
| C3 vs C4 | 4 | 0 | 0.1250 |

### By CEFR level

| Config. | Level | n | Correct (%) | Level fit (%) | Too difficult (%) |
| --- | --- | --- | --- | --- | --- |
| C1 | A1 | 60 | 96.7 | 100.0 | 0.0 |
| C1 | A2 | 60 | 95.0 | 100.0 | 0.0 |
| C1 | B1 | 40 | 92.5 | 100.0 | 0.0 |
| C2 | A1 | 60 | 98.3 | 100.0 | 0.0 |
| C2 | A2 | 60 | 98.3 | 100.0 | 0.0 |
| C2 | B1 | 40 | 97.5 | 100.0 | 0.0 |
| C3 | A1 | 60 | 98.3 | 100.0 | 0.0 |
| C3 | A2 | 60 | 100.0 | 100.0 | 0.0 |
| C3 | B1 | 40 | 100.0 | 100.0 | 0.0 |
| C4 | A1 | 60 | 96.7 | 100.0 | 0.0 |
| C4 | A2 | 60 | 95.0 | 100.0 | 0.0 |
| C4 | B1 | 40 | 100.0 | 100.0 | 0.0 |

## Table IV. Retrieval quality by question language

| Question language | n | Hit@5 | MRR |
| --- | --- | --- | --- |
| English | 96 | 1.000 | 0.977 |
| German | 38 | 1.000 | 0.890 |
| Mixed | 26 | 1.000 | 0.949 |
| All in-scope | 160 | 1.000 | 0.952 |

## Table V. Abstention

| Config. | Out-of-scope flagged (%) | Precision | False abstentions on in-scope (%) |
| --- | --- | --- | --- |
| C1 Closed-book | 0.0 | n/a | 0.0 |
| C2 RAG | 15.0 | 1.000 | 0.0 |
| C3 RAG + level | 10.0 | 1.000 | 0.0 |
| C4 RAG + level + abstain | 90.0 | 0.947 | 0.6 |
