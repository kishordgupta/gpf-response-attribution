# Response-attribution confound audit

Canonical input: 7,056 rows, 28 columns; SHA-256 `c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da`. The input was read only.

**Direct UI leakage:** 386 of 588 Gemini web responses (65.65%) have a leading `Gemini said` label; no other class has it. Conservative leading-wrapper cleaning removes all 386 observed explicit model/company token matches. No such token matches remain in cleaned text.

An additional 12 Claude Sonnet responses have a generic leading `Response:` wrapper. The fixed cleaning removes 398 leading wrappers in total; these generic wrappers do not explicitly name the class.

**Duplicates:** no exact raw or cleaned/casefolded duplicate groups. The all-pairs lexical check uses 61,935 eligible features and finds 0 pairs at cosine similarity ≥0.95. Maximum observed similarity is 0.905215. This does not establish response independence.

**Known alignment issues:** five Gemini observations have previously identified target inconsistencies. A conservative screen removes their entire cells across all models, leaving 583 cells and 6,996 rows. They are preserved in the canonical data.

**Company grouping:** model-family developers are explicitly mapped; serving routers are not treated as developers. Model classes are balanced, whereas the company task has class imbalance.

## Evaluation requirements

- Leading UI headers directly expose a model class and must be removed before both fitting and inference.
- After header cleaning, formatting, length, refusal style, and collection-interface behavior may still identify a captured product stream; attribution is not proof of model architecture or training lineage.
- Use response text only. Model/provider IDs, run/cell IDs, timestamps, source metadata, and collection-stage columns are forbidden input features.
- All fourteen article/family cells and all twelve models for an identity should stay in the same primary split. A row-random split is not an independent prompt-generalization evaluation.
- Exact duplicate hashes must not cross a fitted model's train/test boundary. Near-duplicate screening is descriptive and threshold dependent.
- Model classes are balanced; company classes are not. Google is 25 percent and xAI 8.33 percent of the corpus. Report macro-F1 and balanced accuracy alongside accuracy and a dummy baseline.
- Two articles only; article/source/topic are confounded. Article holdout measures a shift between these two conditions, not broad news-domain generalization.
- Labels are recorded UI/router aliases, not authenticated backend checkpoints. Inference is closed-set among the supplied labels and cannot establish the origin of arbitrary text.
