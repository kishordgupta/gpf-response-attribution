# Model card: GPF response attribution

## Model purpose

These local classifiers predict either one of twelve recorded model labels or one of six developer-family labels from a single response's cleaned text. They support a controlled attribution experiment on the GPF Matched-News snapshot. They do not detect whether arbitrary writing is AI-generated, identify a human author, verify a serving checkpoint, or provide an unknown-source option.

The label set is closed. A new text is assigned among known labels even if none is appropriate. SVM decision margins are ranking scores, not probabilities or evidence of provenance. A high margin is not a validated confidence guarantee.

## Training data and labels

The source is the unchanged 7,056-row GPF Matched-News CSV: twelve model labels, two articles, 42 synthetic identity phrases, seven question families, and one response per condition. There are 588 matched prompts. The raw dataset is obtained with a pinned downloader and checksum, not bundled here.

| Developer family | Recorded model labels |
|---|---|
| OpenAI | GPT 5.4; GPT 5.4 Mini |
| Anthropic | Claude Sonnet 4.5; Claude Haiku 4.5 |
| Meta | Meta Llama 3.3 Turbo; Meta Llama Maverick 4 |
| Google | Gemini 2.5 Flash; Gemini 2.5 Pro; 3.5 Flash-Lite |
| Mistral | Mistral Large 3; Mistral Medium 3 |
| xAI | Grok Fast |

The developer family is derived from the recorded model family. `model_provider` sometimes identifies a router and is not used as the company ground truth. These aliases are collection labels, not independently authenticated checkpoints. The two articles concern Brewdog insolvency and Denmark/Ukraine military aid. Source, topic, event, and article length are confounded.

## Inputs and preprocessing

Only response text is supplied to each feature pipeline. Metadata is used to construct labels and split groups. Known leading `Gemini said` and `Response:`/numbered response wrappers are removed, then whitespace is normalized. This matters because the Gemini display wrapper appears in 386 of its 588 web outputs and no other stream. No article text or prompt field is appended.

Literal model/vendor names are audited after cleaning. No literal model/company terms remain after cleaning; masking is therefore a disclosed no-op. This does not rule out indirect identifiers, contextual habits, citation artifacts, or stylistic clues. The numerical style baseline includes length, vocabulary diversity, sentence-ending, capitalization, digit, punctuation, and Markdown features of the cleaned response; it does not use metadata.

## Algorithms and evaluated conditions

- Word unigram/bigram TF-IDF with a linear support-vector classifier.
- Character-within-word 3–5-gram TF-IDF with a linear support-vector classifier.
- Eighteen surface/style measurements, train-fitted standardization, and logistic regression.
- Most-frequent dummy baseline.

Exact settings and the random seed are in `config/experiment.json`. Features and classifiers are trained separately within each evaluation fold. The primary split holds out identity labels in five folds; character-model stress tests hold out each article or each question family. The screened character analysis removes all model rows at five suspect prompt cells and repeats identity/article tests.

The final demonstration artifacts are fitted on the full allowed training population after evaluation. Their training-set fit is not the reported evaluation. Do not mix their predictions back into out-of-fold tables or evaluate them on their own training rows as evidence of transfer.

<!-- METRICS_START -->
## Measured held-out performance

| Representation / holdout | Model accuracy | Company accuracy |
|---|---|---|
| Word TF-IDF + SVM / identity | 98.77% | 99.94% |
| Character TF-IDF + SVM / identity | 97.15% | 99.86% |
| Style + logistic / identity | 74.06% | 82.53% |
| Most frequent / identity | 8.33% | 25.00% |
| Character / article | 64.37% | 92.16% |
| Character / family | 93.34% | 99.35% |

All values pool held-out predictions on the full 7,056-row corpus. For the character article test, macro F1 is 61.09% (model) and 91.64% (company); balanced accuracy is 64.37% and 90.90%. Removing all twelve models at five suspect cells changes character identity model accuracy from 97.15% to 97.26% and article model accuracy from 64.37% to 63.64%. Consult the full report for all sixteen conditions and per-label errors. These scores do not evaluate the final all-data demonstration artifacts on independent new data.
<!-- METRICS_END -->

## Evaluation and appropriate interpretation

Accuracy is the fraction of correct labels. Macro F1 weights classes equally; balanced accuracy averages per-class recall. Company class sizes are unequal: Google's three streams account for 25% of the rows. Report the measured dummy and class-sensitive metrics along with accuracy.

The identity split measures transfer to unseen Y phrases while retaining the same two topics and seven question types. Article transfer has only two directions. Family transfer has seven held-out tasks. These are different estimands; none constitutes an unrestricted model-identification benchmark. Fold variation is descriptive, not independent replicate or population-level uncertainty.

Five known Gemini records have content inconsistent with assigned identity/task. Their cause could be capture alignment, context contamination, or generation behavior. Screening excludes complete cells for all twelve labels rather than relabeling responses. It is not a proof that every retained record is correct.

## Intended use

Use for reproducible closed-set benchmark comparison, teaching about grouped validation and collection leakage, examining text-style signals, and generating hypotheses for broader attribution studies. Inspect per-class errors, not only pooled scores. Record the artifact version, source checksum, split, and preprocessing when reporting a result.

Do not use this classifier for academic misconduct allegations, employment decisions, content enforcement, human-versus-AI judgments, attribution outside the known label panel, or conclusions about model fairness or quality. It has no validated abstention rule or calibrated source probabilities. Short fragments, paraphrases, edited text, new topics, languages, generation settings, and later checkpoints are outside demonstrated scope.

## Known limitations and confounds

The dataset has one output per condition, two articles, overlapping identity labels, and no human-written or unknown-model examples. Label identity is partly entangled with collection platform, date, serving route, system prompt, and hidden generation settings. Generic cleaning removes known wrappers but cannot isolate an intrinsic fingerprint of model weights. Same-company labels differ by size, tier, version, and sometimes platform; this is not longitudinal model evolution.

Identity labels are experimental prompt phrases, not attributes measured from real participants. Successful classification does not establish harmful bias, stereotype endorsement, factuality, or capability. Failure to attribute can reflect similar response styles rather than shared underlying weights.

## Provenance, rights, and maintenance

The source SHA-256 is `c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da`. See [CITATION.md](../CITATION.md) for the Haque–Gupta 2026 preprint and [RIGHTS.md](../RIGHTS.md) for separate third-party content rights. Model artifacts should be loaded only from trusted sources because Python serialization is executable. Retrain and reevaluate before adapting the system to a materially different data source or label panel.
