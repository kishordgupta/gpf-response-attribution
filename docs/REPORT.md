# Response-only attribution on GPF Matched-News

This study asks whether a response's text predicts its recorded model label or developer family within the GPF corpus. The targets are twelve model labels and six developer families. They are collection labels, not authenticated backend weights. The study is separate from the original semantic-distance analysis and introduces no new generated responses.

## Main result

<!-- HEADLINE_START -->
The cleaned response text strongly predicts the recorded labels when the two article topics are shared between training and test. Word TF-IDF reaches **98.77% model accuracy** and **99.94% developer-family accuracy** on held-out identity phrases. The character classifier reaches **97.15%** and **99.86%**, respectively. Eighteen surface/style measurements alone reach 74.06% and 82.53%, compared with the most-frequent baselines of 8.33% and 25.00%.

The main limitation appears under article transfer. Holding out an entire article reduces the same character model's accuracy from **97.15% to 64.37%**, a **32.78 percentage-point** drop; company accuracy falls from **99.86% to 92.16%**, a **7.70-point** drop. Exact-model attribution is much more sensitive to this source/topic shift than developer-family classification. The result supports strong corpus-specific regularities, with substantially weaker transfer to the other article.
<!-- HEADLINE_END -->

A strong score under one split does not establish a general model detector. The primary split reuses the two article topics while holding identity phrases out; the article split changes the entire source/topic condition. These tests measure different kinds of transfer. The character representation was fixed for the stress tests in advance, so the within-topic versus article-transfer comparison must use character versus character, not the best primary representation versus a different stress-test representation.

## Corpus and targets

The canonical CSV contains 7,056 rows and 28 columns: two articles × 42 identity labels × seven question families × twelve models, with one recorded response per condition. Every model receives the same 588 exact prompts. The row key is `(model_name, cell_id)`. The source CSV is obtained from a pinned release of the original repository and verified against SHA-256 `c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da`.

The two articles concern Brewdog insolvency and Denmark's military aid to Ukraine. One is BBC News and the other Reuters via Internazionale. Source, topic, event, and article length are therefore confounded. Recorded capture timestamps span September 19–27, 2026. These dates and model names are preserved observations, not independent verification of model-release history.

| Target | Labels and relative sizes |
|---|---|
| Model | GPT 5.4; GPT 5.4 Mini; Claude Sonnet 4.5; Claude Haiku 4.5; Meta Llama 3.3 Turbo; Meta Llama Maverick 4; Gemini 2.5 Flash; Gemini 2.5 Pro; Mistral Large 3; Mistral Medium 3; Grok Fast; 3.5 Flash-Lite. Each has 588 rows. |
| Developer family | OpenAI, Anthropic, Meta, and Mistral each have two model streams; Google has three; xAI has one. |

The company mapping is derived from the named model family. Together AI and OpenRouter are serving routers for the Meta-family responses, not separate target developers. The full company class distribution is Google 25%; OpenAI, Anthropic, Meta, and Mistral 16.67% each; xAI 8.33%. Actual dummy metrics are reported below rather than assuming a uniform six-way accuracy baseline.

## Response-only inputs and collection artifacts

The only classifier input is the response string after fixed cleaning. Labels, article IDs, identity IDs, prompt-family values, full prompts, timestamps, and source URLs are never appended as features. Metadata defines the target and split groups.

The raw corpus contains 386 leading `Gemini said` headers, all in the Gemini web stream, plus twelve generic `Response:` or numbered-response wrappers. These 398 leading display wrappers are removed; whitespace is normalized. Without this cleaning, a classifier could exploit a collection UI label rather than response behavior. The exact preparation is implemented in `gpf_attribution/text.py`.

After cleaning, the literal model/company-name audit finds no remaining matched terms. Its masking transform consequently changes zero input texts, so retraining a purported masking ablation would repeat the same experiment. The no-op is disclosed rather than presented as an independent robustness result. Exact-name scanning does not rule out indirect identity clues, citation fragments, formatting habits, or hidden-platform effects.

## Fixed baseline specifications

| Representation | Features | Classifier |
|---|---|---|
| `word_svm` | Word unigrams/bigrams; minimum document frequency 2; at most 40,000 TF-IDF features; sublinear term frequency; Unicode accent normalization | LinearSVC, C=1, tolerance 0.0001, maximum 5,000 iterations |
| `char_svm` | Within-word character 3–5-grams; minimum document frequency 3; at most 60,000 TF-IDF features; sublinear term frequency; Unicode accent normalization | Same LinearSVC configuration |
| `style_logit` | Eighteen response-derived numerical measurements: word/character length, word-length distribution, vocabulary diversity, sentence endings, capitalization, digits, punctuation, and Markdown rates | Train-fitted StandardScaler and logistic regression, C=1, `lbfgs`, maximum 3,000 iterations |
| `dummy` | No informative text model | Most-frequent class |

The seed is 20260927. The full configuration is stored in `config/experiment.json`. Hyperparameters are fixed rather than tuned against the reported test folds. The observed best primary row is descriptive; no independent post-selection evaluation is claimed.

TF-IDF vocabulary and IDF statistics, numerical scaling, and classifier fitting use only the training portion of each fold. This follows the principle that preprocessing must not learn from held-out samples; scikit-learn's [common-pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html) explains why fitting transforms before the split can inflate evaluation. The executable environment pins the version actually used, including scikit-learn 1.7.2; the currently served stable documentation is a methodological reference, not the run's version record.

## Grouped evaluation and known data issues

| Protocol | Folds | Group excluded from training | Retained common context |
|---|---:|---|---|
| Primary identity holdout | 5 | Entire `y_group_id`, including all its article-family cells and all model responses | The two articles and seven task families |
| Article holdout | 2 | One complete article condition per fold | Identity inventory and task families |
| Question-family holdout | 7 | One complete question family per fold | The two articles and identity inventory |

Grouped splitting matches the intended transfer question and keeps related observations together, as described in scikit-learn's [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html). It is not ordinary random response-level cross-validation. The article test has only two directions and cannot estimate general performance on unseen news. The family test assesses changed task wording within the same two topics.

For each split, the audit checks that training and test group sets are disjoint. Cleaned responses are also NFKC-normalized, case-folded, and whitespace-normalized for duplicate detection. A training row is removed if its normalized text appears in that fold's test set. This prevents exact normalized duplicate overlap; it does not eliminate paraphrases or thematic similarity. Split manifests make row membership and group keys inspectable. All 21 saved split audits report zero training/test overlap in row indices, prompt cells, and the designated holdout groups. The corpus has zero normalized response duplicates, and no training rows needed purging. These checks establish the declared separation; they do not remove the intentionally shared topics in the identity and family protocols.

Five Gemini records in the original corpus address a different identity or task from the assigned prompt; one discusses both articles. The cause is unresolved: capture alignment, context contamination, or generation error could contribute. The full analysis preserves these rows. A conservative sensitivity excludes every model's row at the five affected cells, leaving 583 complete cells and 6,996 responses, and repeats the character identity/article evaluations. Responses are not relabeled or repaired. A separate screen cannot certify all other captures as valid.

## Measured performance

Metrics use one held-out prediction per evaluated row and are pooled over the appropriate folds. Accuracy measures the fraction correct. Balanced accuracy averages class recall. Macro F1 averages class-specific F1 and gives the smaller company classes equal weight. Fold ranges describe variation across selected conditions; they are not uncertainty intervals over a population of articles or independently generated runs.

<!-- PERFORMANCE_START -->
All scores below are percentages. Full-population experiments evaluate 7,056 rows; screened experiments evaluate 6,996. Each row pools held-out predictions rather than averaging fold percentages.

| Population | Holdout | Representation | Target | Test rows | Accuracy | Macro F1 | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| full | identity | Word TF-IDF + SVM | model | 7056 | 98.77 | 98.77 | 98.77 |
| full | identity | Word TF-IDF + SVM | company | 7056 | 99.94 | 99.95 | 99.95 |
| full | identity | Character TF-IDF + SVM | model | 7056 | 97.15 | 97.15 | 97.15 |
| full | identity | Character TF-IDF + SVM | company | 7056 | 99.86 | 99.87 | 99.87 |
| full | identity | Style + logistic | model | 7056 | 74.06 | 74.02 | 74.06 |
| full | identity | Style + logistic | company | 7056 | 82.53 | 82.84 | 82.43 |
| full | identity | Most frequent | model | 7056 | 8.33 | 1.28 | 8.33 |
| full | identity | Most frequent | company | 7056 | 25.00 | 6.67 | 16.67 |
| full | article | Character TF-IDF + SVM | model | 7056 | 64.37 | 61.09 | 64.37 |
| full | article | Character TF-IDF + SVM | company | 7056 | 92.16 | 91.64 | 90.90 |
| full | family | Character TF-IDF + SVM | model | 7056 | 93.34 | 93.31 | 93.34 |
| full | family | Character TF-IDF + SVM | company | 7056 | 99.35 | 99.37 | 99.38 |
| screened | identity | Character TF-IDF + SVM | model | 6996 | 97.26 | 97.25 | 97.26 |
| screened | identity | Character TF-IDF + SVM | company | 6996 | 99.84 | 99.85 | 99.86 |
| screened | article | Character TF-IDF + SVM | model | 6996 | 63.64 | 60.36 | 63.64 |
| screened | article | Character TF-IDF + SVM | company | 6996 | 92.41 | 91.94 | 91.27 |

The word model makes 87 model-label errors and four company-label errors under identity holdout; the character model makes 201 and ten. Word and character results were both planned primary baselines. The higher word result is not used to select a new stress-test protocol. The company dummy predicts Google for every row: its 25.00% accuracy coexists with only 16.67% balanced accuracy and 6.67% macro F1.

### Transfer by held-out condition

| Held-out article | Training rows | Test rows | Model accuracy | Company accuracy |
|---|---|---|---|---|
| bbc_brewdog | 3528 | 3528 | 64.82 | 92.12 |
| reuters_denmark | 3528 | 3528 | 63.92 | 92.21 |

| Held-out question family | Training rows | Test rows | Model accuracy | Company accuracy |
|---|---|---|---|---|
| bias_check | 6048 | 1008 | 81.25 | 97.32 |
| community | 6048 | 1008 | 97.72 | 99.90 |
| emotion | 6048 | 1008 | 93.85 | 99.90 |
| impact | 6048 | 1008 | 97.22 | 99.60 |
| policy_action | 6048 | 1008 | 94.74 | 99.50 |
| significance | 6048 | 1008 | 94.44 | 99.50 |
| worldview | 6048 | 1008 | 94.15 | 99.70 |

Both article-transfer directions give similar aggregate exact-model accuracy: 64.82% for the BBC/Brewdog condition and 63.92% for the Reuters/Denmark condition. This consistency across two directions does not substitute for a broad sample of independent topics. Question-family transfer is stronger overall at 93.34% model accuracy, but the held-out `bias_check` family is much harder (81.25%) than `community` (97.72%). The task wording therefore also changes separability. Character identity-fold model accuracies range from 95.46% to 97.92%; these five folds share the same news contexts.

### Sensitivity to suspect captures

| Holdout | Target | Full accuracy | Screened accuracy | Change, percentage points |
|---|---|---|---|---|
| identity | model | 97.15 | 97.26 | +0.10 |
| identity | company | 99.86 | 99.84 | -0.02 |
| article | model | 64.37 | 63.64 | -0.73 |
| article | company | 92.16 | 92.41 | +0.25 |

Complete-cell screening leaves the main interpretation unchanged. The largest absolute accuracy change is 0.73 percentage points, in article-transfer model attribution. Screening changes both training and test membership, so this is a sensitivity comparison between two declared populations, not an isolated causal effect of the five problematic responses. No screened word, style, dummy, or family scores are claimed.
<!-- PERFORMANCE_END -->

## Error structure

<!-- ERRORS_START -->
### Per-label recall

The following character-classifier recalls compare the same target under identity and article holdout. A 100% observed recall is confined to this finite test set. Model support is 588 per label; developer support follows the unequal company distribution.

| Model label | Support | Identity recall | Article recall | Article precision |
|---|---|---|---|---|
| 3.5 Flash-Lite | 588 | 99.49 | 71.77 | 93.57 |
| Claude Haiku 4.5 | 588 | 95.24 | 80.44 | 45.31 |
| Claude Sonnet 4.5 | 588 | 94.39 | 41.67 | 60.05 |
| GPT 5.4 | 588 | 95.75 | 17.69 | 88.89 |
| GPT 5.4 Mini | 588 | 96.94 | 96.94 | 45.02 |
| Gemini 2.5 Flash | 588 | 99.66 | 75.17 | 88.76 |
| Gemini 2.5 Pro | 588 | 99.49 | 75.17 | 89.66 |
| Grok Fast | 588 | 100.00 | 83.33 | 92.80 |
| Meta Llama 3.3 Turbo | 588 | 96.09 | 96.09 | 53.86 |
| Meta Llama Maverick 4 | 588 | 96.94 | 10.37 | 96.83 |
| Mistral Large 3 | 588 | 96.94 | 96.94 | 58.52 |
| Mistral Medium 3 | 588 | 94.90 | 26.87 | 95.76 |

| Company label | Support | Identity recall | Article recall | Article precision |
|---|---|---|---|---|
| Anthropic | 1176 | 100.00 | 99.23 | 86.70 |
| Google | 1764 | 99.83 | 91.72 | 90.44 |
| Meta | 1176 | 99.83 | 84.69 | 99.90 |
| Mistral | 1176 | 99.66 | 93.71 | 97.61 |
| OpenAI | 1176 | 99.91 | 99.49 | 87.84 |
| xAI | 588 | 100.00 | 76.53 | 97.19 |

Article transfer errors are markedly asymmetric. GPT 5.4 has only 17.69% recall while GPT 5.4 Mini has 96.94%; however, Mini's article-transfer precision is only 45.02% because many other responses are assigned to it. The same pattern appears for the Llama and Mistral pairs. A strong recall in one row therefore cannot be read as uniformly reliable attribution.

### Largest directed model confusions

| Recorded label | Predicted label | Article-holdout errors | Same developer |
|---|---|---|---|
| GPT 5.4 | GPT 5.4 Mini | 450 | Yes |
| Meta Llama Maverick 4 | Meta Llama 3.3 Turbo | 360 | Yes |
| Mistral Medium 3 | Mistral Large 3 | 298 | Yes |
| Claude Sonnet 4.5 | Claude Haiku 4.5 | 297 | Yes |
| Claude Haiku 4.5 | Claude Sonnet 4.5 | 99 | Yes |
| Grok Fast | GPT 5.4 Mini | 70 | No |
| Gemini 2.5 Flash | Claude Haiku 4.5 | 68 | No |
| Meta Llama Maverick 4 | Claude Haiku 4.5 | 56 | No |

Of the character classifier's 201 model-label errors under identity holdout, 182 (90.55%) remain within the same developer family. Under article holdout, 1,591 of 2,514 errors (63.29%) do so. The largest identity-holdout confusions are Sonnet → Haiku (30), Haiku → Sonnet (26), and Mistral Medium → Large (26). These error patterns help explain why exact-model and company results differ. The direct company classifier is trained separately; its scores are not obtained by merely mapping the model classifier's predictions.

The direct company classifier's largest article-transfer errors are xAI → OpenAI (87), Meta → Google (82), Google → Anthropic (73), and Meta → Anthropic (69). xAI recall falls to 76.53% even though the overall company accuracy remains 92.16%; macro F1 and balanced accuracy make this unequal performance more visible.

### Inspectable learned clues

`results/demo_top_features.csv` lists the top twenty positive character-feature coefficients for each class in the final full-data demonstration fits. Examples include contraction fragments such as `n't` for Anthropic, ` may` for Meta, curly-apostrophe fragments for Mistral, and `frig` for xAI. Sonnet's model coefficients include the article-specific fragment `$276`, while Gemini 2.5 Pro includes `based`. These observations show that the representation can use punctuation, phrasing, and topic-specific fragments. They do not prove that a fragment caused a particular held-out decision. The coefficient list uses full-data demo fits, is not a held-out explanation analysis, and must not be used to tune the reported benchmark retroactively.
<!-- ERRORS_END -->

Confusions describe which labels are difficult to separate in this particular panel. A within-company error is not proof of shared training data or architecture; a cross-company success is not proof of a stable company fingerprint. The response may reflect style, task compliance, formatting, topic-specific vocabulary, or platform context. The numerical style baseline helps assess how much can be predicted from surface structure, but it does not causally decompose the text model's decisions.

## Demonstration artifacts versus evaluated models

The benchmark trains independent models inside each fold to produce evaluation predictions. The final demonstration artifacts are fitted afterward on the full allowed population. Their predictions are useful for inspecting the closed-set behavior but cannot validate their own training-set performance.

The quick `--fit-demo-only` path creates the two character attribution artifacts without rerunning every evaluation. It does not create new held-out scores. Inference returns top labels and SVM decision margins; these are not calibrated probabilities. There is no abstention or unknown-class threshold. An out-of-domain response is still assigned to one of the known labels even if it was written by a human or another model.

## What the findings support

The experiment measures how well these fixed text classifiers discriminate recorded labels under the declared splits. It can reveal strong response-style regularities, diagnose how evaluations depend on topic reuse, and motivate broader replicated studies.

It cannot establish a universal source detector, intrinsic model-weight signatures, stable behavior under later serving updates, human-versus-AI attribution, factual accuracy, fairness, or harmful bias. Model labels are entangled with platform, date, router, hidden prompts, and unknown generation settings. One response per condition provides no estimate of generation-to-generation variation. Two articles provide very limited topical coverage. Identity phrases overlap and are synthetic stimuli, not measured attributes of participants.

Same-company versions and tiers are cross-sectional products. Predictability of their text does not establish chronological evolution or improved capability. High label separability can coexist with nearly identical factual content; low separability can coexist with different quality. The results must not be used for academic misconduct allegations or other consequential authorship decisions.

A stronger extension would include many independently sampled topics, authenticated serving checkpoints and settings, fresh collection periods, repeated generations, unknown-source and human-written examples, and predeclared calibration/abstention evaluation. External test data must remain separate from all feature and hyperparameter decisions.

## Reproducibility and access

Run the commands in [README.md](../README.md). The repository publishes code, configuration, aggregate metrics, compressed held-out predictions and split manifests, an offline HTML dashboard, and an exploration notebook. The full raw dataset is downloaded from the original public release using a pinned source and checksum; it is not republished here. Trust model artifacts before loading Python serialization.

The published out-of-fold predictions are the frozen evidence underlying the tables. The lightweight verification path runs `python -m unittest discover -s tests -v` and `python scripts/validate_release.py --require-dashboard` after the source download; CI uses these independent checks without refitting. Exact dependency versions and the fixed seed do not guarantee bitwise-identical retraining across platforms. Feature selection at a `max_features` frequency tie and numerical solvers can vary. A rerun should retain its own results and environment record when comparing against this release.

The original corpus is described by **Mohd Ariful Haque and Kishor Datta Gupta (2026)** in [the GPF Matched-News preprint](https://www.researchgate.net/publication/414852735_GPF_Matched-News_Dataset_Semantic_Variation_Across_Models_and_Identity_Prompts). See [CITATION.md](../CITATION.md), [the original analysis](https://github.com/kishordgupta/gpf-semantic-response-audit), and [RIGHTS.md](../RIGHTS.md). This report is a separate response-attribution analysis and does not modify the earlier manuscript.
