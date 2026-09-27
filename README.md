# GPF Response Attribution

Can a classifier identify a response's **recorded model label** or **developer family** from its text alone? This project tests that question on the 7,056-response GPF Matched-News corpus, using held-out identities, articles, and question families.

The input is cleaned response text. The targets are twelve recorded model labels or six explicitly mapped developer families. This is a **closed-set research experiment**, not a general detector of AI writing, human authorship, or an authenticated backend checkpoint.

- [Interactive offline results dashboard](dashboard/index.html) — download and open the HTML file in a browser; GitHub's file view shows its source.
- [Analysis report](docs/REPORT.md) and [model card](docs/MODEL_CARD.md).
- [Starter notebook](notebooks/GPF_Response_Attribution.ipynb).
- [Original semantic analysis and dataset](https://github.com/kishordgupta/gpf-semantic-response-audit), [Kaggle dataset](https://www.kaggle.com/datasets/kishor1123/gpf-matched-news-7056-responses-from-12-models), and [preprint](https://www.researchgate.net/publication/414852735_GPF_Matched-News_Dataset_Semantic_Variation_Across_Models_and_Identity_Prompts).

## What is evaluated

The original design contains `2 articles × 42 identity labels × 7 question families = 588 exact prompts`, each answered by twelve model labels. All 7,056 rows have one recorded generation; there are no independent repeats.

| Evaluation | Held-out groups | Question it answers |
|---|---|---|
| Primary, five folds | Identity label (`y_group_id`) | Does attribution transfer to unseen identity phrases within the same two articles and seven tasks? |
| Article stress test, two folds | Article | Does a model trained on one article transfer to the other article/topic? |
| Family stress test, seven folds | Question family | Does attribution transfer to an unseen question objective within the same articles and identities? |

The primary comparison uses word TF-IDF + LinearSVC, character TF-IDF + LinearSVC, numerical style features + logistic regression, and a most-frequent dummy. Article/family stress tests use the fixed character baseline. A conservative screen excludes five suspect complete prompt cells, leaving 6,996 rows, and repeats character identity/article evaluations.

<!-- RESULTS_START -->
| Representation / holdout | Model accuracy | Company accuracy |
|---|---|---|
| Word TF-IDF + SVM / identity | 98.77% | 99.94% |
| Character TF-IDF + SVM / identity | 97.15% | 99.86% |
| Style + logistic / identity | 74.06% | 82.53% |
| Most frequent / identity | 8.33% | 25.00% |
| Character / article | 64.37% | 92.16% |
| Character / family | 93.34% | 99.35% |

The same character classifier falls **32.78 percentage points** in model accuracy when an entire article is held out (97.15% → 64.37%). High performance on unseen identity phrases within these two stories does not establish a general detector. Company results use a separately trained classifier, and the 25% majority baseline reflects Google's larger class. Full macro F1, balanced accuracy, screening results, and error analysis are in the report and dashboard.
<!-- RESULTS_END -->

## Reproduce the analysis

Use Python 3.12 and the exact dependencies in `requirements.txt`. This lightweight path downloads the pinned source and independently verifies the published evidence without refitting the classifiers:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py --output data/responses.csv
python -m unittest discover -s tests -v
python scripts/validate_release.py --require-dashboard
```

The downloader fetches the original public dataset at a pinned GitHub commit, decompresses it, and verifies SHA-256 before saving. It refuses to overwrite a different existing file. The full raw corpus is **not republished in this repository**. Training and prediction run locally; there are no model-generation or embedding API calls.

The expected source CSV SHA-256 is:

```text
c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da
```

To rerun the full benchmark and rebuild its dashboard:

```sh
python scripts/train.py --data data/responses.csv --output results
python scripts/build_dashboard.py --results results --output dashboard/index.html
```

Pinned dependencies and a fixed seed improve reproducibility but do not guarantee bitwise-identical retraining across platforms: feature selection at a `max_features` frequency tie and numerical solvers can vary. The saved out-of-fold predictions are the frozen evidence for the published tables. The validator and CI independently check that evidence without refitting.

Precomputed result tables make the report and dashboard usable without retraining. Serialized models are produced locally by the training workflow; only load artifacts you trust. To fit only the two final demonstration artifacts without rerunning the full evaluation, then inspect a response:

```sh
python scripts/train.py --data data/responses.csv --output results --fit-demo-only
python scripts/predict.py \
  --artifact artifacts/model_char_full.joblib \
  --text-file response.txt \
  --top-k 3
```

Returned decision margins rank the known labels; they are **not calibrated probabilities**. The final fitted artifact is distinct from the held-out models used to report evaluation scores. It cannot determine that a response came from an unknown model or a human, and it should not be used to make consequential authorship decisions.

The starter notebook opens precomputed results first; downloading data, retraining, prediction, and exports are separate opt-in cells. Open it in a Jupyter environment with the same project dependencies (Jupyter itself is an optional interface and is not needed for the command-line benchmark). Its default path makes no network requests and does not modify the source data.

## Controls that matter

- Only `response` text becomes a feature. Prompt, article, identity, model, provider, company, and timestamp fields define labels, groups, or audits, not classifier inputs.
- Known leading display wrappers are removed before vectorization. The `Gemini said` prefix appears in 386 Gemini web outputs and zero other streams; twelve generic response wrappers are also removed. Leaving them in would reward recognition of collection artifacts.
- Vocabulary, IDF, and style scaling are fitted within each training fold. Entire held-out groups remain unseen during training. Normalized train/test text duplicates are audited and purged from training if present.
- Hyperparameters are fixed rather than chosen from these test scores. Screening does not silently correct or reassign the original responses.
- Company is an explicit developer-family mapping, not the original serving-provider field: Together AI and OpenRouter are routers for Meta-family labels.

Company labels are unequal: Google supplies three of twelve model streams, OpenAI/Anthropic/Meta/Mistral two each, and xAI one. Therefore Google's majority share is 25%; a six-way uniform accuracy baseline is not the correct most-frequent reference. Inspect the actual dummy, macro F1, and balanced accuracy results.

## Files

| Path | Purpose |
|---|---|
| `config/experiment.json` | Fixed feature, estimator, target, and split specifications |
| `gpf_attribution/` | Data checks, text preparation, estimators, and prediction helpers |
| `scripts/download_data.py` | Pinned, checksum-verified source acquisition |
| `scripts/train.py` | Held-out evaluation and final local artifacts |
| `scripts/predict.py` | Closed-set prediction for supplied response text |
| `results/metrics.csv` | Fold and pooled accuracy, macro F1, and balanced accuracy |
| `results/per_class.csv` | Per-label precision, recall, F1, and support |
| `results/confusion_counts.csv` | Exact pooled confusion counts |
| `results/predictions.csv.gz` | Held-out predictions and margins, without response text |
| `results/split_manifest.csv.gz` | Row membership and group keys for each split |
| `results/split_audits.json` | Separation and duplicate-purge audits |
| `results/demo_top_features.csv` | Descriptive positive character coefficients from full-data demo fits |
| `dashboard/index.html` | Self-contained interactive aggregate results |
| `docs/REPORT.md` | Methods, findings, and limits |
| `docs/MODEL_CARD.md` | Intended use, behavior, and restrictions |

## Interpretation and rights

Labels also differ in platform, collection date, route, and possibly hidden settings. Two articles and one generation per condition cannot establish a stable model fingerprint. High attribution scores do not mean that a model is more biased, truthful, fair, or capable. Cross-sectional version/tier differences do not demonstrate temporal evolution. See the [report](docs/REPORT.md) for the exact scope.

Code and original documentation use the [MIT license](LICENSE). Dataset and publisher-content rights are separate; see [RIGHTS.md](RIGHTS.md). Please cite **Mohd Ariful Haque and Kishor Datta Gupta (2026)** and the original GPF preprint as detailed in [CITATION.md](CITATION.md).
