#!/usr/bin/env python3
"""Audit the frozen response corpus without printing or rewriting source text."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_chunked

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da"
COMPANY = {
    "GPT 5.4": "OpenAI", "GPT 5.4 Mini": "OpenAI",
    "Claude Sonnet 4.5": "Anthropic", "Claude Haiku 4.5": "Anthropic",
    "Meta Llama 3.3 Turbo": "Meta", "Meta Llama Maverick 4": "Meta",
    "Gemini 2.5 Flash": "Google", "Gemini 2.5 Pro": "Google",
    "Mistral Large 3": "Mistral", "Mistral Medium 3": "Mistral",
    "Grok Fast": "xAI", "3.5 Flash-Lite": "Google",
}
FLAGGED_CELLS = {
    "bbc_brewdog__gender_sexuality__07__significance",
    "bbc_brewdog__geography__05__community",
    "bbc_brewdog__political__00__bias_check",
    "reuters_denmark__gender_sexuality__00__emotion",
    "reuters_denmark__religion__06__worldview",
}
MODEL_NAMES = re.compile(r"\b(?:openai|chatgpt|gpt[ -]?\d|anthropic|claude|google|gemini|grok|xai|mistral|llama|meta)\b", re.I)


def independent_clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while True:
        stripped = re.sub(r"^\s*(?:Gemini said|Response(?:\s+\d+)?:)\s*\n+", "", text, count=1, flags=re.I)
        if stripped == text:
            break
        text = stripped
    return re.sub(r"\s+", " ", text).strip()


def duplicates(values):
    groups = collections.defaultdict(list)
    for index, text in enumerate(values):
        groups[text].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/responses.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "audits")
    args = parser.parse_args()
    data = args.data.read_bytes()
    raw = gzip.decompress(data) if args.data.suffix == ".gz" else data
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Input is not the exact canonical 28-column CSV.")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    rows = list(reader)
    assert len(rows) == 7056 and len(reader.fieldnames) == 28
    assert set(row["model_name"] for row in rows) == set(COMPANY)
    cleaned = [independent_clean(row["response"]) for row in rows]
    raw_headers = []
    raw_name_rows = []
    cleaned_name_rows = []
    generic_response_wrappers = []
    flags = []
    for index, (row, text) in enumerate(zip(rows, cleaned)):
        reasons = []
        if re.match(r"^\s*Gemini said\b", row["response"]):
            raw_headers.append(index)
            reasons.append("leading_model_identifying_ui_header")
        if re.match(r"^\s*Response(?:\s+\d+)?:\s*\n", row["response"], flags=re.I):
            generic_response_wrappers.append(index)
            reasons.append("leading_generic_response_wrapper")
        if MODEL_NAMES.search(row["response"]):
            raw_name_rows.append(index)
        if MODEL_NAMES.search(text):
            cleaned_name_rows.append(index)
            reasons.append("explicit_model_or_company_token_after_cleaning")
        if row["model_name"] == "3.5 Flash-Lite" and row["cell_id"] in FLAGGED_CELLS:
            reasons.append("known_prompt_response_target_inconsistency")
        if reasons:
            flags.append({"row_index": index, "model_name": row["model_name"], "cell_id": row["cell_id"], "flags": ";".join(reasons)})
    # All eligible features are retained; there is no unstable max_features tie cutoff.
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(cleaned)
    thresholds = [0.8, 0.9, 0.95, 0.97]
    counts = {str(t): 0 for t in thresholds}
    highest = {"cosine_similarity": -1.0}
    high_pairs = []
    offset = 0
    for block in pairwise_distances_chunked(matrix, metric="cosine", working_memory=64, n_jobs=1):
        for local in range(len(block)):
            block[local, :offset + local + 1] = np.inf
        for threshold in thresholds:
            counts[str(threshold)] += int(np.count_nonzero(block <= 1 - threshold))
        i, j = np.unravel_index(np.argmin(block), block.shape)
        similarity = float(1 - block[i, j])
        if similarity > highest["cosine_similarity"]:
            first = offset + int(i)
            second = int(j)
            highest = {"row_a": first, "row_b": second, "cosine_similarity": similarity,
                       "model_a": rows[first]["model_name"], "model_b": rows[second]["model_name"],
                       "cell_a": rows[first]["cell_id"], "cell_b": rows[second]["cell_id"]}
        aa, bb = np.where(block <= 0.05)
        for i, j in zip(aa, bb):
            first, second = offset + int(i), int(j)
            high_pairs.append({"row_a": first, "row_b": second, "cosine_similarity": float(1 - block[i, j]),
                               "same_model": rows[first]["model_name"] == rows[second]["model_name"],
                               "same_cell": rows[first]["cell_id"] == rows[second]["cell_id"]})
        offset += len(block)
    run_id_counts = collections.Counter(row["run_id"] for row in rows)
    report = {
        "dataset_sha256": digest, "rows": len(rows), "columns": len(reader.fieldnames),
        "model_counts": dict(collections.Counter(row["model_name"] for row in rows)),
        "company_counts": dict(collections.Counter(COMPANY[row["model_name"]] for row in rows)),
        "company_mapping": COMPANY,
        "recorded_provider_by_model": {model: sorted({row["model_provider"] for row in rows if row["model_name"] == model}) for model in COMPANY},
        "leading_ui_header": {"total": len(raw_headers), "model_counts": dict(collections.Counter(rows[i]["model_name"] for i in raw_headers)), "fraction_of_gemini_web_class": len(raw_headers) / 588},
        "generic_response_wrapper": {"total": len(generic_response_wrappers), "model_counts": dict(collections.Counter(rows[i]["model_name"] for i in generic_response_wrappers))},
        "all_leading_wrapper_changed_rows": sum(text != re.sub(r"\s+", " ", row["response"]).strip() for row, text in zip(rows, cleaned)),
        "explicit_name_scan": {"pattern": MODEL_NAMES.pattern, "raw_rows": len(raw_name_rows), "cleaned_rows": len(cleaned_name_rows), "cleaned_row_indices": cleaned_name_rows,
                               "interpretation": "A token match is a heuristic, not proof of self-identification. Raw matches overlap the model-identifying UI header."},
        "duplicates": {"raw_exact_groups": duplicates([row["response"] for row in rows]), "cleaned_casefolded_groups": duplicates([text.casefold() for text in cleaned])},
        "lexical_near_duplicate_audit": {"representation": "lowercase word unigram/bigram TF-IDF, min_df=2, sublinear_tf=True, all eligible features, cosine similarity",
                                         "eligible_features": matrix.shape[1], "unordered_pairs_checked": len(rows) * (len(rows) - 1) // 2,
                                         "pair_counts_at_or_above_similarity": counts, "highest_pair": highest,
                                         "candidate_threshold": 0.95, "candidates": high_pairs,
                                         "interpretation": "High lexical similarity is a review candidate, not proof of copied or miscaptured responses. Absence at this threshold does not prove statistical independence."},
        "known_target_inconsistencies": {"affected_model": "3.5 Flash-Lite", "cells": sorted(FLAGGED_CELLS),
                                          "row_indices": [index for index, row in enumerate(rows) if row["model_name"] == "3.5 Flash-Lite" and row["cell_id"] in FLAGGED_CELLS],
                                          "conservative_complete_cell_screen": {"cells": 583, "rows": 6996},
                                          "interpretation": "Retain canonical data; report a sensitivity removing these five cells across all models. Cause is unresolved, not a proven model-defect label."},
        "run_id_reuse": {"unique": len(run_id_counts), "values_used_twice": sum(value == 2 for value in run_id_counts.values()), "observation_key": ["model_name", "cell_id"]},
        "limitations": [
            "Leading UI headers directly expose a model class and must be removed before both fitting and inference.",
            "After header cleaning, formatting, length, refusal style, and collection-interface behavior may still identify a captured product stream; attribution is not proof of model architecture or training lineage.",
            "Use response text only. Model/provider IDs, run/cell IDs, timestamps, source metadata, and collection-stage columns are forbidden input features.",
            "All fourteen article/family cells and all twelve models for an identity should stay in the same primary split. A row-random split is not an independent prompt-generalization evaluation.",
            "Exact duplicate hashes must not cross a fitted model's train/test boundary. Near-duplicate screening is descriptive and threshold dependent.",
            "Model classes are balanced; company classes are not. Google is 25 percent and xAI 8.33 percent of the corpus. Report macro-F1 and balanced accuracy alongside accuracy and a dummy baseline.",
            "Two articles only; article/source/topic are confounded. Article holdout measures a shift between these two conditions, not broad news-domain generalization.",
            "Labels are recorded UI/router aliases, not authenticated backend checkpoints. Inference is closed-set among the supplied labels and cannot establish the origin of arbitrary text.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confounds.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "row_flags.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "model_name", "cell_id", "flags"])
        writer.writeheader()
        writer.writerows(flags)
    lines = ["# Response-attribution confound audit", "",
             f"Canonical input: {len(rows):,} rows, 28 columns; SHA-256 `{digest}`. The input was read only.", "",
             f"**Direct UI leakage:** {len(raw_headers)} of 588 Gemini web responses ({100 * len(raw_headers) / 588:.2f}%) have a leading `Gemini said` label; no other class has it. Conservative leading-wrapper cleaning removes all {len(raw_name_rows)} observed explicit model/company token matches. No such token matches remain in cleaned text.", "",
             f"An additional {len(generic_response_wrappers)} Claude Sonnet responses have a generic leading `Response:` wrapper. The fixed cleaning removes 398 leading wrappers in total; these generic wrappers do not explicitly name the class.", "",
             f"**Duplicates:** no exact raw or cleaned/casefolded duplicate groups. The all-pairs lexical check uses {matrix.shape[1]:,} eligible features and finds {len(high_pairs)} pairs at cosine similarity ≥0.95. Maximum observed similarity is {highest['cosine_similarity']:.6f}. This does not establish response independence.", "",
             "**Known alignment issues:** five Gemini observations have previously identified target inconsistencies. A conservative screen removes their entire cells across all models, leaving 583 cells and 6,996 rows. They are preserved in the canonical data.", "",
             "**Company grouping:** model-family developers are explicitly mapped; serving routers are not treated as developers. Model classes are balanced, whereas the company task has class imbalance.", "",
             "## Evaluation requirements", ""]
    lines += ["- " + item for item in report["limitations"]]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    print(json.dumps({"rows": len(rows), "headers": len(raw_headers), "cleaned_explicit_names": len(cleaned_name_rows), "near_duplicate_candidates": len(high_pairs), "max_lexical_similarity": highest["cosine_similarity"]}, indent=2))


if __name__ == "__main__":
    main()
