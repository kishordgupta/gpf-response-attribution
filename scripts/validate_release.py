#!/usr/bin/env python3
"""Verify saved attribution evidence without refitting the research models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audits.audit_confounds import EXPECTED_SHA256, COMPANY, FLAGGED_CELLS, independent_clean
from gpf_attribution.data import load_dataset
from gpf_attribution.text import clean_response, normalized_response_hash, mask_model_names

EXPECTED_PLANS = {
    ("full", "identity", representation, target)
    for representation in ["word_svm", "char_svm", "style_logit", "dummy"]
    for target in ["model", "company"]
} | {
    (population, strategy, "char_svm", target)
    for population, strategy in [("full", "article"), ("full", "family"), ("screened", "identity"), ("screened", "article")]
    for target in ["model", "company"]
}


def independently_score(truth, predicted, labels):
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guessed in zip(truth, predicted):
        matrix[label_index[actual], label_index[guessed]] += 1
    true_positives = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(true_positives, predicted_support, out=np.zeros_like(true_positives), where=predicted_support != 0)
    recall = np.divide(true_positives, support, out=np.zeros_like(true_positives), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positives), where=(precision + recall) != 0)
    return {"accuracy": float(true_positives.sum() / matrix.sum()), "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean())}, matrix, precision, recall, f1, support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/responses.csv")
    parser.add_argument("--require-dashboard", action="store_true")
    args = parser.parse_args()
    checks = []

    def check(name, condition, detail=None):
        entry = {"check": name, "passed": bool(condition)}
        if detail is not None:
            entry["detail"] = detail
        checks.append(entry)
        if not condition:
            raise AssertionError(name)

    output = ROOT / "audits/release_validation.json"
    try:
        before = hashlib.sha256(args.data.read_bytes()).hexdigest()
        check("Original dataset has the immutable expected checksum", before == EXPECTED_SHA256)
        data = load_dataset(args.data)
        check("Canonical shape, class balance and developer mapping", len(data) == 7056 and data.groupby("model_name").size().eq(588).all() and data.company.equals(data.model_name.map(COMPANY)))
        expected_text = data.response.map(independent_clean)
        actual_text = data.response.map(clean_response)
        check("Production cleaning equals independent cleaning for every row", expected_text.equals(actual_text))
        hashes = expected_text.map(lambda text: hashlib.sha256(unicodedata.normalize("NFKC", text).casefold().encode()).hexdigest())
        check("Production duplicate hashes equal independently normalized hashes", hashes.equals(data.response.map(normalized_response_hash)))
        check("Independent normalized-text audit finds no duplicate rows", hashes.is_unique)
        check("Leading model-identifying UI labels do not survive cleaning", not actual_text.str.match(r"^Gemini said\s*", case=False).any())
        check("Literal model/vendor masking changes no cleaned corpus row", actual_text.map(mask_model_names).equals(actual_text))
        wrapper_count = sum(text != re.sub(r"\s+", " ", raw).strip() for raw, text in zip(data.response, actual_text))
        check("All 398 recorded leading wrappers are removed", wrapper_count == 398)

        summary = json.loads((ROOT / "results/summary.json").read_text())
        status = json.loads((ROOT / "results/status.json").read_text())
        check("Training evidence reports completed frozen experiment", status["phase"] == "complete" and summary["data_sha256"] == EXPECTED_SHA256 and summary["rows"] == 7056)
        check("Five-cell sensitivity matches independently audited cells", set(summary["screened_excluded_cells"]) == FLAGGED_CELLS and summary["screened_rows"] == 6996)
        check("Company imbalance and header audit are recorded accurately", summary["target_company_counts"] == data.company.value_counts().to_dict() and summary["cleaning"]["leading_wrapper_changed_rows"] == 398 and summary["cleaning"]["gemini_said_leading_rows"] == 386)

        splits = pd.read_csv(ROOT / "results/split_manifest.csv.gz", keep_default_na=False)
        check("Split records use valid canonical row indices and roles", splits.row_index.between(0, len(data) - 1).all() and set(splits.role) <= {"train", "test", "purged"})
        check("Split records are unique within each fold", not splits.duplicated(["population", "split_strategy", "fold", "row_index"]).any())
        for saved, canonical in [("cell_id", "cell_id"), ("y_group_id", "y_group_id"), ("article_id", "x_group_id"), ("prompt_family", "prompt_family")]:
            check(f"Split {saved} values match the immutable CSV", np.array_equal(splits[saved].to_numpy(), data.loc[splits.row_index, canonical].to_numpy()))
        split_index = {}
        split_count = 0
        for key, group in splits.groupby(["population", "split_strategy", "fold"], sort=False):
            population, strategy, fold = key
            permitted = set(data.index if population == "full" else data.index[~data.cell_id.isin(FLAGGED_CELLS)])
            train = set(group.loc[group.role.eq("train"), "row_index"])
            test = set(group.loc[group.role.eq("test"), "row_index"])
            purged = set(group.loc[group.role.eq("purged"), "row_index"])
            column = {"identity": "y_group_id", "article": "x_group_id", "family": "prompt_family"}[strategy]
            check(f"{population}/{strategy}/{fold}: roles partition the correct population", train | test | purged == permitted and not (train & test or train & purged or test & purged) and bool(train) and bool(test))
            check(f"{population}/{strategy}/{fold}: no cell or held-out-group leakage", not (set(data.loc[list(train), "cell_id"]) & set(data.loc[list(test), "cell_id"])) and not (set(data.loc[list(train), column]) & set(data.loc[list(test), column])))
            check(f"{population}/{strategy}/{fold}: no normalized response overlap after purge", not (set(hashes.loc[list(train)]) & set(hashes.loc[list(test)])))
            check(f"{population}/{strategy}/{fold}: all target classes represented in training", set(data.loc[list(train), "model_name"]) == set(COMPANY) and set(data.loc[list(train), "company"]) == set(COMPANY.values()))
            if purged:
                check(f"{population}/{strategy}/{fold}: every purged row actually collides with test", set(hashes.loc[list(purged)]) <= set(hashes.loc[list(test)]))
            split_index[key] = {"train": train, "test": test, "purged": purged}
            split_count += 1
        check("All 21 planned held-out folds are present", split_count == 21 and summary["split_count"] == 21)
        for (population, strategy), group in splits.loc[splits.role.eq("test")].groupby(["population", "split_strategy"]):
            permitted = set(data.index if population == "full" else data.index[~data.cell_id.isin(FLAGGED_CELLS)])
            check(f"{population}/{strategy}: every eligible row is tested exactly once", set(group.row_index) == permitted and not group.row_index.duplicated().any())

        predictions = pd.read_csv(ROOT / "results/predictions.csv.gz", keep_default_na=False, low_memory=False)
        metrics = pd.read_csv(ROOT / "results/metrics.csv", keep_default_na=False)
        per_class = pd.read_csv(ROOT / "results/per_class.csv", keep_default_na=False)
        confusions = pd.read_csv(ROOT / "results/confusion_counts.csv", keep_default_na=False)
        check("Prediction outputs are complete and response-text-free", len(predictions) == 112656 and summary["prediction_rows"] == len(predictions) and not ({"response", "full_prompt", "prompt_question"} & set(predictions.columns)))
        check("No row is predicted twice in an experiment", not predictions.duplicated(["experiment_id", "row_index"]).any())
        seen_plans = set()
        recomputed_metrics = 0
        max_error = 0.0
        for experiment, group in predictions.groupby("experiment_id", sort=False):
            descriptors = group[["population", "split_strategy", "representation", "target"]].drop_duplicates()
            check(f"{experiment}: unambiguous experiment identifiers", len(descriptors) == 1)
            population, strategy, representation, target = descriptors.iloc[0]
            plan = (population, strategy, representation, target)
            seen_plans.add(plan)
            check(f"{experiment}: experiment label matches its design", experiment == f"{population}__{strategy}__{representation}__{target}")
            column = {"model": "model_name", "company": "company"}[target]
            labels = sorted(data[column].unique())
            check(f"{experiment}: truth and identifiers match canonical observations", np.array_equal(group.true_label.to_numpy(), data.loc[group.row_index, column].to_numpy()) and np.array_equal(group.cell_id.to_numpy(), data.loc[group.row_index, "cell_id"].to_numpy()) and np.array_equal(group.y_group_id.to_numpy(), data.loc[group.row_index, "y_group_id"].to_numpy()))
            correct = group.correct.astype(str).str.casefold().map({"true": True, "false": False})
            check(f"{experiment}: saved correctness and closed-set labels are valid", correct.notna().all() and np.array_equal(correct, group.true_label.eq(group.predicted_label)) and set(group.predicted_label) <= set(labels))
            expected_rows = set(data.index if population == "full" else data.index[~data.cell_id.isin(FLAGGED_CELLS)])
            check(f"{experiment}: pooled test coverage is exact", set(group.row_index) == expected_rows)
            grouped = list(group.groupby("fold", sort=False)) + [("pooled", group)]
            for fold, fold_rows in grouped:
                if fold != "pooled":
                    expected_split = split_index[(population, strategy, fold)]
                    check(f"{experiment}/{fold}: predictions come only from that held-out fold", set(fold_rows.row_index) == expected_split["test"])
                measured, matrix, precision, recall, f1, support = independently_score(fold_rows.true_label, fold_rows.predicted_label, labels)
                saved = metrics.loc[metrics.experiment_id.eq(experiment) & metrics.fold.eq(fold)]
                check(f"{experiment}/{fold}: one matching metric row and sample count", len(saved) == 1 and int(saved.n_test.iloc[0]) == len(fold_rows))
                if fold != "pooled":
                    check(f"{experiment}/{fold}: training count agrees with manifest", float(saved.n_train.iloc[0]) == len(expected_split["train"]))
                for name, value in measured.items():
                    error = abs(float(saved[name].iloc[0]) - value)
                    max_error = max(max_error, error)
                    check(f"{experiment}/{fold}: independently recomputed {name}", error < 1e-12)
                recomputed_metrics += 1
                if fold == "pooled":
                    by_class = per_class.loc[per_class.experiment_id.eq(experiment)].set_index("label").reindex(labels)
                    check(f"{experiment}: per-class metrics and supports agree", len(by_class) == len(labels) and np.allclose(by_class.precision, precision, rtol=0, atol=1e-12) and np.allclose(by_class.recall, recall, rtol=0, atol=1e-12) and np.allclose(by_class.f1, f1, rtol=0, atol=1e-12) and np.array_equal(by_class.support, support))
                    cm = confusions.loc[confusions.experiment_id.eq(experiment)].pivot(index="true_label", columns="predicted_label", values="count").reindex(index=labels, columns=labels).to_numpy()
                    check(f"{experiment}: every confusion-matrix count agrees", np.array_equal(cm, matrix))
            if representation != "dummy":
                margins = group[["top1_margin", "top2_margin", "top3_margin"]].apply(pd.to_numeric, errors="raise").to_numpy()
                check(f"{experiment}: finite descending top-three margins and distinct labels", np.isfinite(margins).all() and (margins[:, 0] >= margins[:, 1] - 1e-12).all() and (margins[:, 1] >= margins[:, 2] - 1e-12).all() and group.predicted_label.ne(group.top2_label).all() and group.predicted_label.ne(group.top3_label).all() and group.top2_label.ne(group.top3_label).all() and set(group.top2_label) <= set(labels) and set(group.top3_label) <= set(labels))
        check("Exact fixed experiment plan, metric and class-table coverage", seen_plans == EXPECTED_PLANS and len(metrics) == 88 and recomputed_metrics == 88 and len(per_class) == 144 and len(confusions) == 1440)
        check("Source CSV is unchanged after all validation", hashlib.sha256(args.data.read_bytes()).hexdigest() == before)
        if args.require_dashboard:
            path = ROOT / "dashboard/index.html"
            check("Standalone dashboard exists", path.is_file() and path.stat().st_size > 10000 and "<html" in path.read_text().lower())
        result = {"status": "pass", "checks_passed": len(checks), "data_sha256": before,
                  "split_folds_verified": split_count, "prediction_rows_verified": len(predictions),
                  "metric_rows_independently_recomputed": recomputed_metrics, "maximum_absolute_metric_error": max_error,
                  "research_models_retrained": False, "checks": checks}
    except Exception as error:
        result = {"status": "fail", "checks_passed": sum(item["passed"] for item in checks), "error": f"{type(error).__name__}: {error}", "checks": checks}
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "split_folds_verified", "prediction_rows_verified", "metric_rows_independently_recomputed", "maximum_absolute_metric_error", "research_models_retrained"]}, indent=2))


if __name__ == "__main__":
    main()
