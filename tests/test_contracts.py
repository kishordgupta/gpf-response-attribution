"""Small, meaningful checks; no full corpus model training or network calls."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpf_attribution.data import DATA_SHA256, MODEL_TO_COMPANY, EXCLUDED_CELLS, load_dataset
from gpf_attribution.text import CLEANING_VERSION, StyleFeatures, clean_response, mask_model_names, normalized_response_hash
from gpf_attribution.estimators import make_classifier
from gpf_attribution.inference import predict_text
from gpf_attribution.training import build_splits


class CleaningContracts(unittest.TestCase):
    def test_ui_wrapper_invariance_and_idempotence(self):
        substantive = "Reported facts: costs rose. The implications remain uncertain."
        wrapped = "  Gemini said\r\n\r\nResponse 7:\n\n" + substantive
        self.assertEqual(clean_response(wrapped), substantive)
        self.assertEqual(clean_response(clean_response(wrapped)), substantive)
        self.assertEqual(normalized_response_hash(wrapped), normalized_response_hash(substantive))

    def test_substantive_inline_names_are_not_silently_removed(self):
        text = "The writer says Gemini is a product name, not proof of authorship."
        self.assertEqual(clean_response(text), text)
        self.assertEqual(clean_response("Gemini said something unusual today."), "Gemini said something unusual today.")

    def test_literal_name_mask_does_not_erase_substrings(self):
        text = "Google and Claude discuss metabolism and metadata."
        masked = mask_model_names(text)
        self.assertNotIn("Google", masked)
        self.assertNotIn("Claude", masked)
        self.assertIn("metabolism", masked)
        self.assertIn("metadata", masked)

    def test_duplicate_hash_normalizes_unicode_and_case(self):
        self.assertEqual(normalized_response_hash("Café and ＡＢＣ"), normalized_response_hash("CAFÉ and ABC"))

    def test_style_features_are_finite_for_edge_inputs(self):
        transformer = StyleFeatures()
        features = transformer.fit_transform(["", "...", "Simple words. Simple words."])
        self.assertEqual(features.shape[1], len(transformer.get_feature_names_out()))
        self.assertTrue(np.isfinite(features).all())


class InferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Three classes preserve the deployed 6/12-class margin shape while
        # keeping this regression check independent of the research corpus.
        texts = ["orchard apple fruit sweet", "apple orchard fruit fresh", "apple sweet orchard fruit",
                 "ocean ship sea waves", "sea ocean ship tide", "ship ocean waves sea",
                 "engine wheel road motor", "motor road wheel engine", "road engine motor vehicle"]
        labels = ["fruit"] * 3 + ["sea"] * 3 + ["road"] * 3
        cls.estimator = make_classifier("word_svm")
        cls.estimator.fit(texts, labels)

    def artifact(self, path, **changes):
        data = {"estimator": self.estimator, "target": "model", "classes": self.estimator.classes_.tolist(),
                "cleaning_version": CLEANING_VERSION, "data_sha256": DATA_SHA256, "masked_names": False}
        data.update(changes)
        joblib.dump(data, path)

    def test_vocabulary_is_not_fitted_on_prediction_text(self):
        vectorizer = self.estimator.named_steps["features"]
        before = dict(vectorizer.vocabulary_)
        self.estimator.predict(["unseenleakagetoken orchard apple"])
        self.assertEqual(before, vectorizer.vocabulary_)
        self.assertNotIn("unseenleakagetoken", vectorizer.vocabulary_)

    def test_persisted_inference_has_same_header_cleaning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.joblib"
            self.artifact(path)
            plain = predict_text(path, "orchard apple fruit", top_k=3)
            wrapped = predict_text(path, "Gemini said\n\norchard apple fruit", top_k=3)
            self.assertEqual(plain, wrapped)
            self.assertEqual(plain["predicted_label"], "fruit")
            self.assertEqual(len(plain["top_predictions"]), 3)
            margins = [item["margin"] for item in plain["top_predictions"]]
            self.assertEqual(margins, sorted(margins, reverse=True))
            self.assertIn("not probability", plain["score_type"])
            self.assertIn("Closed set", plain["scope"])

    def test_empty_substance_and_wrong_cleaning_version_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.joblib"
            self.artifact(path)
            for text in ["", " \n", "Gemini said\n\n"]:
                with self.assertRaises(ValueError):
                    predict_text(path, text)
            self.artifact(path, cleaning_version="incompatible-fixture")
            with self.assertRaises(ValueError):
                predict_text(path, "orchard apple")


class CanonicalDataContracts(unittest.TestCase):
    def test_canonical_read_and_company_mapping_preserve_input(self):
        path = ROOT / "data/responses.csv"
        if not path.exists():
            self.skipTest("Run scripts/download_data.py before the full release checks.")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        data = load_dataset(path)
        self.assertEqual(before, DATA_SHA256)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(len(data), 7056)
        self.assertEqual(data.groupby("model_name").size().unique().tolist(), [588])
        self.assertEqual(data.company.value_counts()["Google"], 1764)
        self.assertEqual(data.company.value_counts()["xAI"], 588)
        self.assertEqual(set(MODEL_TO_COMPANY.values()), {"OpenAI", "Anthropic", "Meta", "Google", "Mistral", "xAI"})
        self.assertEqual(data.loc[~data.cell_id.isin(EXCLUDED_CELLS)].shape[0], 6996)

    def test_altered_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("model_name,response\nfictional,answer\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_dataset(path)


class SplitContracts(unittest.TestCase):
    def test_group_isolation_and_duplicate_purge(self):
        path = ROOT / "data/responses.csv"
        if not path.exists():
            self.skipTest("Canonical data needed for the split-contract fixture.")
        data = load_dataset(path)
        identities = sorted(data.y_group_id.unique())[:2]
        small = data.loc[data.y_group_id.isin(identities)].copy()
        small["normalized_hash"] = small.response.map(normalized_response_hash)
        first = small.index[small.y_group_id.eq(identities[0])][0]
        second = small.index[small.y_group_id.eq(identities[1])][0]
        # Synthetic collision tests defensive purging without altering responses
        # or the canonical file. Train/test must never share this derived hash.
        small.loc[second, "normalized_hash"] = small.loc[first, "normalized_hash"]
        splits = build_splits(small, "full", "identity", [[identities[0]], [identities[1]]])
        self.assertEqual(len(splits), 2)
        for fold, train, test, purged, audit in splits:
            self.assertEqual(len(purged), 1)
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(small.loc[train, "cell_id"]) & set(small.loc[test, "cell_id"]))
            self.assertFalse(set(small.loc[train, "y_group_id"]) & set(small.loc[test, "y_group_id"]))
            self.assertFalse(set(small.loc[train, "normalized_hash"]) & set(small.loc[test, "normalized_hash"]))
            self.assertEqual(set(small.loc[train, "model_name"]), set(MODEL_TO_COMPANY))
            self.assertEqual(audit["purged_train_rows"], 1)


if __name__ == "__main__":
    unittest.main()
