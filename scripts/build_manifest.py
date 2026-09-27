#!/usr/bin/env python3
"""Hash the explicit public release allowlist; never include local data or models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    ".github/workflows/validate.yml", ".gitignore", ".python-version",
    "CITATION.md", "LICENSE", "README.md", "RIGHTS.md", "requirements.txt",
    "artifacts/metadata.json", "audits/README.md", "audits/audit_confounds.py",
    "audits/confounds.json", "audits/release_validation.json", "audits/row_flags.csv",
    "config/experiment.json", "dashboard/index.html", "dashboard/template.html",
    "docs/MODEL_CARD.md", "docs/REPORT.md", "gpf_attribution/__init__.py",
    "gpf_attribution/data.py", "gpf_attribution/estimators.py", "gpf_attribution/inference.py",
    "gpf_attribution/text.py", "gpf_attribution/training.py",
    "notebooks/GPF_Response_Attribution.ipynb", "results/cleaning_audit.json",
    "results/confusion_counts.csv", "results/demo_top_features.csv", "results/fit_warnings.json",
    "results/identity_fold_assignments.json", "results/metrics.csv", "results/per_class.csv",
    "results/predictions.csv.gz", "results/split_audits.json", "results/split_manifest.csv.gz",
    "results/status.json", "results/summary.json", "results/timings.json",
    "scripts/build_dashboard.py", "scripts/build_manifest.py", "scripts/download_data.py",
    "scripts/predict.py", "scripts/train.py", "scripts/validate_release.py", "tests/test_contracts.py",
]


def main():
    files = []
    for name in sorted(PUBLIC_FILES):
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked publication file: {name}")
        content = path.read_bytes()
        files.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {
        "schema_version": 1,
        "description": "Exact publication allowlist; the manifest itself is not recursively hashed.",
        "dataset_sha256": "c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da",
        "excluded": ["data/responses.csv", "data/responses.csv.gz", "artifacts/*.joblib", "logs", "caches", "local validation staging"],
        "files": files,
    }
    target = ROOT / "MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": "MANIFEST.json", "listed_files": len(files), "publication_files_including_manifest": len(files) + 1}))


if __name__ == "__main__":
    main()
