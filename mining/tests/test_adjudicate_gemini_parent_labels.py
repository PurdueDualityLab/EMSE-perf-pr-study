from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq1_optimization_patterns"))

from adjudicate_gemini_parent_labels import adjudicate  # noqa: E402


def test_adjudicate_preserves_sub_pattern_and_uses_unique_catalog_parent(tmp_path):
    labels_path = tmp_path / "labels.parquet"
    catalog_path = tmp_path / "catalog.csv"
    provider_dir = tmp_path / ".provider"
    audit_path = tmp_path / "audit.csv"
    provider_dir.mkdir()
    pd.DataFrame(
        [{"key": "1:2", "repo_id": 1, "number": 2, "classification_status": "error", "high_level_pattern": None, "sub_pattern": None, "explanation": None, "optimization_comparison": None, "error": "Response label is not in the catalog."}]
    ).to_parquet(labels_path, index=False)
    pd.DataFrame(
        [{"High-level Pattern": "Correct Parent", "Sub pattern": "Child"}]
    ).to_csv(catalog_path, index=False)
    (provider_dir / "output-0001-attempt-01.jsonl").write_text(
        '{"key":"1:2","response":{"candidates":[{"content":{"parts":[{"text":"{\\"explanation\\":\\"why\\",\\"optimization_comparison\\":\\"before/after\\",\\"high_level_pattern\\":\\"Wrong Parent\\",\\"sub_pattern\\":\\"Child\\"}"}]}}]}}\n'
    )

    result = adjudicate(labels_path, catalog_path, provider_dir, audit_path)

    assert result.loc[0, "high_level_pattern"] == "Correct Parent"
    assert result.loc[0, "sub_pattern"] == "Child"
    assert result.loc[0, "classification_status"] == "classified"
    audit = pd.read_csv(audit_path)
    assert audit.loc[0, "original_high_level_pattern"] == "Wrong Parent"
    assert audit.loc[0, "corrected_high_level_pattern"] == "Correct Parent"
