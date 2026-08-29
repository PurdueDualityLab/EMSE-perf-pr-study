from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq2_validation"))

from analyze_rq2_agreement import analyze  # noqa: E402


def test_analyze_accepts_parquet_array_validation_types():
    gpt = _labels()
    gemini = _labels()
    gpt["validation_types"] = gpt["validation_types"].map(np.array)
    gemini["validation_types"] = gemini["validation_types"].map(np.array)

    comparison, summary = analyze(gpt, gemini)

    assert len(comparison) == 4
    assert summary["overall"]["validation_type_set"]["matches"] == 4


def test_analyze_reports_presence_primary_and_multilabel_agreement():
    gpt = _labels()
    gemini = gpt.copy()
    gemini.at[2, "validation_types"] = ["benchmark", "static-reasoning"]

    comparison, summary = analyze(gpt, gemini)

    assert len(comparison) == 4
    assert summary["overall"]["validation_present"]["matches"] == 4
    assert summary["overall"]["primary_validation_type"]["matches"] == 4
    assert summary["overall"]["validation_type_set"]["matches"] == 3
    assert summary["overall"]["per_validation_type"]["static-reasoning"]["matches"] == 3


def test_analyze_rejects_incomplete_official_sample():
    labels = _labels()
    expected = pd.concat(
        [labels[["repo_id", "number"]], pd.DataFrame([{"repo_id": 5, "number": 5}])],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="expected sample identities"):
        analyze(labels, labels.copy(), expected)


def test_analyze_rejects_errors():
    gpt = _labels()
    gemini = _labels()
    gemini.loc[0, "classification_status"] = "error"

    with pytest.raises(ValueError, match="All model errors must be resolved"):
        analyze(gpt, gemini)


def test_primary_type_disagreement_requires_adjudication():
    gpt = _labels()
    gemini = _labels()
    gpt.at[0, "validation_types"] = ["benchmark", "static-reasoning"]
    gemini.at[0, "validation_types"] = ["benchmark", "static-reasoning"]
    gemini.at[0, "primary_validation_type"] = "static-reasoning"

    comparison, _ = analyze(gpt, gemini)

    assert not comparison.iloc[0]["full_match"]


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repo_id": number,
                "number": number,
                "sample_arm": "agentic" if number <= 2 else "human_candidate",
                "html_url": f"https://example.test/{number}",
                "evidence_status": "complete",
                "evidence_complete": True,
                "classification_status": "classified",
                "validation_present": True,
                "validation_types": ["benchmark"],
                "primary_validation_type": "benchmark",
                "evidence_sources": ["description"],
                "metrics": ["latency"],
                "evidence_quotes": ["latency improved"],
                "validation_description": "Reported benchmark.",
                "prompt_version": "rq2-performance-validation",
                "study_contract_sha256": "study",
                "provider_config_sha256": "provider",
                "input_row_sha256": f"input-{number}",
                "prompt_sha256": f"prompt-{number}",
            }
            for number in range(1, 5)
        ]
    )
