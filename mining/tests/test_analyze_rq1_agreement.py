from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq1_optimization_patterns"))

from analyze_agreement import analyze, cohen_kappa  # noqa: E402


def test_cohen_kappa_known_example():
    first = pd.Series(["a", "a", "b", "b"])
    second = pd.Series(["a", "b", "b", "b"])

    assert cohen_kappa(first, second) == pytest.approx(0.5)


def test_analyze_reports_overall_and_per_arm_agreement():
    gpt = _labels()
    gemini = gpt.copy()
    gemini.loc[gemini["repo_id"].eq(3), "sub_pattern"] = "different"

    gpt["prompt_version"] = "legacy-v2"
    gemini["prompt_version"] = "legacy-v2"
    comparison, summary = analyze(gpt, gemini)

    assert len(comparison) == 4
    assert summary["overall"]["high_level_pattern"]["agreement_rate"] == 1.0
    assert summary["overall"]["sub_pattern"]["matches"] == 3
    assert summary["overall"]["hierarchical_label"]["matches"] == 3


def test_analyze_rejects_different_prompt_versions():
    base = _labels().drop(columns="prompt_version")
    gpt = base.assign(prompt_version="v1")
    gemini = base.assign(prompt_version="v2")

    with pytest.raises(ValueError, match="different prompt versions"):
        analyze(gpt, gemini)


def test_analyze_rejects_errors_instead_of_shrinking_denominator():
    gpt = _labels()
    gemini = _labels()
    gemini.loc[0, "classification_status"] = "error"

    with pytest.raises(ValueError, match="All model errors must be resolved"):
        analyze(gpt, gemini)


def test_analyze_rejects_different_identities():
    gpt = _labels()
    gemini = _labels()
    gemini.loc[0, "repo_id"] = 999999

    with pytest.raises(ValueError, match="different PR identities"):
        analyze(gpt, gemini)


def test_analyze_rejects_different_study_contracts():
    gpt = _labels()
    gemini = _labels()
    gemini["study_contract_sha256"] = "different"

    with pytest.raises(ValueError, match="different study contract hashes"):
        analyze(gpt, gemini)


def test_analyze_accepts_different_provider_configs():
    gpt = _labels()
    gemini = _labels()
    gemini["provider_config_sha256"] = "gemini-provider"

    comparison, _ = analyze(gpt, gemini)

    assert len(comparison) == 4


def test_analyze_rejects_balanced_subset_of_expected_sample():
    gpt = _labels()
    gemini = _labels()
    expected = pd.concat(
        [gpt[["repo_id", "number"]], pd.DataFrame([{"repo_id": 5, "number": 5}])],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="expected sample identities"):
        analyze(gpt, gemini, expected)


def _labels():
    return pd.DataFrame(
        [
            {
                "repo_id": number,
                "number": number,
                "sample_arm": "agentic" if number <= 2 else "human_candidate",
                "classification_status": "classified",
                "high_level_pattern": "A" if number % 2 else "B",
                "sub_pattern": "a" if number % 2 else "b",
                "prompt_version": "legacy-v2",
                "study_contract_sha256": "study-v1",
                "provider_config_sha256": "provider-v1",
                "input_row_sha256": f"input-{number}",
                "prompt_sha256": f"prompt-{number}",
            }
            for number in range(1, 5)
        ]
    )
