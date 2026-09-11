from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq1_optimization_patterns"))

from build_rq1_consensus import build_consensus  # noqa: E402


CATALOG = pd.DataFrame(
    {
        "High-level Pattern": ["A", "A", "B", "C"],
        "Sub pattern": ["a1", "a2", "b1", "c1"],
    }
)


def _sample(rows=6):
    return pd.DataFrame(
        [
            {
                "repo_id": number,
                "number": number * 10,
                "repo_full_name": f"org/repo-{number}",
                "sample_arm": "agentic" if number % 2 else "human_candidate",
            }
            for number in range(1, rows + 1)
        ]
    )


def _labels(sample, votes):
    rows = []
    for row, label in zip(sample.to_dict("records"), votes, strict=True):
        rows.append(
            {
                "repo_id": row["repo_id"],
                "number": row["number"],
                "sample_arm": row["sample_arm"],
                "classification_status": "classified",
                "input_row_sha256": f"input-{row['repo_id']}",
                "prompt_sha256": f"prompt-{row['repo_id']}",
                "prompt_version": "v2",
                "high_level_pattern": label[0],
                "sub_pattern": label[1],
            }
        )
    return pd.DataFrame(rows)


def test_atomic_consensus_covers_all_coalitions_and_unresolved_cases():
    sample = _sample()
    gpt_votes = [("A", "a1"), ("A", "a1"), ("A", "a1"), ("A", "a1"), ("A", "a1"), ("A", "a1")]
    gemini_votes = [("A", "a1"), ("A", "a1"), ("B", "b1"), ("B", "b1"), ("B", "b1"), ("A", "a2")]
    qwen_votes = [("A", "a1"), ("B", "b1"), ("A", "a1"), ("B", "b1"), ("C", "c1"), ("B", "b1")]

    qwen = _labels(sample, qwen_votes)
    qwen["prompt_version"] = "v1-whitespace-variant"
    qwen["prompt_sha256"] = [f"qwen-prompt-{number}" for number in range(len(qwen))]
    result, summary = build_consensus(
        _labels(sample, gpt_votes), _labels(sample, gemini_votes), qwen, sample, CATALOG
    )

    assert result["consensus_status"].tolist() == [
        "unanimous", "majority_2_of_3", "majority_2_of_3", "majority_2_of_3",
        "no_2_of_3_consensus", "no_2_of_3_consensus",
    ]
    assert result["consensus_coalition"].tolist() == [
        "gpt+gemini+qwen", "gpt+gemini", "gpt+qwen", "gemini+qwen", "", ""
    ]
    assert result["consensus_vote_count"].tolist() == [3, 2, 2, 2, 1, 1]
    assert result.loc[5, "included_in_analysis"] == False  # noqa: E712
    assert pd.isna(result.loc[5, "consensus_high_level_pattern"])
    assert summary["included_rows"] == 4
    assert summary["prompt_compatibility"]["exact_prompt_equality_claimed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(classification_status="error"), "non-classified"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate PR identities"),
        (lambda frame: frame.assign(input_row_sha256="different"), "different input_row_sha256"),
        (lambda frame: frame.assign(high_level_pattern="wrong"), "outside the catalog hierarchy"),
    ],
)
def test_consensus_rejects_invalid_model_inputs(mutation, message):
    sample = _sample(2)
    base = _labels(sample, [("A", "a1"), ("B", "b1")])
    gpt, gemini, qwen = base.copy(), base.copy(), base.copy()
    qwen = mutation(qwen)

    with pytest.raises(ValueError, match=message):
        build_consensus(gpt, gemini, qwen, sample, CATALOG)


def test_consensus_rejects_identity_or_arm_mismatch():
    sample = _sample(2)
    base = _labels(sample, [("A", "a1"), ("B", "b1")])
    changed = base.copy()
    changed.loc[0, "sample_arm"] = "human_candidate"

    with pytest.raises(ValueError, match="official sample IDs and arms"):
        build_consensus(base, changed, base, sample, CATALOG)


def test_consensus_rejects_gpt_gemini_prompt_mismatch():
    sample = _sample(2)
    base = _labels(sample, [("A", "a1"), ("B", "b1")])
    gemini = base.copy()
    gemini.loc[0, "prompt_sha256"] = "different"

    with pytest.raises(ValueError, match="GPT and Gemini use different prompt_sha256"):
        build_consensus(base, gemini, base, sample, CATALOG)


def test_consensus_rejects_invalid_catalog_hierarchy():
    sample = _sample(2)
    base = _labels(sample, [("A", "a1"), ("B", "b1")])
    invalid_catalog = pd.DataFrame({"High-level Pattern": ["A"], "Sub pattern": [None]})

    with pytest.raises(ValueError, match="missing labels"):
        build_consensus(base, base, base, sample, invalid_catalog)
