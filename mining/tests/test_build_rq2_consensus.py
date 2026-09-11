from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq2_validation"))

from build_rq2_consensus import build_consensus  # noqa: E402


def _labels(votes):
    rows = []
    for number, (present, primary, types) in enumerate(votes, 1):
        rows.append({
            "repo_id": number // 2,
            "number": number,
            "sample_arm": "agentic" if number % 2 else "human_candidate",
            "classification_status": "classified",
            "validation_present": present,
            "validation_types": types,
            "primary_validation_type": primary,
            "evidence_sources": ["description"] if present else [],
            "metrics": [],
            "evidence_quotes": ["measured"] if present else [],
            "input_row_sha256": f"input-{number}",
            "prompt_sha256": f"prompt-{number}",
            "model_provenance": "retained",
        })
    return pd.DataFrame(rows)


def _build(gpt_votes, gemini_votes, qwen_votes):
    votes = {"gpt": _labels(gpt_votes), "gemini": _labels(gemini_votes), "qwen": _labels(qwen_votes)}
    sample = votes["gpt"][["repo_id", "number", "sample_arm"]]
    return build_consensus(votes, sample, enforce_official_controls=False)


ABSENT = (False, "none", [])
BENCHMARK = (True, "benchmark", ["benchmark"])
PROFILING = (True, "profiling", ["profiling"])
STATIC = (True, "static-reasoning", ["static-reasoning"])


@pytest.mark.parametrize(
    ("votes", "present", "coalition"),
    [
        ([BENCHMARK, BENCHMARK, BENCHMARK], True, "gpt+gemini+qwen"),
        ([BENCHMARK, BENCHMARK, ABSENT], True, "gpt+gemini"),
        ([BENCHMARK, ABSENT, BENCHMARK], True, "gpt+qwen"),
        ([ABSENT, BENCHMARK, BENCHMARK], True, "gemini+qwen"),
        ([ABSENT, ABSENT, BENCHMARK], False, "gpt+gemini"),
    ],
)
def test_binary_majority_and_all_coalitions(votes, present, coalition):
    result, _ = _build([votes[0]], [votes[1]], [votes[2]])
    row = result.iloc[0]
    assert bool(row["consensus_validation_present"]) is present
    assert row["stage1_coalition"] == coalition
    if present:
        assert row["stage2_coalition"] == coalition
    assert row["gpt_model_provenance"] == "retained"


def test_stage2_canonicalizes_order_and_votes_atomic_tuple():
    first = (True, "benchmark", ["benchmark", "profiling"])
    second = (True, "benchmark", ["profiling", "benchmark"])
    result, _ = _build([first], [second], [PROFILING])
    row = result.iloc[0]
    assert row["consensus_status"] == "exact_majority"
    assert row["stage2_coalition"] == "gpt+gemini"
    assert row["consensus_validation_types"] == ["benchmark", "profiling"]


def test_positive_without_tuple_majority_is_unresolved_not_synthetic_label():
    # Components have majorities, but no two models supplied the synthesized combination.
    mixed = (True, "benchmark", ["benchmark", "profiling"])
    result, summary = _build([BENCHMARK], [mixed], [PROFILING])
    row = result.iloc[0]
    assert row["consensus_status"] == "unresolved"
    assert row["consensus_primary_validation_type"] is None
    assert row["consensus_validation_types"] == []
    assert not row["included_in_stage2_analysis"]
    assert summary["unresolved"] == 1


def test_negative_majority_emits_canonical_absent_label():
    result, _ = _build([ABSENT], [BENCHMARK], [ABSENT])
    row = result.iloc[0]
    assert row["consensus_primary_validation_type"] == "none"
    assert row["consensus_validation_types"] == []
    assert row["inclusion_reason"] == "stage1_consensus_absent"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("classification_status", "error", "must be classified"),
        ("prompt_sha256", "different", "different input_row_sha256 or prompt_sha256"),
        ("primary_validation_type", "unknown", "unknown primary"),
        ("validation_types", [], "positive label violates"),
    ],
)
def test_rejects_errors_hash_mismatch_and_invalid_semantics(field, value, message):
    frames = {model: _labels([BENCHMARK]) for model in ("gpt", "gemini", "qwen")}
    frames["gpt"].at[0, field] = value
    with pytest.raises(ValueError, match=message):
        build_consensus(frames, frames["gemini"][["repo_id", "number", "sample_arm"]], enforce_official_controls=False)


def test_rejects_duplicate_or_wrong_identity_and_arm():
    frames = {model: _labels([BENCHMARK]) for model in ("gpt", "gemini", "qwen")}
    sample = frames["gpt"][["repo_id", "number", "sample_arm"]]
    frames["qwen"].loc[0, "sample_arm"] = "human_candidate"
    with pytest.raises(ValueError, match="identities and arms"):
        build_consensus(frames, sample, enforce_official_controls=False)
    duplicate = pd.concat([sample, sample], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate PR identities"):
        build_consensus({model: _labels([BENCHMARK]) for model in frames}, duplicate, enforce_official_controls=False)
