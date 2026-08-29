from pathlib import Path
import sys

import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq2_validation"))

from run_rq2 import (  # noqa: E402
    ValidationLabel,
    _group_text,
    prepare_analysis_sample,
    prompt_for,
    response_format,
)


def test_response_format_uses_strict_validation_schema():
    value = response_format()["format"]

    assert value["strict"] is True
    assert value["schema"]["additionalProperties"] is False
    assert set(value["schema"]["required"]) == {
        "validation_present",
        "validation_types",
        "primary_validation_type",
        "evidence_sources",
        "metrics",
        "evidence_quotes",
        "validation_description",
    }


def test_validation_label_enforces_absence_invariants():
    with pytest.raises(ValidationError, match="absent validation requires no validation types"):
        ValidationLabel(
            validation_present=False,
            validation_types=["benchmark"],
            primary_validation_type="benchmark",
            evidence_sources=[],
            metrics=[],
            evidence_quotes=[],
            validation_description="No evidence.",
        )


def test_validation_label_requires_quotes_for_present_validation():
    with pytest.raises(ValidationError, match="requires sources and quotes"):
        ValidationLabel(
            validation_present=True,
            validation_types=["benchmark"],
            primary_validation_type="benchmark",
            evidence_sources=["description"],
            metrics=["latency"],
            evidence_quotes=[],
            validation_description="Benchmark reported.",
        )


def test_prompt_disallows_ci_name_inference_and_supports_multiple_types():
    prompt = prompt_for({"title": "Improve speed", "ci": "name=benchmark | conclusion=success", "evidence_complete": True})

    assert "Categories are non-exclusive" in prompt
    assert "workflow/check name contains words such as benchmark" in prompt
    assert "CI success alone establishes correctness testing" in prompt


def test_group_text_orders_comments_deterministically():
    frame = pd.DataFrame(
        [
            {"repo_id": 1, "number": 2, "comment_id": 2, "created_at": "2026-02-01", "body": "second"},
            {"repo_id": 1, "number": 2, "comment_id": 1, "created_at": "2026-01-01", "body": "first"},
        ]
    )

    result = _group_text(frame, columns=["created_at", "body"], order=["created_at", "comment_id"])

    assert result.iloc[0]["text"].index("first") < result.iloc[0]["text"].index("second")


def test_prepare_analysis_sample_excludes_incomplete_row_and_weekly_counterpart(tmp_path):
    sample_path = tmp_path / "source.parquet"
    evidence_dir = tmp_path / "evidence"
    output_dir = tmp_path / "rq2_sample"
    evidence_dir.mkdir()
    rows = []
    for arm in ("agentic", "human_candidate"):
        for number, selection_hash in ((1, "a"), (2, "z")):
            rows.append(
                {
                    "repo_id": 1 if arm == "agentic" else 2,
                    "number": number,
                    "sample_arm": arm,
                    "sampling_stratum": "2026-W01",
                    "selection_hash": selection_hash,
                    "created_at": f"2026-01-0{number}T00:00:00Z",
                }
            )
    sample = pd.DataFrame(rows)
    sample.to_parquet(sample_path, index=False)
    status = sample[["repo_id", "number"]].copy()
    for column in (
        "pull_request_files_status", "commits_status", "issue_comments_status",
        "review_comments_status", "reviews_status",
        "workflow_runs_status", "check_runs_status",
    ):
        status[column] = "complete"
    status["status"] = "complete"
    status.loc[
        (status["repo_id"] == 2) & (status["number"] == 1),
        [column for column in status.columns if column.endswith("_status")],
    ] = "not_requested"
    status.loc[(status["repo_id"] == 2) & (status["number"] == 1), "status"] = "not_found"
    status.to_parquet(evidence_dir / "collection_status.parquet", index=False)

    result_path = prepare_analysis_sample(sample_path, evidence_dir, output_dir)

    result = pd.read_parquet(result_path)
    exclusions = pd.read_parquet(output_dir / "evidence_exclusions.parquet")
    assert set(map(tuple, result[["repo_id", "number"]].to_numpy())) == {(1, 1), (2, 2)}
    assert set(map(tuple, exclusions[["repo_id", "number"]].to_numpy())) == {(2, 1), (1, 2)}
