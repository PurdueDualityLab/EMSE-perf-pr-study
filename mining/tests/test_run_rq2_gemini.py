import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq2_validation"))

from run_rq2_gemini import _parse_outputs, request_for, split_jsonl_lines  # noqa: E402


def test_internal_jobs_are_deterministic_disjoint_and_complete():
    lines = [f'{{"key": "{number}"}}\n' for number in range(5)]

    jobs = split_jsonl_lines(lines, max_requests=2, max_bytes=1_000)

    assert [len(job) for job in jobs] == [2, 2, 1]
    assert [line for job in jobs for line in job] == lines


def test_internal_jobs_honor_estimated_token_budget():
    lines = ["x" * 40 + "\n" for _ in range(5)]

    jobs = split_jsonl_lines(
        lines, max_requests=100, max_bytes=10_000, max_estimated_tokens=20
    )

    assert [len(job) for job in jobs] == [1, 1, 1, 1, 1]


def test_request_uses_shared_semantic_schema():
    request = request_for("Classify this PR")
    config = request["generation_config"]

    assert config["temperature"] == 0
    assert config["thinking_config"] == {"thinking_level": "MEDIUM"}
    assert config["max_output_tokens"] == 4096
    assert config["response_json_schema"]["additionalProperties"] is False
    assert "validation_types" in config["response_json_schema"]["required"]
    json.dumps(request)


def test_missing_output_becomes_explicit_error(tmp_path):
    pd.DataFrame([{"repo_id": 1, "number": 2, "key": "1:2"}]).to_parquet(
        tmp_path / "batch_manifest.parquet", index=False
    )
    (tmp_path / "prepare_metadata.json").write_text(json.dumps({"model": "gemini"}))

    result = _parse_outputs(tmp_path, [], {"1:2"})

    assert result["classification_status"].tolist() == ["error"]
    assert result["key"].tolist() == ["1:2"]


def test_non_stop_response_becomes_error(tmp_path):
    pd.DataFrame([{"repo_id": 1, "number": 2, "key": "1:2"}]).to_parquet(
        tmp_path / "batch_manifest.parquet", index=False
    )
    (tmp_path / "prepare_metadata.json").write_text(json.dumps({"model": "gemini"}))
    output = json.dumps({"key": "1:2", "response": {"candidates": [{"finishReason": "SAFETY"}]}})

    result = _parse_outputs(tmp_path, [output], {"1:2"})

    assert result["classification_status"].tolist() == ["error"]
    assert "SAFETY" in result.iloc[0]["error"]
